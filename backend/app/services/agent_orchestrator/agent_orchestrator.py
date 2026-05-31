from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================================================
# SCHEMAS
# =========================================================

class PlannerOutputSchema(BaseModel):
    objective: str
    complexity: str
    requires_decomposition: bool
    strategy: List[str]
    success_criteria: List[str]
    constraints: List[str]


class TaskSchema(BaseModel):
    id: str
    description: str
    required_capability: str
    matched_skill: str = ""
    worker_type: str = ""
    dependencies: List[str] = Field(default_factory=list)
    priority: str
    estimated_complexity: str


class TaskDecomposerOutputSchema(BaseModel):
    tasks: List[TaskSchema]


class CapabilityMatcherOutputSchema(BaseModel):
    tasks: List[TaskSchema]


class SupervisorOutputSchema(BaseModel):
    approved: bool = True
    execution_plan: List[dict] = Field(default_factory=list)
    notes: Optional[str] = None


class AggregatorOutputSchema(BaseModel):
    execution_plan: List[dict] = Field(default_factory=list)


# =========================================================
# STATE
# =========================================================

class OrchestratorState(TypedDict, total=False):
    user_request: str

    planner_output: Dict[str, Any]
    decomposed_tasks: Dict[str, Any]
    capability_matches: Dict[str, Any]
    supervisor_output: Dict[str, Any]
    aggregator_output: Dict[str, Any]

    final_plan: List[dict]


# =========================================================
# PARSERS (DSL → STRUCTURED DICT)
# =========================================================

def extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_list_block(block: str) -> List[str]:
    return [
        line.replace("-", "").strip()
        for line in block.split("\n")
        if line.strip().startswith("-")
    ]


def parse_planner_dsl(text: str) -> Dict[str, Any]:
    return {
        "objective": extract_tag(text, "objective"),
        "complexity": extract_tag(text, "complexity"),
        "requires_decomposition": extract_tag(text, "requires_decomposition") == "true",
        "strategy": extract_list_block(extract_tag(text, "strategy")),
        "success_criteria": extract_list_block(extract_tag(text, "success_criteria")),
        "constraints": extract_list_block(extract_tag(text, "constraints")),
    }


def parse_task_dsl(text: str) -> Dict[str, Any]:
    # expects JSON-like tasks embedded but still tolerant
    import json
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError("Invalid task decomposition output")


# =========================================================
# ORCHESTRATOR
# =========================================================

class AgentOrchestrator:

    # =========================
    # PROMPTS (DSL FOR PLANNER)
    # =========================

    PLANNER_PROMPT = """
You are a mission-level planner.

User request:
{user_request}


OUTPUT FORMAT (STRICT DSL - NO JSON):
<objective>...</objective>
<complexity>low|medium|high</complexity>
<requires_decomposition>true|false</requires_decomposition>

<strategy>
- step 1
- step 2
</strategy>

<success_criteria>
- criterion 1
- criterion 2
</success_criteria>

<constraints>
- constraint 1
</constraints>

RULES:
- No JSON allowed
- No explanation
- Only tags
"""

    # =========================
    # TASK DECOMPOSER (DSL OUTPUT)
    # =========================

    TASK_DECOMPOSER_PROMPT = """
You are a task decomposition system.

Planner input:
{planner_output}

OUTPUT FORMAT (STRICT JSON ONLY):

{
  "tasks": [
    {
      "id": "task_1",
      "description": "",
      "required_capability": "",
      "matched_skill": "",
      "worker_type": "",
      "dependencies": [],
      "priority": "high",
      "estimated_complexity": "medium"
    }
  ]
}

RULES:
- Break strategy into atomic tasks
- Do NOT assign skills or workers yet (keep blank "")
- Must be valid JSON only
"""

    # =========================
    # CAPABILITY MATCHER (JSON)
    # =========================

    CAPABILITY_MATCHER_PROMPT = """
You are a capability matching system.

Tasks:
{tasks}

Available skills:
{skills}

Available workers:
{workers}

RULES:
- Assign best matched_skill per task
- Assign worker_type per task
- If no match, keep ""

OUTPUT JSON ONLY:
{{
  "tasks": {tasks}
}}
"""

    # =========================
    # INIT
    # =========================

    def __init__(self, llm, checkpointer=None):
        self.llm = llm
        self.graph = self._build_graph(checkpointer)

    # =========================
    # PUBLIC
    # =========================

    def invoke(self, user_request: str, thread_id: str):
        state: OrchestratorState = {"user_request": user_request}

        return self.graph.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}}
        )

    # =========================
    # GRAPH
    # =========================

    def _build_graph(self, checkpointer):
        g = StateGraph(OrchestratorState)

        g.add_node("planner_node", self._planner_node)
        g.add_node("task_decomposer_node", self._task_decomposer_node)
        g.add_node("capability_matcher_node", self._capability_matcher_node)

        g.add_node("supervisor_node", self._supervisor_node)
        g.add_node("aggregator_node", self._aggregator_node)

        g.set_entry_point("planner_node")

        g.add_edge("planner_node", "task_decomposer_node")
        g.add_edge("task_decomposer_node", "capability_matcher_node")
        g.add_edge("capability_matcher_node", "supervisor_node")
        g.add_edge("supervisor_node", "aggregator_node")
        g.add_edge("aggregator_node", END)

        return g.compile(checkpointer=checkpointer)

    # =========================
    # NODES
    # =========================

    # -------------------------
    # PLANNER (DSL)
    # -------------------------

    def _planner_node(self, state: OrchestratorState):
        logger.info("planner_node")

        prompt = self.PLANNER_PROMPT.format(
            user_request=state["user_request"]
        )

        raw = self._invoke_llm(prompt)
        parsed = parse_planner_dsl(raw)

        validated = PlannerOutputSchema.model_validate(parsed)

        return {"planner_output": validated.model_dump()}

    # -------------------------
    # TASK DECOMPOSER (JSON)
    # -------------------------

    def _task_decomposer_node(self, state: OrchestratorState):
        logger.info("task_decomposer_node")

        prompt = self.TASK_DECOMPOSER_PROMPT.format(
            planner_output=state["planner_output"]
        )

        data = self._invoke_llm(prompt)

        validated = TaskDecomposerOutputSchema.model_validate(data)

        return {"decomposed_tasks": validated.model_dump()}

    # -------------------------
    # CAPABILITY MATCHER (JSON)
    # -------------------------

    def _capability_matcher_node(self, state: OrchestratorState):
        logger.info("capability_matcher_node")

        prompt = self.CAPABILITY_MATCHER_PROMPT.format(
            tasks=state["decomposed_tasks"]["tasks"],
            skills=[],
            workers=[]
        )

        data = self._invoke_llm(prompt)

        validated = CapabilityMatcherOutputSchema.model_validate(data)

        return {"capability_matches": validated.model_dump()}

    # -------------------------
    # PLACEHOLDERS
    # -------------------------

    def _supervisor_node(self, state: OrchestratorState):
        logger.info("supervisor_node (placeholder)")

        return {
            "supervisor_output": SupervisorOutputSchema().model_dump()
        }

    def _aggregator_node(self, state: OrchestratorState):
        logger.info("aggregator_node (placeholder)")

        output = AggregatorOutputSchema(execution_plan=[])

        return {
            "aggregator_output": output.model_dump(),
            "final_plan": output.execution_plan
        }

    # =========================
    # LLM WRAPPER
    # =========================

    def _invoke_llm(self, prompt: str) -> str:
        raw = self.llm.invoke(prompt)
        return getattr(raw, "content", str(raw))