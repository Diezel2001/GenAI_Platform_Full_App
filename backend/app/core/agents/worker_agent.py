from __future__ import annotations

import json
import re
import logging

from typing import (
    Any,
    Dict,
    List,
    Optional,
    TypedDict,
    Annotated,
    Literal,
)

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    create_model,
)

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from langchain_core.messages import (
    BaseMessage,
)

from app.services.skill_manager import Skill


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================================================
# SCHEMAS
# =========================================================

class PlannerSchema(BaseModel):

    thought: str

    needs_tool: bool

    done: bool

    goal: str

    final_answer: Optional[str] = None


class ObservationSchema(BaseModel):

    summary: str

    key_facts: List[str] = Field(default_factory=list)

    unresolved_questions: List[str] = Field(default_factory=list)

    important_entities: List[str] = Field(default_factory=list)

    risks_or_uncertainties: List[str] = Field(default_factory=list)

    recommended_next_actions: List[str] = Field(default_factory=list)

    success: bool

    confidence: float = Field(..., ge=0.0, le=1.0)


class FinalResponseSchema(BaseModel):

    response: str

    confidence: float = Field(..., ge=0.0, le=1.0)

    sources_used: List[str] = Field(default_factory=list)


# =========================================================
# STATE
# =========================================================

class WorkerState(TypedDict, total=False):

    messages: Annotated[List[BaseMessage], add_messages]

    task: str

    skill: Optional[Skill]

    planner: Dict[str, Any]

    action: Dict[str, Any]

    pending_tool: Dict[str, Any]

    tool_result: Any

    observations: List[dict]

    validation_error: Optional[str]

    failed_validation_count: int

    tool_call_history: List[str]

    step_count: int

    max_steps: int

    final_answer: str


# =========================================================
# TOOL RUNTIME
# =========================================================

class ToolRuntime:

    def __init__(self, tools: Dict[str, Any]):

        self.tools = tools

    def validate(self, tool_name: str, args: dict):

        if tool_name not in self.tools:
            raise ValueError(f"Tool not found: {tool_name}")

        tool = self.tools[tool_name]

        schema = tool["schema"]

        if isinstance(args, str):
            args = json.loads(args)

        validated = schema(**args)

        return validated.model_dump()

    def execute(self, tool_name: str, args: dict):

        tool = self.tools[tool_name]

        func = tool["func"]

        return func(**args)

    def tool_descriptions(
        self,
        allowed_tools: Optional[List[str]] = None,
    ) -> str:

        tools = self.tools

        if allowed_tools:
            tools = {
                k: v
                for k, v in tools.items()
                if k in allowed_tools
            }

        if not tools:
            return "No tools available."

        out = []

        for name, tool in tools.items():

            desc = tool.get(
                "description",
                "No description"
            )

            schema = tool.get("schema")

            fields = []

            if schema:

                for field_name, field_info in schema.model_fields.items():

                    fields.append(
                        f"{field_name}: {field_info.annotation}"
                    )

            fields_text = ", ".join(fields)

            out.append(
                f"- {name}\n"
                f"  Description: {desc}\n"
                f"  Arguments: {fields_text}"
            )

        return "\n\n".join(out)


# =========================================================
# WORKER AGENT
# =========================================================

class WorkerAgent:

    PLANNER_PROMPT_TEMPLATE = """
You are a planner agent.

Your job:
- determine next objective and or goal,
- determine whether a tool is needed,
- determine whether task is complete.

CONVERSATION CONTEXT:
{conversation_context}

TASK:
{task}

RECENT OBSERVATIONS:
{observations}

RULES:
- Do NOT select tools.
- Do NOT generate tool arguments.
- Think at planning level only.
- If the task requires a tool, set needs_tool=true and describe
  the goal clearly.
- If no tool available for the task or 
  the task can be answered from your own knowledge or from
  existing observations alone, set needs_tool=false, done=true,
  and write the full answer in final_answer.
- If enough information from tools has been gathered to answer
  the task, set done=true and write the full answer in
  final_answer.

OUTPUT JSON:
{{
  "thought": "string",
  "needs_tool": true,
  "done": false,
  "goal": "string",
  "final_answer": null
}}
"""

    SKILL_PLANNER_PROMPT_TEMPLATE = """
You are executing a predefined skill workflow.

TASK:
{task}

CONVERSATION CONTEXT:
{conversation_context}

SKILL:
{skill_context}

RECENT OBSERVATIONS:
{observations}

RULES:
- Follow the skill instructions carefully.
- Determine only the NEXT objective.
- Do NOT select tools.
- Do NOT generate arguments.
- If task is complete, mark done=true.

OUTPUT JSON:
{{
  "thought": "string",
  "needs_tool": true,
  "done": false,
  "goal": "string",
  "final_answer": null
}}
"""

    ACTION_PROMPT_TEMPLATE = """
You are an action selector.

Your job:
- choose ONE allowed tool,
- generate valid arguments,
- or finalize.

GOAL:
{goal}

RECENT OBSERVATIONS:
{observations}

AVAILABLE TOOLS:
{tools}

RULES:
- ONLY use listed tools.
- NEVER invent tools.
- If no tool needed choose FINAL.

OUTPUT JSON:
{{
  "thought": "string",
  "action": "TOOL" | "FINAL",
  "tool_name": "string or null",
  "tool_input": {{}},
  "final_answer": "string or null"
}}
"""

    FINALIZE_PROMPT_TEMPLATE = """
You are generating the final response.

TASK:
{task}

OBSERVATIONS:
{observations}

RULES:
- Use observations as source of truth.
- Do not invent unsupported facts.

OUTPUT JSON:
{{
  "response": "string",
  "confidence": 0.0,
  "sources_used": []
}}
"""

    def __init__(
        self,
        llm,
        tools: Dict[str, Any],
        checkpointer,
        *,
        max_steps: int = 10,
        max_validation_failures: int = 5,
    ):

        self.llm = llm

        self.runtime = ToolRuntime(tools)

        self.max_steps = max_steps

        self.max_validation_failures = max_validation_failures

        self.ActionSchema = self._build_action_schema()

        self.graph = self._build_graph(checkpointer)

    # =====================================================
    # DYNAMIC ACTION SCHEMA
    # =====================================================

    def _build_action_schema(self):

        tool_names = tuple(self.runtime.tools.keys())

        if tool_names:

            ToolLiteral = Literal.__getitem__(tool_names)

        else:

            ToolLiteral = str

        return create_model(
            "ActionSchema",
            thought=(str, ...),
            action=(Literal["TOOL", "FINAL"], ...),
            tool_name=(Optional[ToolLiteral], None),
            tool_input=(Optional[Dict[str, Any]], None),
            final_answer=(Optional[str], None),
        )

    # =====================================================
    # PUBLIC
    # =====================================================

    def invoke(
        self,
        task: str,
        thread_id: str,
        skill: Optional[Skill] = None,
    ):

        state: WorkerState = {
            "task": task,
            "skill": skill,
            "observations": [],
            "tool_call_history": [],
            "step_count": 0,
            "failed_validation_count": 0,
            "max_steps": self.max_steps,
        }

        result = self.graph.invoke(
            state,
            config={
                "configurable": {
                    "thread_id": thread_id
                }
            }
        )

        return result.get("final_answer", "")

    # =====================================================
    # GRAPH
    # =====================================================

    def _build_graph(self, checkpointer):

        g = StateGraph(WorkerState)

        g.add_node("planner", self._planner)

        g.add_node("action_selector", self._action_selector)

        g.add_node("validate", self._validate)

        g.add_node("observe_invalid", self._observe_invalid)

        g.add_node("act", self._act)

        g.add_node("observe", self._observe)

        g.add_node("finalize", self._finalize)

        g.set_entry_point("planner")

        g.add_conditional_edges(
            "planner",
            self._planner_router,
            {
                "action": "action_selector",
                "final": "finalize",
            }
        )

        g.add_conditional_edges(
            "action_selector",
            self._action_router,
            {
                "validate": "validate",
                "final": "finalize",
            }
        )

        g.add_conditional_edges(
            "validate",
            self._validate_router,
            {
                "valid": "act",
                "invalid": "observe_invalid",
            }
        )

        g.add_edge("act", "observe")

        g.add_edge("observe", "planner")

        g.add_edge("observe_invalid", "planner")

        g.add_edge("finalize", END)

        return g.compile(checkpointer=checkpointer)

    # =====================================================
    # ROUTERS
    # =====================================================

    def _planner_router(self, state: WorkerState):

        planner_dict = state["planner"]

        planner = PlannerSchema.model_validate(planner_dict)

        if planner.done:
            return "final"

        return "action"

    def _action_router(self, state: WorkerState):

        action = self.ActionSchema.model_validate(
            state["action"]
        )

        if action.action == "FINAL":
            return "final"

        return "validate"

    def _validate_router(self, state: WorkerState):

        if state.get("validation_error"):
            return "invalid"

        return "valid"

    # =====================================================
    # PLANNER
    # =====================================================

    def _planner(self, state: WorkerState):

        step_count = state.get("step_count", 0)

        if step_count >= state.get(
            "max_steps",
            self.max_steps
        ):

            logger.warning("Max steps reached.")

            return {
                "planner": PlannerSchema(
                    thought="Maximum steps reached.",
                    needs_tool=False,
                    done=True,
                    goal="Stop execution.",
                    final_answer=(
                        "Task stopped because maximum "
                        "reasoning steps were reached."
                    )
                ).model_dump()
            }

        if state.get(
            "failed_validation_count",
            0
        ) >= self.max_validation_failures:

            return {
                "planner": PlannerSchema(
                    thought="Too many validation failures.",
                    needs_tool=False,
                    done=True,
                    goal="Stop execution.",
                    final_answer=(
                        "Task stopped due to repeated "
                        "invalid tool selections."
                    )
                ).model_dump()
            }

        observations = state.get("observations", [])[-3:]

        conversation_context = self._build_conversation_context(
            state.get("messages", []),
            max_pairs=5,
        )

        skill = state.get("skill")

        if skill:

            skill_context = getattr(
                skill,
                "body",
                str(skill)
            )

            prompt = self._build_skill_planner_prompt(
                task=state["task"],
                conversation_context=conversation_context,
                observations=observations,
                skill_context=skill_context,
            )

        else:

            prompt = self._build_planner_prompt(
                task=state["task"],
                conversation_context=conversation_context,
                observations=observations,
            )

        data = self._invoke_llm_with_retry(prompt)

        planner = PlannerSchema.model_validate(data)

        logger.info(
            "Planner step=%s done=%s",
            step_count,
            planner.done
        )

        return {
            "planner": planner.model_dump(),
            "step_count": step_count + 1,
        }

    # =====================================================
    # ACTION SELECTOR
    # =====================================================

    def _action_selector(self, state: WorkerState):

        planner = PlannerSchema.model_validate(
            state["planner"]
        )

        observations = state.get("observations", [])[-3:]

        skill = state.get("skill")

        allowed_tools = None

        if skill and hasattr(skill, "allowed_tools"):

            allowed_tools = [
                t for t in skill.allowed_tools
                if t in self.runtime.tools
            ]

        tools_text = self.runtime.tool_descriptions(
            allowed_tools=allowed_tools
        )

        prompt = self._build_action_prompt(
            goal=planner.goal,
            observations=observations,
            tools=tools_text,
        )

        data = self._invoke_llm_with_retry(prompt)

        action = self.ActionSchema.model_validate(data)

        logger.info(
            "Action selected=%s",
            action.action
        )

        return {
            "action": action.model_dump()
        }

    # =====================================================
    # VALIDATE
    # =====================================================

    def _validate(self, state: WorkerState):

        action = self.ActionSchema.model_validate(
            state["action"]
        )

        try:

            clean_args = self.runtime.validate(
                action.tool_name,
                action.tool_input or {},
            )

            logger.info(
                "Validated tool=%s",
                action.tool_name
            )

            return {
                "pending_tool": {
                    "name": action.tool_name,
                    "args": clean_args,
                },
                "validation_error": None,
            }

        except Exception as e:

            logger.warning(
                "Validation failed: %s",
                str(e)
            )

            return {
                "validation_error": str(e),
                "failed_validation_count": (
                    state.get(
                        "failed_validation_count",
                        0
                    ) + 1
                )
            }

    # =====================================================
    # OBSERVE INVALID
    # =====================================================

    def _observe_invalid(self, state: WorkerState):

        observations = state.get("observations", [])

        new_obs = ObservationSchema(
            summary=(
                f"Invalid tool usage: "
                f"{state.get('validation_error')}"
            ),
            risks_or_uncertainties=[
                "Model selected unavailable or invalid tool."
            ],
            recommended_next_actions=[
                "Choose only from allowed tools."
            ],
            success=False,
            confidence=1.0,
        ).model_dump()

        observations = (observations + [new_obs])[-20:]

        return {
            "observations": observations,
            "validation_error": None,
        }

    # =====================================================
    # ACT
    # =====================================================

    def _act(self, state: WorkerState):

        tool = state["pending_tool"]

        logger.info(
            "Executing tool=%s",
            tool["name"]
        )

        try:

            result = self.runtime.execute(
                tool["name"],
                tool["args"]
            )

        except Exception as e:

            result = (
                f"TOOL_EXECUTION_ERROR: {str(e)}"
            )

        history = state.get(
            "tool_call_history",
            []
        )

        history.append(tool["name"])

        return {
            "tool_result": result,
            "tool_call_history": history,
        }

    # =====================================================
    # OBSERVE
    # =====================================================

    def _observe(self, state: WorkerState):

        tool = state["pending_tool"]

        tool_result = state.get("tool_result")

        observations = state.get("observations", [])

        summary = str(tool_result)

        truncated = False
        if len(summary) > 1000:
            summary = summary[:1000] + "... [TRUNCATED]"
            truncated = True

        new_obs = ObservationSchema(
            summary=summary,
            risks_or_uncertainties=(
                ["Tool result was truncated; full output may contain additional relevant data."]
                if truncated else []
            ),
            success="TOOL_EXECUTION_ERROR" not in summary,
            confidence=1.0,
        ).model_dump()

        new_obs["tool"] = tool["name"]

        observations = (observations + [new_obs])[-20:]

        logger.info(
            "Observation stored tool=%s",
            tool["name"]
        )

        return {
            "observations": observations
        }

    # =====================================================
    # FINALIZE
    # =====================================================

    def _finalize(self, state: WorkerState):

        planner = state.get("planner", {})

        if planner.get("final_answer"):

            return {
                "final_answer": planner["final_answer"]
            }

        prompt = self._build_finalize_prompt(
            task=state["task"],
            observations=state.get(
                "observations",
                []
            )[-5:]
        )

        data = self._invoke_llm_with_retry(prompt)

        final = FinalResponseSchema.model_validate(
            data
        )

        logger.info(
            "Finalized response confidence=%.2f",
            final.confidence
        )

        return {
            "final_answer": final.response
        }

    def _build_planner_prompt(
        self,
        task: str,
        conversation_context: str,
        observations: list,
    ) -> str:

        return self.PLANNER_PROMPT_TEMPLATE.format(
            task=task,
            conversation_context=conversation_context,
            observations=json.dumps(observations, indent=2),
        )

    def _build_skill_planner_prompt(
        self,
        task: str,
        conversation_context: str,
        observations: list,
        skill_context: str,
    ) -> str:

        return self.SKILL_PLANNER_PROMPT_TEMPLATE.format(
            task=task,
            conversation_context=conversation_context,
            observations=json.dumps(observations, indent=2),
            skill_context=skill_context,
        )

    def _build_action_prompt(
        self,
        goal: str,
        observations: list,
        tools: str,
    ) -> str:

        return self.ACTION_PROMPT_TEMPLATE.format(
            goal=goal,
            observations=json.dumps(observations, indent=2),
            tools=tools,
        )

    def _build_finalize_prompt(
        self,
        task: str,
        observations: list,
    ) -> str:

        return self.FINALIZE_PROMPT_TEMPLATE.format(
            task=task,
            observations=json.dumps(observations, indent=2),
        )

    # =====================================================
    # HELPERS
    # =====================================================

    def _extract_json(self, text: str):
        """
        Robust JSON extraction that correctly handles nested objects.
        Walks character-by-character to find the outermost JSON object,
        correctly handling nesting, strings, and escape sequences.
        """

        if not text:
            raise ValueError("Empty LLM output")

        # Try fenced block first (```json ... ```)
        fenced = re.search(
            r"```json\s*(\{.*\})\s*```",
            str(text),
            re.DOTALL,
        )

        if fenced:
            try:
                parsed = json.loads(fenced.group(1))
                return parsed
            except json.JSONDecodeError as e:
                pass

        s = str(text)
        start = s.find("{")

        if start == -1:
            raise ValueError(
                f"No JSON found in output: {text[:300]}"
            )

        depth = 0
        in_str = False
        escape = False

        for i, ch in enumerate(s[start:], start):

            if escape:
                escape = False
                continue

            if ch == "\\" and in_str:
                escape = True
                continue

            if ch == '"':
                in_str = not in_str

            if not in_str:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed = json.loads(s[start:i + 1])
                            return parsed
                        except json.JSONDecodeError as e:
                            raise

        raise ValueError(
            f"Malformed JSON in output: {text[:300]}"
        )

    def _extract_json_safe(self, text: str):

        try:

            return self._extract_json(text)

        except Exception:

            return {
                "summary": "Failed to parse JSON.",
                "success": False,
            }

    def _invoke_llm_with_retry(
        self,
        prompt: str,
        *,
        max_attempts: int = 2,
    ) -> dict:

        last_error: Exception = ValueError("No attempts made")

        for attempt in range(max_attempts):

            try:

                raw = self.llm.invoke(prompt)

                text = getattr(raw, "content", str(raw))

                result = self._extract_json(text)

                return result

            except (ValueError, json.JSONDecodeError) as e:

                last_error = e

                logger.warning(
                    "LLM parse failure attempt=%s/%s: %s",
                    attempt + 1,
                    max_attempts,
                    str(e),
                )

        raise last_error

    def _build_conversation_context(
        self,
        messages: List[BaseMessage],
        *,
        max_pairs: int = 5,
    ) -> str:

        return "No conversation history."
