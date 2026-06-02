from typing import TypedDict, Annotated, Literal, Type, TypeVar, List
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
import legacy_documents_files.workflow.prompts_v2 as prompts
from app.services.llm import Llmwrapper
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, ValidationError
import json, uuid, re
from app.tools.registry import TOOL_REGISTRY

# =============================================================================
# HELPERS
# =============================================================================

T = TypeVar("T", bound=BaseModel)
def parse_and_validate(json_string: str, model: Type[T]) -> T:
    try:
        data = json.loads(json_string)
        return model(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Invalid structured JSON: {e}") from e

def format_messages(messages):
    formatted = []
    for m in messages:
        if isinstance(m, HumanMessage):
            formatted.append(f"USER: {m.content}")
        elif isinstance(m, AIMessage):
            formatted.append(f"ASSISTANT: {m.content}")
        elif isinstance(m, ToolMessage):
            formatted.append(f"TOOL RESULT:\n{m.content}")
        else:
            formatted.append(f"UNKNOWN: {m.content}")
    return "\n".join(formatted)


def _make_human_message(content: str) -> HumanMessage:
    return HumanMessage(content=content)

def _make_ai_message(content: str) -> AIMessage:
    return AIMessage(content=content)

def build_agent_context(messages):
    last_user_idx = None

    for i in reversed(range(len(messages))):
        if isinstance(messages[i], HumanMessage):
            last_user_idx = i
            break

    if last_user_idx is None:
        return None, []

    user_msg = messages[last_user_idx].content


    return user_msg

def get_tool_descriptions():
    desc = []
    for name, tool in TOOL_REGISTRY.items():
        desc.append(f"{name}: {tool.schema.model_json_schema()}")
    return "\n".join(desc)

def format_observations(observations):
    return "\n".join(
        f"- {obs['summary']} | hint: {obs['next_hint']}"
        for obs in observations
    )

# =============================================================================
# STATE
# =============================================================================

class WorkflowState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]

    # Reprompter
    prompt: str
    task_type: str
    intent: str
    will_decompose: str

    # Decomposer
    tasks: list[str]
    current_task_index: int
    task_outputs: list[str]

    # Router
    route: str

    # Reasoner
    step_count: int
    max_steps: int
    last_thought: str
    tool_name: str
    is_final: bool

    # Tool_builder
    valid: bool

    # Actor
    tool_call_result: str

    # Observer
    observations: list[dict]

    
# =============================================================================
# LLM Output Schemas
# =============================================================================

class StructuredPrompt(BaseModel):
    intent: str
    task_type: str
    task_breakdown: str
    prompt: str

class DecomposerOutput(BaseModel):
    tasks: List[str]
    
class ReasonerOutput(BaseModel):
    thought: str
    action: str  # "TOOL" or "FINAL"
    tool_name: str | None = None
    tool_input: dict | None = None
    final_answer: str | None = None

class ObserverOutput(BaseModel):
    summary: str
    key_points: list[str]
    next_hint: str

class AggregatorOutput(BaseModel):
    final_answer: str
    
# =============================================================================
# NODES
# =============================================================================

def analyzer_node(state: WorkflowState) -> dict:
    # Formulates a structured prompt from user message and session context
    messages = state.get("messages", [])

    conversation = messages[-10:]
    latest_user_msg = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)),
        ""
    )
    response = Llmwrapper.llm.invoke(
        prompts.analyzer_p.format(
            history=format_messages(conversation),
            user_input=latest_user_msg
        )
    )

    raw_text = getattr(response, "content", str(response))
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in analyzer output: {raw_text}")
    json_str = match.group(0)
    parsed = parse_and_validate(json_str, StructuredPrompt)
    
    return {
        "prompt": parsed.prompt,
        "task_type": parsed.task_type,
        "intent": parsed.intent,
        "will_decompose": parsed.task_breakdown
    }

def direct_node(state: WorkflowState) -> dict:
    prompt = state.get("prompt", "")
    tasks = state.get("tasks", [])
    current_task_index = state.get("current_task_index", 0)

    # Determine what to execute
    if tasks:
        if current_task_index >= len(tasks):
            return {"status": "done"}
        task = tasks[current_task_index]
    else:
        task = prompt

    response = Llmwrapper.llm.invoke(
        prompts.direct_p.format(task=task)
    )
    output = str(response)

    # CASE 1: Decomposed → treat as task completion
    if tasks:
        return {
            "task_outputs": state.get("task_outputs", []) + [output],
            "current_task_index": current_task_index + 1,
            "step_count": 0
        }

    # CASE 2: No decomposition → final answer
    return {
        "messages": [AIMessage(content=output)],
        "status": "done" 
    }

def decomposer_node(state: WorkflowState) -> dict:
    prompt = state.get("prompt", "")
    goal = state.get("intent", "")

    if not goal:
        raise ValueError("Decomposer received empty goal")

    response = Llmwrapper.llm.invoke(
        prompts.decomposer_p.format(
            prompt=prompt, 
            intent=goal)
    )

    raw_text = getattr(response, "content", str(response))
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in decomposer output: {raw_text}")
    json_str = match.group(0)
    parsed = parse_and_validate(json_str, DecomposerOutput)

    return {
        "tasks": parsed.tasks,
        "current_task_index": 0
    }

def router_node(state: WorkflowState) -> dict:
    # keeps track of current task being done (task execution for now is not parallel)
    # determines if needs tool call or no
    tasks = state.get("tasks", [])
    current_task_index = state.get("current_task_index", 0)

    if state.get("status") == "done":
        if state.get("tasks"):
            return {"route": "tasks_done"}
        else:
            return {"route": "end"}  # NEW

    # HANDLE NON-DECOMPOSED CASE
    if not tasks:
        # fallback to original prompt
        task = state.get("prompt", "")
    else:
        # TASK COMPLETION CHECK 
        if current_task_index >= len(tasks):
            #goal achived clear tasks, reset index
            return {
                "tasks": [],
                "current_task_index": 0,
                "status": "done",
                "route": "tasks_done"
            }

        task = tasks[current_task_index]

    # 4. ROUTE DECISION (LLM)
    response = Llmwrapper.llm.invoke(
        prompts.router_p.format(task=task)
    )

    raw_text = getattr(response, "content", str(response))
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in router output: {raw_text}")
    json_str = match.group(0)
    data = json.loads(json_str)

    route = data.get("route", "DIRECT")

    # RETURN ROUTE
    return {
        "route": route,
    }


def reasoner_node(state: WorkflowState) -> dict:
    # determines what to do in step, what tool to use given contexts and list of tools and previous steps
    messages = state.get("messages", [])
    observations = state.get("observations", [])
    observations=format_observations(observations)

    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", 10)

    tasks = state.get("tasks", [])
    current_task_index = state.get("current_task_index", 0)

    # STEP LIMIT (hard stop)
    if step_count >= max_steps:
        return {
            "current_task_index": current_task_index + 1,
            "task_outputs": state.get("task_outputs", []) + ["Max steps reached"],
            "step_count": 0,
            "messages": [
                AIMessage(content="Reached max reasoning steps. Stopping.")
            ]
        }

    # DETERMINE CURRENT TASK
    if tasks:
        task = tasks[current_task_index]
    else:
        task = state.get("prompt", "")

    # BUILD CONTEXT (user + tool observations)
    user_msg = build_agent_context(messages)
    tools=get_tool_descriptions()

    # CALL LLM
    response = Llmwrapper.llm.invoke(
        prompts.reasoner_p.format(
            context=user_msg,
            task=task,
            tools=tools,
            observations=observations
        )
    )

    raw_text = getattr(response, "content", str(response))
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in reasoner output: {raw_text}")
    json_str = match.group(0)
    parsed = parse_and_validate(json_str, ReasonerOutput)

    # HANDLE FINAL ANSWER
    if parsed.action == "FINAL":
        if tasks:
            return {
            "current_task_index": current_task_index + 1,
            "task_outputs": state.get("task_outputs", []) + [parsed.final_answer],
            "step_count": 0
            }
        
        return {
            "current_task_index": current_task_index + 1,
            "messages": [
                AIMessage(content=parsed.final_answer or "Done.")
            ],
            "step_count": 0
        }

    # HANDLE TOOL CALL
    if parsed.action == "TOOL":
        tool_call_id = str(uuid.uuid4())

        tool_call = {
            "id": tool_call_id,
            "name": parsed.tool_name,
            "args": parsed.tool_input or {}
        }

        return {
            "messages": [
                AIMessage(
                    content=parsed.thought,
                    tool_calls=[tool_call]
                )
            ],
            "last_thought": parsed.thought,
            "tool_name": parsed.tool_name,
            "step_count": step_count + 1
        }

    # FALLBACK (safety)
    task_outputs = state.get("task_outputs") or []

    return {
        "current_task_index": current_task_index + 1,
        "task_outputs": task_outputs + ["Failed to complete task"],
        "step_count": 0
    }


def tool_call_builder_node(state: WorkflowState) -> dict:
    # validates tool and determines arguments to be passed, creates tool call 
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None

    tool_calls = getattr(last_message, "tool_calls", None)

    if not tool_calls:
        return {"valid": "no"}

    tool_call = tool_calls[0]
    tool_name = tool_call.get("name")
    raw_args = tool_call.get("args", {})

    # TOOL EXISTENCE CHECK (dynamic)
    tool_def = TOOL_REGISTRY.get(tool_name)

    if not tool_def:
        return {
            "valid": "no",
            "messages": [
                ToolMessage(
                    content=f"Tool '{tool_name}' not found.",
                    tool_call_id=tool_call.get("id")
                )
            ]
        }

    schema = tool_def.schema

    # ENSURE ARGS IS DICT
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except:
            return {
                "valid": "no",
                "messages": [
                    ToolMessage(
                        content="Tool arguments must be valid JSON.",
                        tool_call_id=tool_call.get("id")
                    )
                ]
            }

    if not isinstance(raw_args, dict):
        return {
            "valid": "no",
            "messages": [
                ToolMessage(
                    content="Tool arguments must be a JSON object.",
                    tool_call_id=tool_call.get("id")
                )
            ]
        }

    # VALIDATION (schema-driven, dynamic)
    try:
        validated = schema(**raw_args)
    except ValidationError as e:
        return {
            "valid": "no",
            "messages": [
                ToolMessage(
                    content=f"Validation error: {str(e)}",
                    tool_call_id=tool_call.get("id")
                )
            ]
        }

    clean_args = validated.model_dump()

    return {
        "valid": "yes",
        "tool_name": tool_name,
        "tool_args": clean_args,
        "tool_call_id": tool_call.get("id")
    }

def actor_node(state: WorkflowState) -> dict:
    tool_name = state.get("tool_name")
    tool_args = state.get("tool_args", {})

    # TOOL LOOKUP
    tool_def = TOOL_REGISTRY.get(tool_name)

    tool_call_id = state.get("tool_call_id", str(uuid.uuid4()))

    if not tool_def:
        return {
            "messages": [
                ToolMessage(
                    content=f"Execution error: Tool '{tool_name}' not found.",
                    tool_call_id=tool_call_id
                )
            ]
        }

    tool_func = tool_def.func

    # EXECUTE TOOL
    try:
        result = tool_func(**tool_args)

        # normalize result → string (important for LLM consumption)
        if not isinstance(result, str):
            result = json.dumps(result, default=str)

    except Exception as e:
        return {
            "messages": [
                ToolMessage(
                    content=f"Execution error: {str(e)}",
                    tool_call_id=tool_call_id
                )
            ]
        }

    # RETURN TOOL RESULT
    return {
        "messages": [
            ToolMessage(
                content=f"{tool_name}({tool_args}) -> {result}",
                tool_call_id=tool_call_id
            )
        ],
        "tool_call_result": result
    }

def observer_node(state: WorkflowState) -> dict:
    messages = state.get("messages", [])
    tool_result = state.get("tool_call_result", "")
    tool_name = state.get("tool_name", "")

    last_thought = state.get("last_thought", "")

    tasks = state.get("tasks") or []
    index = state.get("current_task_index", 0)
    if 0 <= index < len(tasks):
        task = tasks[index]
    else:
        task = state.get("prompt", "")

    # Get last tool message for context
    last_tool_msg = None
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            last_tool_msg = m
            break

    tool_context = last_tool_msg.content if last_tool_msg else ""

    # Build prompt
    prompt = prompts.observer_p.format(
        task=task,
        tool_name=tool_name,
        reasoner_thought=last_thought,
        tool_context=tool_context,
        tool_result=tool_result,
        
    )

    response = Llmwrapper.llm.invoke(prompt)
    raw_text = getattr(response, "content", str(response))

    # Extract JSON safely
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        # HARD FALLBACK (deterministic structure)
        fallback = ObserverOutput(
            summary="Failed to parse tool result.",
            key_points=[tool_result[:200]],
            next_hint="Retry or use a different tool."
        )
        parsed = fallback
    else:
        json_str = match.group(0)
        try:
            parsed = parse_and_validate(json_str, ObserverOutput)
        except Exception:
            # SECOND FALLBACK
            parsed = ObserverOutput(
                summary="Invalid structured observation.",
                key_points=[json_str[:200]],
                next_hint="Re-evaluate tool output."
            )

    # Store structured observation (not just string!)
    observations = state.get("observations", [])
    observations.append(parsed.model_dump())

    return {
        "observations": observations,

        # Feed structured observation back into loop
        "messages": [
            ToolMessage(
                content=parsed.summary,
                tool_call_id=getattr(last_tool_msg, "tool_call_id", "unknown")
            )
        ]
    }

def aggregator_node(state: WorkflowState) -> dict:
    """
    Combines outputs from all tasks into a single cohesive final answer.
    If only one output exists, return it directly.
    """

    task_outputs = state.get("task_outputs") or []
    intent = state.get("intent", "")
    prompt = state.get("prompt", "")

    # CASE 1: No task outputs (fallback safety)
    if not task_outputs:
        return {
            "messages": [
                AIMessage(content="No results were generated.")
            ]
        }

    # CASE 2: Single task → return directly (no LLM needed)
    if len(task_outputs) == 1:
        return {
            "messages": [
                AIMessage(content=task_outputs[0])
            ]
        }

    # CASE 3: Multiple tasks → aggregate via LLM
    # Build structured input
    formatted_outputs = "\n".join(
        f"{i+1}. {out}" for i, out in enumerate(task_outputs)
    )

    agg_prompt = prompts.aggregator_p.format(
        intent=intent,
        prompt=prompt,
        task_outputs=formatted_outputs
    )

    response = Llmwrapper.llm.invoke(agg_prompt)
    raw_text = getattr(response, "content", str(response))

    # Extract JSON
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)

    if not match:
        # HARD FALLBACK: naive join
        combined = "\n".join(task_outputs)
        return {
            "messages": [
                AIMessage(content=combined)
            ]
        }

    json_str = match.group(0)

    try:
        parsed = parse_and_validate(json_str, AggregatorOutput)
        final_answer = parsed.final_answer
    except Exception:
        # SECOND FALLBACK
        final_answer = "\n".join(task_outputs)

    return {
        "messages": [
            AIMessage(content=final_answer)
        ]
    }

# =============================================================================
# ROUTING
# =============================================================================
def should_decompose(state: WorkflowState) -> Literal["yes", "no"]:
    return  state.get("will_decompose", "no")

def is_tool_valid(state: WorkflowState) -> Literal["yes", "no"]:
    return  state.get("valid", "no")

def should_continue(state: WorkflowState) -> Literal["act", "end"]:
    messages = state.get("messages", [])
    if not messages:
        return "end"
    last_message = messages[-1]
    if getattr(last_message, "tool_calls", None):
        return "act"
    return "end"

def route_router(state) -> Literal["direct", "react", "tasks_done", "end"]:
    if state.get("status", "") == "done":
        return "tasks_done"

    return state.get("route", "DIRECT").lower()

# =============================================================================
# WORKFLOW
# =============================================================================

def create_agent_workflow() -> StateGraph:
    workflow = StateGraph(WorkflowState)

    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("decomposer", decomposer_node)
    workflow.add_node("router", router_node)
    workflow.add_node("direct", direct_node)

    workflow.add_node("reasoner", reasoner_node)
    workflow.add_node("tool_call_builder", tool_call_builder_node)
    workflow.add_node("actor", actor_node)
    workflow.add_node("observer", observer_node)

    workflow.add_node("aggregator", aggregator_node)

    workflow.add_edge(START, "analyzer")

    workflow.add_conditional_edges(
        "analyzer",
        should_decompose,
        {
            "yes": "decomposer",
            "no": "router"
        }
    )
    workflow.add_edge("decomposer", "router")

    workflow.add_conditional_edges(
        "router",
        route_router,
        {
            "direct": "direct",
            "react": "reasoner",
            "tasks_done": "aggregator",
            "end": END   # 👈 direct termination
        }
    )
    workflow.add_edge("direct", "router")
    workflow.add_conditional_edges(
        "reasoner",
        should_continue,
        {
            "act": "tool_call_builder",
            "end": "router"
        }
    )
    workflow.add_conditional_edges(
        "tool_call_builder",
        is_tool_valid,
        {
            "yes": "actor",
            "no": "reasoner"
        }
    )
    workflow.add_edge("actor", "observer")
    workflow.add_edge("observer", "reasoner")

    workflow.add_edge("aggregator", END)

    return workflow







