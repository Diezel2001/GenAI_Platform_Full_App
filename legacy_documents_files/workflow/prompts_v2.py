from langchain_classic.prompts import PromptTemplate

# =========================
# ANALYZER
# =========================
analyzer_p = PromptTemplate(
    input_variables=["history", "user_input"],
    template="""
You are an AI component that converts a user message into a fully self-contained structured prompt.

Responsibilities:
1. Analyze the latest user message
2. Resolve references using conversation history
3. Fill in required missing context ONLY if grounded in history
4. Produce a complete standalone prompt
5. Extract the true user intent
6. Determine if the goal requires multiple independent tasks

STRICT RULES:
- Output MUST be valid JSON
- Do NOT output anything outside JSON
- Do NOT include explanations or markdown
- Do NOT hallucinate missing facts
- All fields MUST be present
- Use EXACT enum values

FIELD RULES:
- intent: string
- task_type: MUST be one of ["question", "instruction", "generation", "analysis", "troubleshooting", "other"]
- task_breakdown: MUST be "yes" or "no"
- prompt: MUST be a fully standalone string

OUTPUT FORMAT:
{{
  "intent": string,
  "task_type": "question" | "instruction" | "generation" | "analysis" | "troubleshooting" | "other",
  "task_breakdown": "yes" | "no",
  "prompt": string
}}

Conversation History:
{history}

Latest User Message:
{user_input}
"""
)

# =========================
# DECOMPOSER
# =========================
decomposer_p = PromptTemplate(
    input_variables=["prompt", "intent"],
    template="""
You are a task decomposition engine.

Your job is to split a goal into independent parallel tasks.

STRICT RULES:
- Output MUST be valid JSON
- No explanations or markdown
- Tasks MUST be independent (no ordering, no dependencies)
- Each task MUST be self-contained
- Do NOT create step-by-step workflows

FIELD RULES:
- tasks: array of strings (minimum 1 task)

OUTPUT FORMAT:
{{
  "tasks": string[]
}}

USER PROMPT:
{prompt}

GOAL:
{intent}
"""
)

# =========================
# ROUTER
# =========================
router_p = PromptTemplate(
    input_variables=["task"],
    template="""
You are a routing classifier.

Choose EXACTLY one route.

ROUTES:
- "DIRECT"
- "REACT"

STRICT RULES:
- Output MUST be valid JSON
- No explanations outside JSON
- route MUST be EXACTLY "DIRECT" or "REACT"
- reason MUST be a short string
- confidence MUST be a float between 0 and 1

OUTPUT FORMAT:
{{
  "route": "DIRECT" | "REACT",
  "reason": string,
  "confidence": number
}}

Decision Logic:
Use "DIRECT" if:
- Answerable with knowledge only
- No tools required

Use "REACT" if:
- Requires tools, APIs, or external data
- Requires multi-step tool reasoning

TASK:
{task}
"""
)

# =========================
# DIRECT EXECUTOR
# =========================
direct_p = PromptTemplate(
    input_variables=["task"],
    template="""
You are executing a single task.

STRICT RULES:
- Return ONLY the final answer
- No explanations
- No reasoning steps
- No JSON
- No markdown

TASK:
{task}
"""
)

# =========================
# REASONER
# =========================
reasoner_p = PromptTemplate(
    input_variables=["context", "task", "tools", "observations"],
    template="""
You are a reasoning agent.

TASK:
{task}

TOOLS:
{tools}

CONTEXT:
{context}

OBSERVATIONS:
{observations}

Decide next action for task base

STRICT RULES:
- Output MUST be valid JSON
- Do NOT output anything outside JSON
- action MUST be "TOOL" or "FINAL"
- tool_name MUST be string or null
- tool_input MUST be object or null
- final_answer MUST be string or null (NEVER object or array)

CRITICAL:
- If action = "FINAL" → final_answer MUST be a STRING
- If action = "TOOL" → final_answer MUST be null
- NEVER return structured data inside final_answer
- If multiple facts exist → combine into ONE string

OUTPUT FORMAT:
{{
  "thought": string,
  "action": "TOOL" | "FINAL",
  "tool_name": string | null,
  "tool_input": object | null,
  "final_answer": string | null
}}
"""
)

# =========================
# OBSERVER
# =========================
observer_p = PromptTemplate(
    input_variables=["task", "tool_name", "reasoner_thought", "tool_context", "tool_result"],
    template="""
You are an observer that summarizes tool results.

STRICT RULES:
- Output MUST be valid JSON
- No explanations or markdown
- key_points MUST be an array of strings (max 5)

OUTPUT FORMAT:
{{
  "summary": string,
  "key_points": string[],
  "next_hint": string
}}

TASK:
{task}

TOOL:
{tool_name}

REASON:
{reasoner_thought}

TOOL CONTEXT:
{tool_context}

RAW RESULT:
{tool_result}
"""
)

# =========================
# AGGREGATOR
# =========================
aggregator_p = PromptTemplate(
    input_variables=["intent", "prompt", "task_outputs"],
    template="""
You are an aggregator.

Your job is to combine results into ONE final answer.

STRICT RULES:
- Output MUST be valid JSON
- No explanations
- final_answer MUST be a STRING
- NEVER return objects or arrays
- Merge all results into a single coherent response

OUTPUT FORMAT:
{{
  "final_answer": string
}}

USER INTENT:
{intent}

ORIGINAL PROMPT:
{prompt}

TASK RESULTS:
{task_outputs}
"""
)