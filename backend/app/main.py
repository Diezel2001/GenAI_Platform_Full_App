from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# from app.api.auth import router as auth_router
from app.api.documents import router as document_router
from app.api.query import router as query_router
from app.api.agent import router as agent_router
# from app.api.health import router as health_router
from langgraph.checkpoint.redis import RedisSaver  

# from app.core.config import settings
# from app.core.logging import configure_logging

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge

import os
from dotenv import load_dotenv

load_dotenv()


# Environment Variables
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DATABASE_URL = os.getenv("DATABASE_URL")


# -------------------------
# Metrics
# -------------------------

# Agent Metrics
AGENT_RUNS = Counter("agent_runs_total", "Total agent runs")
AGENT_ERRORS = Counter("agent_errors_total", "Agent failures")
AGENT_LATENCY = Histogram("agent_latency_seconds", "Agent execution time")
AGENT_STEPS = Histogram("agent_steps", "Steps per execution")

# llm usage metrics
LLM_CALLS = Counter("llm_calls_total", "Total LLM calls")
TOKENS_INPUT = Counter("llm_input_tokens_total", "Input tokens")
TOKENS_OUTPUT = Counter("llm_output_tokens_total", "Output tokens")
LLM_LATENCY = Histogram("llm_latency_seconds", "LLM response time")

# tool usage
TOOL_CALLS = Counter("tool_calls_total", "Tool usage", ["tool_name"])
TOOL_ERRORS = Counter("tool_errors_total", "Tool errors", ["tool_name"])
TOOL_LATENCY = Histogram("tool_latency_seconds", "Tool latency", ["tool_name"])

# rag retrieval metrics
RETRIEVAL_LATENCY = Histogram("retrieval_latency_seconds", "RAG retrieval time")
DOCS_RETRIEVED = Histogram("documents_retrieved", "Docs per query")
CACHE_HITS = Counter("rag_cache_hits_total", "Cache hits")
CACHE_MISSES = Counter("rag_cache_misses_total", "Cache misses")

# memory
MEMORY_READ_LATENCY = Histogram("memory_read_seconds", "Memory read time")
MEMORY_WRITE_LATENCY = Histogram("memory_write_seconds", "Memory write time")
ACTIVE_SESSIONS = Gauge("active_sessions", "Active sessions")


# Lifespan Events
@asynccontextmanager
async def lifespan(app: FastAPI):

    with RedisSaver.from_conn_string(REDIS_URL) as checkpointer:
        checkpointer.setup()

        from app.services.agents.worker_agent import WorkerAgent
        from app.services.llm.Llmwrapper import llm
        from app.services.tools import get_tools, _FILE_TOOLS, _CLI_TOOLS

        myAgent = WorkerAgent(
            llm=llm,
            checkpointer=checkpointer,
            tools={**_FILE_TOOLS, **_CLI_TOOLS })

        app.state.agent = myAgent
        app.state.checkpointer = checkpointer

        from app.utility.langgraphViz import visualize_langgraph

        visualize_langgraph(myAgent.graph, "worker_agent.png")

        print("🚀 Application starting up...")

        yield

        print("🛑 Application shutting down...")


# -------------------------
# App Initialization
# -------------------------

app = FastAPI(
    title="Agentic Generative AI Platform",
    description="Multi-tenant Agentic AI Platform API",
    version="1.0.0",
    lifespan=lifespan,
)



# -------------------------
# Middleware
# -------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# Metrics (Prometheus)
# -------------------------
Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    excluded_handlers=["/metrics", "/"],
).instrument(app).expose(app, include_in_schema=False)



# -------------------------
# Routers
# -------------------------

# app.include_router(auth_router, prefix="/auth", tags=["Auth"])
app.include_router(document_router, prefix="/documents", tags=["Documents"])
app.include_router(query_router, prefix="/query", tags=["Query"])
app.include_router(agent_router, prefix="/agent", tags=["Agent"])

# -------------------------
# Root Endpoint
# -------------------------

@app.get("/")
async def root():
    return {"status": "ok", "service": "Agentic Generative AI Platform"}
