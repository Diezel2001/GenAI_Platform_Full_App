from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from typing import Optional
import uuid

# from app.core.workflows import _make_human_message, _make_ai_message

class AgentRequestSchema(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    k: int = Field(default=5, ge=1, le=100)
    session_id: str


class AgentResponseSchema(BaseModel):
    message: str
    results: str
    route: str
    analysis: str


router = APIRouter()


@router.post("/", response_model=AgentResponseSchema)
async def agent_input_injestion(request: Request, input: AgentRequestSchema):

    # graph = request.app.state.graph
    agent = request.app.state.agent
    # human_message = _make_human_message(input.message)
    try:
        # result = graph.invoke(
        #     {
        #         "messages": [human_message],
        #         "step_count": 0,
        #         "max_steps": 5
        #     },
        #     config={"configurable": {"thread_id": input.session_id}}
        # )

        result = agent.invoke(
            task=input.message,
            thread_id=input.session_id
        )

        response_text = result

        # messages = result.get("messages", [])
        # response_text = ""

        # if messages:
        #     last_message = messages[-1]
        #     response_text = (
        #         last_message.content
        #         if hasattr(last_message, "content")
        #         else str(last_message)
        #     )

        return AgentResponseSchema(
            message=input.message,
            results=response_text,
            route = "",
            analysis = ""
        )

    except Exception as e:
        raise e