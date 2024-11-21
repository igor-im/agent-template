import json
import os
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4
import gradio as gr

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph
from psycopg import AsyncConnection
from dotenv import load_dotenv

from agent import research_assistant
from schema.schema import (
    ChatMessage,
    Feedback,
    FeedbackResponse,
    StreamInput,
    UserInput,
    convert_message_content_to_string,
)
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register

load_dotenv()

PORT = int(os.environ.get("PORT"))
PHOENIX_API_KEY = os.environ.get("PHOENIX_API_KEY")
PHOENIX_PROJECT_NAME = os.environ.get("PHOENIX_PROJECT_NAME")
if not PHOENIX_API_KEY:
    raise ValueError("PHOENIX_API_KEY is not set")
if not PHOENIX_PROJECT_NAME:
    raise ValueError("PHOENIX_PROJECT_NAME is not set")

os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = PHOENIX_API_KEY
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

tracer_provider = register(
    project_name=os.getenv('PHOENIX_PROJECT_NAME'),
    endpoint="https://app.phoenix.arize.com/v1/traces"
)

LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

DB_URI = os.environ.get("CONVERSATIONS_POSTGRES_URL")
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    async with await AsyncConnection.connect(DB_URI, **connection_kwargs) as l_conn:
        research_assistant.checkpointer = AsyncPostgresSaver(l_conn)
        app.state.agent = research_assistant
        yield
    # context manager will clean up the AsyncSqliteSaver on exit


# async def lifespan(app: FastAPI):
#     research_assistant.checkpointer = MemorySaver()
#     app.state.agent = research_assistant
#     yield


# context manager will clean up the AsyncSqliteSaver on exit

app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def check_auth_header(request: Request, call_next: Callable) -> Response:
    if auth_secret := os.getenv("AUTH_SECRET"):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return Response(status_code=401, content="Missing or invalid token")
        if auth_header[7:] != auth_secret:
            return Response(status_code=401, content="Invalid token")
    return await call_next(request)


def _parse_input(user_input: UserInput) -> tuple[dict[str, Any], str]:
    run_id = uuid4()
    thread_id = user_input.thread_id or str(uuid4())
    input_message = ChatMessage(type="human", content=user_input.message)
    kwargs = {
        "input": {"messages": [input_message.to_langchain()]},
        "config": RunnableConfig(
            configurable={"thread_id": thread_id, "model": user_input.model}, run_id=run_id
        ),
    }
    return kwargs, run_id


def _remove_tool_calls(content: str | list[str | dict]) -> str | list[str | dict]:
    """Remove tool calls from content."""
    if isinstance(content, str):
        return content
    # Currently only Anthropic models stream tool calls, using content item type tool_use.
    return [
        content_item
        for content_item in content
        if isinstance(content_item, str) or content_item["type"] != "tool_use"
    ]


@app.post("/invoke")
async def invoke(user_input: UserInput) -> ChatMessage:
    """
    Invoke the agent.py with user input to retrieve a final response.

    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    """
    agent: CompiledStateGraph = app.state.agent
    kwargs, run_id = _parse_input(user_input)
    try:
        response = await agent.ainvoke(**kwargs)
        output = ChatMessage.from_langchain(response["messages"][-1])
        output.run_id = str(run_id)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def message_generator(user_input: StreamInput) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.py.

    This is the workhorse method for the /stream endpoint.
    """
    agent: CompiledStateGraph = app.state.agent
    kwargs, run_id = _parse_input(user_input)

    # Process streamed events from the graph and yield messages over the SSE stream.
    async for event in agent.astream_events(**kwargs, version="v2"):
        if not event:
            continue

        # Yield messages written to the graph state after node execution finishes.
        if (
                event["event"] == "on_chain_end"
                # on_chain_end gets called a bunch of times in a graph execution
                # This filters out everything except for "graph node finished"
                and any(t.startswith("graph:step:") for t in event.get("tags", []))
                and "messages" in event["data"]["output"]
        ):
            new_messages = event["data"]["output"]["messages"]
            for message in new_messages:
                try:
                    chat_message = ChatMessage.from_langchain(message)
                    chat_message.run_id = str(run_id)
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'content': f'Error parsing message: {e}'})}\n\n"
                    continue
                # LangGraph re-sends the input message, which feels weird, so drop it
            if chat_message.type == "human" and chat_message.content == user_input.message:
                continue
            yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

        # Yield tokens streamed from LLMs.
        if (
                event["event"] == "on_chat_model_stream"
                and user_input.stream_tokens
                and "llama_guard" not in event.get("tags", [])
        ):
            content = _remove_tool_calls(event["data"]["chunk"].content)
            if content:
                # Empty content in the context of OpenAI usually means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content.
                yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)})}\n\n"
            continue

    yield "data: [DONE]\n\n"


def _sse_response_example() -> dict[int, Any]:
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": "data: {'type': 'token', 'content': 'Hello'}\n\ndata: {'type': 'token', 'content': ' World'}\n\ndata: [DONE]\n\n",
                    "schema": {"type": "string"},
                }
            },
        }
    }


@app.post("/stream", response_class=StreamingResponse, responses=_sse_response_example())
async def stream_agent(user_input: StreamInput) -> StreamingResponse:
    """
    Stream the agent.py's response to a user input, including intermediate messages and tokens.

    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.
    """
    return StreamingResponse(message_generator(user_input), media_type="text/event-stream")


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox()
    thread = gr.Dropdown(label="Thread ID", choices=["thread1", "thread2"])
    clear = gr.ClearButton([msg, chatbot])

    async def respond(message, chat_history, threadarg):
        print("respond called", threadarg)
        bot_message = await invoke(UserInput(message=message, thread_id=threadarg))
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message.content})
        return "", chat_history

    msg.submit(respond, [msg, chatbot, thread], [msg, chatbot])

app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8110)
