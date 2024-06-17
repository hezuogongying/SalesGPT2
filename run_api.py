import json
import os
from typing import List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from salesgpt.salesgptapi import SalesGPTAPI

# Load environment variables
load_dotenv()

# Access environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://react-frontend:80",
    "https://sales-gpt-frontend-git-main-filip-odysseypartns-projects.vercel.app",
    "https://sales-gpt-frontend.vercel.app",
]
CORS_METHODS = ["GET", "POST"]

# Initialize FastAPI app
app = FastAPI()

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=CORS_METHODS,
    allow_headers=["*"],
)


class AuthenticatedResponse(BaseModel):
    message: str


def get_auth_key(authorization: str = Header(...)) -> None:
    print(f"Authorization header: {authorization}")
    auth_key = os.getenv("AUTH_KEY")
    if not auth_key:
        raise HTTPException(status_code=500, detail="AUTH_KEY not configured")
    expected_header = f"Bearer {auth_key}"
    if authorization != expected_header:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/")
async def say_hello():
    return {"message": "Hello World"}


class MessageList(BaseModel):
    session_id: str
    human_say: str


sessions = {}


@app.get("/botname", response_model=None)
async def get_bot_name(authorization: Optional[str] = Header(None)):
    load_dotenv()
    if os.getenv("ENVIRONMENT") == "production":
        get_auth_key(authorization)

    sales_api = SalesGPTAPI(
        config_path=os.getenv("CONFIG_PATH", "examples/example_agent_setup.json"),
        product_catalog=os.getenv(
            "PRODUCT_CATALOG", "examples/sample_product_catalog.txt"
        ),
        verbose=True,
        model_name=os.getenv("GPT_MODEL", "gpt-3.5-turbo-0613"),
    )
    name = sales_api.sales_agent.salesperson_name
    return {"name": name, "model": sales_api.sales_agent.model_name}


@app.post("/chat")
async def chat_with_sales_agent(
    req: MessageList,
    stream: bool = Query(False),
    authorization: Optional[str] = Header(None),
):
    """
        处理与销售代理的聊天互动。

        该端点接收来自用户的消息并返回销售代理的响应。它支持会话管理，以维护与同一用户的多次交互的上下文。

        参数：
            req (MessageList)：包含会话 ID 和来自人类用户的消息的请求对象。
            Stream（布尔型，可选）：指示是否应流式传输响应的标志。目前，流媒体尚未实现。
    返回：
            如果请求流式传输，它将返回 StreamingResponse 对象（尚未实现）。否则，它返回销售代理对用户消息的响应。

        笔记：
            流媒体功能已计划但尚未可用。当前的实现仅支持同步响应。
    """
    sales_api = None
    if os.getenv("ENVIRONMENT") == "production":
        get_auth_key(authorization)
    # print(f"Received request: {req}")
    if req.session_id in sessions:
        print("Session is found!")
        sales_api = sessions[req.session_id]
        print(f"Are tools activated: {sales_api.sales_agent.use_tools}")
        print(f"Session id: {req.session_id}")
    else:
        print("Creating new session")
        sales_api = SalesGPTAPI(
            config_path=os.getenv("CONFIG_PATH", "examples/example_agent_setup.json"),
            verbose=True,
            product_catalog=os.getenv(
                "PRODUCT_CATALOG", "examples/sample_product_catalog.txt"
            ),
            model_name=os.getenv("GPT_MODEL", "gpt-3.5-turbo-0613"),
            use_tools=os.getenv("USE_TOOLS_IN_API", "True").lower()
            in ["true", "1", "t"],
        )
        print(f"TOOLS?: {sales_api.sales_agent.use_tools}")
        sessions[req.session_id] = sales_api

    # TODO stream not working
    if stream:

        async def stream_response():
            stream_gen = sales_api.do_stream(req.conversation_history, req.human_say)
            async for message in stream_gen:
                data = {"token": message}
                yield json.dumps(data).encode("utf-8") + b"\n"

        return StreamingResponse(stream_response())
    else:
        response = await sales_api.do(req.human_say)
        return response


# Main entry point
if __name__ == "__main__":
    uvicorn.run("run_api:app", host="127.0.0.1", port=8000, reload=True)
