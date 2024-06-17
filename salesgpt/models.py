import aioboto3
import aiohttp
import os
import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import run_in_executor
from langchain_openai import ChatOpenAI

from salesgpt.tools import completion_bedrock


class BedrockCustomModel(ChatOpenAI):
    """自定义聊天模型，回显输入的前“n”个字符。

    在向 LangChain 贡献实现时，请仔细记录
    该模型包括初始化参数，包括
    如何初始化模型并包含任何相关的示例
    指向底层模型文档或 API 的链接。

    Example:

        .. code-block:: python

            model = CustomChatModel(n=2)
            result = model.invoke([HumanMessage(content="hello")])
            result = model.batch([[HumanMessage(content="hello")],
                                 [HumanMessage(content="world")]])
    """

    model: str
    system_prompt: str
    """要回显的提示的最后一条消息的字符数."""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        """重写 _generate 方法来实现聊天模型逻辑。

        这可以是对 API 的调用、对本地模型的调用或任何其他
        生成对输入提示的响应的实现。

        Args:
            messages: 由消息列表组成的提示。
            stop: 模型应停止生成的字符串列表。
                  如果由于停止令牌而停止生成，则停止令牌本身
                  应作为输出的一部分包含在内。这并没有强制执行
                  现在跨模型，但这是一个很好的实践，因为
                  它使解析模型的输出变得更加容易
                  下游并了解发电停止的原因。
            run_manager: 带有 LLM 回调的运行管理器.
        Returns:
            ChatResult: 包含生成的消息的 ChatResult 对象。
        """
        last_message = messages[-1]

        print(messages)
        response = completion_bedrock(
            model_id=self.model,
            system_prompt=self.system_prompt,
            messages=[{"content": last_message.content, "role": "user"}],
            max_tokens=1000,
        )
        print("output", response)
        content = response["content"][0]["text"]
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """async 版本的 _generate 方法

        Args:
            messages (List[BaseMessage]): _description_
            stop (Optional[List[str]], optional): _description_. Defaults to None.
            run_manager (Optional[AsyncCallbackManagerForLLMRun], optional): _description_. Defaults to None.
            stream (Optional[bool], optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_

        Returns:
            ChatResult: _description_
        """
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            raise NotImplementedError("Streaming not implemented")

        last_message = messages[-1]

        print(messages)
        response = await acompletion_bedrock(
            model_id=self.model,
            system_prompt=self.system_prompt,
            messages=[{"content": last_message.content, "role": "user"}],
            max_tokens=1000,
        )
        print("output", response)
        content = response["content"][0]["text"]
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

        # message_dicts, params = self._create_message_dicts(messages, stop)
        # params = {
        #     **params,
        #     **({"stream": stream} if stream is not None else {}),
        #     **kwargs,
        # }
        # response = await self.async_client.create(messages=message_dicts, **params)
        # return self._create_chat_result(response)


async def acompletion_bedrock2(model_id, system_prompt, messages, max_tokens=1000):
    """
    使用 Anthropic Claude 生成消息的高级 API 调用，针对异步进行了重构。
    """
    session = aioboto3.Session()
    async with session.client(
        service_name="bedrock-runtime", region_name=os.environ.get("AWS_REGION_NAME")
    ) as bedrock_runtime:

        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": messages,
            }
        )

        response = await bedrock_runtime.invoke_model(body=body, modelId=model_id)

        # print('RESPONSE', response)

        # Correctly handle the streaming body
        response_body_bytes = await response["body"].read()
        # print('RESPONSE BODY', response_body_bytes)
        response_body = json.loads(response_body_bytes.decode("utf-8"))
        # print('RESPONSE BODY', response_body)

        return response_body


async def acompletion_bedrock(model_id, system_prompt, messages, max_tokens=1000):
    """
    使用 Anthropic Claude 生成消息的高级 API 调用，针对异步进行了重构。
    """
    # 建立请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f'Bearer {os.environ.get("AWS_ACCESS_KEY_ID")}',
    }

    # 请求体
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages,
        }
    )

    # 异步 HTTP 请求
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"https://bedrock-runtime.{os.environ.get('AWS_REGION_NAME')}.amazonaws.com/models/{model_id}/invoke",
            headers=headers,
            data=body,
        ) as response:
            response_text = await response.text()
            return json.loads(response_text)
