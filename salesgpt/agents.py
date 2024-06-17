from copy import deepcopy
from typing import Any, Callable, Dict, List, Union

from langchain.agents import (
    AgentExecutor,
    LLMSingleActionAgent,
    create_openai_tools_agent,
)
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.base import Chain
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.agents import (
    _convert_agent_action_to_messages,
    _convert_agent_observation_to_messages,
)
from langchain_core.language_models.llms import create_base_retry_decorator
from litellm import acompletion
from pydantic import Field

from salesgpt.chains import SalesConversationChain, StageAnalyzerChain
from salesgpt.custom_invoke import CustomAgentExecutor
from salesgpt.logger import time_logger
from salesgpt.parsers import SalesConvoOutputParser
from salesgpt.prompts_cn import SALES_AGENT_TOOLS_PROMPT
from salesgpt.stages import CONVERSATION_STAGES
from salesgpt.templates import CustomPromptTemplateForTools
from salesgpt.tools_cn import get_tools, setup_knowledge_base


def _create_retry_decorator(llm: Any):
    """
    创建一个重试装饰器来处理 OpenAI API 错误。

    该函数创建一个重试装饰器，它将重试函数调用
    如果它引发任何指定的 OpenAI API 错误。最大重试次数
    由“llm”对象的“max_retries”属性确定。

    Args:
        llm (Any): An object that has a 'max_retries' attribute specifying the maximum number of retries.

    Returns:
        Callable[[Any], Any]: A retry decorator.
    """

    import openai

    errors = [
        openai.Timeout,
        openai.APIError,
        openai.APIConnectionError,
        openai.RateLimitError,
        openai.APIStatusError,
    ]
    return create_base_retry_decorator(error_types=errors, max_retries=llm.max_retries)


class SalesGPT(Chain):
    """销售代理的控制器模型."""

    conversation_history: List[str] = []
    conversation_stage_id: str = "1"
    current_conversation_stage: str = CONVERSATION_STAGES.get("1")
    stage_analyzer_chain: StageAnalyzerChain = Field(...)
    sales_agent_executor: Union[CustomAgentExecutor, None] = Field(...)
    knowledge_base: Union[RetrievalQA, None] = Field(...)
    sales_conversation_utterance_chain: SalesConversationChain = Field(...)
    conversation_stage_dict: Dict = CONVERSATION_STAGES

    model_name: str = "gpt-3.5-turbo-0613"  # TODO - make this an env variable

    use_tools: bool = False
    salesperson_name: str = "Ted Lasso"
    salesperson_role: str = "Business Development Representative"
    company_name: str = "Sleep Haven"
    company_business: str = (
        "Sleep Haven is a premium mattress company that provides customers with the most comfortable and supportive sleeping experience possible. We offer a range of high-quality mattresses, pillows, and bedding accessories that are designed to meet the unique needs of our customers."
    )
    company_values: str = (
        "Our mission at Sleep Haven is to help people achieve a better night's sleep by providing them with the best possible sleep solutions. We believe that quality sleep is essential to overall health and well-being, and we are committed to helping our customers achieve optimal sleep by offering exceptional products and customer service."
    )
    conversation_purpose: str = (
        "find out whether they are looking to achieve better sleep via buying a premier mattress."
    )
    conversation_type: str = "call"

    def retrieve_conversation_stage(self, key):
        """
        根据提供的密钥检索对话阶段。

        该函数使用key在conversation_stage_dict字典中查找相应的对话阶段。
        如果字典中没有找到该键，则默认为“1”.

        Args:
            key (str): The key to look up in the conversation_stage_dict dictionary.

        Returns:
            str: The conversation stage corresponding to the key, or "1" if the key is not found.
        """
        return self.conversation_stage_dict.get(key, "1")

    @property
    def input_keys(self):
        """
        返回输入键列表的属性。

        该属性当前设置为返回空列表。可以在子类中重写它以返回键列表
        用于从字典中提取输入数据.

        Returns:
            List[str]: An empty list.
        """
        return []

    @property
    def output_keys(self):
        """
        返回输出键列表的属性。

        该属性当前设置为返回空列表。可以在子类中重写它以返回键列表
        用于从字典中提取输出数据。

        Returns:
            List[str]: An empty list.
        """
        return []

    @time_logger
    def seed_agent(self):
        """
        此方法通过设置初始对话阶段并清除对话历史记录来播种对话。

        使用键“1”检索初始对话阶段。对话历史记录将重置为空列表。

        Returns:
            None
        """
        self.current_conversation_stage = self.retrieve_conversation_stage("1")
        self.conversation_history = []

    @time_logger
    def determine_conversation_stage(self):
        """
                根据对话历史记录判断当前对话阶段。

                该方法使用 stage_analyzer_chain 来分析对话历史并确定当前阶段。
                对话历史记录被连接成一个字符串，每个条目由换行符分隔。
                当前对话阶段 ID 也会传递到 stage_analyzer_chain。
        然后该方法打印确定的会话阶段 ID 并检索相应的会话阶段
                使用retrieve_conversation_stage 方法从conversation_stage_dict 字典中获取。

                最后，该方法打印确定的对话阶段.

                Returns:
                    None
        """
        print(f"Conversation Stage ID before analysis: {self.conversation_stage_id}")
        print("Conversation history:")
        print(self.conversation_history)
        stage_analyzer_output = self.stage_analyzer_chain.invoke(
            input={
                "conversation_history": "\n".join(self.conversation_history).rstrip(
                    "\n"
                ),
                "conversation_stage_id": self.conversation_stage_id,
                "conversation_stages": "\n".join(
                    [
                        str(key) + ": " + str(value)
                        for key, value in CONVERSATION_STAGES.items()
                    ]
                ),
            },
            return_only_outputs=False,
        )
        print("Stage analyzer output")
        print(stage_analyzer_output)
        self.conversation_stage_id = stage_analyzer_output.get("text")

        self.current_conversation_stage = self.retrieve_conversation_stage(
            self.conversation_stage_id
        )

        print(f"Conversation Stage: {self.current_conversation_stage}")

    @time_logger
    async def adetermine_conversation_stage(self):
        """
                根据对话历史记录判断当前对话阶段。

                该方法使用 stage_analyzer_chain 来分析对话历史并确定当前阶段。
                对话历史记录被连接成一个字符串，每个条目由换行符分隔。
                当前对话阶段 ID 也会传递到 stage_analyzer_chain。
        然后该方法打印确定的会话阶段 ID 并检索相应的会话阶段
                使用retrieve_conversation_stage 方法从conversation_stage_dict 字典中获取。

                最后，该方法打印确定的对话阶段.

                Returns:
                    None
        """
        print(f"Conversation Stage ID before analysis: {self.conversation_stage_id}")
        print("Conversation history:")
        print(self.conversation_history)
        stage_analyzer_output = await self.stage_analyzer_chain.ainvoke(
            input={
                "conversation_history": "\n".join(self.conversation_history).rstrip(
                    "\n"
                ),
                "conversation_stage_id": self.conversation_stage_id,
                "conversation_stages": "\n".join(
                    [
                        str(key) + ": " + str(value)
                        for key, value in CONVERSATION_STAGES.items()
                    ]
                ),
            },
            return_only_outputs=False,
        )
        print("Stage analyzer output")
        print(stage_analyzer_output)
        self.conversation_stage_id = stage_analyzer_output.get("text")

        self.current_conversation_stage = self.retrieve_conversation_stage(
            self.conversation_stage_id
        )

        print(f"Conversation Stage: {self.current_conversation_stage}")

    def human_step(self, human_input):
        """
        处理人工输入并将其附加到对话历史记录中。

        此方法将人工输入作为字符串，通过在开头添加“User:”和在末尾添加“<END_OF_TURN>”来格式化它，然后将此格式化字符串附加到对话历史记录中.

        Args:
            human_input (str): The input string from the human user.

        Returns:
            None
        """
        human_input = "User: " + human_input + " <END_OF_TURN>"
        self.conversation_history.append(human_input)

    @time_logger
    def step(self, stream: bool = False):
        """
        执行对话中的一个步骤。如果流参数设置为 True，
         它返回一个流生成器对象，用于在下游应用程序中操作流块。
         如果流参数设置为 False，它将使用空字典作为输入来调用 _call 方法。

         Args:
             stream (bool, optional): A flag indicating whether to return a streaming generator object.
             Defaults to False.

         Returns:
             Generator: A streaming generator object if stream is set to True. Otherwise, it returns None.
        """
        if not stream:
            return self._call(inputs={})
        else:
            return self._streaming_generator()

    @time_logger
    async def astep(self, stream: bool = False):
        """
        在对话中执行异步步骤。

        如果流参数设置为 False，它将使用空字典作为输入来调用 _acall 方法。
        如果流参数设置为 True，它将返回一个流生成器对象，用于在下游应用程序中操作流块。

        Args:
            stream (bool, optional): A flag indicating whether to return a streaming generator object.
            Defaults to False.

        Returns:
            Generator: A streaming generator object if stream is set to True. Otherwise, it returns None.
        """
        if not stream:
            return await self.acall(inputs={})
        else:
            return await self._astreaming_generator()

    @time_logger
    async def acall(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行销售代理的一个步骤。

        此函数用对话的当前状态暂时覆盖输入，
        使用销售代理执行器或销售对话话语链生成代理的话语，
        将代理的响应添加到对话历史记录中，并返回 AI 消息。

        Parameters
        ----------
        inputs : Dict[str, Any]
            The initial inputs for the sales agent.

        Returns
        -------
        Dict[str, Any]
            The AI message generated by the sales agent.

        """
        # override inputs temporarily
        inputs = {
            "input": "",
            "conversation_stage": self.current_conversation_stage,
            "conversation_history": "\n".join(self.conversation_history),
            "salesperson_name": self.salesperson_name,
            "salesperson_role": self.salesperson_role,
            "company_name": self.company_name,
            "company_business": self.company_business,
            "company_values": self.company_values,
            "conversation_purpose": self.conversation_purpose,
            "conversation_type": self.conversation_type,
        }

        # Generate agent's utterance
        if self.use_tools:
            ai_message = await self.sales_agent_executor.ainvoke(inputs)
            output = ai_message["output"]
        else:
            ai_message = await self.sales_conversation_utterance_chain.ainvoke(
                inputs, return_intermediate_steps=True
            )
            output = ai_message["text"]

        # Add agent's response to conversation history
        agent_name = self.salesperson_name
        output = agent_name + ": " + output
        if "<END_OF_TURN>" not in output:
            output += " <END_OF_TURN>"
        self.conversation_history.append(output)

        if self.verbose:
            tool_status = "USE TOOLS INVOKE:" if self.use_tools else "WITHOUT TOOLS:"
            print(f"{tool_status}\n#\n#\n#\n#\n------------------")
            print(f"AI Message: {ai_message}")
            print()
            print(f"Output: {output.replace('<END_OF_TURN>', '')}")

        return ai_message

    @time_logger
    def _prep_messages(self):
        """
        为流生成器准备消息列表。

        此方法根据对话的当前状态准备消息列表。
        这些消息是使用“sales_conversation_utterance_chain”对象的“prep_prompts”方法准备的。
        准备好的消息包括有关当前对话阶段、对话历史记录、销售人员的姓名和角色的详细信息，
        公司名称、业务、价值观、对话目的和对话类型。

        Returns:
            list: A list of prepared messages to be passed to a streaming generator.
        """

        prompt = self.sales_conversation_utterance_chain.prep_prompts(
            [
                dict(
                    conversation_stage=self.current_conversation_stage,
                    conversation_history="\n".join(self.conversation_history),
                    salesperson_name=self.salesperson_name,
                    salesperson_role=self.salesperson_role,
                    company_name=self.company_name,
                    company_business=self.company_business,
                    company_values=self.company_values,
                    conversation_purpose=self.conversation_purpose,
                    conversation_type=self.conversation_type,
                )
            ]
        )

        inception_messages = prompt[0][0].to_messages()

        message_dict = {"role": "system", "content": inception_messages[0].content}

        if self.sales_conversation_utterance_chain.verbose:
            pass
            # print("\033[92m" + inception_messages[0].content + "\033[0m")
        return [message_dict]

    @time_logger
    def _streaming_generator(self):
        """
        生成用于部分 LLM 输出操作的流生成器​​。

        当销售代理需要在完整的 LLM 输出可用之前采取行动时，可以使用此方法。
        例如，在部分 LLM 输出上执行文本转语音时。该方法返回一个流生成器
        它可以操纵正在生成的法学硕士的部分输出。

        Returns
        -------
        generator
            A streaming generator for manipulating partial LLM output.

        Examples
        --------
        >>> streaming_generator = self._streaming_generator()
        >>> for chunk in streaming_generator:
        ...     print(chunk)
        Chunk 1, Chunk 2, ... etc.

        See Also
        --------
        https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb
        """

        messages = self._prep_messages()

        return self.sales_conversation_utterance_chain.llm.completion_with_retry(
            messages=messages,
            stop="<END_OF_TURN>",
            stream=True,
            model=self.model_name,
        )

    async def acompletion_with_retry(self, llm: Any, **kwargs: Any) -> Any:
        """
        使用坚韧重试异步完成调用。

        此方法使用 tenacity 库在失败时重试异步完成调用。
        它使用 '_create_retry_decorator' 方法创建一个重试装饰器，并将其应用于
        '_completion_with_retry' 函数进行实际的异步完成调用。

        Parameters
        ----------
        llm : Any
            The language model to be used for the completion.
        **kwargs : Any
            Additional keyword arguments to be passed to the completion function.

        Returns
        -------
        Any
            The result of the completion function call.

        Raises
        ------
        Exception
            If the completion function call fails after the maximum number of retries.
        """
        retry_decorator = _create_retry_decorator(llm)

        @retry_decorator
        async def _completion_with_retry(**kwargs: Any) -> Any:
            # Use OpenAI's async api https://github.com/openai/openai-python#async-api
            return await acompletion(**kwargs)

        return await _completion_with_retry(**kwargs)

    async def _astreaming_generator(self):
        """
        异步生成器在处理多个数据时减少 I/O 阻塞
         客户同时。

         该函数返回一个流生成器，它可以操作 LLM 的部分输出
         正在飞翔的一代。这在销售代理想要采取行动的场景中非常有用
         在完整的法学硕士输出可用之前。例如，如果我们想对部分 LLM 输出进行文本到语音转换。

         Returns
         -------
         AsyncGenerator
             A streaming generator which can manipulate partial output from an LLM in-flight of the generation.

         Examples
         --------
         >>> streaming_generator = self._astreaming_generator()
         >>> async for chunk in streaming_generator:
         >>>     await chunk ...
         Out: Chunk 1, Chunk 2, ... etc.

         See Also
         --------
         https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb
        """

        messages = self._prep_messages()

        return await self.acompletion_with_retry(
            llm=self.sales_conversation_utterance_chain.llm,
            messages=messages,
            stop="<END_OF_TURN>",
            stream=True,
            model=self.model_name,
        )

    def _call(self, inputs: Dict[str, Any]):
        """
        执行销售代理的一个步骤。

         此函数用对话的当前状态暂时覆盖输入，
         使用销售代理执行器或销售对话话语链生成代理的话语，
         将代理的响应添加到对话历史记录中，并返回 AI 消息。
         Parameters
         ----------
         inputs : Dict[str, Any]
             The initial inputs for the sales agent.

         Returns
         -------
         Dict[str, Any]
             The AI message generated by the sales agent.

        """
        # override inputs temporarily
        inputs = {
            "input": "",
            "conversation_stage": self.current_conversation_stage,
            "conversation_history": "\n".join(self.conversation_history),
            "salesperson_name": self.salesperson_name,
            "salesperson_role": self.salesperson_role,
            "company_name": self.company_name,
            "company_business": self.company_business,
            "company_values": self.company_values,
            "conversation_purpose": self.conversation_purpose,
            "conversation_type": self.conversation_type,
        }

        # Generate agent's utterance
        if self.use_tools:
            ai_message = self.sales_agent_executor.invoke(inputs)
            output = ai_message["output"]
        else:
            ai_message = self.sales_conversation_utterance_chain.invoke(
                inputs, return_intermediate_steps=True
            )
            output = ai_message["text"]

        # Add agent's response to conversation history
        agent_name = self.salesperson_name
        output = agent_name + ": " + output
        if "<END_OF_TURN>" not in output:
            output += " <END_OF_TURN>"
        self.conversation_history.append(output)

        if self.verbose:
            tool_status = "USE TOOLS INVOKE:" if self.use_tools else "WITHOUT TOOLS:"
            print(f"{tool_status}\n#\n#\n#\n#\n------------------")
            print(f"AI Message: {ai_message}")
            print()
            print(f"Output: {output.replace('<END_OF_TURN>', '')}")

        return ai_message

    @classmethod
    @time_logger
    def from_llm(cls, llm: ChatLiteLLM, verbose: bool = False, **kwargs):
        """
               用于从给定 ChatLiteLLM 实例初始化 SalesGPT 控制器的类方法。

                该方法建立了阶段分析链和销售对话话语链。它还检查是否有自定义提示
                是否要使用以及是否要为代理设置工具。如果要使用工具，它会建立知识库，
                获取工具、设置提示并使用工具初始化代理。如果不使用工具，则设置
        销售代理执行者和知识库为“无”。
                Parameters
                ----------
                llm : ChatLiteLLM
                    The ChatLiteLLM instance to initialize the SalesGPT Controller from.
                verbose : bool, optional
                    If True, verbose output is enabled. Default is False.
                **kwargs : dict
                    Additional keyword arguments.

                Returns
                -------
                SalesGPT
                    The initialized SalesGPT Controller.
        """
        stage_analyzer_chain = StageAnalyzerChain.from_llm(llm, verbose=verbose)
        sales_conversation_utterance_chain = SalesConversationChain.from_llm(
            llm, verbose=verbose
        )

        # Handle custom prompts
        use_custom_prompt = kwargs.pop("use_custom_prompt", False)
        custom_prompt = kwargs.pop("custom_prompt", None)

        sales_conversation_utterance_chain = SalesConversationChain.from_llm(
            llm,
            verbose=verbose,
            use_custom_prompt=use_custom_prompt,
            custom_prompt=custom_prompt,
        )

        # Handle tools
        use_tools_value = kwargs.pop("use_tools", False)
        if isinstance(use_tools_value, str):
            if use_tools_value.lower() not in ["true", "false"]:
                raise ValueError("use_tools must be 'True', 'False', True, or False")
            use_tools = use_tools_value.lower() == "true"
        elif isinstance(use_tools_value, bool):
            use_tools = use_tools_value
        else:
            raise ValueError(
                "use_tools must be a boolean or a string ('True' or 'False')"
            )
        sales_agent_executor = None
        knowledge_base = None

        if use_tools:
            product_catalog = kwargs.pop("product_catalog", None)
            tools = get_tools(product_catalog)

            prompt = CustomPromptTemplateForTools(
                template=SALES_AGENT_TOOLS_PROMPT,
                tools_getter=lambda x: tools,
                input_variables=[
                    "input",
                    "intermediate_steps",
                    "salesperson_name",
                    "salesperson_role",
                    "company_name",
                    "company_business",
                    "company_values",
                    "conversation_purpose",
                    "conversation_type",
                    "conversation_history",
                ],
            )
            llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)
            tool_names = [tool.name for tool in tools]
            output_parser = SalesConvoOutputParser(
                ai_prefix=kwargs.get("salesperson_name", ""), verbose=verbose
            )
            sales_agent_with_tools = LLMSingleActionAgent(
                llm_chain=llm_chain,
                output_parser=output_parser,
                stop=["\nObservation:"],
                allowed_tools=tool_names,
            )

            sales_agent_executor = CustomAgentExecutor.from_agent_and_tools(
                agent=sales_agent_with_tools,
                tools=tools,
                verbose=verbose,
                return_intermediate_steps=True,
            )

        return cls(
            stage_analyzer_chain=stage_analyzer_chain,
            sales_conversation_utterance_chain=sales_conversation_utterance_chain,
            sales_agent_executor=sales_agent_executor,
            knowledge_base=knowledge_base,
            model_name=llm.model,
            verbose=verbose,
            use_tools=use_tools,
            **kwargs,
        )
