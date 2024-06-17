import re
from typing import Union

from langchain.agents.agent import AgentOutputParser
from langchain.agents.conversational.prompt import FORMAT_INSTRUCTIONS
from langchain.schema import AgentAction, AgentFinish  # OutputParserException


class SalesConvoOutputParser(AgentOutputParser):
    """
    SalesConvoOutputParser 是 SalesConvo 代理输出的解析器。

    Args:
        AgentOutputParser (_type_): _description_

    Returns:
        _type_: _description_
    """

    ai_prefix: str = "AI"  # change for salesperson_name
    verbose: bool = False

    def get_format_instructions(self):
        """
        获取代理的格式说明。

        Returns:
            str: 代理的格式说明.
        """
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if self.verbose:
            print("TEXT")
            print(text)
            print("-------")
        regex = r"Action: (.*?)[\n]*Action Input: (.*)"
        match = re.search(regex, text)
        if not match:
            return AgentFinish(
                {"output": text.split(f"{self.ai_prefix}:")[-1].strip()}, text
            )
        action = match.group(1)
        action_input = match.group(2)
        return AgentAction(action.strip(), action_input.strip(" ").strip('"'), text)

    @property
    def _type(self):
        """
        Returns: str
        """
        return "sales-agent"
