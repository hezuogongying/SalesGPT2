import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatLiteLLM

from salesgpt.agents import SalesGPT

load_dotenv()

llm = ChatLiteLLM(temperature=0.9, model_name="gpt-3.5-turbo-0613")

sales_agent = SalesGPT.from_llm(
    llm,
    verbose=False,
    salesperson_name="Ted Lasso",
    salesperson_role="Sales Representative",
    company_name="Sleep Haven",
    company_business="""睡眠天堂 
                            是一家提供优质床垫的公司
                            为顾客提供最舒适、最
                            可能提供支持性睡眠体验。 
                            我们提供一系列高品质床垫，
                            枕头和床上用品 
                            旨在满足独特的 
                            我们客户的需求。""",
)

sales_agent.seed_agent()

# get generator of the LLM output
generator = sales_agent.step(stream=True)

# operate on streaming LLM output in near-real time
# for instance, do something after each full sentence is generated
for chunk in generator:
    print(chunk)
