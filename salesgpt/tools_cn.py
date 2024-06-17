import json
import os

import boto3
import requests
from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import BedrockChat
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from litellm import completion
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def setup_knowledge_base(
    product_catalog: str = None, model_name: str = "gpt-3.5-turbo"
):
    """
    我们假设产品目录只是一个文本字符串。
    """
    # 加载产品目录
    with open(product_catalog, "r") as f:
        product_catalog = f.read()

    text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
    texts = text_splitter.split_text(product_catalog)

    llm = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0)

    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_texts(
        texts, embeddings, collection_name="product-knowledge-base"
    )

    knowledge_base = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
    )
    return knowledge_base


def completion_bedrock(model_id, system_prompt, messages, max_tokens=1000):
    """
    使用 Anthropic Claude 调用高级 API 来生成消息。
    """
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime", region_name=os.environ.get("AWS_REGION_NAME")
    )

    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages,
        }
    )

    response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
    response_body = json.loads(response.get("body").read())

    return response_body


def get_product_id_from_query(query, product_price_id_mapping_path):
    # 从 JSON 文件加载 Product_price_id_mapping
    with open(product_price_id_mapping_path, "r") as f:
        product_price_id_mapping = json.load(f)

    # 将product_price_id_mapping序列化为JSON字符串以包含在提示中
    product_price_id_mapping_json_str = json.dumps(product_price_id_mapping)

    # 从product_price_id_mapping键动态创建枚举列表
    enum_list = list(product_price_id_mapping.values()) + [
        "No relevant product id found"
    ]
    enum_list_str = json.dumps(enum_list)

    prompt = f"""
    您是一名专家数据科学家，您正在开展一个项目，根据客户的需求向他们推荐产品。
    给出以下查询:
    {query}
   以及以下产品价格 id 映射:
    {product_price_id_mapping_json_str}
   返回与查询最相关的价格 ID。
    仅返回价格 ID，无其他文本。如果未找到相关价格 ID，则返回“未找到相关价格 ID”。
    您的输出将遵循此架构:
    {{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Price ID Response",
    "type": "object",
    "properties": {{
        "price_id": {{
        "type": "string",
        "enum": {enum_list_str}
        }}
    }},
    "required": ["price_id"]
    }}
    返回一个有效的可直接解析的 json，不要在代码片段中返回它或添加任何类型的解释！
    """
    prompt += "{"
    model_name = os.getenv("GPT_MODEL", "gpt-3.5-turbo-1106")

    if "anthropic" in model_name:
        response = completion_bedrock(
            model_id=model_name,
            system_prompt="你是一个得力的助手.",
            messages=[{"content": prompt, "role": "user"}],
            max_tokens=1000,
        )

        product_id = response["content"][0]["text"]

    else:
        response = completion(
            model=model_name,
            messages=[{"content": prompt, "role": "user"}],
            max_tokens=1000,
            temperature=0,
        )
        product_id = response.choices[0].message.content.strip()
    return product_id


def generate_stripe_payment_link(query: str) -> str:
    """Generate a stripe payment link for a customer based on a single query string."""

    # example testing payment gateway url
    PAYMENT_GATEWAY_URL = os.getenv(
        "PAYMENT_GATEWAY_URL", "https://agent-payments-gateway.vercel.app/payment"
    )
    PRODUCT_PRICE_MAPPING = os.getenv(
        "PRODUCT_PRICE_MAPPING", "example_product_price_id_mapping.json"
    )

    # use LLM to get the price_id from query
    price_id = get_product_id_from_query(query, PRODUCT_PRICE_MAPPING)
    price_id = json.loads(price_id)
    payload = json.dumps(
        {"prompt": query, **price_id, "stripe_key": os.getenv("STRIPE_API_KEY")}
    )
    headers = {
        "Content-Type": "application/json",
    }

    response = requests.request(
        "POST", PAYMENT_GATEWAY_URL, headers=headers, data=payload
    )
    return response.text


def get_mail_body_subject_from_query(query):
    prompt = f"""
   鉴于查询: "{query}", 分析内容并提取发送电子邮件所需的信息。所需信息包括收件人的电子邮件地址、电子邮件主题和电子邮件正文内容。 
    根据分析，返回一个Python格式的字典，其中键是“recipient”、“subject”和“body”，值是从查询中提取的相应信息。
例如，如果查询是关于发送电子邮件以通知某人即将发生的事件，则输出应如下所示：
    {{
        "recipient": "example@example.com",
        "subject": "即将举行的活动通知",
        "body": "亲爱的[Name]，我们想提醒您下周即将举行的活动。我们期待在那里见到您."
    }}
    现在，根据提供的查询，返回所描述的结构化信息。
    返回一个有效的可直接解析的 json，不要在代码片段中返回它或添加任何类型的解释！
    """
    model_name = os.getenv("GPT_MODEL", "gpt-3.5-turbo-1106")

    if "anthropic" in model_name:
        response = completion_bedrock(
            model_id=model_name,
            system_prompt="You are a helpful assistant.",
            messages=[{"content": prompt, "role": "user"}],
            max_tokens=1000,
        )

        mail_body_subject = response["content"][0]["text"]

    else:
        response = completion(
            model=model_name,
            messages=[{"content": prompt, "role": "user"}],
            max_tokens=1000,
            temperature=0.2,
        )
        mail_body_subject = response.choices[0].message.content.strip()
    print(mail_body_subject)
    return mail_body_subject


def send_email_with_gmail(email_details):
    """.env 应包含 GMAIL_MAIL 和 GMAIL_APP_PASSWORD 才能正常工作"""
    try:
        sender_email = os.getenv("GMAIL_MAIL")
        app_password = os.getenv("GMAIL_APP_PASSWORD")
        recipient_email = email_details["recipient"]
        subject = email_details["subject"]
        body = email_details["body"]
        # 创建 MIME 消息
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        # 使用 SSL 选项创建服务器对象
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(sender_email, app_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        return "邮件发送成功."
    except Exception as e:
        return f"电子邮件未成功发送，错误: {e}"


def send_email_tool(query):
    """根据单个查询字符串发送电子邮件"""
    email_details = get_mail_body_subject_from_query(query)
    if isinstance(email_details, str):
        email_details = json.loads(email_details)  # Ensure it's a dictionary
    print("EMAIL DETAILS")
    print(email_details)
    result = send_email_with_gmail(email_details)
    return result


def generate_calendly_invitation_link(query):
    """根据单个查询字符串生成日历邀请链接"""
    event_type_uuid = os.getenv("CALENDLY_EVENT_UUID")
    api_key = os.getenv("CALENDLY_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = "https://api.calendly.com/scheduling_links"
    payload = {
        "max_event_count": 1,
        "owner": f"https://api.calendly.com/event_types/{event_type_uuid}",
        "owner_type": "EventType",
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 201:
        data = response.json()
        return f"url: {data['resource']['booking_url']}"
    else:
        return "无法创建 Calendly 链接： "


def get_tools(product_catalog):
    # 查询get_tools可用于嵌入并找到相关工具
    # see here: https://langchain-langchain.vercel.app/docs/use_cases/agents/custom_agent_with_plugin_retrieval#tool-retriever

    # 我们目前只使用四个工具，但这是高度可扩展的！
    knowledge_base = setup_knowledge_base(product_catalog)
    tools = [
        Tool(
            name="ProductSearch",
            func=knowledge_base.run,
            description="useful for when you need to answer questions about product information or services offered, availability and their costs.",
        ),
        Tool(
            name="GeneratePaymentLink",
            func=generate_stripe_payment_link,
            description="useful to close a transaction with a customer. You need to include product name and quantity and customer name in the query input.",
        ),
        Tool(
            name="SendEmail",
            func=send_email_tool,
            description="Sends an email based on the query input. The query should specify the recipient, subject, and body of the email.",
        ),
        Tool(
            name="SendCalendlyInvitation",
            func=generate_calendly_invitation_link,
            description="""Useful for when you need to create invite for a personal meeting in Sleep Heaven shop. 
            Sends a calendly invitation based on the query input.""",
        ),
    ]

    return tools
