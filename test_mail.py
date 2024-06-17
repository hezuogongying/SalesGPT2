import os
from salesgpt.tools import send_email_with_gmail  # 根据需要调整导入路径
from dotenv import load_dotenv
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

load_dotenv()


# 设置测试环境变量
def send_simple_email(recipient_email, subject, body):
    try:
        sender_email = os.getenv("GMAIL_MAIL")
        app_password = os.getenv("GMAIL_APP_PASSWORD")
        print(sender_email)
        print(app_password)
        # 确保发件人电子邮件和应用程序密码不是“无”
        if not sender_email or not app_password:
            return "Sender email or app password not set."

        # 创建 MIME 消息
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        # 使用 SSL 选项创建服务器对象
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, app_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        return "Email sent successfully."
    except Exception as e:
        return f"Failed to send email: {e}"


# 测试电子邮件详细信息
recipient_email = "makovoz.ilja@gmail.com"
subject = "Test Email"
body = "This is a test email sent from the Python script without using LLM."

# 发送测试电子邮件
result = send_simple_email(recipient_email, subject, body)
print(result)
