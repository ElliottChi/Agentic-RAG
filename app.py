# app.py
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import os
from dotenv import load_dotenv
from agent_logic import get_traffic_agent
from rag_chain import get_initialized_chain
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

app = Flask(__name__)

line_bot_api = LineBotApi(os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('LINE_CHANNEL_SECRET'))

print("正在進行系統預熱 (Pre-loading)...這會花一點時間，請耐心等待...")

get_initialized_chain()

print("RAG Chain 預熱完成！")

print("正在啟動 Agentic RAG 系統...")
traffic_agent = get_traffic_agent()
print("系統啟動完成！Agent Ready.")

# 建立一個全域變數來存記憶
# 格式: { "User_ID_123": [HumanMessage, AIMessage, ...], ... }
user_chat_history = {}

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_msg = event.message.text
    user_id = event.source.user_id
    
    try:
        print(f"Agent 收到訊息: {user_msg}")
        
        current_history = user_chat_history.get(user_id, [])
        result = traffic_agent.invoke({
            "input": user_msg,
            "chat_history": current_history
        })
        reply_text = result["output"]

        current_history.append(HumanMessage(content=user_msg))
        current_history.append(AIMessage(content=reply_text))
        
        if len(current_history) > 10:
            current_history = current_history[-10:]
            
        user_chat_history[user_id] = current_history
        
    except Exception as e:
        print(f"Agent Error: {e}")
        reply_text = "抱歉，系統運算中發生錯誤，請稍後再試。"

    try:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply_text)
        )
    except Exception as e:
        print(f"LINE API Error: {e}")

if __name__ == "__main__":
    app.run(port=5000)