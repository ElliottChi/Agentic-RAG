import streamlit as st
import httpx
import asyncio
import uuid

# --- Page Config ---
st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Constants ---
API_URL = "http://localhost:8000/chat"

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ 系統狀態 (System Status)")
    st.success("🟢 Orchestration Layer: Active")
    st.success("🟢 Qdrant Vector DB: Connected")
    st.success("🟢 Neo4j Graph DB: Connected")
    st.info("此展示介面連接至本地 Agentic Workflow 引擎，展示多輪自動檢索與生成 (Agentic RAG) 能力。")
    
    if st.button("清除對話紀錄"):
        st.session_state.messages = []
        st.rerun()

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# --- Main UI ---
st.title("🤖 Deep Research Agent")
st.markdown("請在下方輸入您的技術或特定領域問題。Agent 將自主拆解問題、跨資料庫多輪檢索，並融合生成最終解答。")

# 顯示對話歷史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # 若為 Assistant 訊息且帶有隱藏的推論過程/來源，則使用 Expander 展示
        if msg["role"] == "assistant" and msg.get("reasoning"):
            with st.expander("🕵️ Agent 思考與檢索過程", expanded=False):
                # 顯示 Reasoning Logs
                st.markdown("**執行軌跡 (Execution Plan):**")
                for log in msg["reasoning"].get("logs", []):
                    st.markdown(f"- {log}")
                
                st.divider()
                
                # 顯示 Sources
                st.markdown("**檢索來源 (Retrieved Documents):**")
                sources = msg["reasoning"].get("sources", [])
                if not sources:
                    st.markdown("未檢索到相關文檔。")
                else:
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Document {i+1}**")
                        st.json(doc["metadata"])
                        st.text(doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"])

# --- Chat Input ---
if user_input := st.chat_input("請輸入問題"):
    # 1. 顯示 User 訊息
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    # 2. 顯示 Assistant Loading 狀態 & 請求 API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🧠 正在思考並檢索資料中 (Planning & Researching)...")
        
        try:
            # 同步呼叫 httpx (Streamlit 預設環境為同步)
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    API_URL, 
                    json={"message": user_input, "thread_id": st.session_state.thread_id}
                )
                response.raise_for_status()
                data = response.json()
                
                answer = data.get("answer", "無法取得回答。")
                logs = data.get("reasoning_logs", [])
                sources = data.get("sources", [])
                
                # 更新畫面字體
                message_placeholder.markdown(answer)
                
                # 即時展示 Expander (不存入 session_state 的重繪，自己先畫一次)
                with st.expander("🕵️ Agent 思考與檢索過程", expanded=False):
                    st.markdown("**執行軌跡 (Execution Plan):**")
                    for log in logs:
                        st.markdown(f"- {log}")
                    st.divider()
                    st.markdown("**檢索來源 (Retrieved Documents):**")
                    if not sources:
                        st.markdown("未檢索到相關文檔。")
                    else:
                        for i, doc in enumerate(sources):
                            st.markdown(f"**Document {i+1}**")
                            st.json(doc["metadata"])
                            st.text(doc["content"][:200] + "...")
                            
                # 存回 Session State 以供重繪
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "reasoning": {
                        "logs": logs,
                        "sources": sources
                    }
                })
                
        except Exception as e:
            error_msg = f"❌ API 請求失敗或超時: {str(e)}"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
