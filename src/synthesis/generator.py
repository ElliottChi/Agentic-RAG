from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
# 實務上會使用真實的 LLM，例如 ChatOpenAI
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

from src.orchestration.state import AgentState

def generator_node(state: AgentState) -> dict:
    """
    Synthesis Layer (Generator 節點)
    負責接收 retrieved_docs 與原始問題 (messages)，
    使用嚴格的 Prompt 指示 LLM 依據檢索文本進行回答並標示來源。
    """
    messages = state.get("messages", [])
    docs = state.get("retrieved_docs", [])
    
    # 取出使用者的最後一個問題
    question = messages[-1].content if messages else ""
    
    # 整理檢索到的文檔內容與來源
    formatted_docs = []
    for i, doc in enumerate(docs):
        content = doc.get("page_content", "")
        source = doc.get("metadata", {}).get("source", "Unknown")
        formatted_docs.append(f"Document {i+1} (Source: {source}):\n{content}")
        
    context_str = "\n\n".join(formatted_docs)
    
    system_prompt = f"""你是一個聰明的交通法規助理。你的任務是專注於協助使用者解決「交通法律與法規」相關的問題。
1. 如果使用者是在閒聊或打招呼 (例如：你好、早安、謝謝)，請直接親切回覆。
2. 【重要限制】如果使用者詢問的問題完全無關「交通、車輛、道路、法規、罰單」等領域 (例如：詢問信用卡優惠、股市、旅遊景點等)，你必須禮貌地「拒絕回答」，並主動聲明你是一個交通法規助理，只能回答交通相關問題。絕對不可以回答領域外的知識。
3. 如果使用者是在問交通法規問題，請優先查詢下方知識庫文件來回答問題。
即使文檔內容未直接提及具體金額，若你有確切的台灣交通法規常識（例如：機車未戴安全帽罰鍰為新台幣500元），請直接提供答案，並向使用者說明這是通用法規常識，而檢索文檔僅提供補充。
如果真的完全無法判斷，再回答無法回答。

[檢索文檔開始]
{context_str}
[檢索文檔結束]"""

    # 將 System Prompt 插入作為首筆訊息，後接所有對話歷史
    final_messages = [SystemMessage(content=system_prompt)] + messages
    
    # === LLM 呼叫區塊 ===
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(final_messages)
    answer = response.content
            
    # 將生成結果包裝成 AIMessage 附加到 messages 狀態中
    return {
        "messages": [AIMessage(content=answer)]
    }
