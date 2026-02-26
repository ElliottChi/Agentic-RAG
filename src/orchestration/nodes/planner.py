import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.orchestration.state import AgentState

def planner_node(state: AgentState) -> dict:
    """
    Planner 節點
    職責：讀取歷史對話 (messages)，判斷使用者的意圖，
    如果有多輪對話，會將代名詞改寫成獨立檢索詞 (Standalone Query)。
    """
    messages = state.get("messages", [])
    
    if not messages:
        return {"current_plan": "No questions asked."}

    last_message = messages[-1].content
    
    # 若對話只有一句，無須參考上下文
    if len(messages) <= 1:
        plan = f"Plan to research about: {last_message}"
    else:
        # 使用 LLM 進行 Query Rewriting
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        chat_history = []
        for msg in messages[:-1]:
            role = "User" if msg.type == "human" else "Assistant"
            chat_history.append(f"{role}: {msg.content}")
        history_str = "\n".join(chat_history)
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "你是一個查詢重寫助理。請閱讀以下的對話歷史，並將使用者的最新問題，改寫成一個不需要上下文也能完全理解的獨立檢索詞（Standalone Query）。如果不需要改寫，請直接輸出原問題。\n\n[對話歷史開始]\n{history}\n[對話歷史結束]"),
            ("user", "最新問題: {question}")
        ])
        
        prompt_value = prompt_template.invoke({"history": history_str, "question": last_message})
        print("Planner 正在進行 Query Rewriting...")
        response = llm.invoke(prompt_value)
        rewritten_query = response.content.strip()
        print(f"原問題: {last_message} -> 重寫後: {rewritten_query}")
        
        plan = f"Plan to research about: {rewritten_query}"

    return {
        "current_plan": plan,
        "search_count": 0,
        "retrieved_docs": []
    }
