from src.orchestration.state import AgentState

def reviewer_node(state: AgentState) -> dict:
    """
    Reviewer 節點
    職責：檢查 retrieved_docs 的內容是否已經足夠回答使用者的問題。
    """
    docs = state.get("retrieved_docs", [])
    current_count = state.get("search_count", 0)
    
    print(f"Reviewer check. We have {len(docs)} documents and searched {current_count} times.")
    
    # 此節點目前只是檢視，並將資訊交由條件路由判斷 (StateGraph 的 conditional edge)。
    # 實務上這裡可以呼叫 LLM 對資料進行評分 (Grader 工具)。
    # 我們並不需要修改任何狀態，回傳空字典或本身即可，但必須有此節點作為路由分支點。
    return {}
