from typing import TypedDict, Annotated, List, Any
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    Agentic RAG 系統的全局狀態定義 (State)。
    傳遞於 LangGraph 各節點之間。
    """
    
    # 對話歷史，新訊息會附加到現有清單，而非覆蓋
    messages: Annotated[List[BaseMessage], operator.add]
    
    # 當前拆解出的檢索計畫/子問題
    current_plan: str
    
    # 從資料庫中檢索出的各類文件集合
    # 使用 operator.add 來累積不同次 Multi-hop 檢索結果
    retrieved_docs: Annotated[List[Any], operator.add]
    
    # 已執行的 Multi-hop 檢索次數
    search_count: int
