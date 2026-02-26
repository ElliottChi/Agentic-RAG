from typing import TypedDict, Annotated, List, Any
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    Agentic RAG 系統的全局狀態定義 (State)。
    傳遞於 LangGraph 各節點之間。
    """
    
    # 對話歷史，使用 Annotated 確保新訊息會附加(append)到現有清單，而非覆蓋
    messages: Annotated[List[BaseMessage], operator.add]
    
    # 當前拆解出的檢索計畫/子問題
    current_plan: str
    
    # 從資料庫中檢索出的各類文件集合
    # 同樣使用 operator.add 來累積不同次 Multi-hop 檢索結果
    retrieved_docs: Annotated[List[Any], operator.add]
    
    # 記錄檢索次數，避免路由無限迴圈
    search_count: int
