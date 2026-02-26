from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.orchestration.state import AgentState
from src.orchestration.nodes.planner import planner_node
from src.orchestration.nodes.researcher import researcher_node
from src.orchestration.nodes.reviewer import reviewer_node
from src.synthesis.generator import generator_node

def route_after_review(state: AgentState) -> Literal["researcher", "generator"]:
    """
    條件路由判斷：
    如果檢索資料不足，且 search_count < 3，則回到 researcher 繼續檢索。
    否則結束檢索，進入 generator 節點生成最終答案。
    """
    count = state.get("search_count", 0)
    docs = state.get("retrieved_docs", [])
    
    # 簡化邏輯：如果檢索不到 2 篇且嘗試次數小於 3 則繼續找
    if len(docs) < 2 and count < 3:
        return "researcher"
    
    return "generator"

def build_graph() -> StateGraph:
    """
    建構 LangGraph 編排層。
    流程: START -> planner -> researcher -> reviewer -> (route) -> generator -> END
    """
    workflow = StateGraph(AgentState)
    
    # 註冊 Nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("generator", generator_node)
    
    # 定義 Edges
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "reviewer")
    
    # 條件邊緣 (Conditional Edge) 從 reviewer 切出
    workflow.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "researcher": "researcher", # 繼續檢索
            "generator": "generator"    # 結束檢索，進入生成層
        }
    )
    
    # 生成完畢後結束
    workflow.add_edge("generator", END)
    
    return workflow

# 實例化 MemorySaver 作為 Checkpointer
memory = MemorySaver()

# 編譯並對外提供圖譜實例，加入 checkpointer
graph = build_graph().compile(checkpointer=memory)
