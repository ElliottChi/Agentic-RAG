import os
import json
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage
from src.orchestration.graph import graph
from src.db.qdrant_store import QdrantStore
from src.db.neo4j_store import Neo4jStore
from src.orchestration.nodes.researcher import get_embeddings

def prepare_mock_dbs():
    """模擬在 Qdrant 與 Neo4j 中寫入初始資料"""
    docs = [
        {"page_content": "React 是一個用於建構使用者介面的 JavaScript 函式庫。", "metadata": {"source": "react_doc.txt"}},
        {"page_content": "LangGraph 是一個建立在 LangChain 之上，用於建立有狀態、多主體應用程式的框架。", "metadata": {"source": "langgraph_doc.txt"}},
        {"page_content": "Neo4j 是一個強大的圖資料庫，適合處理高度關聯的資料。", "metadata": {"source": "neo4j_doc.txt"}}
    ]
    
    # 使用 Memory Qdrant 寫入資料
    qdrant = QdrantStore(
        collection_name="deep_research_rag", 
        embedding_model=get_embeddings(), 
        vector_size=384,
        qdrant_url=":memory:"
    )
    qdrant.add_documents(docs)
    
    return docs

@patch('src.orchestration.nodes.researcher.Neo4jStore')
def main(mock_neo4j_class):
    print("開始執行端到端整合測試...")
    
    # 設定環境變數供 Node 使用 Memory Qdrant
    os.environ["QDRANT_URL"] = ":memory:"
    
    # 準備資料到 Qdrant
    prepare_mock_dbs()
    print("已成功將資料寫入 Qdrant (:memory:).")
    
    # 設定 Mock Neo4j 的行為
    mock_neo4j_instance = MagicMock()
    mock_neo4j_instance.similarity_search.return_value = [
        {"page_content": "LangGraph 是一個建立在 LangChain 之上，用於建立有狀態、多主體應用程式的框架。", "metadata": {"source": "neo4j_mock_db", "engine": "neo4j"}}
    ]
    mock_neo4j_class.return_value = mock_neo4j_instance

    # 初始化狀態與問題
    initial_state = {
        "messages": [HumanMessage(content="請問什麼是 LangGraph？它與 React 有什麼不同？")],
        "current_plan": "",
        "retrieved_docs": [],
        "search_count": 0
    }
    
    print("啟動 LangGraph 工作流...")
    
    # 執行工作流
    final_state = graph.invoke(initial_state)
    
    # 記錄日誌
    log_data = {
        "final_plan": final_state.get("current_plan"),
        "total_searches": final_state.get("search_count"),
        "total_retrieved_docs": len(final_state.get("retrieved_docs", [])),
        "docs": [doc for doc in final_state.get("retrieved_docs", [])],
        "final_answer": final_state.get("messages", [])[-1].content if final_state.get("messages") else "N/A"
    }
    
    with open("execution_log.json", "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
        
    print(f"工作流執行完畢！執行軌跡已寫入 execution_log.json。")

if __name__ == "__main__":
    main()
