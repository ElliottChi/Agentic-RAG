import os
from langchain_huggingface import HuggingFaceEmbeddings
from src.orchestration.state import AgentState
from src.db.qdrant_store import QdrantStore
from src.db.neo4j_store import Neo4jStore
from src.db.bm25_store import BM25Store

# 模組層級初始化 Embeddings (避免每次呼叫節點都重新載入模型)
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        # 改用更強大且開源免費的繁中嵌入模型 BAAI/bge-base-zh-v1.5
        _embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-zh-v1.5",
            encode_kwargs={'normalize_embeddings': True}
        )
    return _embeddings

def researcher_node(state: AgentState) -> dict:
    """
    Researcher 節點
    職責：根據 current_plan，前往 Qdrant (Vector DB) 與 Neo4j (Graph DB) 檢索文獻，並合併結果。
    """
    current_plan = state.get("current_plan", "")
    current_count = state.get("search_count", 0)
    
    print(f"Researcher invoked. Current count: {current_count}. Plan: {current_plan}")
    
    docs = []
    
    # 初始化資料庫連線
    # 測試環境下，如果是 :memory: 將由環境變數或上層決定
    qdrant_path = os.environ.get("QDRANT_PATH", "data/qdrant_db")
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    
    try:
        # 1. 檢索 Qdrant (向量)
        qdrant_store = QdrantStore(
            collection_name="deep_research_rag_bge", 
            embedding_model=get_embeddings(),
            vector_size=768, # BAAI/bge-base-zh-v1.5 維度為 768
            qdrant_path=qdrant_path
        )
        qdrant_results = qdrant_store.similarity_search(current_plan, k=5)
        docs.extend(qdrant_results)
    except Exception as e:
        print(f"Qdrant search error: {e}")

    try:
        # 2. 檢索 Neo4j (圖譜)
        neo4j_store = Neo4jStore(uri=neo4j_uri)
        neo4j_results = neo4j_store.similarity_search(current_plan, k=5)
        # 確保結果為字典格式並標記來源引擎
        for res in neo4j_results:
            if "metadata" not in res:
                res["metadata"] = {}
            res["metadata"]["engine"] = "neo4j"
        docs.extend(neo4j_results)
        neo4j_store.close()
    except Exception as e:
        print(f"Neo4j search error: {e}")
    
    try:
        # 3. 檢索 BM25 (關鍵字)
        bm25_store = BM25Store()
        bm25_results = bm25_store.similarity_search(current_plan, k=5)
        for res in bm25_results:
            if "metadata" not in res:
                res["metadata"] = {}
            res["metadata"]["engine"] = "bm25"
        docs.extend(bm25_results)
    except Exception as e:
        print(f"BM25 search error: {e}")

    # ===== 去重邏輯 (Deduplication) =====
    # 基於 page_content 進行簡單去重，確保 LLM 不會吃到重複的字串
    unique_docs = []
    seen_contents = set()
    for doc in docs:
        content = doc.get("page_content", "")
        if content not in seen_contents:
            seen_contents.add(content)
            unique_docs.append(doc)
    
    print(f"Total retrieved docs before dedup: {len(docs)}, after dedup: {len(unique_docs)}")

    # 回傳更新的狀態
    return {
        "retrieved_docs": unique_docs,  # 搭配 operator.add 會自動 append
        "search_count": current_count + 1
    }
