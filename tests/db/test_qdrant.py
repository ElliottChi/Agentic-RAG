import os
import pytest
from typing import List

from langchain_core.embeddings import Embeddings
from src.db.qdrant_store import QdrantStore

class DummyEmbeddings(Embeddings):
    """用於測試的虛擬 Embedding 模型"""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 固定回傳長度 10 的虛擬向量
        return [[0.1] * 10 for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1] * 10

def test_qdrant_store_workflow():
    """測試 QdrantStore 的基本儲存與檢索流程"""
    # 強制載入記憶體模式以進行無狀態測試
    os.environ["QDRANT_URL"] = ":memory:"
    
    embeddings = DummyEmbeddings()
    store = QdrantStore(
        collection_name="test_collection", 
        embedding_model=embeddings, 
        vector_size=10,
        qdrant_url=":memory:"
    )
    
    # 準備測試資料
    docs = [
        {"page_content": "這是一篇關於 AI 的測試文章", "metadata": {"source": "ai.txt"}},
        {"page_content": "這是另一篇關於 RAG 的測試文章", "metadata": {"source": "rag.txt"}},
    ]
    
    # 執行新增
    store.add_documents(docs)
    
    # 執行檢索
    results = store.similarity_search("AI", k=1)
    
    # 基本斷言驗證
    assert len(results) == 1
    assert "page_content" in results[0]
    assert "score" in results[0]
    assert results[0]["score"] > 0
    
    # 測試 Reranker 預留方法 (BaseRetriever 提供)
    reranked = store.rerank("AI", results, top_k=1)
    assert len(reranked) == 1
    assert reranked[0]["page_content"] == results[0]["page_content"]
