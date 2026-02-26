import os
import pickle
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from src.db.base import BaseRetriever

class BM25Store(BaseRetriever):
    """
    BM25 關鍵字檢索實作，繼承統一的 BaseRetriever 介面。
    使用 pickle 將 LangChain 的 BM25Retriever 暫存於本地端。
    """
    def __init__(self, index_path: str = "data/processed/bm25_index.pkl"):
        self.index_path = index_path
        self.retriever: Optional[BM25Retriever] = None
        self._load_index()

    def _load_index(self):
        """嘗試從硬碟載入已建立的 BM25 索引"""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "rb") as f:
                    self.retriever = pickle.load(f)
                print(f"[BM25Store] Loaded index from {self.index_path}")
            except Exception as e:
                print(f"[BM25Store] Failed to load index: {e}")
                self.retriever = None

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        將文件加入 BM25 索引並序列化儲存。
        :param documents: 包含 'page_content' 與 'metadata' 的字典列表
        """
        if not documents:
            return

        # 轉換回 LangChain Document 格式供 BM25Retriever 使用
        lc_docs = [
            Document(page_content=doc.get("page_content", ""), metadata=doc.get("metadata", {}))
            for doc in documents
        ]

        # 重新建立整個索引 (BM25 通常是靜態建立，若需增量需特別處理，此處採重建)
        self.retriever = BM25Retriever.from_documents(lc_docs)

        # 確保目錄存在
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # 存檔
        with open(self.index_path, "wb") as f:
            pickle.dump(self.retriever, f)
        print(f"[BM25Store] Saved index with {len(lc_docs)} documents to {self.index_path}")

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        BM25 關鍵字搜尋。
        :param query: 檢索關鍵字
        :param k: 返回數量限制
        :return: 格式化的結果列表
        """
        if not self.retriever:
            print("[BM25Store] Index not initialized. Return empty.")
            return []

        # 設定單次檢索數量
        self.retriever.k = k
        try:
            lc_docs = self.retriever.invoke(query)
        except Exception as e:
            print(f"[BM25Store] Search error: {e}")
            return []

        results = []
        # LangChain 的 BM25Retriever 回傳的結果預設沒有明確曝露內部 score 屬性
        # 若需要精確分數量化，需要深入底層或直接回傳順序當作參考。此處先給予預設值。
        for i, doc in enumerate(lc_docs):
            results.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "score": 1.0 / (i + 1) # 暫以倒數排名當作偽分數
            })
            
        return results
