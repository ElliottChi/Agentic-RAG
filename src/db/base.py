from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRetriever(ABC):
    """
    統一的檢索抽象層介面
    所有參與 RAG 檢索的資料庫 (例如 Qdrant, Neo4j) 必須實作此介面。
    """

    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        將文件新增至資料庫
        :param documents: 包含 'page_content' 和 'metadata' 的字典列表
        """
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        根據查詢字串進行相似度檢索
        :param query: 查詢字串
        :param k: 返回數量
        :param kwargs: 其他特定資料庫的過濾條件
        :return: 包含 'page_content' 和 'metadata' (可能包含 'score') 的字典列表
        """
        pass

    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        預留的 Cross-Encoder Reranking 模組呼叫位置
        子類別可選擇性覆寫，或統一由外部 Orchestration 層呼叫
        :param query: 原始查詢
        :param results: 初步檢索結果
        :param top_k: 重排後返回數量
        :return: 重排後的文件列表
        """
        # TODO: 實作與 Cross-Encoder 整合的通用重排序邏輯
        # 目前暫時直接回傳截斷後的結果
        return results[:top_k]
