import os
import uuid
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from langchain_core.embeddings import Embeddings

from src.db.base import BaseRetriever

# 確保讀取環境變數 (不硬編碼憑證)
load_dotenv()

class QdrantStore(BaseRetriever):
    """
    Qdrant 向量檢索實作，繼承統一的 BaseRetriever 介面。
    遵循嚴格 Type Hinting，不 Hardcode 任何密碼。
    """

    def __init__(self, collection_name: str, embedding_model: Embeddings, vector_size: int = 1536, qdrant_url: Optional[str] = None, qdrant_api_key: Optional[str] = None, qdrant_path: Optional[str] = None):
        """
        初始化 Qdrant 連線與設定。
        :param collection_name: Collection 名稱
        :param embedding_model: LangChain 的 Embeddings 模型實例
        :param vector_size: 向量維度
        :param qdrant_url: Qdrant 服務位址 (優先使用)
        :param qdrant_api_key: Qdrant API Key (優先使用)
        :param qdrant_path: Qdrant 本地端持久化路徑
        """
        url = qdrant_url or os.environ.get("QDRANT_URL", None)
        path = qdrant_path or os.environ.get("QDRANT_PATH", None)
        api_key = qdrant_api_key or os.environ.get("QDRANT_API_KEY", None)

        if path:
            self.client = QdrantClient(path=path)
        elif url == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(url=url or "http://localhost:6333", api_key=api_key)

        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # 確保 Collection 存在
        self._ensure_collection_exists(vector_size)

    def _ensure_collection_exists(self, vector_size: int) -> None:
        """
        內部方法：檢查 Collection 是否存在，若無則建立。
        :param vector_size: 向量維度
        """
        try:
            collections_response = self.client.get_collections()
            collection_names = [col.name for col in collections_response.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
        except Exception as e:
            raise RuntimeError(f"Failed to ensure Qdrant collection: {e}")

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        將文件加入向量資料庫。
        :param documents: 包含 'page_content' (str) 與 'metadata' (dict) 鍵的字典列表。
        """
        if not documents:
            return

        texts = [doc.get("page_content", "") for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]

        # 批次取得嵌入向量
        vectors = self.embedding_model.embed_documents(texts)

        points = []
        for i, vector in enumerate(vectors):
            point_id = str(uuid.uuid4())
            payload = {
                "page_content": texts[i],
                "metadata": metadatas[i]
            }
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        向量相似度搜尋。
        :param query: 使用者查詢字串
        :param k: 返回數量限制
        :param kwargs: 其他預留查詢參數
        :return: 包含內容、依賴資料與相似度分數的字典列表
        """
        query_vector = self.embedding_model.embed_query(query)
        
        # 使用最新 Query API 進行檢索
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=k,
            with_payload=True,
            **kwargs
        )

        results = []
        for scored_point in search_result.points:
            payload = scored_point.payload or {}
            results.append({
                "page_content": payload.get("page_content", ""),
                "metadata": payload.get("metadata", {}),
                "score": scored_point.score
            })
        
        return results
