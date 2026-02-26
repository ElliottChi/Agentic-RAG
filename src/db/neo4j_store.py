import os
import re
import uuid
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from neo4j import GraphDatabase
from src.db.base import BaseRetriever

load_dotenv()

class Neo4jStore(BaseRetriever):
    """
    Neo4j 圖資料庫檢索實作，繼承統一的 BaseRetriever 介面。
    主要職責為將文件與提取出的簡單實體寫入圖譜中，並提供基於 Cypher 的關鍵字相似度/關聯檢索。
    """

    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        """
        初始化 Neo4j Driver 連線。
        支援顯式傳入以利測試或手動覆寫。
        """
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.environ.get("NEO4J_USERNAME", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "password")

        # 使用 driver 管理連線池
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        """關閉連線"""
        self.driver.close()

    def _extract_entities(self, text: str) -> List[str]:
        """
        簡單的 Entity Extraction：擷取文本中連續的英文字母大寫詞彙或中文詞彙(這裡以簡單 Regex 處理大寫英文字)。
        未來的深度識別應由 Agent (Orchestration Layer) 處理後寫入 metadata，此處僅為底層保底邏輯。
        """
        # 簡單擷取: 大寫開頭的英文單字，長度 >= 2
        entities = set(re.findall(r'\b[A-Z][a-zA-Z]+\b', text))
        return list(entities)

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        將文件以 Node 形式寫入 Neo4j，並將擷取出的 Entity 也建立 Node，以 :MENTIONS 關聯。
        """
        if not documents:
            return

        with self.driver.session() as session:
            for doc in documents:
                content = doc.get("page_content", "")
                metadata = doc.get("metadata", {})
                doc_id = str(uuid.uuid4())
                source = metadata.get("source", "unknown")

                # 提取實體
                entities = self._extract_entities(content)

                # Cypher: 建立 Document 節點
                create_doc_query = """
                MERGE (d:Document {id: $doc_id})
                SET d.content = $content, d.source = $source
                RETURN d
                """
                session.run(create_doc_query, doc_id=doc_id, content=content, source=source)

                # Cypher: 建立 Entity 節點並與 Document 建立關聯
                for entity in entities:
                    # 使用 MERGE 避免實體重複
                    create_entity_query = """
                    MATCH (d:Document {id: $doc_id})
                    MERGE (e:Entity {name: $entity_name})
                    MERGE (d)-[:MENTIONS]->(e)
                    """
                    session.run(create_entity_query, doc_id=doc_id, entity_name=entity)

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        執行圖譜檢索。
        基本策略：根據 Query 提取實體，找出包含這些實體的 Document。
        """
        query_entities = self._extract_entities(query)
        
        # 若無實體，直接回傳空 (這裡可後續擴充全文檢索或向量擴充)
        if not query_entities:
            return []

        results = []
        with self.driver.session() as session:
            # Cypher: 尋找符合任一實體的 Document，並依照符合的實體數量(關聯度)排序
            search_query = """
            MATCH (d:Document)-[:MENTIONS]->(e:Entity)
            WHERE e.name IN $entities
            WITH d, count(e) as score
            ORDER BY score DESC
            LIMIT $k
            RETURN d.content AS content, d.source AS source, score
            """
            
            records = session.run(search_query, entities=query_entities, k=k)
            for record in records:
                results.append({
                    "page_content": record["content"],
                    "metadata": {"source": record["source"]},
                    "score": record["score"]  # 在這裡 score 是命中的實體數
                })
        
        return results
