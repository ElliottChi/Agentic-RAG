import unittest
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any

from src.db.neo4j_store import Neo4jStore

class TestNeo4jStore(unittest.TestCase):

    @patch('src.db.neo4j_store.GraphDatabase.driver')
    def test_neo4j_store_workflow(self, mock_driver_class):
        """測試 Neo4jStore 的實體擷取、寫入與檢索邏輯 (Mocked)"""
        # 設定 Mock
        mock_driver_instance = MagicMock()
        mock_session_instance = MagicMock()
        
        # 設定 driver.session() 回傳我們建立的 mock_session
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session_instance
        mock_driver_class.return_value = mock_driver_instance

        # 初始化 Store
        store = Neo4jStore(uri="bolt://mock", user="mock", password="pwd")
        
        # 確認 driver 已被呼叫
        mock_driver_class.assert_called_once_with("bolt://mock", auth=("mock", "pwd"))

        # 整理測試資料 (包含大寫實體 'OpenAI', 'LangChain', 'Agent')
        docs = [
            {"page_content": "OpenAI provides powerful models.", "metadata": {"source": "doc1.txt"}},
            {"page_content": "LangChain helps build Agent systems.", "metadata": {"source": "doc2.txt"}},
        ]

        # 測試 1: 驗證實體擷取邏輯
        entities1 = store._extract_entities(docs[0]["page_content"])
        self.assertIn("OpenAI", entities1)
        
        entities2 = store._extract_entities(docs[1]["page_content"])
        self.assertIn("LangChain", entities2)
        self.assertIn("Agent", entities2)

        # 測試 2: 驗證新增邏輯 (add_documents)
        store.add_documents(docs)
        
        # 由於寫入了兩個 files, 每次寫入會有 1 次 MERGE Document，
        # 和對應數量實體的 MERGE Entity 動作
        # doc1 有 1 個實體 -> 2 calls
        # doc2 有 2 個實體 -> 3 calls
        # 總共預期 session.run 會被呼叫 5 次
        self.assertEqual(mock_session_instance.run.call_count, 5)

        # 測試 3: 驗證檢索邏輯 (similarity_search)
        
        # 設定 Mock session.run 的回傳值，模擬 Cypher 查詢結果
        mock_record_1 = {"content": "LangChain helps build Agent systems.", "source": "doc2.txt", "score": 2}
        
        mock_result_cursor = MagicMock()
        mock_result_cursor.__iter__.return_value = [mock_record_1]
        mock_session_instance.run.return_value = mock_result_cursor
        
        # 執行檢索 (查詢字串包含 'LangChain', 'Agent')
        results = store.similarity_search("Tell me about LangChain and Agent.", k=1)
        
        # 確認結果
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["page_content"], "LangChain helps build Agent systems.")
        self.assertEqual(results[0]["metadata"]["source"], "doc2.txt")
        self.assertEqual(results[0]["score"], 2)

        # 測試 4: BaseRetriever Rerank 方法
        reranked = store.rerank("query", results, top_k=1)
        self.assertEqual(len(reranked), 1)
        
        # 關閉連線測試
        store.close()
        mock_driver_instance.close.assert_called_once()
