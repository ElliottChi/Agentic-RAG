import unittest
from langchain_core.messages import HumanMessage
from unittest.mock import patch

from src.orchestration.graph import build_graph
from langgraph.checkpoint.memory import MemorySaver


class TestLangGraph(unittest.TestCase):
    @patch('src.orchestration.graph.planner_node')
    @patch('src.orchestration.graph.researcher_node')
    @patch('src.orchestration.graph.reviewer_node')
    @patch('src.orchestration.graph.generator_node')
    def test_graph_routing_multi_hop(self, mock_generator, mock_reviewer, mock_researcher, mock_planner):
        """
        測試 Graph 的路由邏輯：
        由於 Reviewer 後的 router 邏輯判斷：若 docs < 2 且 count < 3 則重複檢索。
        在此模擬情境下：
        第 1 次 researcher 執行後：docs 有 1 篇，count = 1 -> router 接回 researcher
        第 2 次 researcher 執行後：docs 有 2 篇，count = 2 -> router 將結束，跳到 END
        因此最終 search_count 應為 2，並會收集到 2 篇 Mock Retrieved Docs
        """
        
        # 設定 Planner Mock
        mock_planner.return_value = {"current_plan": "Plan to research about: What is LangGraph?"}
        
        # 設定 Researcher Mock (模擬兩次迭代)
        # 第一次回傳增加 1 篇 doc，第二次再回傳增加 1 篇 doc (LangGraph 會自動把陣列 extend)
        mock_researcher.side_effect = [
            {"retrieved_docs": [{"page_content": "Attempt 1"}], "search_count": 1},
            {"retrieved_docs": [{"page_content": "Attempt 2"}], "search_count": 2},
            {"retrieved_docs": [{"page_content": "Attempt 3"}], "search_count": 3},
        ]
        
        # 設定 Reviewer Mock (只是過渡，不改變狀態)
        mock_reviewer.return_value = {}
        
        # 設定 Generator Mock
        mock_generator.return_value = {"messages": [HumanMessage(content="Mocked Answer")]}

        # 重新編譯測試用的圖譜，這樣才能吃到 patch 後的節點
        memory = MemorySaver()
        test_graph = build_graph().compile(checkpointer=memory)

        initial_state = {
            "messages": [HumanMessage(content="What is LangGraph?")],
            "current_plan": "",
            "retrieved_docs": [],
            "search_count": 0
        }
        
        # 執行編排流程一直到結束
        final_state = test_graph.invoke(initial_state, config={"configurable": {"thread_id": "test_thread_routing"}})

        
        # 驗證 Planner 邏輯有生效
        self.assertEqual(final_state["current_plan"], "Plan to research about: What is LangGraph?")
        
        # 驗證 Researcher 的迴圈邏輯與 Annotated(operator.add) 是否有合併清單
        self.assertEqual(final_state["search_count"], 2)
        self.assertEqual(len(final_state["retrieved_docs"]), 2)
        
        # 驗證每一輪迴圈加進去的內容
        self.assertIn("Attempt 1", final_state["retrieved_docs"][0]["page_content"])
        self.assertIn("Attempt 2", final_state["retrieved_docs"][1]["page_content"])
