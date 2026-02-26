import unittest
from langchain_core.messages import HumanMessage

from src.orchestration.graph import graph

class TestLangGraph(unittest.TestCase):
    def test_graph_routing_multi_hop(self):
        """
        測試 Graph 的路由邏輯：
        由於 Reviewer 後的 router 邏輯判斷：若 docs < 2 且 count < 3 則重複檢索。
        在此模擬情境下：
        第 1 次 researcher 執行後：docs 有 1 篇，count = 1 -> router 接回 researcher
        第 2 次 researcher 執行後：docs 有 2 篇，count = 2 -> router 將結束，跳到 END
        因此最終 search_count 應為 2，並會收集到 2 篇 Mock Retrieved Docs
        """
        initial_state = {
            "messages": [HumanMessage(content="What is LangGraph?")],
            "current_plan": "",
            "retrieved_docs": [],
            "search_count": 0
        }
        
        # 執行編排流程一直到結束
        # .invoke 會依據流程自動往下走
        final_state = graph.invoke(initial_state)
        
        # 驗證 Planner 邏輯有生效
        self.assertEqual(final_state["current_plan"], "Plan to research about: What is LangGraph?")
        
        # 驗證 Researcher 的迴圈邏輯與 Annotated(operator.add) 是否有合併清單
        self.assertEqual(final_state["search_count"], 2)
        self.assertEqual(len(final_state["retrieved_docs"]), 2)
        
        # 驗證每一輪迴圈加進去的內容
        self.assertIn("Attempt 1", final_state["retrieved_docs"][0]["page_content"])
        self.assertIn("Attempt 2", final_state["retrieved_docs"][1]["page_content"])
