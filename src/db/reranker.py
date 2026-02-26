from sentence_transformers import CrossEncoder

class BGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "cpu"):
        """
        初始化交叉注意力模型。
        為了保持中文精準度，選用智源研究院開源的 BGE Reranker。
        """
        print(f"Loading Cross-Encoder Reranker: {model_name}...")
        self.model = CrossEncoder(model_name, max_length=512, device=device)

    def rerank(self, query: str, docs: list, top_k: int = 3) -> list:
        """
        對已經檢索出來的文件 (docs) 進行重新打分排序
        """
        if not docs:
            return []

        # 1. 將 (query, doc_content) 配對成模型吃得進去的格式
        pairs = []
        for doc in docs:
            content = doc.get("page_content", "")
            pairs.append([query, content])

        # 2. 丟給 Cross-Encoder 進行精密打分
        # scores 會是一個陣列，例如: [0.95, -1.2, 5.4, 0.3 ...]，分數越高越相關
        scores = self.model.predict(pairs)

        # 3. 將分數與原本的 doc 綁定
        scored_docs = []
        for score, doc in zip(scores, docs):
            doc_copy = doc.copy()
            # 我們將打分結果附著在 metadata 裡，方便 LangSmith 追蹤除錯
            if "metadata" not in doc_copy:
                doc_copy["metadata"] = {}
            doc_copy["metadata"]["rerank_score"] = float(score)
            scored_docs.append(doc_copy)

        # 4. 依照分數由高至低降冪排序
        scored_docs.sort(key=lambda x: x["metadata"]["rerank_score"], reverse=True)

        # 5. 只取最精華的 top_k 筆資料回傳，幫 LLM 省 Token 又能避開雜亂資訊
        return scored_docs[:top_k]
