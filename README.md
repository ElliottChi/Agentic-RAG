# Enterprise Agentic RAG Framework - 智慧交通法規諮詢助理 🚦

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.2-green)](https://www.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-orange)](https://openai.com/)
[![RAG](https://img.shields.io/badge/RAG-Advanced-purple)]()

## 專案簡介 (Introduction)
本專案是一個基於 **Agentic Workflow (代理人工作流)** 的企業級知識問答系統原型。

針對傳統 RAG (Retrieval-Augmented Generation) 系統常見的「檢索不精準」與「資源浪費」痛點，本系統導入了 **Router-Retriever 架構**。透過 LLM 進行意圖識別 (Intent Recognition)，自主判斷該進行閒聊、查詢知識庫，或是進行多步驟推理 (Reasoning)。

系統落地於 **LINE Bot** 平台，並解決了 **Cold Start (冷啟動延遲)** 與 **Stateless (無狀態記憶)** 等問題，實現了高精準度、低延遲的法規諮詢服務。

---

## 核心功能與架構 (Key Features & Architecture)

### 1. Agentic Workflow (代理人工作流)
系統不再被動檢索，而是具備「大腦」。
- **Router (路由機制)**：自動判斷使用者意圖（閒聊 vs. 查法規 vs. 計算罰款）。
- **ReAct Pattern**：具備多步驟推理能力。例如：「闖紅燈加沒戴安全帽罰多少？」Agent 會自主拆解任務，分別查詢兩條法規後，再呼叫計算工具加總。

### 2. Advanced RAG (進階檢索增強生成)
為了解決法律條文「差一字失之千里」的問題，採用 **Two-Stage Retrieval (兩階段檢索)**：
- **Stage 1 (Recall)**：使用 **Hybrid Search**，結合 **BM25 (關鍵字)** 與 **ChromaDB (語意向量)**，確保檢索廣度。
- **Stage 2 (Precision)**：引入 **Cross-Encoder Reranker** 模型，對檢索結果進行深度語意評分，大幅降低幻覺 (Hallucination)。

### 3. Engineering Optimization (工程優化)
- **Model Pre-loading (預熱機制)**：在 Server 啟動時強制載入 Embedding 與 Reranker 模型，解決首次推論超時 (Cold Start) 問題。
- **Session Memory (對話記憶)**：自行實作對話狀態管理，將 Chat History 動態注入 Prompt，賦予 Agent 上下文理解能力。

---

## 技術棧 (Tech Stack)

* **LLM Orchestration**: LangChain, LangGraph
* **Model**: OpenAI GPT-4o-mini (Reasoning), BAAI/bge-reranker-base (Reranking)
* **Vector DB**: ChromaDB
* **Retrieval**: BM25, EnsembleRetriever, ContextualCompressionRetriever
* **Backend**: Flask
* **Interface**: LINE Messaging API
* **Tools**: Python 3.10+, Docker (Optional)

---

## 專案結構 (Project Structure)

```bash
.
├── app.py                # 主程式：Flask Server, LINE Webhook, Memory Management
├── agent_logic.py        # Agent 大腦：定義 Tools, Router Prompt, ReAct 邏輯
├── rag_chain.py          # 底層 RAG：實作 Hybrid Search & Reranking 邏輯
├── requirements.txt      # 專案依賴套件
├── .env                  # 環境變數 (API Keys)
└── data/                 # 知識庫 PDF 存放區

```

---

# Enterprise Agentic RAG Framework - 智慧交通法規諮詢助理 🚦

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.2-green)](https://www.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-orange)](https://openai.com/)
[![RAG](https://img.shields.io/badge/RAG-Advanced-purple)]()

## 專案簡介 (Introduction)
本專案是一個基於 **Agentic Workflow (代理人工作流)** 的企業級知識問答系統原型。

針對傳統 RAG (Retrieval-Augmented Generation) 系統常見的「檢索不精準」與「資源浪費」痛點，本系統導入了 **Router-Retriever 架構**。透過 LLM 進行意圖識別 (Intent Recognition)，自主判斷該進行閒聊、查詢知識庫，或是進行多步驟推理 (Reasoning)。

系統落地於 **LINE Bot** 平台，並解決了 **Cold Start (冷啟動延遲)** 與 **Stateless (無狀態記憶)** 等工程難題，實現了高精準度、低延遲的法規諮詢服務。

---

## 核心功能與架構 (Key Features & Architecture)

### 1. Agentic Workflow (代理人工作流)
系統不再被動檢索，而是具備「大腦」。
- **Router (路由機制)**：自動判斷使用者意圖（閒聊 vs. 查法規 vs. 計算罰款）。
- **ReAct Pattern**：具備多步驟推理能力。例如：「闖紅燈加沒戴安全帽罰多少？」Agent 會自主拆解任務，分別查詢兩條法規後，再呼叫計算工具加總。

### 2. Advanced RAG (進階檢索增強生成)
為了解決法律條文「差一字失之千里」的問題，採用 **Two-Stage Retrieval (兩階段檢索)**：
- **Stage 1 (Recall)**：使用 **Hybrid Search**，結合 **BM25 (關鍵字)** 與 **ChromaDB (語意向量)**，確保檢索廣度。
- **Stage 2 (Precision)**：引入 **Cross-Encoder Reranker** 模型，對檢索結果進行深度語意評分，大幅降低幻覺 (Hallucination)。

### 3. Engineering Optimization (工程優化)
- **Model Pre-loading (預熱機制)**：在 Server 啟動時強制載入 Embedding 與 Reranker 模型，解決首次推論超時 (Cold Start) 問題。
- **Session Memory (對話記憶)**：自行實作對話狀態管理，將 Chat History 動態注入 Prompt，賦予 Agent 上下文理解能力。

---

## 技術棧 (Tech Stack)

* **LLM Orchestration**: LangChain, LangGraph
* **Model**: OpenAI GPT-4o-mini, BAAI/bge-reranker-base (Reranking)
* **Vector DB**: ChromaDB
* **Retrieval**: BM25, EnsembleRetriever, ContextualCompressionRetriever
* **Backend**: Flask
* **Interface**: LINE Messaging API
* **Tools**: Python 3.10+, Docker (Optional)

---

## 專案結構 (Project Structure)

```bash
.
├── app.py                # 主程式：Flask Server, LINE Webhook, Memory Management
├── agent_logic.py        # Agent 大腦：定義 Tools, Router Prompt, ReAct 邏輯
├── rag_chain.py          # 底層 RAG：實作 Hybrid Search & Reranking 邏輯
├── requirements.txt      # 專案依賴套件
├── .env                  # 環境變數 (API Keys)
└── data/                 # 知識庫 PDF 存放區

```

---

## 快速開始 (Quick Start)

### 1. Clone 專案

```bash
git clone [https://github.com/your-username/Agentic-RAG-Demo.git](https://github.com/your-username/Agentic-RAG-Demo.git)
cd Agentic-RAG-Demo

```

### 2. 安裝依賴

建議使用虛擬環境 (Virtual Environment)：

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

```

### 3. 設定環境變數

請建立 `.env` 檔案，並填入以下資訊：

```ini
OPENAI_API_KEY=sk-xxxxxx
LINE_CHANNEL_ACCESS_TOKEN=xxxxxx
LINE_CHANNEL_SECRET=xxxxxx

```

### 4. 啟動伺服器

```bash
python app.py

```

*注意：啟動時會進行系統預熱 (Pre-loading)，請等待約 30-60 秒直到出現 "Agent Ready" 字樣。*

---

## 實測案例 (Demo Scenarios)

### 情境 A：意圖識別 (Router)

> **User:** "你好，你是誰？"
> **Agent:** (不呼叫工具) "你好！我是你的交通法規助理..."
> *(節省檢索成本，回應速度 < 1s)*

### 情境 B：進階 RAG (Hybrid + Rerank)

> **User:** "機車闖紅燈罰多少？"
> **Agent:** (呼叫 `search_traffic_law`) "根據道路交通管理處罰條例..."
> *(精準引用法條，過濾掉汽車或行人的無關資訊)*

### 情境 C：多步推理 (Reasoning)

> **User:** "闖紅燈跟沒戴安全帽總共要罰多少？"
> **Agent:** > 1. 查詢闖紅燈罰款
> 2. 查詢未戴安全帽罰款
> 3. 呼叫 `calculate_fine` 計算總和
> 4. 回覆最終金額



## Author

**Elliott Chi**

* Master's in Information Management, NKUST
* Focus: NLP, Transformer Architecture, Agentic AI

