from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uuid

# 如果需要跨域請求，這在容器化架構也有用
from fastapi.middleware.cors import CORSMiddleware

from langchain_core.messages import HumanMessage
from src.orchestration.graph import graph

app = FastAPI(title="Deep Research Agent API", description="Agentic RAG Backend Server", version="1.0.0")

# 允許 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None

class SourceDoc(BaseModel):
    content: str
    metadata: dict

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDoc]
    reasoning_logs: List[str]

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    
    # 建立 LangGraph 初始狀態
    thread_id = request.thread_id or str(uuid.uuid4())
    inputs = {
        "messages": [HumanMessage(content=request.message)],
        "current_plan": "",
        "retrieved_docs": [],
        "search_count": 0
    }
    
    try:
        final_state = graph.invoke(inputs)
        
        # 解析返回狀態
        final_messages = final_state.get("messages", [])
        answer = final_messages[-1].content if final_messages else "No answer generated."
        
        docs = final_state.get("retrieved_docs", [])
        sources = [
            SourceDoc(content=doc.get("page_content", ""), metadata=doc.get("metadata", {}))
            for doc in docs
        ]
        
        # 準備 Reasoning Logs 供前端展示
        plan = final_state.get("current_plan", "")
        search_count = final_state.get("search_count", 0)
        logs = [
            f"🎯 意圖分析與計畫 (Planner): {plan}",
            f"🔍 檢索執行次數 (Researcher): 進行了 {search_count} 次 Multi-hop 檢索",
            f"📄 收集到文件總數 (Reviewer): {len(docs)} 份指引",
            f"🤖 答案綜合生成 (Generator): 完成生成"
        ]
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            reasoning_logs=logs
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Agent Execution Error: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "ok"}
