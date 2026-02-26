@echo off
echo Starting Deep Research Agentic RAG Services...

:: 設定環境變數 (使用硬碟持久化 Qdrant 以保留資料)
set PYTHONPATH=%cd%
set QDRANT_PATH=data/qdrant_db

:: 1. 背景啟動 FastAPI Server (API Layer)
echo [1/2] Starting FastAPI Backend on port 8000...
start cmd /k "C:\Users\USER\anaconda3\envs\prRAG\python.exe -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload"

:: 等待幾秒讓 API 先起來
timeout /t 3 /nobreak >nul

:: 2. 啟動 Streamlit Frontend (UI Layer)
echo [2/2] Starting Streamlit UI...
start cmd /k "C:\Users\USER\anaconda3\envs\prRAG\python.exe -m streamlit run src/ui/app.py"

echo All services started! Close the command windows to shut them down.
pause
