@echo off
echo Starting Deep Research Agentic RAG Services...


set PYTHONPATH=%cd%
set QDRANT_PATH=data/qdrant_db


echo [1/2] Starting FastAPI Backend on port 8000...
start cmd /k "C:\Users\USER\anaconda3\envs\prRAG\python.exe -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload"


timeout /t 3 /nobreak >nul


echo [2/2] Starting Streamlit UI...
start cmd /k "C:\Users\USER\anaconda3\envs\prRAG\python.exe -m streamlit run src/ui/app.py"

echo All services started! Close the command windows to shut them down.
pause
