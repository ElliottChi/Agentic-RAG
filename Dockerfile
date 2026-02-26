FROM python:3.9-slim

WORKDIR /app

# 複製依賴清單並安裝
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製原始碼與必要資源
COPY src/ src/
COPY data/ data/
COPY scripts/ scripts/

# 預設啟動 FastAPI 服務
EXPOSE 8000
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
