import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from glob import glob
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, CSVLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.db.qdrant_store import QdrantStore
from src.db.neo4j_store import Neo4jStore
from src.db.bm25_store import BM25Store
from src.orchestration.nodes.researcher import get_embeddings

load_dotenv()

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    raw_dir = os.path.join(project_root, "data", "raw")
    
    # 建立 Loaders
    txt_loader = DirectoryLoader(raw_dir, glob="**/*.txt", loader_cls=TextLoader)
    pdf_loader = DirectoryLoader(raw_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    csv_loader = DirectoryLoader(raw_dir, glob="**/*.csv", loader_cls=CSVLoader)
    # JSONLoader 需要指定 jq schema，這裡用最簡單的 '.' 取出所有內容，並設定 text_content=False 將 dict 轉成文字
    json_loader = DirectoryLoader(raw_dir, glob="**/*.json", loader_cls=JSONLoader, loader_kwargs={'jq_schema': '.', 'text_content': False})

    docs = []
    print("正在載入 TXT 檔案...")
    docs.extend(txt_loader.load())
    
    print("正在載入 PDF 檔案...")
    docs.extend(pdf_loader.load())
    
    print("正在載入 CSV 檔案...")
    docs.extend(csv_loader.load())
    
    print("正在載入 JSON 檔案...")
    docs.extend(json_loader.load())

    print(f"總共載入 {len(docs)} 份文件。")

    # 分塊 (Split)
    print("開始分割文本...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    
    # 轉換成 Store 規範的格式
    formatted_docs = []
    for d in split_docs:
        formatted_docs.append({
            "page_content": d.page_content,
            "metadata": d.metadata
        })

    print(f"分割完畢，產生 {len(formatted_docs)} 個文本區塊 (Chunks)。")

    # 連線設定
    qdrant_path = os.environ.get("QDRANT_PATH", "data/qdrant_db")
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")

    try:
        print(f"正在連線至 Qdrant (Path: {qdrant_path})...")
        qdrant_store = QdrantStore(
            collection_name="deep_research_rag_bge",
            embedding_model=get_embeddings(),
            vector_size=768,
            qdrant_path=qdrant_path
        )
        print("將 Chunk 寫入 Qdrant...")
        qdrant_store.add_documents(formatted_docs)
        print("Qdrant 寫入成功！")
    except Exception as e:
        print(f"[錯誤] Qdrant 寫入失敗: {e}")

    try:
        print(f"正在連線至 Neo4j (URI: {neo4j_uri})...")
        neo4j_store = Neo4jStore(uri=neo4j_uri)
        print("將 Entity 與 Chunk 關聯寫入 Neo4j...")
        neo4j_store.add_documents(formatted_docs)
        neo4j_store.close()
        print("Neo4j 寫入成功！")
    except Exception as e:
        print(f"[錯誤] Neo4j 寫入失敗: {e}")

    try:
        print("正在建立並寫入 BM25 索引...")
        bm25_store = BM25Store()
        bm25_store.add_documents(formatted_docs)
        print("BM25 寫入成功！")
    except Exception as e:
        print(f"[錯誤] BM25 寫入失敗: {e}")

    print("=== 資料匯入匯入管線 (Ingestion Pipeline) 執行完畢 ===")

if __name__ == "__main__":
    main()
