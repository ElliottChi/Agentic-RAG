import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain.agents import tool, AgentExecutor, create_tool_calling_agent
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

def load_and_process_data(directory_path="./data"):
    if not os.path.exists(directory_path):
        print(f"警告: 找不到 {directory_path} 資料夾")
        return []

    print("正在載入知識庫文件 (PDF)...")
    
    loader = DirectoryLoader(
        directory_path, 
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    
    try:
        documents = loader.load()
    except Exception as e:
        print(f"載入檔案時發生錯誤: {e}")
        return []
    
    if not documents:
        print("警告: 知識庫中沒有文件")
        return []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    return splits

# 建立 Hybrid Retriever + Reranker (Two-Stage)
def get_hybrid_retriever(splits):
    
    embedding_function = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    
    # 準備 Vector Retriever (Chroma)
    # 如果已經有建立過，它會自動讀取舊的，若有新文件則會加入
    vectorstore = Chroma.from_documents(
        documents=splits, # 確保向量庫跟 BM25 用的文件一致
        embedding=embedding_function,
        persist_directory="./chroma_db"
    )
    
    # 第一階段 - 粗篩：將搜尋範圍擴大，設 k=50 (從向量資料庫撈)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 50})

    # 準備 BM25 Retriever
    print("正在建立 BM25 索引...")
    bm25_retriever = BM25Retriever.from_documents(splits)
    # 第一階段 - 粗篩：關鍵字檢索也擴大至 50 筆
    bm25_retriever.k = 50

    # 結合兩者 - 這是我們的 Base Retriever
    print("正在整合 Hybrid Retriever (粗篩階段)...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        # weights 控制權重: [BM25權重, Vector權重]
        weights=[0.5, 0.5]
    )
    
    # 第二階段 - 精排：使用 Cross-Encoder Reranker
    print("正在初始化 Cross-Encoder Reranker (精排階段)...")
    
    # 使用 BAAI 的 Reranker 模型，這對中文的語意排序效果較好
    reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    
    # 設定壓縮器：取最高的 Top 3
    compressor = CrossEncoderReranker(model=reranker_model, top_n=3)
    
    # 構建兩階段檢索器: 先跑 ensemble (粗篩), 再跑 compressor (精排)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )
    
    return compression_retriever

# 初始化 RAG Chain
def get_rag_chain():
    splits = load_and_process_data("./data")
    
    if not splits:
        raise ValueError("無法初始化 RAG：沒有資料")

    retriever = get_hybrid_retriever(splits)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    system_prompt = (
        "你是一個專業、嚴謹的交通法規諮詢助手。請根據以下提供的【法規資料】，用繁體中文回答使用者的問題。"
        "\n\n"
        "【回答原則】\n"
        "1. 引用依據：回答時請明確指出是根據哪一條法規（例如：根據《道路交通管理處罰條例》第XX條）。\n"
        "2. 解釋清晰：法條文字通常較為生硬，請用白話文解釋其含義，讓一般民眾能聽懂。\n"
        "3. 罰則說明：如果涉及違規，請明確列出罰款金額範圍或記點規定。\n"
        "4. 嚴謹客觀：請基於事實回答，不要加入個人情感或不確定的推測。如果【法規資料】中沒有相關條文，請直接說「抱歉，目前的資料庫中沒有相關法規資訊」。\n"
        "\n\n"
        "【法規資料】\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain


# 全域變數，用來儲存已經初始化好的 Chain
_cached_chain = None

def get_initialized_chain():
    """
    Singleton 模式：確保 RAG Chain 只會被初始化一次。
    避免每次 Agent 呼叫工具時都要重新讀取 PDF。
    """
    global _cached_chain
    if _cached_chain is None:
        print("正在初始化 RAG Chain (全域)...")
        _cached_chain = get_rag_chain()
        print("RAG Chain 初始化完成！")
    return _cached_chain

# --- 測試用 ---
if __name__ == "__main__":
    chain = get_rag_chain()
    # 測試一個既需要語意又需要關鍵字的查詢
    query = "機車闖紅燈罰多少錢?" 
    print(f"使用者問題: {query}")
    
    response = chain.invoke({"input": query}) 
    print("回答:", response["answer"])


    print("\n====== 檢索到的參考資料 (Context) ======")
    if "context" in response:
        for i, doc in enumerate(response["context"]):
            print(f"\n[文件 {i+1}] 來源: {doc.metadata.get('source', '未知')}")
            print(f"內容摘要: {doc.page_content[:150]}...") # 只印出前 150 字避免太長
    else:
        print("沒有檢索到相關文件。")
    print("==========================================")