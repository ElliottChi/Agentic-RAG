import os
from dotenv import load_dotenv

from src.db.qdrant_store import QdrantStore
from src.orchestration.nodes.researcher import get_embeddings

def main():
    load_dotenv(override=True)
    qdrant_path = os.environ.get('QDRANT_PATH', 'data/qdrant_db')
    store = QdrantStore(collection_name='deep_research_rag_bge', embedding_model=get_embeddings(), vector_size=768, qdrant_path=qdrant_path)
    res = store.similarity_search('沒戴安全帽要罰錢嗎?', k=5)
    
    print('===== QDRANT SEARCH =====')
    print(f'Total retrieved: {len(res)}')
    for r in res:
        print(r.get('score'), r.get('metadata', {}).get('source'), r.get('page_content')[:100])

if __name__ == "__main__":
    main()
