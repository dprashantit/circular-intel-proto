"""bge-small-en-v1.5 embeddings -> Chroma persistent collection."""
import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "circular_chunks"

_model = None


def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_collection(chroma_dir: str):
    client = chromadb.PersistentClient(path=chroma_dir)
    return client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


def upsert_chunks(chroma_dir: str, chunk_ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
    collection = get_collection(chroma_dir)
    embeddings = embed_texts(texts)
    collection.upsert(ids=chunk_ids, embeddings=embeddings, documents=texts, metadatas=metadatas)


def dense_search(chroma_dir: str, query: str, top_k: int = 20, where: dict | None = None) -> list[dict]:
    collection = get_collection(chroma_dir)
    query_embedding = embed_texts([query])[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for i in range(len(result["ids"][0])):
        hits.append(
            {
                "chunk_id": result["ids"][0][i],
                "text": result["documents"][0][i],
                "metadata": result["metadatas"][0][i],
                "score": 1 - result["distances"][0][i],  # cosine distance -> similarity
            }
        )
    return hits
