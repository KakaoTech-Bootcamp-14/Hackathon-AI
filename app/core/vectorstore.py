from langchain_chroma import Chroma
from app.config import CHROMA_DIR, CHROMA_COLLECTION
from app.core.embeddings import get_embeddings

_vs = None

def get_vectorstore() -> Chroma:
    global _vs
    if _vs is None:
        _vs = _load()
    return _vs

def reload_vectorstore() -> Chroma:
    global _vs
    _vs = _load()
    return _vs

def _load() -> Chroma:
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
    )
