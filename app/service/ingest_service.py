from __future__ import annotations
import os, hashlib
from typing import Dict, Any, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.vectorstore import get_vectorstore

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def ingest_pdf(
    study_session_id: str,
    file_path: str,
    original_filename: str,
    doc_id: Optional[str] = None,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    sha = _sha256(file_path)
    if doc_id is None:
        doc_id = f"{original_filename}#sha256:{sha[:12]}"

    loader = PyPDFLoader(file_path)
    pages = loader.load()  # page/source 메타 포함

    for d in pages:
        d.metadata["study_session_id"] = study_session_id
        d.metadata["doc_id"] = doc_id
        d.metadata["source_filename"] = original_filename

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    docs = splitter.split_documents(pages)

    vs = get_vectorstore()
    vs.add_documents(docs)

    # chunks는 굳이 응답으로 안 보내도 된다고 했으니 최소 정보만 반환
    return {
        "study_session_id": study_session_id,
        "doc_id": doc_id,
        "source_filename": original_filename,
        "sha256": sha,
    }
