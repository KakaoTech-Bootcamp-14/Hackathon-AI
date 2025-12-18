from __future__ import annotations
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.llm import get_llm
from app.core.vectorstore import get_vectorstore

MATERIAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "너는 업로드된 학습 자료를 바탕으로 '학습 노트(교재 스타일)'를 작성하는 튜터다.\n"
     "구어체를 쓰지 말고, 정리된 설명문으로 작성해라.\n"
     "반드시 context에 있는 내용으로만 작성하고, 없으면 '문서에서 근거를 찾을 수 없습니다'라고 명시해라.\n"
     "출력은 마크다운으로 작성하라.\n"),
    ("human",
     "학습 주제: {topic}\n"
     "필요하면 하위 항목(소제목)도 구성해라.\n\n"
     "context:\n{context}\n")
])

def create_material(study_session_id: str, topic: str, k: int = 6) -> Dict[str, Any]:
    vs = get_vectorstore()
    retriever = vs.as_retriever(
        search_kwargs={
            "k": k,
            "filter": {"study_session_id": study_session_id},
        }
    )
    docs = retriever.invoke(topic)

    parts = []
    sources = []
    for d in docs:
        src = d.metadata.get("source_filename", d.metadata.get("source", "unknown"))
        page = d.metadata.get("page")
        text = (d.page_content or "").strip()
        if len(text) > 1200:
            text = text[:1200] + "..."
        parts.append(f"[{src} page={page}]\n{text}")
        sources.append({"source": src, "page": page})

    context = "\n\n".join(parts)

    llm = get_llm(temperature=0.2)
    chain = MATERIAL_PROMPT | llm | StrOutputParser()
    md = chain.invoke({"topic": topic, "context": context})

    return {
        "study_session_id": study_session_id,
        "topic": topic,
        "content_md": md,
        "sources": sources,
    }
