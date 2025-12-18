from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from app.core.llm import get_llm
from app.core.vectorstore import get_vectorstore
from app.core.memory import get_history  # 세션 메모리(아래 참고)

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "너는 문서 검색을 위한 질문 재작성기다. 대화 기록을 참고해서 "
     "사용자의 마지막 질문을 문서 검색에 적합한 '단일 질문'으로 재작성해라. "
     "대명사(그거/이거/위 내용)는 구체적으로 풀어써라. 질문만 출력해라."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "너는 문서 기반 한국어 튜터다. 반드시 제공된 문서(context)에 있는 내용만 사용해서 답변해라. "
        "출력 형식은 반드시 다음 규칙을 지켜라. "
        "1) 줄바꿈(\\n), 목록, 마크다운, 제목, 기호(-, *, #)를 절대 사용하지 마라. "
        "2) 한 문단, 한 줄의 자연스러운 설명문으로만 작성하라. "
        "3) 불필요한 반복이나 장황한 표현은 피하고 핵심과 5문장 이하의 부연설명만 간결하게 설명하라. "
        "4) 문서에서 근거를 찾을 수 없으면 반드시 '문서에서 근거를 찾을 수 없습니다'라고만 답하라."
    ),
    MessagesPlaceholder("chat_history"),
    (
        "human",
        "질문: {question}\n\ncontext:\n{context}"
    )
])

def _format_context(docs, limit_each: int = 1200) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", d.metadata.get("source_filename", "unknown"))
        page = d.metadata.get("page")
        page_info = f", page={page}" if page is not None else ""
        text = (d.page_content or "").strip().replace("\x00", "")
        if len(text) > limit_each:
            text = text[:limit_each] + "..."
        parts.append(f"[{i}] source={src}{page_info}\n{text}")
    return "\n\n".join(parts)

def chat_answer(study_session_id: str, question: str, top_k: int = 4) -> Dict[str, Any]:
    sid = (study_session_id or "").strip() or "demo11"
    q = question.strip()
    if not q:
        raise ValueError("question is empty")

    llm = get_llm()
    history = get_history(sid)

    # 1) 검색용 질문 재작성 (대화 반영)
    rewrite_chain = REWRITE_PROMPT | llm | StrOutputParser()
    search_query = rewrite_chain.invoke({
        "input": q,
        "chat_history": history.messages,
    })

    # 2) 검색
    vs = get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(search_query)

    context = _format_context(docs)

    # 3) 답변 생성 (대화 + 문서 근거)
    answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
    answer_md = answer_chain.invoke({
        "question": q,
        "context": context,
        "chat_history": history.messages,
    })

    # 4) 메모리 업데이트
    history.add_user_message(q)
    history.add_ai_message(answer_md)

    sources: List[dict] = []
    for d in docs:
        sources.append({
            "source": d.metadata.get("source", d.metadata.get("source_filename", "unknown")),
            "page": d.metadata.get("page"),
        })

    return {
        "study_session_id": sid,
        "question": q,
        "answer_md": answer_md,
        "sources": sources,
    }
