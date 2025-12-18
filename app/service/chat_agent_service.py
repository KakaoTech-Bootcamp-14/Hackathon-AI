from __future__ import annotations
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser

from langchain.agents import AgentExecutor, create_openai_tools_agent

from app.core.llm import get_llm
from app.core.vectorstore import get_vectorstore
from app.core.memory import get_history, append_history

from app.tools.quiz_tools import (
    generate_ox_quiz as generate_ox_quiz_tool,
    check_ox_answer as check_ox_answer_tool,
)
from app.tools.youtube_tool import youtube_search as youtube_search_tool



def _format_context(docs, limit_each: int = 900) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source_filename", d.metadata.get("source", "unknown"))
        page = d.metadata.get("page")
        text = (d.page_content or "").replace("\n", " ").strip()
        if len(text) > limit_each:
            text = text[:limit_each] + "..."
        parts.append(f"[{i}] {src} page={page} {text}".strip())
    return " ".join(parts).strip()


@tool
def explain_concept(study_session_id: str, question: str, top_k: int = 4) -> str:
    """
    업로드된 문서(Chroma)에서 study_session_id로 필터링하여 근거 기반으로 설명한다.
    출력은 반드시 한 줄 문장(줄바꿈/마크다운/목록 금지).
    """
    sid = (study_session_id or "").strip() or "demo11"
    q = (question or "").strip()
    if not q:
        return "질문이 비어 있습니다."

    vs = get_vectorstore()
    retriever = vs.as_retriever(
        search_kwargs={
            "k": int(top_k),
            "filter": {"study_session_id": sid},
        }
    )
    docs = retriever.invoke(q)
    if not docs:
        return "문서에서 근거를 찾을 수 없습니다"

    context = _format_context(docs)

    llm = get_llm(temperature=0.2)
    prompt = (
        "너는 문서 기반 한국어 튜터다. 반드시 제공된 context에 있는 내용만 사용해 답해라. 존댓말을 써라. "
        "줄바꿈, 목록, 마크다운, 제목 기호(#,-,*)를 절대 쓰지 말고 3~5문장정도로 답해라. "
        "더 자세히 설명해달라는 류이 표현을 하면 5문장을 초과하여 더 길게 여러 문장으로 답해라. "
        "문서에서 근거를 못 찾으면 '문서에서 근거를 찾을 수 없습니다'라고만 답하라. "
        f"질문={q} context={context}"
    )
    ans = llm.invoke(prompt).content
    return str(ans).replace("\n", " ").strip()


def _overview_context_for_quiz(sid: str, k: int = 6) -> str:
    vs = get_vectorstore()
    retriever = vs.as_retriever(
        search_kwargs={
            "k": k,
            "filter": {"study_session_id": sid},
        }
    )
    docs = retriever.invoke("이 문서에서 학습 핵심 개념을 요약해줘")
    return _format_context(docs, limit_each=700)


AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "너는 학습 도우미 에이전트다. 사용자의 입력과 대화 흐름을 보고 적절한 도구를 선택해 행동하라. "
     "반드시 도구를 사용해서 답변을 만들고, 최종 출력은 answer_md로 사용자에게 보여줄 자연스러운 한국어 설명문만 반환하라. "
     "출력 규칙: 줄바꿈(\\n), 마크다운, 목록, 제목 기호(#,-,*) 사용 금지. 존댓말 사용. "
     ""
     "행동 규칙:"
     "1) 사용자가 입력을 'O' 또는 'X'만 보냈다면 check_ox_answer를 호출하라."
     ""
     "2) 사용자가 여러 번 이해가 안 된다는 표현을 하거나, 혼란스러워 보이거나, "
     "문서 기반 설명만으로 충분하지 않다고 판단되면 youtube_search를 호출하라. "
     "이 경우 최종 answer_md는 다음 흐름을 반드시 포함해야 한다: "
     "  - 먼저 핵심 개념을 한 번 더 간단히 설명한다."
     "  - 글 설명이 어려울 수 있음을 인정한다."
     "  - 이해를 돕기 위해 영상 자료를 함께 보면 좋겠다고 안내한다."
     "  - 마지막에 영상 링크를 자연스럽게 포함한다."
     ""
     "3) 사용자가 개념을 어느 정도 이해한 것으로 보이거나, "
     "'퀴즈', '점검', '확인' 등의 표현을 사용하면 generate_ox_quiz를 호출하라. "
     "이 경우 최종 answer_md는 다음 흐름을 반드시 포함해야 한다: "
     "  - 먼저 '잘 이해하셨습니다' 또는 이에 준하는 긍정적인 피드백을 한다."
     "  - 방금 학습한 핵심 내용을 한두 문장으로 요약해준다."
     "  - 이제 이해도를 점검하기 위해 OX 퀴즈를 풀어보자고 안내한다."
     "  - 마지막에 OX 퀴즈 문장을 제시한다."
     ""
     "4) 그 외의 일반적인 질문에는 explain_concept를 호출해 문서 근거 기반으로 설명하라."
     ""
     "주의: 설명, 요약, 퀴즈 안내, 영상 안내를 포함하더라도 최종 answer_md는 하나의 자연스러운 문단이어야 한다."
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "study_session_id={study_session_id} top_k={top_k} user_input={input}"),
    MessagesPlaceholder("agent_scratchpad"),
])


def chat_agent_answer(study_session_id: str, question: str, top_k: int = 4) -> Dict[str, Any]:
    sid = (study_session_id or "").strip() or "demo11"
    q = (question or "").strip()
    if not q:
        raise ValueError("question is empty")

    history = get_history(sid)

    # 퀴즈 생성 도구에 넣을 context는 넓게 조금만 뽑아줌 (문서 기반 유지)
    quiz_context = _overview_context_for_quiz(sid)

    # tools (quiz tool은 context 필요하니 래핑)
    @tool
    def generate_ox_quiz(study_session_id: str, topic: str) -> str:
        """문서 기반 context를 붙여 OX퀴즈를 생성한다."""
        return str(
            generate_ox_quiz_tool.invoke({
                "study_session_id": study_session_id,
                "topic": topic,
                "context": quiz_context
            })
        ).replace("\n", " ").strip()

    tools = [
        explain_concept,
        youtube_search_tool,
        generate_ox_quiz,
        check_ox_answer_tool,
    ]

    llm = get_llm(temperature=0.2)
    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=AGENT_PROMPT)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    result = executor.invoke({
        "study_session_id": sid,
        "top_k": top_k,
        "input": q,
        "chat_history": history,
    })

    answer_md = str(result.get("output", "")).replace("\n", " ").strip()

    append_history(sid, q, answer_md)

    return {
        "study_session_id": sid,
        "question": q,
        "answer_md": answer_md,
    }
