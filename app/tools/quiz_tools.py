from __future__ import annotations
from typing import Literal
from langchain_core.tools import tool
from app.core.llm import get_llm
from app.core.memory import set_quiz, get_quiz, clear_quiz

@tool
def generate_ox_quiz(study_session_id: str, topic: str, context: str) -> str:
    """
    주어진 topic+context 기반으로 OX 퀴즈 1개를 만든다.
    퀴즈 상태를 서버 메모리에 저장하고, 사용자에게 보여줄 한 줄 문장을 반환한다.
    """
    llm = get_llm(temperature=0.2)

    prompt = (
        "너는 학습 튜터다. 아래 context 내용만 근거로 OX 퀴즈를 1개 만들어라. "
        "출력은 반드시 다음 형식으로만 한 줄로 출력하라: "
        "STATEMENT=<문장> | ANSWER=<O 또는 X> | EXPLANATION=<한줄 해설>. "
        "줄바꿈 금지. 문서에 없는 내용 금지.\n\n"
        f"topic={topic}\ncontext={context}\n"
    )

    out = llm.invoke(prompt).content
    text = str(out).replace("\n", " ").strip()

    # 파싱 (최소한으로)
    statement = ""
    answer = ""
    explanation = ""

    for part in text.split("|"):
        p = part.strip()
        if p.startswith("STATEMENT="):
            statement = p[len("STATEMENT="):].strip()
        elif p.startswith("ANSWER="):
            answer = p[len("ANSWER="):].strip().upper()
        elif p.startswith("EXPLANATION="):
            explanation = p[len("EXPLANATION="):].strip()

    if answer not in ["O", "X"] or not statement:
        # 실패 시 간단 fallback
        statement = f"{topic}에 대한 설명이 문서에 근거하여 옳다."
        answer = "O"
        explanation = "문서 근거로 판단하세요."

    set_quiz(study_session_id, {
        "statement": statement,
        "answer": answer,
        "explanation": explanation,
        "topic": topic,
    })

    return f"OX퀴즈: {statement} (O/X로 답하세요)"


@tool
def check_ox_answer(study_session_id: str, user_answer: Literal["O","X","o","x"]) -> str:
    """
    직전에 낸 OX퀴즈를 채점하고 한 줄로 피드백한다.
    """
    quiz = get_quiz(study_session_id)
    if not quiz:
        return "채점할 퀴즈가 없습니다. 먼저 퀴즈를 생성해 주세요."

    correct = quiz.get("answer", "O")
    statement = quiz.get("statement", "")
    explanation = quiz.get("explanation", "")

    ua = str(user_answer).strip().upper()
    ok = (ua == correct)

    clear_quiz(study_session_id)

    if ok:
        return f"정답입니다. ({ua}) | 문장: {statement} | 해설: {explanation}".replace("\n", " ")
    return f"오답입니다. (당신={ua}, 정답={correct}) | 문장: {statement} | 해설: {explanation}".replace("\n", " ")
