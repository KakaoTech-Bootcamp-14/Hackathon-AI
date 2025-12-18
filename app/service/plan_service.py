from __future__ import annotations
from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import re
from app.core.llm import get_llm
from app.core.vectorstore import get_vectorstore

PLAN_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "너는 업로드된 문서를 기반으로 학습 목차(계획)를 만드는 튜터다.\n"
     "반드시 아래 출력 규칙을 지켜라.\n\n"
     "출력 규칙:\n"
     "1) JSON/마크다운 코드블록(```) 금지, 설명문 금지\n"
     "2) 각 줄은 반드시 다음 형식으로만 출력\n"
     "   {{n}}일차, {{대제목}}, [{{소제목1}}, {{소제목2}}, ...]\n"
     "   (위는 형식 예시이며 실제 출력에서는 중괄호 {{ }}를 쓰지 말 것)\n"
     "3) 소제목 개수는 고정하지 말고, 하루 공부량에 맞게 2~6개로 가변적으로 배치\n"
     "4) total_days 만큼의 줄을 정확히 출력하라(누락/추가 금지)\n"
     "5) hours_per_day에 따라 소제목 개수를 조절하라: 1시간 이하=2~3개, 2시간=3~4개, 3시간 이상=4~6개\n"
     ),
    ("human",
     "total_days={total_days}\n"
     "hours_per_day={hours_per_day}\n\n"
     "context:\n{context}\n")
])


REPLAN_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "너는 학습 계획 재조정(Replan) 엔진이다.\n"
     "입력으로 studied_topics(이미 공부함), pending_topics(남음), remaining_days, hours_per_day가 주어진다.\n"
     "remaining_days와 hours_per_day가 줄어들면 반드시 pending_topics를 '병합/압축'해서 남은 기간에 맞춰라.\n\n"
     "필수 절차(반드시 내부적으로 이 순서로 수행):\n"
     "1) pending_topics를 정리한다: 중복 제거, 표기 통일(예: '~절' 유무), 거의 같은 항목 병합.\n"
     "2) 의미가 겹치는 항목들을 '압축 토픽 그룹'으로 묶는다.\n"
     "   - 규칙 A: 같은 개념의 변형/동의어/표기 차이는 하나로 합친다.\n"
     "   - 규칙 B: '기초 → 응용 → 심화'로 자연스럽게 이어지도록 묶는다.\n"
     "   - 규칙 C: 너무 세부적인 항목이 많으면 상위 개념명으로 그룹화하고 세부는 괄호로 표시할 수 있다.\n"
     "     (예: 상위개념(세부1/세부2/세부3))\n"
     "3) 압축 토픽 그룹을 우선순위로 정렬한다.\n"
     "   - 시험/과제/실무에 공통으로 중요한 '핵심 기초'를 먼저 배치하고,\n"
     "     이후에 응용/심화/부가 주제로 배치한다.\n"
     "   - 단, 입력에 근거 없는 임의의 주제를 새로 만들지 말고 pending_topics에서만 재구성한다.\n"
     "4) remaining_days에 맞춰 하루 단위로 배분한다.\n"
     "   - hours_per_day 기준 권장 배정량:\n"
     "     1시간 이하: 하루 2~3개 그룹\n"
     "     2시간: 하루 3~4개 그룹\n"
     "     3시간 이상: 하루 4~6개 그룹\n"
     "5) 커버리지 규칙:\n"
     "   - 결과 전체가 pending_topics를 가능한 넓게 커버해야 한다.\n"
     "   - '앞부분 몇 개만' 나열하는 방식 금지.\n"
     "   - 결과에 반영된 압축 그룹들이 pending_topics의 키워드를 최대한 많이 포함하도록 구성하라.\n\n"
     "출력 규칙:\n"
     "- 설명문/코드블록/JSON 금지\n"
     "- 정확히 remaining_days 줄 출력\n"
     "- 각 줄 형식: N일차, 대제목, [소제목1, 소제목2, ...]\n"
     "- 소제목 개수는 고정하지 말고 hours_per_day에 맞춰 가변\n"
     "- 소제목은 압축 토픽 그룹 이름으로 쓰고, 필요하면 괄호로 세부를 묶어라.\n"
     ),
    ("human",
     "remaining_days={remaining_days}\n"
     "hours_per_day={hours_per_day}\n"
     "studied_topics={studied_topics}\n"
     "pending_topics={pending_topics}\n")
])



def parse_plan_text(plan_text: str) -> List[Dict]:
    lines = [l.strip() for l in plan_text.splitlines() if l.strip()]
    chapters = []

    for idx, line in enumerate(lines, start=1):
        # 예: 1일차, 데이터 모델링의 이해, [데이터 모델의 이해, 엔터티, 속성]
        m = re.match(r"\d+일차,\s*(.*?),\s*\[(.*)\]", line)
        if not m:
            continue

        chapter_title = m.group(1).strip()
        tasks_raw = m.group(2)
        tasks = [t.strip() for t in tasks_raw.split(",") if t.strip()]

        chapters.append({
            "chapterOrder": idx,
            "chapterTitle": chapter_title,
            "tasks": [
                {"taskOrder": i + 1, "taskTitle": task}
                for i, task in enumerate(tasks)
            ]
        })

    return chapters



def _retrieve_overview_context(study_session_id: str, k: int = 18) -> str:
    """계획 생성을 위해 넓게(overview) 문서를 가져옴"""
    vs = get_vectorstore()
    retriever = vs.as_retriever(
        search_kwargs={
            "k": k,
            "filter": {"study_session_id": study_session_id},
        }
    )
    docs = retriever.invoke("이 문서의 목차, 핵심 주제, 챕터 구조, 학습해야 할 개념을 요약해줘")
    parts = []
    for d in docs:
        src = d.metadata.get("source_filename", d.metadata.get("source", "unknown"))
        page = d.metadata.get("page")
        text = (d.page_content or "").strip()
        if len(text) > 900:
            text = text[:900] + "..."
        parts.append(f"- ({src}, page={page}) {text}")
    return "\n".join(parts)

def create_plan(study_session_id: str, total_days: int, hours_per_day: float) -> Dict[str, Any]:
    context = _retrieve_overview_context(study_session_id)
    llm = get_llm(temperature=0.2)
    chain = PLAN_PROMPT | llm | StrOutputParser()
    out = chain.invoke({"total_days": total_days, "hours_per_day": hours_per_day, "context": context})
    return parse_plan_text(out)

def replan(
    study_session_id: str,
    remaining_days: int,
    hours_per_day: float,
    studied_topics: List[str],
    pending_topics: List[str],
) -> Dict[str, Any]:
    context = _retrieve_overview_context(study_session_id)
    llm = get_llm(temperature=0.2)
    chain = REPLAN_PROMPT | llm | StrOutputParser()
    out = chain.invoke({
        "remaining_days": remaining_days,
        "hours_per_day": hours_per_day,
        "studied_topics": studied_topics,
        "pending_topics": pending_topics,
        "context": context
    })
    return parse_plan_text(out)
