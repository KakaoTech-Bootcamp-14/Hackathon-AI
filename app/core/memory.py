from __future__ import annotations
from typing import Dict, Optional, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# 세션별 대화 기록 (서버 메모리)
_HISTORY: Dict[str, list[BaseMessage]] = {}

# 세션별 퀴즈 상태 (서버 메모리)
# 예: {"statement": "...", "answer": "O", "explanation": "...", "topic": "..."}
_QUIZ_STATE: Dict[str, Dict[str, Any]] = {}


def get_history(session_id: str) -> list[BaseMessage]:
    return _HISTORY.setdefault(session_id, [])


def append_history(session_id: str, user_text: str, ai_text: str) -> None:
    hist = get_history(session_id)
    hist.append(HumanMessage(content=user_text))
    hist.append(AIMessage(content=ai_text))


def clear_history(session_id: str) -> None:
    _HISTORY.pop(session_id, None)
    _QUIZ_STATE.pop(session_id, None)


def set_quiz(session_id: str, quiz: Dict[str, Any]) -> None:
    _QUIZ_STATE[session_id] = quiz


def get_quiz(session_id: str) -> Optional[Dict[str, Any]]:
    return _QUIZ_STATE.get(session_id)


def clear_quiz(session_id: str) -> None:
    _QUIZ_STATE.pop(session_id, None)
