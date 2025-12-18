from typing import Dict
from langchain_community.chat_message_histories import ChatMessageHistory

_SESSION_STORE: Dict[str, ChatMessageHistory] = {}

def get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = ChatMessageHistory()
    return _SESSION_STORE[session_id]

def clear_history(session_id: str) -> None:
    _SESSION_STORE.pop(session_id, None)
