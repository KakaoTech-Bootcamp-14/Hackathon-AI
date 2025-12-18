from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from app.service.chat_service import chat_answer
from app.core.memory import clear_history

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatReq(BaseModel):
    study_session_id: Optional[str] = "demo11"
    question: str = Field(..., min_length=1)
    top_k: int = Field(4, ge=1, le=20)

class ChatResp(BaseModel):
    study_session_id: str
    question: str
    answer_md: str
    sources: List[Dict[str, Any]] = []

class ClearReq(BaseModel):
    study_session_id: Optional[str] = "demo11"

@router.post("", response_model=ChatResp)
def chat(req: ChatReq):
    try:
        return chat_answer(req.study_session_id, req.question, req.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear")
def clear(req: ClearReq):
    sid = (req.study_session_id or "").strip() or "demo11"
    clear_history(sid)
    return {"ok": True, "study_session_id": sid}
