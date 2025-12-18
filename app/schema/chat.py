from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ChatReq(BaseModel):
    study_session_id: Optional[str] = "demo11"
    question: str = Field(..., min_length=1)
    top_k: int = Field(4, ge=1, le=20)

class ChatResp(BaseModel):
    study_session_id: str
    question: str
    answer_md: str
    sources: List[Dict[str, Any]] = []

class ClearChatReq(BaseModel):
    study_session_id: Optional[str] = "demo11"
