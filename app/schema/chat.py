from pydantic import BaseModel
from typing import Optional

class ChatReq(BaseModel):
    study_session_id: Optional[str] = "demo11"
    question: str
    top_k: int = 4

class ChatResp(BaseModel):
    study_session_id: str
    question: str
    answer_md: str
