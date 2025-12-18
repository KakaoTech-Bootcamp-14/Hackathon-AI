from pydantic import BaseModel, Field
from typing import List, Optional

class CreatePlanReq(BaseModel):
    study_session_id: Optional[str] = "demo11"
    total_days: int = Field(..., ge=1, le=365)
    hours_per_day: float = Field(..., gt=0, le=24)

class ReplanReq(BaseModel):
    study_session_id: Optional[str] = "demo11"
    remaining_days: int = Field(..., ge=1, le=365)
    hours_per_day: float = Field(..., gt=0, le=24)
    studied_topics: List[str] = []
    pending_topics: List[str] = []

class PlanResp(BaseModel):
    study_session_id: str
    plan_json: str

class MaterialReq(BaseModel):
    study_session_id: Optional[str] = "demo11"
    topic: str
    top_k: int = 6

class MaterialResp(BaseModel):
    study_session_id: str
    topic: str
    content_md: str
    sources: List[dict]

