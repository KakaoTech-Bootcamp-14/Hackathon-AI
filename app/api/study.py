from fastapi import APIRouter, HTTPException
from app.schema.study import (
    CreatePlanReq, ReplanReq,
    MaterialReq, MaterialResp
)
from app.service.plan_service import create_plan, replan
from app.service.material_service import create_material

from typing import List
from app.schema.study import ChapterItem

router = APIRouter(prefix="/study", tags=["study"])

@router.post("/plan", response_model=List[ChapterItem])
def plan(req: CreatePlanReq):
    try:
        return create_plan(req.study_session_id, req.total_days, req.hours_per_day)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/replan", response_model=List[ChapterItem])
def replan_api(req: ReplanReq):
    try:
        return replan(
            study_session_id=req.study_session_id,
            remaining_days=req.remaining_days,
            hours_per_day=req.hours_per_day,
            studied_topics=req.studied_topics,
            pending_topics=req.pending_topics,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/material", response_model=MaterialResp)
def material(req: MaterialReq):
    try:
        return create_material(req.study_session_id, req.topic, k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
