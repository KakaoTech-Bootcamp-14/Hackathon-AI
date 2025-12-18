from fastapi import APIRouter, HTTPException
from app.schema.chat import ChatReq, ChatResp
from app.service.chat_agent_service import chat_agent_answer

router = APIRouter(prefix="", tags=["chat"])

@router.post("/chat", response_model=ChatResp)
def chat(req: ChatReq):
    try:
        return chat_agent_answer(
            study_session_id=req.study_session_id,
            question=req.question,
            top_k=req.top_k
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))