import os
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.config import UPLOAD_DIR
from app.service.ingest_service import ingest_pdf
from app.schema.documents import UploadDocResp

router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/upload", response_model=UploadDocResp)
async def upload_document(
    study_session_id: Optional[str] = Form(None),
    file: UploadFile = File(...)
):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="only .pdf is supported")

    sid = (study_session_id or "").strip() or "demo11"
    save_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(save_path, "wb") as f:
            f.write(await file.read())

        result = ingest_pdf(
            study_session_id=sid,
            file_path=save_path,
            original_filename=file.filename,
        )
        return result
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)