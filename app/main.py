from fastapi import FastAPI
from app.api.documents import router as documents_router
from app.api.study import router as study_router
from app.api.chat import router as chat_router

app = FastAPI(title="AI Tutor API (Upload / Plan / Material / Replan)")

app.include_router(documents_router)
app.include_router(study_router)
app.include_router(chat_router)

@app.get("/health")
def health():
    return {"ok": True}
