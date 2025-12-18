from pydantic import BaseModel

class UploadDocResp(BaseModel):
    study_session_id: str
    doc_id: str
    source_filename: str
    sha256: str
