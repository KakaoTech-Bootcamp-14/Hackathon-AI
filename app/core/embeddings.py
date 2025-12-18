from langchain_openai import OpenAIEmbeddings
from app.config import OPENAI_EMBED_MODEL

def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
