from langchain_openai import ChatOpenAI
from app.config import OPENAI_CHAT_MODEL

def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=temperature)