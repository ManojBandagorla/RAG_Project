from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from core.http_client import get_http_client
from config.config import API_KEY,LLM_BASE_URL,LLM_MODEL,EMBEDDING_MODEL

def get_llm():
    client = get_http_client()

    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        api_key=API_KEY,
        http_client=get_http_client()
    )

def get_embeddings():
    client = get_http_client()

    return OpenAIEmbeddings(
        base_url=LLM_BASE_URL,
        model=EMBEDDING_MODEL,
        api_key=API_KEY,
        http_client=get_http_client()
    )