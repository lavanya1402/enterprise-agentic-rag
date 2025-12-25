# rag_core/config.py
import os
from pydantic import BaseModel

def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}

class Settings(BaseModel):
    # feature toggles
    ENABLE_HYBRID: bool = _env_bool("ENABLE_HYBRID", True)
    ENABLE_RERANK: bool = _env_bool("ENABLE_RERANK", False)
    ENABLE_QUERY_EXPANSION: bool = _env_bool("ENABLE_QUERY_EXPANSION", False)
    ENABLE_SELF_RAG: bool = _env_bool("ENABLE_SELF_RAG", True)
    ENABLE_CRAG: bool = _env_bool("ENABLE_CRAG", False)  # uses self-rag + retry loop

    # retrieval params
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    ALPHA: float = float(os.getenv("ALPHA", "0.55"))  # hybrid weight: vectors vs bm25
    QUERY_EXPANSION_N: int = int(os.getenv("QUERY_EXPANSION_N", "3"))

    # chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # LLM
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))

    # paths
    RAW_PDF_DIR: str = os.getenv("RAW_PDF_DIR", "data/raw_pdfs")

settings = Settings()
