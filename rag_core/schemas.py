# rag_core/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional

class DocChunk(BaseModel):
    id: str
    text: str
    source: str
    chunk_index: int = 0
    score: float = 0.0
    method: str = "vector"  # vector | bm25 | hybrid | fusion | rerank

class RAGResult(BaseModel):
    query: str
    answer: str
    citations: List[str] = Field(default_factory=list)
    docs: List[DocChunk] = Field(default_factory=list)
    debug: Optional[dict] = None
