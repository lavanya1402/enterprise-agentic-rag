# rag_core/ingestion/chunkers.py
from typing import List

def chunk_text(text: str, size: int = 800, overlap: int = 200) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if size <= 0:
        raise ValueError("chunk size must be > 0")
    if overlap >= size:
        overlap = max(0, size // 4)

    chunks = []
    start = 0
    n = len(text)
    step = size - overlap

    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end])
        start += step

    return chunks
