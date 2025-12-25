# rag_core/generation/self_rag.py

def should_retry(answer: str) -> bool:
    a = (answer or "").lower()
    triggers = [
        "not available in documents",
        "i don't know",
        "cannot find",
        "not sure",
        "insufficient",
    ]
    return any(t in a for t in triggers)
