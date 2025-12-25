# rag_core/retrieval/query_expansion.py
from typing import List
from rag_core.prompts import QUERY_EXPANSION_PROMPT

def expand_queries(llm, query: str, n: int = 3) -> List[str]:
    prompt = QUERY_EXPANSION_PROMPT.format(query=query, n=n)
    text = llm(prompt).strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # ensure original query included
    if query.strip() not in lines:
        lines.insert(0, query.strip())
    return lines[: max(1, n)]
