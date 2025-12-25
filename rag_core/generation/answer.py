# rag_core/generation/answer.py
from __future__ import annotations

from typing import List, Tuple
from rag_core.prompts import ANSWER_PROMPT
from rag_core.schemas import DocChunk


def _format_context(docs: List[DocChunk], max_chars: int = 24000) -> str:
    """
    Build a context string with stable chunk labels.
    Increased max_chars for stronger grounding on academic PDFs.
    """
    parts: List[str] = []
    total = 0

    for d in docs:
        source = getattr(d, "source", "unknown")
        chunk_index = getattr(d, "chunk_index", getattr(d, "chunk_id", "NA"))
        text = (getattr(d, "text", "") or "").strip()

        if not text:
            continue

        block = f"[{source} | chunk {chunk_index}]\n{text}\n"
        if total + len(block) > max_chars:
            break

        parts.append(block)
        total += len(block)

    return "\n".join(parts).strip()


def generate_answer(llm, question: str, docs: List[DocChunk]) -> Tuple[str, List[str]]:
    """
    Returns: (answer, citations_sources_list)
    Answer is grounded and expects inline citations like:
    [file.pdf | chunk 94]
    """
    context = _format_context(docs)

    # If retrieval gave nothing, short-circuit
    if not context:
        return "Not available in documents.", []

    prompt = ANSWER_PROMPT.format(context=context, question=question)
    answer = (llm(prompt) or "").strip()

    # Enforce exact missing policy
    if (not answer) or (answer.strip().lower() == "not available in documents.") or ("not available in documents" in answer.lower() and len(answer) < 60):
        answer = "Not available in documents."

    # Sources list for UI citations section (unique sources)
    citations: List[str] = []
    seen = set()
    for d in docs:
        src = getattr(d, "source", None)
        if src and src not in seen:
            citations.append(src)
            seen.add(src)

    return answer, citations
