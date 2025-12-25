# rag_core/generation/explore.py
from __future__ import annotations

import json
from typing import List, Dict, Any

from rag_core.prompts import EXPLORE_PROMPT
from rag_core.schemas import DocChunk


def _format_context(docs: List[DocChunk], max_chars: int = 16000) -> str:
    parts = []
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


def explore_document(llm, docs: List[DocChunk]) -> Dict[str, Any]:
    """
    Produces:
    {
      snapshot: str,
      topics: [..],
      questions: [{q:..., support:[...]}]
    }
    """
    context = _format_context(docs)

    if not context:
        return {
            "snapshot": "No readable content retrieved from document.",
            "topics": [],
            "questions": [],
        }

    prompt = EXPLORE_PROMPT.format(context=context)
    raw = (llm(prompt) or "").strip()

    # Try parse strict JSON
    try:
        data = json.loads(raw)
    except Exception:
        # Fallback: try to extract JSON substring
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(raw[start : end + 1])
            except Exception:
                data = {}
        else:
            data = {}

    snapshot = data.get("snapshot") or "Document snapshot unavailable."
    topics = data.get("topics") or []
    questions = data.get("questions") or []

    # Validate shape
    clean_q = []
    for item in questions:
        q = (item.get("q") or "").strip()
        support = item.get("support") or []
        if q:
            clean_q.append({"q": q, "support": support})

    return {
        "snapshot": snapshot,
        "topics": topics[:12],
        "questions": clean_q[:20],
    }
