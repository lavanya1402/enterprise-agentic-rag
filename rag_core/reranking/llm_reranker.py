# rag_core/reranking/llm_reranker.py
from __future__ import annotations

import json
from typing import List

from rag_core.prompts import RERANK_PROMPT
from rag_core.schemas import DocChunk


class LLMReranker:
    """
    LLM-based reranker expecting JSON:
    {"ranking":[1,2,3,...]}  (1-based indices referring to P1..Pn)
    """

    def __init__(self, llm):
        self.llm = llm

    def rerank(self, query: str, docs: List[DocChunk], top_k: int = 5) -> List[DocChunk]:
        if not docs:
            return []

        top_k = max(1, min(int(top_k), len(docs)))

        blocks = []
        for i, d in enumerate(docs, start=1):
            text = (getattr(d, "text", "") or "").strip()
            blocks.append(f"[P{i}]\n{text[:1200]}")

        passages = "\n\n".join(blocks)
        prompt = RERANK_PROMPT.format(query=query, passages=passages)
        raw = (self.llm(prompt) or "").strip()

        ranking = None
        try:
            obj = json.loads(raw)
            ranking = obj.get("ranking", None)
            if not isinstance(ranking, list):
                ranking = None
        except Exception:
            ranking = None

        # fallback parse "1,2,3"
        if not ranking:
            cleaned = raw.replace("\n", ",").replace(" ", "")
            parts = [p for p in cleaned.split(",") if p]
            parsed = []
            for p in parts:
                if p.isdigit():
                    parsed.append(int(p))
            ranking = parsed if parsed else None

        if not ranking:
            # fallback original order
            out = []
            for d in docs[:top_k]:
                d2 = d.model_copy() if hasattr(d, "model_copy") else d
                try:
                    d2.method = "rerank"
                except Exception:
                    pass
                out.append(d2)
            return out

        # normalize 1-based -> 0-based, dedupe
        seen = set()
        norm = []
        for idx in ranking:
            if not isinstance(idx, int):
                continue
            j = idx - 1
            if 0 <= j < len(docs) and j not in seen:
                norm.append(j)
                seen.add(j)

        if not norm:
            norm = list(range(len(docs)))

        out: List[DocChunk] = []
        for j in norm[:top_k]:
            d2 = docs[j].model_copy() if hasattr(docs[j], "model_copy") else docs[j]
            try:
                d2.method = "rerank"
            except Exception:
                pass
            out.append(d2)

        return out
