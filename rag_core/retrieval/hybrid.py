# rag_core/retrieval/hybrid.py
from typing import List, Dict, Optional
import numpy as np

from rag_core.schemas import DocChunk


def _minmax_norm(vals: List[float]) -> List[float]:
    if not vals:
        return []
    vmin, vmax = float(min(vals)), float(max(vals))
    if vmax - vmin < 1e-9:
        return [1.0 for _ in vals]
    return [(v - vmin) / (vmax - vmin) for v in vals]


class HybridRetriever:
    def __init__(self, vector_store, bm25_store, alpha: float = 0.55):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.alpha = float(alpha)

    def retrieve(self, query: str, top_k: int = 5, pool_mult: int = 4) -> List[DocChunk]:
        v_docs = self.vector_store.search(query, k=top_k * pool_mult)
        b_docs = self.bm25_store.search(query, k=top_k * pool_mult)

        # normalize scores so they combine meaningfully
        v_scores = _minmax_norm([d.score for d in v_docs])
        b_scores = _minmax_norm([d.score for d in b_docs])

        merged: Dict[str, Dict] = {}
        for d, s in zip(v_docs, v_scores):
            merged[d.id] = {"doc": d, "v": float(s), "b": 0.0}
        for d, s in zip(b_docs, b_scores):
            if d.id not in merged:
                merged[d.id] = {"doc": d, "v": 0.0, "b": float(s)}
            else:
                merged[d.id]["b"] = float(s)

        scored = []
        for item in merged.values():
            score = self.alpha * item["v"] + (1.0 - self.alpha) * item["b"]
            dd = item["doc"].model_copy()
            dd.score = float(score)
            dd.method = "hybrid"
            scored.append(dd)

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]


# ✅ REQUIRED BY pipeline.py
def hybrid_retrieve(
    query: str,
    vector_store,
    bm25_store,
    top_k: int = 5,
    alpha: float = 0.55,
    pool_mult: int = 4,
) -> List[DocChunk]:
    """
    Thin wrapper so pipeline can import:
    from rag_core.retrieval.hybrid import hybrid_retrieve
    """
    retriever = HybridRetriever(vector_store=vector_store, bm25_store=bm25_store, alpha=alpha)
    return retriever.retrieve(query=query, top_k=top_k, pool_mult=pool_mult)
