# rag_core/retrieval/fusion.py
from typing import List, Dict
from rag_core.schemas import DocChunk

def rrf_fusion(result_sets: List[List[DocChunk]], top_k: int = 5, k: int = 60) -> List[DocChunk]:
    """
    Reciprocal Rank Fusion:
    score(doc) = Σ 1 / (k + rank)
    """
    scores: Dict[str, float] = {}
    docs_map: Dict[str, DocChunk] = {}

    for docs in result_sets:
        for rank, d in enumerate(docs):
            scores[d.id] = scores.get(d.id, 0.0) + 1.0 / (k + rank + 1)
            docs_map[d.id] = d

    fused = []
    for doc_id, s in scores.items():
        dd = docs_map[doc_id].model_copy()
        dd.score = float(s)
        dd.method = "fusion"
        fused.append(dd)

    fused.sort(key=lambda x: x.score, reverse=True)
    return fused[:top_k]
