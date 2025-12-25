# rag_core/pipeline.py
from __future__ import annotations

from typing import List, Tuple, Dict, Any

from rag_core.config import settings
from rag_core.schemas import DocChunk

from rag_core.retrieval.hybrid import hybrid_retrieve
from rag_core.reranking.llm_reranker import LLMReranker
from rag_core.generation.answer import generate_answer
from rag_core.generation.explore import explore_document


class Pipeline:
    def __init__(self, vector_store, bm25_store, llm):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.llm = llm
        self.reranker = LLMReranker(llm)

    def retrieve_only(self, query: str) -> List[DocChunk]:
        top_k = getattr(settings, "TOP_K", 5)

        docs = hybrid_retrieve(
            query=query,
            vector_store=self.vector_store,
            bm25_store=self.bm25_store,
            top_k=top_k,
            alpha=getattr(settings, "ALPHA", 0.55),
        )

        if getattr(settings, "ENABLE_RERANK", False):
            docs = self.reranker.rerank(query=query, docs=docs, top_k=top_k)

        return docs

    def run(self, query: str) -> Tuple[str, List[str], List[DocChunk]]:
        docs = self.retrieve_only(query)
        answer, citations = generate_answer(self.llm, query, docs)
        return answer, citations, docs

    def explore(self) -> Dict[str, Any]:
        probe_queries = [
            "summary key points",
            "risk factors recommendations",
            "pregnancy complications outcomes",
            "screening diagnosis monitoring",
            "treatment guidance management",
        ]

        gathered: List[DocChunk] = []
        seen = set()

        for q in probe_queries:
            docs = self.retrieve_only(q)
            for d in docs:
                key = (
                    getattr(d, "source", ""),
                    getattr(d, "chunk_index", getattr(d, "chunk_id", "")),
                )
                if key not in seen:
                    seen.add(key)
                    gathered.append(d)

        gathered = gathered[:18]
        return explore_document(self.llm, gathered)
