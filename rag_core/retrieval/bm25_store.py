# rag_core/retrieval/bm25_store.py
from __future__ import annotations

import os
import glob
import json
import re
from typing import List, Optional

from rank_bm25 import BM25Okapi

from rag_core.ingestion.pdf_loader import load_pdfs
from rag_core.ingestion.chunkers import chunk_text
from rag_core.schemas import DocChunk


def _tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    # simple tokenizer
    return re.findall(r"[a-z0-9]+", text)


class BM25Store:
    """
    BM25 index over chunks.
    - build(pdf_dir) builds corpus
    - search(query, k) returns DocChunk list
    """

    def __init__(
        self,
        index_dir: str = os.path.join("data", "indexes"),
        meta_name: str = "bm25_meta.json",
    ):
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)

        self.meta_path = os.path.join(self.index_dir, meta_name)

        self.bm25: Optional[BM25Okapi] = None
        self.meta: List[dict] = []        # parallel to corpus: {id, source, chunk_index, text}
        self.corpus_tokens: List[List[str]] = []

        # optional load
        self._try_load()

    def _try_load(self) -> None:
        # BM25 model cannot be perfectly restored without corpus tokens
        # so we store meta + tokens, rebuild bm25 on load
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                self.meta = payload.get("meta", [])
                self.corpus_tokens = payload.get("corpus_tokens", [])
                if self.meta and self.corpus_tokens:
                    self.bm25 = BM25Okapi(self.corpus_tokens)
            except Exception:
                self.bm25 = None
                self.meta = []
                self.corpus_tokens = []

    def build(self, pdf_dir: str, chunk_size: int = 800, overlap: int = 200) -> None:
        pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
        if not pdf_paths:
            raise RuntimeError(f"No PDFs found in: {pdf_dir}")

        meta: List[dict] = []
        corpus_tokens: List[List[str]] = []

        for path in pdf_paths:
            source = os.path.basename(path)
            full_text = load_pdfs(path)
            chunks = chunk_text(full_text, size=chunk_size, overlap=overlap)

            for i, ch in enumerate(chunks):
                cid = f"{source}::chunk_{i}"
                meta.append(
                    {
                        "id": cid,
                        "source": source,
                        "chunk_index": i,
                        "text": ch,
                    }
                )
                corpus_tokens.append(_tokenize(ch))

        self.meta = meta
        self.corpus_tokens = corpus_tokens
        self.bm25 = BM25Okapi(self.corpus_tokens)

        # persist
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {"meta": self.meta, "corpus_tokens": self.corpus_tokens},
                f,
                ensure_ascii=False,
                indent=2,
            )

    def search(self, query: str, k: int = 5) -> List[DocChunk]:
        if self.bm25 is None or not self.meta:
            raise RuntimeError("BM25 index not built. Click Build/Refresh Index first.")

        q_tokens = _tokenize(query)
        scores = self.bm25.get_scores(q_tokens)  # array floats
        # get top-k indices
        top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results: List[DocChunk] = []
        for ix in top_idxs:
            m = self.meta[ix]
            results.append(
                DocChunk(
                    id=m["id"],
                    source=m["source"],
                    chunk_index=int(m["chunk_index"]),
                    text=m["text"],
                    score=float(scores[ix]),
                    method="bm25",
                )
            )
        return results
