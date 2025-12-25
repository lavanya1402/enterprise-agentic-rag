# rag_core/retrieval/vector_store.py
from __future__ import annotations

import os
import glob
import json
from typing import List, Optional, Union

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from rag_core.ingestion.pdf_loader import load_pdfs
from rag_core.ingestion.chunkers import chunk_text
from rag_core.schemas import DocChunk


def _norm(v: np.ndarray) -> np.ndarray:
    if v.ndim == 1:
        v = v.reshape(1, -1)
    denom = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / denom


class VectorStore:
    """
    FAISS cosine similarity store (SentenceTransformer embeddings).
    - build(pdf_dir OR pdf_paths) indexes PDFs
    - search(query, k) returns DocChunk list
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_dir: str = os.path.join("data", "indexes"),
        index_name: str = "vector.faiss",
        meta_name: str = "vector_meta.json",
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)

        self.index_path = os.path.join(self.index_dir, index_name)
        self.meta_path = os.path.join(self.index_dir, meta_name)

        self.index: Optional[faiss.Index] = None
        self.meta: List[dict] = []  # parallel to vectors

        self._try_load()

    def _try_load(self) -> None:
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.meta = json.load(f)
            except Exception:
                self.index = None
                self.meta = []

    def build(
        self,
        pdf_dir_or_paths: Union[str, List[str]],
        chunk_size: int = 800,
        overlap: int = 200,
    ) -> None:
        # ✅ accept folder OR list of pdf paths
        if isinstance(pdf_dir_or_paths, list):
            pdf_paths = sorted([p for p in pdf_dir_or_paths if str(p).lower().endswith(".pdf")])
            if not pdf_paths:
                raise RuntimeError("No PDF paths provided to VectorStore.build().")
        else:
            pdf_dir = str(pdf_dir_or_paths)
            pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
            if not pdf_paths:
                raise RuntimeError(f"No PDFs found in: {pdf_dir}")

        texts: List[str] = []
        meta: List[dict] = []

        for path in pdf_paths:
            source = os.path.basename(path)

            full_text = load_pdfs(path)

            # ✅ make sure loader output is string
            if isinstance(full_text, list):
                full_text = "\n".join([str(x) for x in full_text])
            full_text = (full_text or "").strip()

            if not full_text:
                # keep going; we will error later if nothing extracted overall
                continue

            chunks = chunk_text(full_text, size=chunk_size, overlap=overlap)
            for i, ch in enumerate(chunks):
                ch = (ch or "").strip()
                if not ch:
                    continue
                cid = f"{source}::chunk_{i}"
                texts.append(ch)
                meta.append({"id": cid, "source": source, "chunk_index": i, "text": ch})

        # ✅ CRITICAL GUARD (your current error)
        if not texts:
            raise RuntimeError(
                "VectorStore.build(): No extractable text chunks were created.\n"
                "Possible reasons:\n"
                "1) PDF is scanned image (needs OCR)\n"
                "2) pdf_loader.load_pdfs() returning empty text\n"
                "3) chunking settings too strict\n\n"
                "Quick checks:\n"
                "- Print first 500 chars of load_pdfs(pdf)\n"
                "- If scanned PDF, use OCR (pytesseract) or pdfplumber image OCR\n"
            )

        # ✅ embed (always list -> output will be 2D)
        embs = self.model.encode(texts, batch_size=32, show_progress_bar=False)
        embs = np.array(embs, dtype="float32")
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)

        embs = _norm(embs)

        dim = int(embs.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(embs)

        self.index = index
        self.meta = meta

        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def search(self, query: str, k: int = 5) -> List[DocChunk]:
        if self.index is None or not self.meta:
            raise RuntimeError("Vector index not built. Click Build/Refresh Index first.")

        q_emb = self.model.encode([query], show_progress_bar=False)
        q_emb = np.array(q_emb, dtype="float32")
        q_emb = _norm(q_emb)

        scores, idxs = self.index.search(q_emb, k)
        scores = scores[0].tolist()
        idxs = idxs[0].tolist()

        results: List[DocChunk] = []
        for score, ix in zip(scores, idxs):
            if ix < 0 or ix >= len(self.meta):
                continue
            m = self.meta[ix]
            results.append(
                DocChunk(
                    id=m["id"],
                    source=m["source"],
                    chunk_index=int(m["chunk_index"]),
                    text=m["text"],
                    score=float(score),
                    method="vector",
                )
            )
        return results
