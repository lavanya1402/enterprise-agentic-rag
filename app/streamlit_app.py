# =========================
# app/streamlit_app.py
# =========================

import os
import sys
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# --- 🔥 FIX PYTHON PATH ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- UI text ---
from ui_text import (
    APP_TITLE, APP_SUBTITLE, INTRO, SIDEBAR_TITLE,
    UPLOAD_HEADER, UPLOAD_HELP,
    INDEX_HEADER, INDEX_HELP,
    QUERY_HEADER, QUERY_PLACEHOLDER,
    ANSWER_HEADER, CITATIONS_HEADER, CONTEXT_HEADER,
    BUILD_INDEX_BTN, SPINNER_INDEX, SPINNER_ANSWER,
    NEED_INDEX_WARNING, UPLOAD_SUCCESS, INDEX_SUCCESS,
    NO_CITATIONS, FOOTER_NOTE
)

# --- RAG core ---
from rag_core.config import settings
from rag_core.pipeline import Pipeline

# ✅ Your structure: rag_core/retrieval/
from rag_core.retrieval.vector_store import VectorStore
from rag_core.retrieval.bm25_store import BM25Store

load_dotenv()

# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)
st.markdown(INTRO)

RAW_DIR = os.path.join("data", "raw_pdfs")
os.makedirs(RAW_DIR, exist_ok=True)

# -------------------------
# Helpers
# -------------------------
def list_pdf_paths(folder: str):
    if not os.path.exists(folder):
        return []
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".pdf")
    ])

# -------------------------
# LLM wrapper
# -------------------------
def build_llm():
    # prefer Streamlit secrets, else env
    api_key = ""
    try:
        api_key = (st.secrets.get("OPENAI_API_KEY", "") or "").strip()
    except Exception:
        api_key = ""

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key:
        return None, "OPENAI_API_KEY missing. Set it in .env or Streamlit secrets."

    client = OpenAI(api_key=api_key)
    model = getattr(settings, "MODEL", "gpt-4o-mini")
    temperature = float(getattr(settings, "OPENAI_TEMPERATURE", 0.2))

    def _llm(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a strict grounded QA assistant. Use ONLY provided context."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    return _llm, None

llm, llm_err = build_llm()
if llm_err:
    st.error(llm_err)
    st.stop()

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header(SIDEBAR_TITLE)
    st.json({
        "ENABLE_HYBRID": getattr(settings, "ENABLE_HYBRID", True),
        "ENABLE_QUERY_EXPA": getattr(settings, "ENABLE_QUERY_EXPA", False),
        "ENABLE_RERANK": getattr(settings, "ENABLE_RERANK", False),
        "ENABLE_SELF_RAG": getattr(settings, "ENABLE_SELF_RAG", False),
        "ENABLE_CRAG": getattr(settings, "ENABLE_CRAG", False),
        "TOP_K": getattr(settings, "TOP_K", 5),
        "ALPHA": getattr(settings, "ALPHA", 0.55),
        "MODEL": getattr(settings, "MODEL", "gpt-4o-mini"),
    })
    st.caption(FOOTER_NOTE)

# -------------------------
# Upload PDFs
# -------------------------
st.subheader(UPLOAD_HEADER)
uploads = st.file_uploader(UPLOAD_HELP, type=["pdf"], accept_multiple_files=True)

if uploads:
    for f in uploads:
        with open(os.path.join(RAW_DIR, f.name), "wb") as out:
            out.write(f.getbuffer())
    st.success(UPLOAD_SUCCESS)

# -------------------------
# Build Index  ✅ (REPLACED BLOCK)
# -------------------------
st.subheader(INDEX_HEADER)
st.caption(INDEX_HELP)

if st.button(BUILD_INDEX_BTN):
    with st.spinner(SPINNER_INDEX):
        try:
            pdf_paths = list_pdf_paths(RAW_DIR)
            if not pdf_paths:
                st.error("No PDFs found in data/raw_pdfs. Upload at least one PDF first.")
                st.stop()

            # ✅ DEBUG: check extraction (tell scanned vs text)
            from rag_core.ingestion.pdf_loader import load_pdfs
            debug_lines = []
            for p in pdf_paths:
                txt = load_pdfs(p, ocr=False, max_pages=2)  # first try WITHOUT OCR
                preview = (txt[:500] + "..." if len(txt) > 500 else txt)
                debug_lines.append((os.path.basename(p), len(txt), preview))

            with st.expander("🔎 Debug: Extracted text preview (first 2 pages, no OCR)"):
                for name, n_chars, preview in debug_lines:
                    st.markdown(f"**{name}** → extracted chars: `{n_chars}`")
                    st.code(preview if preview else "<<< EMPTY TEXT >>>")

            # ✅ Build stores normally (DIRECT CALL)
            vector_store = VectorStore()
            bm25_store = BM25Store()

            vector_store.build(RAW_DIR)
            bm25_store.build(RAW_DIR)

            st.session_state["vector_store"] = vector_store
            st.session_state["bm25_store"] = bm25_store

        except Exception as e:
            st.error(f"Index build failed: {e}")
            st.stop()

    st.success(INDEX_SUCCESS)

# -------------------------
# Ask Question
# -------------------------
st.subheader(QUERY_HEADER)
query = st.text_input("Ask:", value="", placeholder=QUERY_PLACEHOLDER)

if query:
    if "vector_store" not in st.session_state or "bm25_store" not in st.session_state:
        st.warning(NEED_INDEX_WARNING)
    else:
        pipeline = Pipeline(
            vector_store=st.session_state["vector_store"],
            bm25_store=st.session_state["bm25_store"],
            llm=llm,
        )

        with st.spinner(SPINNER_ANSWER):
            try:
                answer, citations, docs = pipeline.run(query.strip())
            except Exception as e:
                st.error(f"Query failed: {e}")
                st.stop()

        st.markdown(f"## {ANSWER_HEADER}")
        st.write(answer)

        st.markdown(f"## {CITATIONS_HEADER}")
        if citations:
            for i, c in enumerate(citations, 1):
                st.write(f"{i}. {c}")
        else:
            st.write(NO_CITATIONS)

        st.markdown(f"## {CONTEXT_HEADER}")
        for d in docs:
            src = getattr(d, "source", "unknown")
            idx = getattr(d, "chunk_index", getattr(d, "chunk_id", "NA"))
            score = getattr(d, "score", None)
            header = f"HYBRID | {src} | chunk {idx}"
            if score is not None:
                header += f" | score={score:.4f}"

            with st.expander(header):
                st.write(getattr(d, "text", ""))
