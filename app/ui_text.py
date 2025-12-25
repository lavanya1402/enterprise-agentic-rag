# app/ui_text.py

APP_TITLE = "📄 Enterprise Modular RAG"
APP_SUBTITLE = "Hybrid (BM25 + Vector) • Fusion/HyDE • Rerank • Self-RAG/CRAG • Citations"

INTRO = """
This app lets you upload enterprise-style policy PDFs and ask questions.
It retrieves relevant chunks using hybrid retrieval,
optionally expands queries, reranks results,
and generates grounded answers with citations.
"""

SIDEBAR_TITLE = "⚙️ Modes & Settings"

UPLOAD_HEADER = "1) Upload PDFs"
UPLOAD_HELP = "Upload one or more enterprise policy PDFs."

INDEX_HEADER = "2) Build Index"
INDEX_HELP = "Build or refresh BM25 + Vector indexes."

QUERY_HEADER = "3) Ask Questions"
QUERY_PLACEHOLDER = "e.g., What is the notice period?"

ANSWER_HEADER = "🧠 Answer"
CITATIONS_HEADER = "📌 Citations"
CONTEXT_HEADER = "🔎 Retrieved Context (debug)"

NO_CITATIONS = "No citations available."

BUILD_INDEX_BTN = "📌 Build/Refresh Index"
SPINNER_INDEX = "Indexing PDFs..."
SPINNER_ANSWER = "Thinking..."

NEED_INDEX_WARNING = "Please click **Build/Refresh Index** first."
UPLOAD_SUCCESS = "PDFs saved successfully ✅"
INDEX_SUCCESS = "Index built successfully ✅"

FOOTER_NOTE = "Tip: Control features using .env and restart Streamlit."
