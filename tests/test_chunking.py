from rag_core.ingestion.chunkers import chunk_text

def test_chunking_basic():
    text = "a" * 2000
    chunks = chunk_text(text, size=800, overlap=200)
    assert len(chunks) >= 3
    assert all(len(c) <= 800 for c in chunks)
