from rag_core.retrieval.hybrid import HybridRetriever
from rag_core.schemas import DocChunk

class DummyVS:
    def search(self, q, k=5):
        return [DocChunk(id="1", text="alpha beta", source="a.pdf", score=0.9),
                DocChunk(id="2", text="gamma", source="b.pdf", score=0.2)]

class DummyBM:
    def search(self, q, k=5):
        return [DocChunk(id="2", text="gamma", source="b.pdf", score=10.0),
                DocChunk(id="3", text="delta", source="c.pdf", score=8.0)]

def test_hybrid_merge():
    hy = HybridRetriever(DummyVS(), DummyBM(), alpha=0.5)
    res = hy.retrieve("test", top_k=3)
    assert len(res) == 3
    assert any(d.id == "1" for d in res)
    assert any(d.id == "2" for d in res)
    assert any(d.id == "3" for d in res)
