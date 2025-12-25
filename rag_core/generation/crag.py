# rag_core/generation/crag.py
from rag_core.generation.self_rag import should_retry

def crag_run(pipeline, query: str, max_iters: int = 2):
    last = None
    for _ in range(max_iters):
        last = pipeline.run(query)
        if not should_retry(last.answer):
            return last
    return last
