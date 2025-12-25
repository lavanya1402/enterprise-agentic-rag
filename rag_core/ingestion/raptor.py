# rag_core/ingestion/raptor.py
"""
Minimal RAPTOR-style placeholder.
Real RAPTOR builds summary trees. For now we provide a safe interface so you can extend later.
"""

from typing import List, Dict

def raptor_summaries(docs: List[Dict]) -> List[Dict]:
    """
    Input: [{"source":..., "text":...}, ...]
    Output: same structure, optionally with "summary" key.
    """
    out = []
    for d in docs:
        out.append({**d, "summary": ""})
    return out
