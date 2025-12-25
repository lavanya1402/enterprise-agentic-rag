# rag_core/prompts.py

ANSWER_PROMPT = """
You are a grounded, evidence-based QA assistant.

Rules:
- Use ONLY the provided CONTEXT.
- You MAY infer answers when the document discusses related concepts
  (e.g., recommendations, prevention, guidance) even if the exact wording
  of the question is not used.
- Every claim MUST be supported by citations taken from the chunk labels.
- If the document truly does not contain related information, reply exactly:
  Not available in documents.
- Keep answers concise and factual.
- Cite sources inline like: [diabetes of woman.pdf | chunk 94]
- If you provide bullets, put at least one citation in every bullet.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

EXPLORE_PROMPT = """
You are an analyst that generates answerable questions from provided document excerpts.

Goal:
- Produce a short "snapshot" (2-4 lines)
- Extract 6-10 key topics
- Generate 10-15 highly answerable questions that can be answered using ONLY the given excerpts.
- Questions must be specific, not generic.
- Each question MUST reference at least one chunk id that contains the answer.

Output STRICT JSON ONLY with this schema:
{
  "snapshot": "string",
  "topics": ["t1","t2",...],
  "questions": [
    {
      "q": "question text",
      "support": ["source|chunk 47", "source|chunk 198"]
    }
  ]
}

EXCERPTS:
{context}
"""

QUERY_EXPANSION_PROMPT = """Generate {n} alternative search queries for the user question.
Keep them short and varied. One per line. No numbering.

Question:
{query}
"""

RERANK_PROMPT = """You are ranking passages for relevance to a question.
Return ONLY the top {top_k} passage indices as comma-separated numbers (e.g., "2,0,3").

Question:
{query}

Passages:
{passages}
"""
