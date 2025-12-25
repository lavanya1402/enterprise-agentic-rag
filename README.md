# Enterprise Policy Intelligence System (Hybrid RAG)

**Live Application**
[https://rag-playgrounds.streamlit.app/](https://rag-playgrounds.streamlit.app/)

---

## Project Overview

This project implements a **production-grade Hybrid Retrieval-Augmented Generation (RAG) system** for answering **enterprise HR, policy, ethics, compliance, and healthcare guideline questions** with **strict grounding and verifiable citations**.

The system is designed for real-world enterprise usage where:

* Answers must come from the **correct document**
* Responses must be **auditable and traceable**
* Hallucinations are **not acceptable**
* Policy interpretation must be **evidence-backed**

---

## Public Policy Documents Used

* Employee Handbook
  [https://resources.workable.com/wp-content/uploads/2017/09/Employee-Handbook.pdf](https://resources.workable.com/wp-content/uploads/2017/09/Employee-Handbook.pdf)

* Work From Home Policy
  [https://blog.hrpartner.io/wp-content/uploads/Working-from-Home-Policy.pdf](https://blog.hrpartner.io/wp-content/uploads/Working-from-Home-Policy.pdf)

* Flexible / IT Work Policy
  [https://www.delltechnologies.com/asset/en-us/products/servers/industry-market/flexible-work-policy.pdf](https://www.delltechnologies.com/asset/en-us/products/servers/industry-market/flexible-work-policy.pdf)

* Code of Ethics & Conduct
  [https://www.dol.gov/sites/dolgov/files/oasam/legacy/files/ethics-code-of-conduct.pdf](https://www.dol.gov/sites/dolgov/files/oasam/legacy/files/ethics-code-of-conduct.pdf)

* WHO Healthcare Guideline (Diabetes in Pregnancy)
  [https://www.who.int/publications/i/item/9789240091603](https://www.who.int/publications/i/item/9789240091603)

---

## Business-Critical Questions Evaluated

1. What actions are classified as ethical violations and what disciplinary consequences apply?
2. What determines eligibility for remote or hybrid work and who approves it?
3. What leave types are available to teaching staff and which require prior approval?
4. How should employees report handbook inconsistencies?
5. Who has final authority when HR policies conflict?

---

## Screenshot Evidence (Click to Upload)

> ⚠️ **IMPORTANT:**
> Click any image below **while editing README on GitHub** → GitHub will prompt you to **upload screenshot from your system**.

---

### Screenshot 1 – Question Answer with Citations

[![Upload Screenshot – Q\&A](screenshots/answer_with_citations.png)](screenshots/answer_with_citations.png)

---

### Screenshot 2 – Retrieved Context & Source Chunks

[![Upload Screenshot – Retrieved Context](screenshots/retrieved_context.png)](screenshots/retrieved_context.png)

---

### Screenshot 3 – Hybrid Retrieval (FAISS + BM25)

[![Upload Screenshot – Hybrid Retrieval](screenshots/hybrid_retrieval.png)](screenshots/hybrid_retrieval.png)

---

### Screenshot 4 – Self-RAG Validation / Fallback

[![Upload Screenshot – Self RAG](screenshots/self_rag_fallback.png)](screenshots/self_rag_fallback.png)

---

### Screenshot 5 – Multi-Document Resolution

[![Upload Screenshot – Multi Document](screenshots/multi_document_answer.png)](screenshots/multi_document_answer.png)

---

## Why This System Is Enterprise-Relevant

* Uses **hybrid retrieval** for precision + recall
* Enforces **zero hallucination tolerance**
* Supports **multi-document policy resolution**
* Fully **audit-friendly and explainable**
* Suitable for HR, compliance, legal, and healthcare governance

---

## Deployment

* Frontend: Streamlit
* Retrieval: FAISS + BM25
* Generation: GPT-based model with strict grounding
* Hosting: Streamlit Cloud

**Live URL**
[https://rag-playgrounds.streamlit.app/](https://rag-playgrounds.streamlit.app/)

---

## Final Note

This project focuses on **real enterprise GenAI behavior**, not experimental demos:

* Correct document resolution
* Reliable retrieval
* Evidence-backed answers
* Production safety

---
