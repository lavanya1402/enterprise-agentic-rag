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

<img width="939" height="503" alt="image" src="https://github.com/user-attachments/assets/ac1e941c-bfeb-45fc-a96f-93e269545f10" />

---

### Screenshot 2 – Retrieved Context & Source Chunks

<img width="939" height="503" alt="image" src="https://github.com/user-attachments/assets/b3f2f352-45ff-4996-8860-2698c3398337" />

---

### Screenshot 3 – Hybrid Retrieval (FAISS + BM25)

<img width="939" height="503" alt="image" src="https://github.com/user-attachments/assets/2b56d476-bdfd-4c68-99e8-6a29768edc24" />

---

### Screenshot 4 – Self-RAG Validation / Fallback

<img width="939" height="503" alt="image" src="https://github.com/user-attachments/assets/2b56d476-bdfd-4c68-99e8-6a29768edc24" />

---

### Screenshot 5 – Multi-Document Resolution

<img width="939" height="503" alt="image" src="https://github.com/user-attachments/assets/2b56d476-bdfd-4c68-99e8-6a29768edc24" />

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
