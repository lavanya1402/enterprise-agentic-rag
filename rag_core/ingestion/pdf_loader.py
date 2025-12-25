# rag_core/ingestion/pdf_loader.py
from __future__ import annotations

import os
from typing import Optional

import pdfplumber


def load_pdfs(pdf_path: str, ocr: bool = True, max_pages: Optional[int] = None) -> str:
    """
    Extract text from a PDF.
    - Works for normal text PDFs via pdfplumber
    - If text is empty and ocr=True, tries OCR (requires pytesseract + installed Tesseract)
Using directly will help you debug quickly.
    """
    if not pdf_path or not os.path.exists(pdf_path):
        return ""

    texts = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = pdf.pages if max_pages is None else pdf.pages[:max_pages]
            for p in pages:
                t = p.extract_text() or ""
                t = t.strip()
                if t:
                    texts.append(t)

    except Exception:
        # if pdfplumber fails unexpectedly
        texts = []

    extracted = "\n\n".join(texts).strip()

    # ✅ If we got text, return it
    if extracted:
        return extracted

    # ✅ Fallback: OCR (only if enabled)
    if not ocr:
        return ""

    # OCR is optional: only runs if pytesseract is installed
    try:
        import pytesseract
        from PIL import Image
    except Exception:
        # pytesseract / PIL not installed
        return ""

    ocr_texts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = pdf.pages if max_pages is None else pdf.pages[:max_pages]
            for p in pages:
                # render page -> image
                im = p.to_image(resolution=200).original
                # pytesseract OCR
                ocr_t = pytesseract.image_to_string(im) or ""
                ocr_t = ocr_t.strip()
                if ocr_t:
                    ocr_texts.append(ocr_t)
    except Exception:
        ocr_texts = []

    return "\n\n".join(ocr_texts).strip()
