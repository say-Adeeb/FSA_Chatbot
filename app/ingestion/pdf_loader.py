"""Extract text from PDFs using PyMuPDF (imported as `fitz`)."""
import os
import logging

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

PDF_FOLDER = os.getenv("PDF_FOLDER", "data/pdfs")


def extract_pdf_text(file_path: str) -> str:
    texts = []
    with fitz.open(file_path) as doc:
        for page in doc:
            text = page.get_text()
            if text.strip():
                texts.append(text)
    return "\n".join(texts)


def load_all_pdfs(folder: str = PDF_FOLDER) -> list[dict]:
    documents: list[dict] = []
    if not os.path.isdir(folder):
        logger.warning("PDF folder not found: %s", folder)
        return documents

    for file_name in sorted(os.listdir(folder)):
        if not file_name.lower().endswith(".pdf"):
            continue
        path = os.path.join(folder, file_name)
        try:
            documents.append({"source": file_name, "content": extract_pdf_text(path)})
            logger.info("Loaded PDF: %s", file_name)
        except Exception:
            logger.exception("Skipped PDF: %s", file_name)

    return documents
