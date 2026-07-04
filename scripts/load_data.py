"""Build the knowledge base: manual chunks + website + PDFs -> FAISS/BM25 index.

Run from the project root:  python -m scripts.load_data
"""
import os
import logging

from app.ingestion.web_scraper import scrape_url, scrape_website
from app.ingestion.cleaner import clean_text
from app.ingestion.chunker import split_text
from app.ingestion.dedup import drop_frequent_boilerplate
from app.ingestion.pdf_loader import load_all_pdfs
from app.retrieval.vectordb import add_documents, save_index

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CORE_URLS = [
    "https://fullstackacademy.in/",
    "https://fullstackacademy.in/about/",
    "https://fullstackacademy.in/contact/",
]

URL_KEYWORDS = ["course", "data", "science", "analyst", "ai", "devops", "branch", "contact", "about", "registration"]
CONTENT_KEYWORDS = ["course", "training", "batch", "fees", "duration", "placement", "learn", "certification", "syllabus"]

MANUAL_CHUNKS_FILE = "manual_curriculum_chunks.txt"


def score_page(page: dict) -> int:
    url = page["url"].lower()
    text = page["content"].lower()
    return sum(2 for w in URL_KEYWORDS if w in url) + sum(1 for w in CONTENT_KEYWORDS if w in text)


def load_manual_chunks() -> list[str]:
    if not os.path.exists(MANUAL_CHUNKS_FILE):
        logger.info("No manual curriculum chunks found.")
        return []
    with open(MANUAL_CHUNKS_FILE, "r", encoding="utf-8") as f:
        raw = f.read()
    chunks = [c.strip() for c in raw.split("\n\n") if c.strip()]
    logger.info("Loaded %d manual curriculum chunks.", len(chunks))
    return chunks


def collect_pages() -> list[dict]:
    pages, visited = [], set()
    for url in CORE_URLS:
        try:
            pages.append(scrape_url(url))
            visited.add(url)
            logger.info("Loaded: %s", url)
        except Exception:
            logger.exception("Skipped: %s", url)

    for page in scrape_website(limit=30):
        if page["url"] not in visited:
            pages.append(page)
            visited.add(page["url"])
    return pages


def main() -> None:
    all_chunks: list[str] = load_manual_chunks()

    for page in collect_pages():
        if score_page(page) >= 3:
            all_chunks.extend(split_text(clean_text(page["content"])))

    for pdf in load_all_pdfs():
        all_chunks.extend(split_text(clean_text(pdf["content"])))

    all_chunks = [c for c in all_chunks if c and c.strip()]

    before = len(all_chunks)
    all_chunks = drop_frequent_boilerplate(all_chunks)
    logger.info("Dropped %d repeated boilerplate chunks (nav/footer/CTA text).", before - len(all_chunks))

    add_documents(all_chunks)
    save_index()
    logger.info("Knowledge base built: %d chunks", len(all_chunks))


if __name__ == "__main__":
    main()
