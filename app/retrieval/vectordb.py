"""Hybrid retrieval: FAISS dense vectors + BM25 lexical, with optional course boost."""
import os
import json
import logging

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.retrieval.embedder import embed_texts

logger = logging.getLogger(__name__)

INDEX_DIR = os.getenv("INDEX_DIR", "app/data/faiss_index")
INDEX_FILE = os.path.join(INDEX_DIR, "index.faiss")
DOC_FILE = os.path.join(INDEX_DIR, "documents.json")

# ---------- STATE ----------
documents: list[str] = []
bm25: BM25Okapi | None = None
index = faiss.IndexFlatIP(settings.EMBEDDING_DIM)


def rebuild_bm25() -> None:
    global bm25
    tokenized = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized) if tokenized else None


def add_documents(chunks: list[str]) -> None:
    global documents
    chunks = [c for c in (chunks or []) if c and c.strip()]
    if not chunks:
        return

    vectors = embed_texts(chunks)
    if vectors.shape[1] != index.d:
        raise ValueError(
            f"Embedding dim {vectors.shape[1]} != index dim {index.d}. "
            f"Check EMBEDDING_MODEL/EMBEDDING_DIM in .env and rebuild the index."
        )

    index.add(vectors)
    documents.extend(chunks)
    rebuild_bm25()


def search_documents(
    query: str, k: int = 5, course: str | None = None, boost_terms: list[str] | None = None
) -> list[str]:
    if not documents or not query.strip():
        return []

    candidates: list[str] = []
    seen: set[str] = set()

    def _add(text: str) -> None:
        if text not in seen:
            seen.add(text)
            candidates.append(text)

    # ----- Dense (FAISS) -----
    query_vec = embed_texts([query])
    _, indices = index.search(query_vec, min(20, len(documents)))
    for i in indices[0]:
        if 0 <= i < len(documents):
            _add(documents[i])

    # ----- Lexical (BM25) -----
    if bm25 is not None:
        scores = bm25.get_scores(query.lower().split())
        for i in np.argsort(scores)[::-1][:20]:
            if 0 <= i < len(documents):
                _add(documents[i])

    # ----- Boost: put chunks matching the course name and/or intent markers
    # (e.g. location/contact keywords) first, but never drop coverage. Stable
    # sort keeps each group's original relative ranking intact. -----
    markers = [m.lower() for m in ([course] if course else []) + (boost_terms or [])]
    if markers:
        boosted = [c for c in candidates if any(m in c.lower() for m in markers)]
        rest = [c for c in candidates if c not in boosted]
        candidates = boosted + rest

    return candidates[:k]


def save_index() -> None:
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_FILE)
    with open(DOC_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    logger.info("FAISS index saved (%d docs)", len(documents))


def load_index() -> None:
    global index, documents

    if not os.path.exists(INDEX_FILE) or not os.path.exists(DOC_FILE):
        logger.warning("No saved index found at %s. Run scripts/load_data.py first.", INDEX_DIR)
        return

    loaded = faiss.read_index(INDEX_FILE)
    if loaded.d != settings.EMBEDDING_DIM:
        raise ValueError(
            f"Saved index dim {loaded.d} != configured EMBEDDING_DIM {settings.EMBEDDING_DIM}. "
            f"Set EMBEDDING_DIM={loaded.d} (and a matching EMBEDDING_MODEL) or rebuild the index."
        )

    index = loaded
    with open(DOC_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)

    rebuild_bm25()
    logger.info("Hybrid index loaded (%d docs, dim=%d)", len(documents), index.d)
