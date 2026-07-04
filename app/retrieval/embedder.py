"""Sentence-embedding wrapper.

The model is loaded lazily so importing this module (e.g. during tests or app
startup ordering) does not immediately pull a large model into memory.
"""
from functools import lru_cache

import numpy as np

from app.core.config import settings


@lru_cache(maxsize=1)
def get_model():
    # Imported lazily so this module can be imported without the heavy
    # sentence-transformers/torch stack present (e.g. in unit tests).
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.EMBEDDING_MODEL)


def embed_texts(texts) -> np.ndarray:
    """Return L2-normalized float32 embeddings shaped (n, EMBEDDING_DIM)."""
    if isinstance(texts, str):
        texts = [texts]

    embeddings = get_model().encode(
        list(texts),
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")
