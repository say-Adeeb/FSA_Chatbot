import os
from dotenv import load_dotenv

load_dotenv()


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes", "on")


class Settings:
    # --- App ---
    APP_NAME = os.getenv("APP_NAME", "RAG Chatbot")
    DEBUG = _as_bool(os.getenv("DEBUG", "False"))
    PORT = int(os.getenv("PORT", 8000))

    # --- CORS (comma-separated origins; "*" allows all) ---
    ALLOWED_ORIGINS = [
        o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()
    ]

    # --- Secrets ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    HF_TOKEN = os.getenv("HF_TOKEN", "")

    # --- Embeddings ---
    # Model + dimension MUST stay in sync. If you change the model, rebuild the
    # index (python -m scripts.load_data). Default: bge-base (768-dim).
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 768))

    # --- LLM ---
    LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.2))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 1000))

    # --- Retrieval ---
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 8))
    MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", 2000))

    # --- Rate limiting (per client IP) ---
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", 20))

    def validate(self) -> None:
        """Fail fast on misconfiguration."""
        if not self.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and fill it in."
            )


settings = Settings()
