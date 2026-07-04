# Full Stack Academy RAG Chatbot

A FastAPI RAG (Retrieval-Augmented Generation) service that answers questions
about Full Stack Academy's courses. It retrieves context from a hybrid index
(FAISS dense vectors + BM25 lexical search) built from the website, course PDFs,
and a manual curriculum file, then generates grounded answers using Groq (Llama 3.1).

## Architecture

```
Website + PDFs + manual chunks
        │  (scripts/load_data.py)
        ▼
  clean → chunk → embed  ──►  FAISS index + documents.json
        │
User ──► /chat ──► ask_rag ──► hybrid search (FAISS + BM25) ──► Groq LLM ──► reply
```

## Setup

1. Create and fill your environment file:
   ```bash
   cp .env.example .env
   # then edit .env and add your GROQ_API_KEY (and HF_TOKEN if needed)
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Build the knowledge base index (one-time, and whenever content changes):
   ```bash
   python -m scripts.load_data
   ```
   This scrapes the website, reads the PDFs in `data/pdfs/`, embeds everything,
   and writes `app/data/faiss_index/`.

4. Run the API:
   ```bash
   uvicorn app.main:app --reload
   ```
   Docs at http://localhost:8000/docs

## Usage

```bash
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{"message": "What topics are in the Data Science course?"}'
```

## Tests

```bash
pytest
```
Tests mock the embedding model and Groq, so they run fast and need no network.

## Evaluation

```bash
python -m evaluation.run_eval
```

Runs a golden set of questions (`evaluation/golden_dataset.py`) through the real
pipeline (real embeddings, real Groq generation) and reports:

- **Retrieval quality** — hit-rate@k and MRR, using keyword containment as a
  relevance proxy (no hand-labeled chunk IDs).
- **Course-detection accuracy** — does `extract_course_name` pick the right course.
- **Refusal accuracy** — does the bot correctly decline to answer off-topic or
  uncovered questions instead of fabricating curriculum details.
- **Answer quality** — an LLM-as-judge (Groq) scores each answer for
  groundedness (no unsupported claims) and relevance.

Each run is saved to `evaluation/results/eval_<timestamp>.json` for tracking
quality over time as the ingestion pipeline or prompts change. Unlike the
mocked unit tests, this hits the real Groq API and costs a small number of
tokens per run.

## Configuration (.env)

| Variable          | Default                     | Notes                                          |
|-------------------|-----------------------------|------------------------------------------------|
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5`     | Must match `EMBEDDING_DIM`                      |
| `EMBEDDING_DIM`   | `768`                       | 768 for bge-base, 384 for bge-small            |
| `LLM_MODEL`       | `llama-3.1-8b-instant`      | Any Groq-supported model                       |
| `RETRIEVAL_K`     | `8`                         | Chunks passed to the LLM                        |
| `ALLOWED_ORIGINS` | `*`                         | Comma-separated CORS origins                    |
| `RATE_LIMIT_PER_MINUTE` | `20`                  | Max `/chat` requests per client IP per minute   |

> **Important:** `EMBEDDING_MODEL` and `EMBEDDING_DIM` must stay in sync. If you
> change the model, rebuild the index (`python -m scripts.load_data`). The app
> validates this on startup and refuses to load a mismatched index.

## Security

- **Never commit `.env`.** It is gitignored. Use `.env.example` as the template.
- If API keys were ever committed, **rotate them immediately.**
# FSA_Chatbot
