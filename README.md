# Full Stack Academy RAG Chatbot

A FastAPI RAG (Retrieval-Augmented Generation) service that answers questions
about Full Stack Academy's courses, locations, and admissions. It retrieves
context from a hybrid index (FAISS dense vectors + BM25 lexical search) built
from the website, course PDFs, and a manual curriculum file, then generates
grounded answers using Groq (Llama 3.1) — refusing to answer rather than
guessing when the context doesn't support it.

## Highlights

- **A real evaluation harness, not just eyeballed outputs.** `evaluation/`
  runs a golden question set through the live pipeline and reports
  retrieval hit-rate@k, MRR, course-detection accuracy, refusal accuracy,
  and LLM-as-judge groundedness/relevance — with each run archived for
  before/after comparison.
- **The harness found and drove real fixes**, not just cosmetic ones:
  - A crawler bug that followed "Download Curriculum" PDF links as if they
    were HTML pages, corrupting **89.7% of the indexed corpus** with raw PDF
    byte garbage. Traced from a low eval score down to the actual scraped
    bytes, fixed at the source (content-type + extension filtering), and
    the index was rebuilt clean (8,530 → 428 real chunks).
  - A `robots.txt` fetch bug where Python's `urllib` default User-Agent got
    403'd by the site's WAF, and `robotparser` silently treats a 401/403 as
    "disallow everything" — which would have quietly broken every future
    reingest.
  - A course-name regex that hallucinated a "course" out of any question
    containing "for/in/about" (e.g. "fees **for** working professionals" →
    treated as a course called "Working Professionals"), polluting
    retrieval for legitimate non-course questions.
- **Grounded refusal over hallucination.** The system prompt and retrieval
  are both tuned so the bot says "I couldn't find that" rather than
  inventing curriculum details — verified by the eval harness's
  groundedness metric, not just assumed.
- **Session-based conversation memory** so short follow-ups ("what about
  the fees?" right after a course question) stay scoped to the right
  course instead of being treated as a fresh, context-free query.
- **Dialogue-act handling with reply variety** — greetings, acknowledgments,
  thanks, and farewells are answered directly without hitting retrieval or
  burning a Groq call, and each has multiple phrasings so the bot doesn't
  repeat the identical sentence turn after turn.

## Architecture

```
Website + PDFs + manual chunks
        │  (scripts/load_data.py)
        ▼
  clean → chunk → dedupe boilerplate → embed  ──►  FAISS index + documents.json
        │
User ──► /chat ──► ask_rag ──┬─► dialogue-act short-circuit (greeting/thanks/etc.)
                              └─► hybrid search (FAISS + BM25 + intent boost)
                                       │
                                  Groq LLM (grounded, refuses if unsupported)
                                       │
                                    reply + session_id
```

Session state (last discussed course, recent turns) and the `/chat` rate
limiter are both in-memory and per-process — sufficient for a single-instance
deployment; see [Known limitations](#known-limitations) below.

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

The response includes a `session_id` — pass it back on subsequent requests
in the same conversation so follow-up questions stay in context:

```bash
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{"message": "what about the fees?", "session_id": "<id from previous reply>"}'
```

A minimal, self-contained chat widget for manually trying the API in a
browser is at `demo/chatbot_widget.html` — open it directly (no build step)
while the API is running locally.

## Tests

```bash
pytest
```
69 tests. Everything mocks the embedding model and Groq, so the suite runs
fast and needs no network.

## Evaluation

```bash
python -m evaluation.run_eval
```

Runs a golden set of questions (`evaluation/golden_dataset.py`) through the
**real** pipeline (real embeddings, real Groq generation, real Groq-as-judge
scoring) and reports:

- **Retrieval quality** — hit-rate@k and MRR, using keyword containment as a
  relevance proxy (no hand-labeled chunk IDs).
- **Course-detection accuracy** — does `extract_course_name` pick the right course.
- **Refusal accuracy** — does the bot correctly decline to answer off-topic or
  uncovered questions instead of fabricating curriculum details.
- **Answer quality** — an LLM-as-judge (Groq) scores each answer for
  groundedness (no unsupported claims) and relevance.

Each run is saved to `evaluation/results/eval_<timestamp>.json` (gitignored —
regenerate locally) for tracking quality over time as the ingestion pipeline
or prompts change. Unlike the mocked unit tests, this hits the real Groq API
and costs a small number of tokens per run.

**Latest measured run** (after the ingestion fixes above; 15 golden questions):

| Metric | Value |
|---|---|
| Course detection accuracy | 1.0 |
| Retrieval hit-rate@k | 1.0 |
| MRR | 0.80 |
| Refusal accuracy | 1.0 |
| Groundedness rate | 0.93 |
| Relevance rate | 0.87 |
| Overall pass rate | 1.0 (15/15) |

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

## Known limitations

- **Single-instance state.** The rate limiter, session store, and FAISS index
  all live in-process memory. Fine for one instance; scaling to multiple
  workers/replicas would need a shared store (Redis, etc.) for the rate
  limiter and sessions.
- **No persistent conversation history.** Sessions are cleared on restart —
  there's no database backing them.
- **Corpus freshness.** The index only reflects the site/PDFs as of the last
  `python -m scripts.load_data` run; there's no scheduled reingest.
- **No deployed instance yet.** Currently local-only; see the project roadmap
  for containerization and hosting.

## Security

- **Never commit `.env`.** It is gitignored. Use `.env.example` as the template.
- If API keys were ever committed, **rotate them immediately.**
