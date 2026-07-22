# PropRAG-PDF POC

Proposition knowledge-graph RAG over arbitrary PDFs, with a ChatGPT-style GUI and
graph visualization. Reproduces the core PropRAG methodology — proposition
decomposition + LLM-free beam search over a proposition/entity graph +
Personalized PageRank retrieval — adapted to user-supplied PDFs.

## Pipeline

```
PDF → clean text (PyMuPDF, noise filtered) → token chunks
    → NER + proposition extraction (LLM) → embed entities/props/chunks
    → igraph (entity↔entity co-occurrence, entity↔passage, entity↔entity synonymy)
Query → contextualize (if follow-up) → beam search over proposition paths
    → path entities seed PPR#1 (d=0.75) → focused subgraph → beam#2 → PPR#2 (d=0.45)
    → top-k passages → LLM answer (history-aware)
```

## Setup

```bash
pip install -r proprag_poc/requirements.txt
```

### LLM endpoint (default: NVIDIA free API)

The default backend is the free **NVIDIA-hosted API** (`integrate.api.nvidia.com`,
OpenAI-compatible). Get a key at <https://build.nvidia.com>, then copy
`.env.example` to `.env` at the repo root and set it:

```ini
NVIDIA_API_KEY=nvapi-...
PROPRAG_LLM_BACKEND=nvidia
PROPRAG_LLM_MODEL=meta/llama-3.3-70b-instruct
PROPRAG_RPM_LIMIT=38
```

`.env` is loaded automatically by `POCConfig` (env vars win).

**Rate limiting.** By default this only guards the chat LLM — one process-wide
limiter caps it at `PROPRAG_RPM_LIMIT` requests per 60s to stay under NVIDIA's
free-tier allowance (~40 RPM, account-level). Cache hits and local backends are
exempt. Embeddings run locally (see below) and never touch this limiter unless
switched to an online embedding backend, in which case they share it with chat.

Other backends are still supported via env (or `POCConfig`):

```bash
export PROPRAG_LLM_BACKEND=openrouter PROPRAG_LLM_API_KEY=sk-...   # OpenRouter
export PROPRAG_LLM_BACKEND=ollama PROPRAG_LLM_MODEL=qwen2.5        # local Ollama
export PROPRAG_LLM_BACKEND=google PROPRAG_LLM_MODEL=gemini-2.5-flash  # Gemini (genai SDK)
```

### Embeddings

By default embeddings run **locally** via `sentence-transformers`
(`BAAI/bge-large-en-v1.5`), so every RAG system embeds with one model (a fair
comparison), with no API key, no rate limit, and no per-token cost. Uses GPU
automatically if `torch` has CUDA available, else CPU.

```ini
PROPRAG_EMBEDDING_BACKEND=sentence_transformers
PROPRAG_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
```

To embed via the NVIDIA API instead (shares the chat rate limiter):

```ini
PROPRAG_EMBEDDING_BACKEND=nvidia
PROPRAG_EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5
```

`nv-embedqa` models take a query/passage `input_type` automatically when this
backend is selected.

## Run the GUI

```bash
streamlit run proprag_poc/app.py
```

1. **Library** — name a corpus, upload PDFs, **Build index**.
2. **Chat** — New chat in the sidebar, pick the corpus, ask multi-turn questions
   (follow-ups keep context).
3. **Compare** — run one query through **BaseRAG** (dense), **GraphRAG**
   (entity-graph), and **PropRAG** (proposition beam + PPR) side-by-side, with a
   metrics table + charts: per-system token consumption (chat + embedding),
   LLM/embedding call counts, retrieval and QA latency, and rate-limit wait.
4. **Graph** — search an entity, view its neighborhood.

## Headless smoke test

```bash
python -m proprag_poc.scripts.smoke_test
```

(Indexes a tiny built-in corpus and runs one query — needs the LLM endpoint up.)

## Notes / limitations

- Digital-text PDFs only (no OCR for scanned pages).
- Co-occurrence edge weights are integers, synonymy weights are floats < 1 — beam
  search relies on this distinction; do not change it.
- Concurrency (`llm_max_workers`) defaults low for a single local endpoint; raise it
  for hosted APIs.
