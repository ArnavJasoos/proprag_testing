# PropRAG vs GraphRAG vs BaseRAG — 2WikiMultiHopQA Benchmark

Head-to-head comparison of three retrieval-augmented QA systems on a certified
multi-hop benchmark, measuring **answer quality** (EM / F1), **retrieval quality**
(Recall@k) and **token consumption** (indexing + query), fully local and CPU-only.

- **BaseRAG** — dense passage retrieval over shared chunk embeddings.
- **PropRAG** — proposition beam-search + PPR over a knowledge graph (reused from
  `proprag_poc`).
- **GraphRAG** — a faithful Microsoft-style GraphRAG built here (own entity/relationship
  extraction, Leiden communities, community reports, local search), so its indexing
  cost is genuinely measured.

All three share one LLM, one embedding model, and one chunk embedding store — the
only measured differences are each system's index and its retrieval strategy.

## Stack

| Component | Choice |
|---|---|
| Chat LLM | gpt-oss-20b Q4_K_M via **Koboldcpp** at `http://localhost:5001/v1` |
| Embeddings | `BAAI/bge-large-en-v1.5` via sentence-transformers (in-process) |
| Graph | python-igraph (`community_leiden`, no `leidenalg`) |
| Reused library | `../proprag_poc` (imported, never modified) |
| Dataset | `../PropRAG-main/reproduce/dataset/2wikimultihopqa{,_corpus}.json` |

The project imports `proprag_poc` as a library; keep this folder as a **sibling** of
`proprag_poc` under the `PropRAG` directory.

## Setup

```bash
pip install -r requirements.txt
```

Start Koboldcpp (separate terminal), serving gpt-oss-20b:

```bash
koboldcpp.exe --model gpt-oss-20b-Q4_K_M.gguf --port 5001 --contextsize 8192
```

## Verification ladder

Run from this project root:

```bash
# 1. Offline sanity — no LLM needed
python -m benchmark.dataset        # subset/corpus stats, gold-title coverage
python -m benchmark.evaluation     # EM/F1/Recall@k hand cases

# 2. Full pipeline in a handful of LLM calls (Koboldcpp must be up)
python -m benchmark.smoke

# 3. Pilot — measures s/call and projects full-run time
python -m benchmark.run --pilot 10

# 4. Full 50-question run (resumable)
python -m benchmark.run
```

## CLI

```
python -m benchmark.run [--questions 50] [--pilot 10] [--seed 42]
                        [--systems BaseRAG,GraphRAG,PropRAG]
                        [--force-reindex] [--report-only] [--no-charts]
```

## Outputs

Everything lands under `data/`:

- `data/corpora/<corpus_id>/` — shared chunk store, PropRAG graph/maps, `graphrag/`
  artifacts (all reused across runs).
- `data/llm_cache.sqlite`, `data/embedding_cache.sqlite` — every call cached, so
  interruptions cost nothing on resume.
- `data/benchmark/<run_id>/` — `manifest.json`, `index_usage.json`, `results.jsonl`
  (one line per question×system, resume-safe), `metrics.json`, **`report.md`**, and
  `charts/` (if matplotlib is installed).

## Runtime & RAM notes

- ~1,400–1,700 chat calls on a CPU-served 20B → potentially long; the pilot projects
  it. Every call is cached; reruns resume at the SQLite / stage-checkpoint /
  `results.jsonl` levels.
- ~11.5 GB for the 20B Q4 model + ~1.3 GB for bge-large. Under RAM pressure, drop the
  embedder without touching code:

  ```bash
  PROPRAG_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5 python -m benchmark.run --pilot 10
  ```

## Fairness

The report footer records the shared model/embedder, `qa_top_k`, seed, and the
GraphRAG gleanings setting (`gr_max_gleanings=0` by default; Microsoft's default is 1,
which roughly doubles GraphRAG index cost). Recall@k is scored on retrieved documents
for all systems; GraphRAG's QA additionally uses community reports + relationships via
its local-search context assembly, which the report states explicitly.
