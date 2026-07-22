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
| Chat LLM | `gpt-oss-20b-Q4_K_M.gguf` from Hugging Face, via Koboldcpp locally or direct `llama-cpp-python` in Colab |
| Embeddings | `BAAI/bge-large-en-v1.5` via sentence-transformers (in-process) |
| Graph | python-igraph (`community_leiden`, no `leidenalg`) |
| Reused library | `../proprag_poc` (imported, never modified) — a vendored copy also ships at `proprag_poc/` in this repo for standalone use (e.g. Colab) |
| Dataset | Desktop: `../PropRAG-main/reproduce/dataset/2wikimultihopqa{,_corpus}.json`. Colab: same files downloaded from HF dataset `osunlp/HippoRAG_2` (cited in `PropRAG-main/reproduce/README.md`) |

Locally, keep this folder as a **sibling** of `proprag_poc` under the `PropRAG` directory —
`benchmark/_bootstrap.py` prefers that canonical copy over the vendored one whenever both
are present, so local dev always edits the real source.

## Setup

```bash
pip install -r requirements.txt
```

For Colab, open `PropRAG_Benchmark_Colab.ipynb` and run the cells. The notebook downloads `unsloth/gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf` with an optional `HF_API_TOKEN` / `HF_TOKEN`, then uses direct `llama-cpp-python` inference without a localhost API server or load-time quantization.

For local desktop runs, you can still start Koboldcpp separately:

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
- ~11.6 GB for the gpt-oss-20b Q4_K_M GGUF + ~1.3 GB for bge-large. Under RAM pressure, drop the
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


## Colab GGUF mode

`PropRAG_Benchmark_Colab.ipynb` is fully self-contained — no Drive mount, no manual
upload. It clones this repo (vendored `proprag_poc` included), then downloads everything
else from Hugging Face at run time: the dataset (`osunlp/HippoRAG_2`), the GGUF
(`unsloth/gpt-oss-20b-GGUF`), and the embedding model (`BAAI/bge-large-en-v1.5`, via
sentence-transformers as usual). It then sets:

```bash
PROPRAG_LLM_BACKEND=llama_cpp
PROPRAG_GGUF_MODEL_PATH=<hf_hub_download() result>
PROPRAG_DATASET_PATH=<hf_hub_download() result>
PROPRAG_CORPUS_PATH=<hf_hub_download() result>
PROPRAG_LLAMA_N_CTX=4096
```

`llama_cpp` is a benchmark-only backend value — `BenchmarkConfig.make_poc_config()`
constructs `POCConfig` with a valid preset (POC doesn't know "llama_cpp") and relabels it
after init; `BenchLLMClient` then runs `llama_cpp.Llama` in-process instead of using
`LLMClient`'s HTTP path, with its own SQLite-cache + `UsageTracker` wiring so token
accounting and resume-on-interrupt still work. `N_CTX=4096` and `N_GPU_LAYERS=-1`
(all layers offloaded) are chosen for a free Colab T4's 15 GB; lower `N_GPU_LAYERS` if you
hit a CUDA OOM, raise `N_CTX` if you have more headroom.

Colab's local disk is wiped when the runtime recycles — the notebook's last cell zips the
latest run's outputs and triggers a browser download.
