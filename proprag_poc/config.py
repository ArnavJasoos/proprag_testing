"""Single configuration object for the PropRAG-PDF POC.

All tunable parameters live here. Pass one ``POCConfig`` instance everywhere;
never hardcode parameters in the pipeline modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Optional


# Backend presets: name -> (base_url, default model, needs_real_key).
# "google" uses the native google-genai SDK (Gemini); all others are OpenAI-compatible.
# "nvidia" is the free NVIDIA-hosted API (integrate.api.nvidia.com), OpenAI-compatible.
_LLM_PRESETS = {
    "nvidia": ("https://integrate.api.nvidia.com/v1", "meta/llama-3.3-70b-instruct", True),
    "google": (None, "gemini-2.5-flash", True),
    "koboldcpp": ("http://localhost:5001/v1", "gpt-oss-20b-Q4_K_M.gguf", False),
    "ollama": ("http://localhost:11434/v1", "qwen2.5", False),
    "vllm": ("http://localhost:8000/v1", "Qwen/Qwen2.5-7B-Instruct", False),
    "openrouter": ("https://openrouter.ai/api/v1", "meta-llama/llama-3.3-70b-instruct", True),
}

# Backends that hit a remote, rate-limited provider (must go through the limiter
# + usage tracker). Local servers and the on-device embedding model are exempt.
ONLINE_LLM_BACKENDS = {"nvidia", "google", "openrouter"}
ONLINE_EMBEDDING_BACKENDS = {"nvidia", "openai"}

# NVIDIA NeMo Retriever (nv-embedqa) embedding endpoints require an ``input_type``
# discriminator ("query" vs "passage"); plain OpenAI embeddings do not.
_NEEDS_INPUT_TYPE_PREFIXES = ("nvidia/", "nv-")


def _load_dotenv():
    """Minimal .env loader (no dependency): populate os.environ from ./.env.

    Existing environment variables win; only fills in unset keys.
    """
    for path in (".env", os.path.join("proprag_poc", ".env")):
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v


@dataclass
class POCConfig:
    """One and only configuration for the POC."""

    # ----- Paths -----
    data_dir: str = field(default="data")

    # ----- LLM endpoint -----
    # Default backend is the free NVIDIA-hosted API (OpenAI-compatible).
    llm_backend: Literal["nvidia", "google", "koboldcpp", "ollama", "vllm", "openrouter"] = "nvidia"
    llm_base_url: Optional[str] = None          # overrides preset if set
    llm_model: Optional[str] = None             # overrides preset if set
    llm_api_key: Optional[str] = None           # read from env / file if None
    temperature: float = 0.2                    # low temp for parseable JSON extraction
    max_completion_tokens: int = 4096
    seed: Optional[int] = None
    request_timeout: int = 600
    max_retry_attempts: int = 5
    # Concurrency for LLM calls. NVIDIA's free tier serializes requests server-side
    # anyway; concurrent workers just queue on their backend and inflate per-call
    # latency. Single worker keeps calls sequential, paced by rpm_limit.
    llm_max_workers: int = 1
    # gpt-oss emits a reasoning/analysis channel; strip it before JSON parsing.
    strip_reasoning: bool = True
    use_json_response_format: bool = True

    # ----- Rate limiting (chat LLM only, by default) -----
    # Free providers ban on overuse. One global sliding-60s-window limiter caps the
    # chat request rate. Only online backends are limited; cache hits are free.
    # NVIDIA's free tier is ~40 RPM (account-level, per model); 38 stays just under.
    # Embeddings run on the local ``sentence_transformers`` backend by default, so
    # they never touch this limiter (only the ``nvidia``/``openai`` embedding
    # backends would share it).
    rpm_limit: int = 38

    # ----- Embeddings (decoupled from the chat LLM) -----
    # Default: a local sentence-transformers model (no API, no rate limit, no
    # token cost) so every RAG system embeds with one model for a fair compare.
    # ``nvidia`` (NIM-hosted nv-embedqa) is kept as an online alternative.
    embedding_backend: Literal["nvidia", "sentence_transformers", "ollama", "openai"] = "sentence_transformers"
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_base_url: Optional[str] = None    # for nvidia/ollama/openai backends
    embedding_api_key: Optional[str] = None
    embedding_batch_size: int = 32
    embedding_normalize: bool = True
    embedding_query_instruction: str = (
        "Represent this sentence for searching relevant passages: "
    )

    # ----- PDF ingestion / chunking -----
    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 96
    tiktoken_encoding: str = "cl100k_base"
    drop_page_numbers: bool = True
    drop_repeated_headers: bool = True
    # A line repeated on >= this fraction of pages is treated as header/footer noise.
    header_repeat_fraction: float = 0.5

    # ----- Graph construction -----
    synonymy_sim_threshold: float = 0.8
    synonymy_top_k: int = 100
    is_directed_graph: bool = False

    # ----- Beam search -----
    beam_width: int = 4
    max_path_length: int = 3
    second_stage_filter_k: int = 40
    # "concatenate" re-embeds joined path text at query time -> fresh API calls per
    # path (slow + costly under an online, rate-limited embedder). The pooling modes
    # combine *precomputed* proposition embeddings with zero extra API calls, so they
    # are the sane default for an online backend. Keep concatenate only for local.
    embedding_combination: Literal[
        "concatenate", "average", "weighted_average", "max_pool", "attention"
    ] = "weighted_average"
    initial_proposition_seeds: int = 200

    # ----- Retrieval / PPR -----
    passage_node_weight: float = 0.05
    ppr_damping_stage1: float = 0.75
    ppr_damping_stage2: float = 0.45
    focus_top_k: int = 50
    select_top_k_paths: int = 20
    select_top_k_entities: int = 40
    retrieval_top_k: int = 10

    # ----- QA -----
    qa_top_k: int = 5
    history_max_turns: int = 6

    # ----- Indexing control -----
    force_index_from_scratch: bool = False
    force_extraction_from_scratch: bool = False

    def __post_init__(self):
        _load_dotenv()
        self._apply_env_overrides()

        base_url, model, needs_key = _LLM_PRESETS[self.llm_backend]
        if self.llm_base_url is None:
            self.llm_base_url = base_url
        if self.llm_model is None:
            self.llm_model = model
        if self.llm_api_key is None:
            self.llm_api_key = self._resolve_llm_key(needs_key)

        # Resolve the embedding endpoint for OpenAI-compatible / NVIDIA backends.
        if self.embedding_backend in ONLINE_EMBEDDING_BACKENDS:
            if self.embedding_base_url is None and self.embedding_backend == "nvidia":
                self.embedding_base_url = _LLM_PRESETS["nvidia"][0]
            if self.embedding_api_key is None:
                self.embedding_api_key = self._resolve_embedding_key()

        os.makedirs(self.data_dir, exist_ok=True)

    # -------------------------------------------------------------- env knobs
    def _apply_env_overrides(self):
        """Let the documented ``PROPRAG_*`` env vars (from .env) drive config.

        These were declared in ``.env.example`` but previously never read; wiring
        them here makes the .env file actually control the backend/model/rate limit.
        Env wins for these specific knobs (no caller in this repo sets them in code).
        """
        backend = os.environ.get("PROPRAG_LLM_BACKEND")
        if backend in _LLM_PRESETS:
            self.llm_backend = backend  # type: ignore[assignment]
        if os.environ.get("PROPRAG_LLM_MODEL"):
            self.llm_model = os.environ["PROPRAG_LLM_MODEL"]
        emb_backend = os.environ.get("PROPRAG_EMBEDDING_BACKEND")
        if emb_backend in ("nvidia", "sentence_transformers", "ollama", "openai"):
            self.embedding_backend = emb_backend  # type: ignore[assignment]
        if os.environ.get("PROPRAG_EMBEDDING_MODEL"):
            self.embedding_model = os.environ["PROPRAG_EMBEDDING_MODEL"]
        rpm = os.environ.get("PROPRAG_RPM_LIMIT")
        if rpm and rpm.strip().isdigit():
            self.rpm_limit = int(rpm.strip())

    # -------------------------------------------------------------- helpers
    @property
    def llm_is_online(self) -> bool:
        return self.llm_backend in ONLINE_LLM_BACKENDS

    @property
    def embedding_is_online(self) -> bool:
        return self.embedding_backend in ONLINE_EMBEDDING_BACKENDS

    @property
    def embedding_needs_input_type(self) -> bool:
        """nv-embedqa models require a query/passage ``input_type`` discriminator."""
        return self.embedding_backend == "nvidia" and self.embedding_model.startswith(
            _NEEDS_INPUT_TYPE_PREFIXES
        )

    def _resolve_llm_key(self, needs_key: bool) -> str:
        # Env var wins. NVIDIA -> NVIDIA_API_KEY; Gemini -> GEMINI/GOOGLE; else
        # PROPRAG_LLM_API_KEY / OPENAI_API_KEY. Then openrouter_api_key.txt; else dummy.
        env_names = []
        if self.llm_backend == "nvidia":
            env_names = ["NVIDIA_API_KEY", "PROPRAG_LLM_API_KEY"]
        elif self.llm_backend == "google":
            env_names = ["GEMINI_API_KEY", "GOOGLE_API_KEY", "PROPRAG_LLM_API_KEY"]
        else:
            env_names = ["PROPRAG_LLM_API_KEY", "OPENAI_API_KEY"]
        for name in env_names:
            key = os.environ.get(name)
            if key:
                return key
        if os.path.isfile("openrouter_api_key.txt"):
            with open("openrouter_api_key.txt", "r", encoding="utf-8") as f:
                txt = f.read().strip()
                if txt:
                    return txt
        if needs_key:
            hint = {
                "nvidia": "NVIDIA_API_KEY",
                "google": "GEMINI_API_KEY",
            }.get(self.llm_backend, "PROPRAG_LLM_API_KEY")
            raise ValueError(
                f"Backend '{self.llm_backend}' needs an API key. Set {hint} "
                "(e.g. in the .env file at the repo root)."
            )
        return "not-needed"

    def _resolve_embedding_key(self) -> str:
        # For the NVIDIA embedding backend reuse the NVIDIA / LLM key by default.
        for name in ("NVIDIA_API_KEY", "PROPRAG_EMBEDDING_API_KEY", "OPENAI_API_KEY"):
            key = os.environ.get(name)
            if key:
                return key
        if self.embedding_backend == "nvidia" and self.llm_backend == "nvidia" and self.llm_api_key:
            return self.llm_api_key
        if self.embedding_backend == "nvidia":
            raise ValueError(
                "Embedding backend 'nvidia' needs an API key. Set NVIDIA_API_KEY "
                "in the .env file at the repo root."
            )
        return "not-needed"
