"""LLM client wrapper for the gpt-oss-20b Koboldcpp backend.

Two things the reused ``LLMClient`` does not do for us:

1. ``strip_reasoning`` is a declared but unimplemented ``POCConfig`` flag. gpt-oss
   emits a Harmony "analysis" channel that, depending on the Koboldcpp build, can
   leak into ``message.content`` and break JSON parsing / ``Answer:`` extraction.
   ``BenchLLMClient`` strips it AFTER ``super().infer()`` so the SQLite cache keeps
   the raw response (still reusable) while callers get clean text.

2. A fail-fast health check so a run aborts immediately with a launch hint when
   Koboldcpp is not up, instead of after a 10-minute request timeout.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request

from . import _bootstrap  # noqa: F401
from proprag_poc.config import POCConfig
from proprag_poc.llm.client import LLMClient

logger = logging.getLogger(__name__)

_FINAL_MARKER = "<|channel|>final<|message|>"
_TERMINATORS = ("<|end|>", "<|return|>", "<|start|>")


def strip_gpt_oss_reasoning(content: str) -> str:
    """Remove the gpt-oss Harmony reasoning channel, keep the final answer text."""
    if not content:
        return content
    if _FINAL_MARKER in content:
        content = content.split(_FINAL_MARKER)[-1]
        for term in _TERMINATORS:
            content = content.split(term)[0]
        return content.strip()
    stripped = content.lstrip()
    if stripped.startswith("analysis") and "assistantfinal" in stripped:
        return stripped.split("assistantfinal")[-1].strip()
    return content


class BenchLLMClient(LLMClient):
    """``LLMClient`` that post-processes gpt-oss reasoning out of every response."""

    def infer(self, *args, **kwargs):
        content, meta, cache_hit = super().infer(*args, **kwargs)
        if self.config.strip_reasoning:
            content = strip_gpt_oss_reasoning(content)
        return content, meta, cache_hit


def check_backend(poc_cfg: POCConfig, timeout: float = 5.0) -> None:
    """Ping ``{base_url}/models``; raise a launch hint on failure."""
    base = (poc_cfg.llm_base_url or "").rstrip("/")
    url = f"{base}/models"
    hint = (
        "Cannot reach the LLM backend at "
        f"{poc_cfg.llm_base_url!r}. Start Koboldcpp first:\n"
        "  koboldcpp.exe --model gpt-oss-20b-Q4_K_M.gguf --port 5001 --contextsize 8192"
    )
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"{url} returned HTTP {resp.status}")
    except (urllib.error.URLError, OSError, RuntimeError) as e:
        raise RuntimeError(f"{hint}\n(underlying error: {e})") from e
    logger.info("LLM backend reachable at %s", poc_cfg.llm_base_url)
