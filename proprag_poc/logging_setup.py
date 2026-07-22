"""One-call logging setup so progress is visible in the terminal.

All package modules log under the ``proprag_poc`` logger tree, so configuring that
parent once routes NER/proposition extraction, embedding batches, LLM calls,
rate-limit waits and the compare run to stdout. Streamlit forwards stdout to the
terminal it was launched from.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def setup_logging(level: int | None = None) -> None:
    """Attach a stdout handler to the ``proprag_poc`` logger (idempotent).

    Level comes from the ``PROPRAG_LOG_LEVEL`` env var (default INFO).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    if level is None:
        level = getattr(logging, os.environ.get("PROPRAG_LOG_LEVEL", "INFO").upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-5s | %(name)s | %(message)s", "%H:%M:%S")
    )
    root = logging.getLogger("proprag_poc")
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True
