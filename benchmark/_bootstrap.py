"""Put the ``proprag_poc`` package on ``sys.path``.

``proprag_poc`` is not pip-installable and uses package-relative imports, so it
imports cleanly only when its PARENT directory is importable. This file lives at
``PropRAG/PropRAG Testing/benchmark/_bootstrap.py``; ``parents[2]`` is the
``PropRAG`` directory that holds ``proprag_poc``. Import this module first from
every entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROPRAG_PARENT = Path(__file__).resolve().parents[2]

if str(_PROPRAG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PROPRAG_PARENT))

# Fail loudly and early if the layout assumption is wrong.
if not (_PROPRAG_PARENT / "proprag_poc" / "config.py").is_file():
    raise ImportError(
        f"proprag_poc not found next to the project. Expected it at "
        f"{_PROPRAG_PARENT / 'proprag_poc'}. Keep 'PropRAG Testing' as a sibling "
        f"of 'proprag_poc' under the PropRAG directory."
    )
