"""Put the proprag_poc package on sys.path.

Locally, proprag_poc is expected next to this project under the PropRAG folder.
In Colab, the notebook sets PROPRAG_MAIN to the cloned PropRAG repository and
this module imports proprag_poc from there.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_candidates = []
_env_main = os.environ.get("PROPRAG_MAIN")
if _env_main:
    _candidates.append(Path(_env_main).resolve())

_here = Path(__file__).resolve()
_candidates.extend([
    _here.parents[2],
    Path("/content/PropRAG_main"),
    Path("/content/PropRAG"),
])

_PROPRAG_PARENT = None
for _cand in _candidates:
    if (_cand / "proprag_poc" / "config.py").is_file():
        _PROPRAG_PARENT = _cand
        break

if _PROPRAG_PARENT is None:
    expected = ", ".join(str(p / "proprag_poc") for p in _candidates)
    raise ImportError(
        "proprag_poc not found. Set PROPRAG_MAIN to the cloned PropRAG "
        f"repository. Checked: {expected}"
    )

if str(_PROPRAG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PROPRAG_PARENT))
