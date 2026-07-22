"""Put the proprag_poc package on sys.path.

Locally, proprag_poc is expected next to this project under the PropRAG folder
(the canonical, actively-edited copy). A vendored copy also ships inside this
repo at ``proprag_poc/`` (sibling of ``benchmark/``) so a fresh clone - e.g. in
Colab, with no sibling checkout available - works standalone. The sibling copy
takes priority when both exist, so local dev always uses the canonical source.

PROPRAG_MAIN can still override the search entirely if needed.
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
    _here.parents[2],   # sibling layout: PropRAG/{proprag_poc, PropRAG Testing}
    _here.parents[1],   # vendored copy: PropRAG Testing/proprag_poc
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
