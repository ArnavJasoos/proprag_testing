"""Stable content-hash node identifiers (ported from reference ``compute_mdhash_id``)."""

from __future__ import annotations

import hashlib


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Return ``{prefix}{md5(content)}``.

    Used for every graph node id (entity-, chunk-, proposition-). Never use raw
    strings as node keys.
    """
    return prefix + hashlib.md5(content.encode("utf-8")).hexdigest()
