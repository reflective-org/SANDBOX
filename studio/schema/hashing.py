# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Canonical serialisation and the config hash (ADR-006).

A run's identity is the SHA-256 of the canonical JSON of its ``RunConfig``. That hash is also the
cache key, the golden-fixture key and half of the idempotence check, so **a hash that drifts
silently invalidates everything at once** -- and does it quietly, which is worse. Hence a canonical
form pinned by explicit rules and a test that asserts a known hash rather than merely asserting
self-consistency.

Canonical form:

1. ``model_dump(mode="json")`` -- enums become their string values, tuples become lists.
2. ``sort_keys=True`` -- insertion order cannot leak into identity. This is what makes two configs
   built by different code paths hash the same.
3. ``separators=(",", ":")`` -- no incidental whitespace.
4. ``ensure_ascii=False`` with a UTF-8 encode -- one representation per string, not two.
5. ``allow_nan=False`` -- ``NaN``/``Infinity`` are not JSON and are not a valid configuration
   either. This RAISES rather than emitting a non-standard token (ADR-005).

Float formatting is Python's ``repr``, which has produced the shortest round-tripping decimal since
3.1 and is therefore stable across the versions this project supports. That is the one assumption
here that is a property of the interpreter rather than of this module, and the pinned-hash test is
what would catch it changing.

There is no hashing anywhere else in the repository -- this is the first -- so nothing constrains
the choice except the need for it to never change silently.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel

#: Named so a future change is a visible migration rather than an invisible one. If the canonical
#: form ever has to change, bump this, bump the schema version, and re-baseline the fixtures on
#: purpose -- never quietly.
CANONICAL_FORM_VERSION = 1


def canonical_payload(model: BaseModel) -> dict[str, Any]:
    """The JSON-mode dict that gets serialised.

    Exposed for tests, and for debugging a hash change: diffing two payloads says which field moved.
    """
    payload = model.model_dump(mode="json")
    if not isinstance(payload, dict):  # pragma: no cover -- pydantic models always dump to a dict
        raise TypeError(f"expected a dict from model_dump, got {type(payload).__name__}")
    return payload


def canonical_json(model: BaseModel) -> str:
    """Canonical JSON string for ``model``.

    Raises:
        ValueError: If the config contains NaN or Infinity, which are neither valid JSON nor a valid
            configuration. Failing here beats writing a token no other parser will read back.
    """
    try:
        return json.dumps(
            canonical_payload(model),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except ValueError as exc:
        raise ValueError(
            f"{type(model).__name__} is not canonically serialisable: {exc}. NaN and Infinity are "
            f"not valid JSON and not a valid configuration; fix the value rather than the encoder."
        ) from exc


def config_hash(model: BaseModel) -> str:
    """Stable SHA-256 (hex) over the canonical JSON of ``model``."""
    return hashlib.sha256(canonical_json(model).encode("utf-8")).hexdigest()


def short_hash(model: BaseModel, length: int = 12) -> str:
    """First ``length`` hex characters of the config hash, for display and directory names.

    Display only. Twelve hex characters is ~48 bits, fine for a human reading a list and not fine as
    an identity; equality checks use the full hash.
    """
    if not 4 <= length <= 64:
        raise ValueError(f"length must be in [4, 64], got {length}")
    return config_hash(model)[:length]


__all__ = [
    "CANONICAL_FORM_VERSION",
    "canonical_json",
    "canonical_payload",
    "config_hash",
    "short_hash",
]
