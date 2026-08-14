# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Run configuration schema -- the single source of truth (ADR-002).

Defined once here in Pydantic v2, exported as JSON Schema, consumed by the web client for form
generation and validation. Nothing in the UI may invent a field that does not exist here.

Contents (task 0.2, not yet implemented):

* ``SciField`` -- the field-metadata carrier: canonical unit, valid range or enum, label,
  description, default, PROVENANCE of that default, citation, and ``derived_from``.
* ``RunConfig`` -- one validated simulation description; versioned, canonically serialisable,
  hashable.
* ``RunSet`` + axes (GRID / ZIP / LIST) -- the primary user-facing object. A single run is a RunSet
  with zero axes; there is no separate N=1 code path.
* Canonical JSON serialisation and stable SHA-256 hashing (ADR-006).

Canonical units are the MODEL's native units (mbar, ppm, pptv, K, s, um^2/cm^3), not SI -- see
ADR-003 and ASSUMPTION-1. ``pint`` is used for display conversion at the presentation boundary and
for property tests, never inside the model interface.

This package must not import ``coupled``, the API, or the database (see ``studio/__init__.py``).
"""

from __future__ import annotations

__all__: list[str] = []
