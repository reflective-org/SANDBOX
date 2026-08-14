# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Run configuration schema -- the single source of truth (ADR-002).

Defined once here in Pydantic v2, exported as JSON Schema, consumed by the web client for form
generation and validation. Nothing in the UI may invent a field that does not exist here.

Contents:

* ``SciField`` (``fields.py``) -- the field-metadata carrier: canonical unit, valid range, label,
  description, default, PROVENANCE of that default, citation, ``derived_from``, and any caveat that
  must travel with the value. Inconsistent metadata raises at import time.
* ``RunConfig`` (``config.py``) -- one validated simulation description; versioned, canonically
  serialisable, hashable. Minimum viable for the Phase-0 slice: what the golden case
  ``30N_20km__sabr220__D2med__a1p0__nuc1__cg1`` needs, plus the provenance and derivation inputs the
  model has no field for.
* ``RunSet`` + axes (``runset.py``) -- the primary user-facing object. A single run is a RunSet with
  zero axes; there is no separate N = 1 code path.
* Canonical JSON and stable SHA-256 (``hashing.py``) -- run identity and cache key (ADR-006).
* JSON Schema export and the flat field catalogue (``export.py``).

Canonical units are the MODEL's native units (mbar, ppmv, pptv, K, s, um^2/cm^3), not SI -- see
ADR-003 and ASSUMPTION-1. ``pint`` is used for display conversion at the presentation boundary and
for property tests, never inside the model interface.

**No physics happens here.** Derived fields declare what they are computed from and stay unset; the
dependency-graph engine is task 0.3 and the derivations are task 0.5.

This package must not import ``coupled``, the API, or the database (see ``studio/__init__.py``).
"""

from __future__ import annotations

from studio.schema.config import (
    PAPER_BACKGROUND_GAS_PPTV,
    SCHEMA_VERSION,
    Background,
    Chemistry,
    Dilution,
    Injection,
    Microphysics,
    Numerics,
    ProcessSwitches,
    RunConfig,
    Schedule,
    SchemaModel,
    Site,
    Termination,
)
from studio.schema.enums import AxisKind, BackgroundAerosol, DilutionRegime, PhotolysisMode
from studio.schema.export import (
    SCHEMA_ID,
    field_catalogue,
    iter_leaf_fields,
    run_config_json_schema,
)
from studio.schema.fields import EXTENSION_KEY, Provenance, SciField, field_metadata
from studio.schema.hashing import (
    CANONICAL_FORM_VERSION,
    canonical_json,
    canonical_payload,
    config_hash,
    short_hash,
)
from studio.schema.runset import (
    Axis,
    AxisPoint,
    ExpandedRun,
    RunSet,
    apply_assignments,
    resolve_path,
)
from studio.schema.units import PINT_EXPRESSION, Unit

__all__ = [
    "CANONICAL_FORM_VERSION",
    "EXTENSION_KEY",
    "PAPER_BACKGROUND_GAS_PPTV",
    "PINT_EXPRESSION",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "Axis",
    "AxisKind",
    "AxisPoint",
    "Background",
    "BackgroundAerosol",
    "Chemistry",
    "Dilution",
    "DilutionRegime",
    "ExpandedRun",
    "Injection",
    "Microphysics",
    "Numerics",
    "PhotolysisMode",
    "ProcessSwitches",
    "Provenance",
    "RunConfig",
    "RunSet",
    "Schedule",
    "SchemaModel",
    "SciField",
    "Site",
    "Termination",
    "Unit",
    "apply_assignments",
    "canonical_json",
    "canonical_payload",
    "config_hash",
    "field_catalogue",
    "field_metadata",
    "iter_leaf_fields",
    "resolve_path",
    "run_config_json_schema",
    "short_hash",
]
