# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The dependency graph and override semantics (task 0.3).

What makes "go back and edit stage 1 without losing your stage 6 choices" correct by construction.
It is a data-model problem, not a UI problem, so it lives here and the UI merely renders the result.

Three pieces:

* ``graph.py`` -- the DAG, built from the schema's ``derived_from`` metadata. Pure graph work: no
  values, no derivations, no physics.
* ``registry.py`` -- which function in ``studio/science`` computes which derived field, with the
  binding checked against the schema rather than trusted.
* ``resolver.py`` -- resolution in topological order, the auto / user_override / stale states, and
  the two explicit ways to settle a stale field.

**Why a package of its own.** ``studio/schema`` is data and stays free of computation;
``studio/science`` is computation and stays free of the config model. Resolution is the composition
of the two, and giving it a name keeps that layering visible -- schema and science remain usable,
and testable, without it.

Like schema and science, this package must not import ``coupled``, the API or the database: the API
resolves a config on every keystroke, and a JAX import on that path would be unaffordable. Enforced
by ``studio/tests/unit/test_import_boundaries.py``.
"""

from __future__ import annotations

from studio.resolve.graph import CyclicDependencyError, DependencyGraph, schema_derived_fields
from studio.resolve.registry import DERIVATIONS, Derivation, derivation_for
from studio.resolve.resolver import (
    ChangedInput,
    InconsistentConfigError,
    OverrideRecord,
    ResolvedConfig,
    StaleField,
    accept_derived,
    apply_change,
    downstream_of,
    keep_override,
    resolve,
    set_override,
)

__all__ = [
    "DERIVATIONS",
    "ChangedInput",
    "CyclicDependencyError",
    "DependencyGraph",
    "Derivation",
    "InconsistentConfigError",
    "OverrideRecord",
    "ResolvedConfig",
    "StaleField",
    "accept_derived",
    "apply_change",
    "derivation_for",
    "downstream_of",
    "keep_override",
    "resolve",
    "schema_derived_fields",
    "set_override",
]
