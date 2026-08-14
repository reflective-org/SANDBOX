# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Plume Studio -- configure, run and compare coupled SAI plume box-model simulations.

Studio orchestrates; it does not reimplement physics. The model is ``coupled/`` and its three
submodules (``tuvx-jax``, ``stratchem-jax``, ``tomas-jax``), consumed in-tree (docs/studio/adr/
ADR-001).

Package boundaries -- enforced by ``studio/tests/unit/test_import_boundaries.py``:

* ``studio.schema`` and ``studio.science`` must import NOTHING from the API, the database, the web
  layer, or ``coupled``. Importing ``coupled`` pulls in JAX and TOMAS (``coupled_scenario.py``
  validates ``background_dist`` against ``coupled.tomas_bridge``), which an API validating a form on
  every keystroke cannot afford. Both packages must be usable from a bare Python session.
* ``studio.modelio`` is the ONLY package permitted to import ``coupled``. It is the seam that would
  be severed if Studio were ever extracted to its own repository.

Nothing here fabricates a number. Unimplemented paths raise ``NotImplementedError``; missing inputs
raise rather than defaulting (ADR-005).
"""

from __future__ import annotations

#: Recorded in every run's provenance alongside the SANDBOX and submodule git SHAs (ADR-006).
#: The git SHAs pin the model; this pins the orchestration layer.
__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
