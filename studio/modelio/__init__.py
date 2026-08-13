# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The model seam: RunConfig <-> CoupledScenario, and state.npz -> RunSummary.

THE ONLY PACKAGE PERMITTED TO IMPORT ``coupled`` (ADR-001). If Studio is ever extracted to its own
repository, this is the seam to sever.

Contents (task 0.4, not yet implemented):

* ``to_scenario(RunConfig) -> CoupledScenario`` -- the single conversion point, and therefore the
  single place unit conversion happens (ADR-003).
* ``RunSummary`` -- the versioned, queryable reduction of a run: scalar time series, final size
  distribution, integrated diagnostics, termination reason, flags, conservation residuals.
  Comparison plots and figures read this; they never read the raw npz (ADR-004).

The equivalence test is the point of this package: ``to_scenario`` applied to a ``RunConfig``
describing case ``30N_20km__sabr220__D2med__a1p0__nuc1__cg1`` must produce a ``CoupledScenario``
FIELD-FOR-FIELD IDENTICAL to ``run_ensemble.build_scenario()`` for that case. That proves the schema
is a faithful superset before anything is run.

Traps this package exists to contain (see docs/studio/CAVEATS.md), each an assertion here:

* ``state.npz`` carries BOTH dry and wet quantities: ``dp_mid_um`` / ``dNdlogDp`` are dry;
  ``SA`` / ``radius_cm`` are wet. Every RunSummary array declares its basis.
* Never reconstruct the time axis as ``i * DT``. Outer intervals snap to the terminator, so the mean
  step is ~592 s against a nominal 600 s -- ~1.4% drift, about 0.5 days by day 36. Use stored ``t``.
* Any resampling is UNIFORM in time. A coarsening grid aliases morning particle-number spikes by up
  to 8x.
* Index gas species by NAME using the npz's own ``species`` list, never by position. (An existing
  analysis script hardcodes ``_SO2, _SO3, _H2SO4 = 32, 34, 35``; that is the failure to avoid.)
"""

from __future__ import annotations

__all__: list[str] = []
