# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The model seam: RunConfig <-> CoupledScenario, and state.npz -> RunSummary.

THE ONLY PACKAGE PERMITTED TO IMPORT ``coupled`` (ADR-001). If Studio is ever extracted to its own
repository, this is the seam to sever.

* ``scenario.py`` -- ``to_scenario(RunConfig) -> CoupledScenario``, the single conversion point and
  therefore the single place unit conversion or name translation happens (ADR-003).
* ``summary.py`` -- ``RunSummary``, the versioned reduction of a run. Comparison plots and figures
  read this; they never read the raw npz (ADR-004). It does not import ``coupled`` -- it only reads
  arrays -- but reading the model's output format is this package's job.

The equivalence test is the point of ``scenario.py``: ``to_scenario`` applied to a ``RunConfig``
describing case ``30N_20km__sabr220__D2med__a1p0__nuc1__cg1`` produces a ``CoupledScenario``
FIELD-FOR-FIELD IDENTICAL to ``run_ensemble.build_scenario()`` for that case. That proves the schema
is a faithful superset before anything is run, which is cheaper and sharper than discovering it from
a diverging result days later.

Traps this package exists to contain (see docs/studio/CAVEATS.md), each an assertion here:

* ``state.npz`` carries BOTH dry and wet quantities: ``dp_mid_um`` / ``dNdlogDp`` are dry;
  ``SA`` / ``radius_cm`` are wet. Every RunSummary series declares its basis.
* Never reconstruct the time axis as ``i * DT``. Outer intervals snap to the terminator, so the mean
  step is ~592 s against a nominal 600 s -- ~1.4% drift, about 0.5 days by day 36. Use stored ``t``.
* Any resampling is UNIFORM in time. A coarsening grid aliases morning particle-number spikes by up
  to 8x.
* Index gas species by NAME using the npz's own ``species`` list, never by position. (An existing
  analysis script hardcodes ``_SO2, _SO3, _H2SO4 = 32, 34, 35``; that is the failure to avoid.)

**``to_scenario`` is deliberately NOT re-exported here.** Import it as
``from studio.modelio.scenario import to_scenario``. Re-exporting it would make every comparison
view that reads a ``RunSummary`` pay for the model it is not using.

Measured, because the cost is not where one would guess: importing ``studio.modelio.scenario``
takes ~0.13 s and pulls in no JAX at all. The first ``to_scenario()`` CALL takes ~1.05 s, because
``CoupledScenario.__post_init__`` imports ``coupled.tomas_bridge`` to validate ``background_dist``
(``coupled_scenario.py:196``) and that is what loads JAX. Subsequent calls are free. So the expense
belongs to constructing a scenario, not to importing this package -- which is precisely why an API
that validates a form on every keystroke must stay on the schema side of this seam. Task 0.8 tracks
making that import lazy in the model.
"""

from __future__ import annotations

from studio.modelio.summary import (
    SECONDS_PER_DAY,
    SUMMARY_SCHEMA_VERSION,
    Basis,
    ConservationCheck,
    RunSummary,
    Series,
    SizeDistribution,
    SummaryFlag,
    TerminationReason,
    summarise_state_npz,
)

__all__ = [
    "SECONDS_PER_DAY",
    "SUMMARY_SCHEMA_VERSION",
    "Basis",
    "ConservationCheck",
    "RunSummary",
    "Series",
    "SizeDistribution",
    "SummaryFlag",
    "TerminationReason",
    "summarise_state_npz",
]
