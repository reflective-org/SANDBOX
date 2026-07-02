# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Bridge from the coupled config to the gas-phase model (CoupledScenario -> ModelConfig + state).

This is the ONE place the coupling layer reaches into ``gas_phase_chemistry``. It maps the single
``CoupledScenario`` onto the gas model's ``ModelConfig`` and builds the initial state vector, so the
duplicated fields (T/P/SA/WTR/... ) have a single, tested mapping and can't drift.

Packaging note (interim): ``gas_phase_chemistry`` is not an installed package (its own tests rely on
a conftest ``sys.path`` insert), so we add it to ``sys.path`` here once, with eyes open. Making it a
real importable package is the tracked follow-up in docs/DEFERRED.md.
"""

from __future__ import annotations

import os
import sys

_GAS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "gas_phase_chemistry"))
if _GAS not in sys.path:
    sys.path.insert(0, _GAS)

import numpy as np

from config import IDX, N_SPECIES, ModelConfig, air_number_density  # noqa: E402  (gas model)

from .coupled_scenario import CoupledScenario, PHOTOLYSIS_MODES as _COUPLED_PHOTOLYSIS_MODES

# Single-source-of-truth guard: the coupled and gas mode lists must stay identical (gas defines them
# on ModelConfig.PHOTOLYSIS_MODES). A true dedup -- removing one definition -- needs
# gas_phase_chemistry to be importable as a package; see DEFERRED.
assert tuple(_COUPLED_PHOTOLYSIS_MODES) == tuple(ModelConfig.PHOTOLYSIS_MODES), (
    f"PHOTOLYSIS_MODES drift: coupled={_COUPLED_PHOTOLYSIS_MODES} vs gas={ModelConfig.PHOTOLYSIS_MODES}")


def to_model_config(sc: CoupledScenario) -> ModelConfig:
    """Map a CoupledScenario onto the gas model's ModelConfig (the shared physical fields)."""
    return ModelConfig(
        T=sc.T, P=sc.P, SA=sc.SA, WTR=sc.WTR, Yn2o5=sc.Yn2o5, opt=sc.opt,
        photolysis=sc.photolysis, latitude=sc.latitude, longitude=sc.longitude,
        day_of_year=sc.day_of_year, start_utc_hour=sc.start_utc_hour,
    )


def initial_state(sc: CoupledScenario) -> np.ndarray:
    """Initial concentrations (molec/cm^3), length N_SPECIES, from ``sc.concentrations`` (pptv).

    Species omitted start at 0. Unknown species names RAISE (no silent typo that contributes nothing --
    closes the #29-review validation gap).
    """
    unknown = sorted(set(sc.concentrations) - set(IDX))
    if unknown:
        raise ValueError(f"unknown species in concentrations: {unknown}")
    M = air_number_density(sc.P, sc.T)
    x = np.zeros(N_SPECIES)
    for name, ppt in sc.concentrations.items():
        x[IDX[name]] = float(ppt) * 1e-12 * M       # pptv -> molec/cm^3
    return x
