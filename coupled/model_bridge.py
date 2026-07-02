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
    """Map a CoupledScenario onto the gas model's ModelConfig (the shared physical fields).

    NOTE: ``switches.sulfur`` is intentionally NOT consumed here -- the sulfur gate is applied by the
    coupled driver (Phase 2.4b) via ``build_params(sulfur_chain=float(switches.sulfur))``, not through
    ModelConfig. Until that driver exists nothing runs a coupled sim, so the switch has no silent
    effect yet; 2.4b wires it. (Tracked in docs/DEFERRED.md.)
    """
    return ModelConfig(
        T=sc.T, P=sc.P, SA=sc.SA, WTR=sc.WTR, Yn2o5=sc.Yn2o5, opt=sc.opt,
        photolysis=sc.photolysis, latitude=sc.latitude, longitude=sc.longitude,
        day_of_year=sc.day_of_year, start_utc_hour=sc.start_utc_hour,
    )


def initial_state(sc: CoupledScenario) -> np.ndarray:
    """Initial concentrations (molec/cm^3), length N_SPECIES, from ``sc.concentrations`` (pptv).

    Guards (no silent assumptions):
      * unknown species names RAISE (no silent typo that contributes nothing);
      * O2 must be present and positive (0 O2 is unphysical -- the commonest "silent 0" foot-gun);
      * **water is single-sourced from WTR** (ppm). If ``concentrations`` also gives H2O it must agree
        with WTR (else RAISE); H2O is then set from WTR regardless, so the O(1D) workaround (cfg.WTR)
        and the HO2+HO2/aerosol terms (conc[H2O]) can never silently diverge.
    Other omitted species start at 0.
    """
    unknown = sorted(set(sc.concentrations) - set(IDX))
    if unknown:
        raise ValueError(f"unknown species in concentrations: {unknown}")
    if float(sc.concentrations.get("O2", 0.0)) <= 0.0:
        raise ValueError("concentrations must include a positive 'O2' (0 O2 is unphysical).")

    M = air_number_density(sc.P, sc.T)
    x = np.zeros(N_SPECIES)
    for name, ppt in sc.concentrations.items():
        x[IDX[name]] = float(ppt) * 1e-12 * M       # pptv -> molec/cm^3

    # Water is single-sourced from WTR (matching the gas model, where H2O is derived from WTR).
    h2o_ppt = sc.WTR * 1.0e6                          # ppm -> pptv
    if "H2O" in sc.concentrations and not np.isclose(
            float(sc.concentrations["H2O"]), h2o_ppt, rtol=1e-3):
        raise ValueError(
            f"concentrations['H2O']={float(sc.concentrations['H2O']):.4g} pptv is inconsistent with "
            f"WTR={sc.WTR} ppm (= {h2o_ppt:.4g} pptv). Make them agree, or omit H2O (derived from WTR).")
    x[IDX["H2O"]] = h2o_ppt * 1e-12 * M
    return x
