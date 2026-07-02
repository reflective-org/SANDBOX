# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Phase 4.3: aerosol->photolysis wiring through the adapter + coupled driver (aerosol_to_j)."""

import numpy as np
import pytest

from coupled import CoupledScenario
from coupled import tomas_bridge as tb
from coupled import aerosol_optics as ao
from coupled.model_bridge import to_model_config

from tuvx_photolysis_adapter import _compute_j_values, calculator_grids


def test_aerosol_band_validation():
    with pytest.raises(ValueError):
        CoupledScenario(aerosol_band_km=(25.0, 15.0))      # lo >= hi
    with pytest.raises(ValueError):
        CoupledScenario(aerosol_band_km=(-1.0, 10.0))      # negative
    sc = CoupledScenario(aerosol_band_km=[15.0, 25.0])     # list from YAML -> tuple
    assert sc.aerosol_band_km == (15.0, 25.0)


def test_aerosol_to_j_switch_allowed():
    from coupled.coupled_scenario import Switches
    CoupledScenario(photolysis="tuvx", switches=Switches(aerosol_to_j=True))  # must not raise


@pytest.mark.slow
def test_adapter_injects_tomas_aerosol_and_changes_j():
    sc = CoupledScenario(T=210.0, P=68.0, WTR=5.0, latitude=0.0, day_of_year=80, start_utc_hour=12.0,
                         days=1, DT=3600.0, dt_couple=3600.0, photolysis="tuvx",
                         aerosol_band_km=(10.0, 30.0),
                         concentrations={"O2": 2.1e11, "O3": 1.18e6, "HCl": 777.0, "ClONO2": 127.0})
    cfg = to_model_config(sc)
    wl_nm, height_edges = calculator_grids(cfg)
    st = tb.initial_tomas_state(sc)
    # scale up the aerosol so the optical depth is appreciable enough to move J measurably
    st = st._replace(Nk=st.Nk * 1.0e4)
    aer = ao.aerosol_optical_props(st, wl_nm, height_edges, sc.aerosol_band_km)
    assert float(np.max(aer.optical_depth)) > 0.0

    j_base = _compute_j_values(cfg, 0.0, aerosol_props=None)
    j_aer = _compute_j_values(cfg, 0.0, aerosol_props=aer)
    # the aerosol perturbs at least one photolysis rate measurably
    assert any(abs(j_aer[k] - j_base[k]) > 1e-3 * abs(j_base[k])
               for k in j_base if abs(j_base[k]) > 0.0)
    # and a no-aerosol call still equals the baseline (no stale state left on the cached calculator)
    j_base2 = _compute_j_values(cfg, 0.0, aerosol_props=None)
    for k in j_base:
        assert j_base2[k] == j_base[k], k

# NOTE: end-to-end run_coupled(aerosol_to_j on vs off) and the JAX-vs-NumPy aerosol parity are NOT
# committed as tests: a real-TUV-x coupled run solves the full radiation field per interval (~30 s
# each), and with nucleation on the runaway aerosol (OPEN item) drives the stiff gas solver to its
# step cap. The end-to-end behavior was confirmed by the Phase-4 verification agent (toggling the
# switch changed the trajectory by up to 1265x) and is exercised by coupled/validate_phase4.py; the
# NumPy mirror runs byte-identical aerosol-optics code to the JAX driver. See VALIDATION.md / DEFERRED.
