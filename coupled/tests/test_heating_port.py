# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Phase 5.1: the TUV-x port's photochemical-heating kernel (O3 channels)."""

import numpy as np
import pytest

from coupled import model_bridge  # noqa: F401  (gas_phase_chemistry on sys.path)
from tuvx_photolysis_adapter import _calculator, _DEFAULT_CONFIG, _TUVX_ROOT
from tuvx_photolysis.radiators import RadiatorOpticalProps


@pytest.mark.slow
def test_heating_rate_profile_physical():
    calc = _calculator(str(_DEFAULT_CONFIG), str(_TUVX_ROOT))
    calc.aerosol_props = None
    h = calc.heating_rate_profile(30.0, 1.0)
    # both O3 channels present, per-height heating-per-absorber, non-negative and finite
    assert set(h) == {"O3+hv->O2+O(1D)", "O3+hv->O2+O(3P)"}
    for name, hr in h.items():
        assert hr.shape == (calc.height_edges_km.size,)
        assert np.all(hr >= 0.0) and np.all(np.isfinite(hr)), name
    assert float(np.max(h["O3+hv->O2+O(1D)"])) > 0.0     # daytime -> nonzero heating


@pytest.mark.slow
def test_overhead_absorbing_aerosol_reduces_heating():
    calc = _calculator(str(_DEFAULT_CONFIG), str(_TUVX_ROOT))
    n_layers = calc.height_edges_km.size - 1
    n_wl = calc.wl_edges.size - 1
    calc.aerosol_props = None
    base = calc.heating_rate_profile(30.0, 1.0)
    calc.aerosol_props = RadiatorOpticalProps(np.full((n_layers, n_wl), 0.5), 0.0, 0.0)  # pure absorber
    try:
        with_aer = calc.heating_rate_profile(30.0, 1.0)
    finally:
        calc.aerosol_props = None
    # an absorbing aerosol removes photons -> column-integrated O3 heating drops
    for name in base:
        assert np.sum(with_aer[name]) <= np.sum(base[name]) + 1e-30, name
    assert np.sum(with_aer["O3+hv->O2+O(1D)"]) < 0.99 * np.sum(base["O3+hv->O2+O(1D)"])
    # cleared state -> baseline reproduced
    assert np.allclose(calc.heating_rate_profile(30.0, 1.0)["O3+hv->O2+O(1D)"],
                       base["O3+hv->O2+O(1D)"], rtol=0, atol=0)


@pytest.mark.slow
def test_heating_responds_to_sza():
    # Like J, the TOP-of-atmosphere heating is SZA-independent (nothing above to attenuate), but the
    # COLUMN-INTEGRATED heating and the lower levels drop as the sun goes down. (Night-gating to
    # exactly 0 at sza>=90 is the coupled heating module's job, tested there.)
    calc = _calculator(str(_DEFAULT_CONFIG), str(_TUVX_ROOT))
    calc.aerosol_props = None
    hi = calc.heating_rate_profile(0.0, 1.0)["O3+hv->O2+O(1D)"]
    lo = calc.heating_rate_profile(85.0, 1.0)["O3+hv->O2+O(1D)"]
    assert np.sum(lo) < np.sum(hi)                 # overhead sun heats the column more
    assert lo[0] < 0.1 * hi[0]                     # near-surface heating strongly SZA-dependent
