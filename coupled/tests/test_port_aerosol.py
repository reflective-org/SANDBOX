# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Phase 4.2: the TUV-x port honors a dynamic aerosol radiator.

An added aerosol must reduce the actinic flux (hence J); setting aerosol_props back to None must
reproduce the no-aerosol J exactly. Uses the real port (slow)."""

import numpy as np
import pytest

from coupled import model_bridge  # noqa: F401  (puts gas_phase_chemistry on sys.path)
from tuvx_photolysis_adapter import _calculator, _DEFAULT_CONFIG, _TUVX_ROOT  # gas-model adapter
from tuvx_photolysis.radiators import RadiatorOpticalProps


@pytest.mark.slow
def test_aerosol_reduces_j_and_none_is_noop():
    calc = _calculator(str(_DEFAULT_CONFIG), str(_TUVX_ROOT))
    n_layers = calc.height_edges_km.size - 1
    n_wl = calc.wl_edges.size - 1
    sza, esd = 30.0, 1.0

    calc.aerosol_props = None
    base = calc.rate_constants_profile(sza, esd, branching=True)

    # A PURE-ABSORBING aerosol slab (SSA=0) can only remove photons -- unlike a scatterer, which can
    # ENHANCE actinic flux at some levels via multiple scattering. So column-integrated J must drop.
    od = np.full((n_layers, n_wl), 0.5)
    aer = RadiatorOpticalProps(od, 0.0, 0.0, is_air=False)   # pure absorber
    calc.aerosol_props = aer
    with_aer = calc.rate_constants_profile(sza, esd, branching=True)

    for name, j0 in base.items():
        assert np.sum(with_aer[name]) <= np.sum(j0) + 1e-30, name   # absorption removes flux
    # and it did something meaningful
    assert any(np.sum(with_aer[n]) < 0.99 * np.sum(base[n]) for n in base if np.sum(base[n]) > 0)

    # resetting to None reproduces the baseline exactly
    calc.aerosol_props = None
    again = calc.rate_constants_profile(sza, esd, branching=True)
    for name, j0 in base.items():
        np.testing.assert_allclose(again[name], j0, rtol=0, atol=0)
