# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Phase 2.5: genuine cross-backend parity of the coupled tuvx run (JAX vs NumPy).

Both backends use the SAME operator split, the SAME frozen step-midpoint J, and the SAME
switches.sulfur gate; only the integrator differs (diffrax Kvaerno5 vs SciPy BDF). So agreement to a
documented tolerance isolates the solver difference. The adapter's J compute is stubbed so no real
TUV-x solve runs -- the test still integrates BOTH backends and compares species (the honest parity
the earlier reviews wanted, minus the slow port; the physical run is validate_coupled.py).
"""

import numpy as np

import coupled.driver as cd
from coupled import CoupledScenario
from coupled.coupled_scenario import Switches
from coupled.reference_numpy import run_coupled_numpy

_COMP = {"O2": 2.1e11, "O3": 1.18e6, "CH4": 1.6e6, "SO2": 9.5e5, "ClO": 10.0,
         "ClONO2": 127.0, "HCl": 777.0, "N2O5": 20.0, "NO": 450.0, "NO2": 450.0,
         "OH": 0.5, "HO2": 3.0}

# stiff, fast-cycling trace species diverge more between the two DIFFERENT stiff solvers -> looser
# per-species bound (same reasoning as test_jax_driver). RHS parity itself is exact (test_jax_tuvx_J).
_TRACE_TOL = {"NO": 5e-2, "NO3": 5e-2, "O": 5e-2, "O1D": 5e-2}


def test_coupled_tuvx_numpy_jax_parity(monkeypatch):
    from config import IDX, SPECIES
    from reactions import MECHANISM
    photo = [r.equation for r in MECHANISM.active if r.kind == "photo"]
    # distinct positive absolute J per photolysis reaction; both backends read this same stub
    monkeypatch.setattr(cd, "compute_j_and_heating",
                        lambda cfg, t, aerosol_props=None: (
                            {eq: 1.0e-4 * (k + 1) for k, eq in enumerate(photo)}, None))

    # gas-only parity (isolate the gas solver): TOMAS/aerosol/heating/dilution OFF -- otherwise the
    # all-on default would move sulfur into particles and the gas-only-S invariant below would not hold
    # (the coupled-with-TOMAS parity is covered separately by test_coupled_parity_tomas.py).
    sc = CoupledScenario(P=68.0, T=210.0, WTR=5.0, latitude=0.0, longitude=0.0, day_of_year=80,
                         start_utc_hour=6.0, days=1, DT=21600.0, dt_couple=21600.0,
                         photolysis="tuvx", concentrations=_COMP,
                         switches=Switches(sulfur=True, nucleation=False, condensation=False,
                                           coagulation=False, aerosol_to_j=False,
                                           heating_to_t=False, dilution=False))

    t_jx, x_jx = cd.run_coupled(sc)
    t_np, x_np = run_coupled_numpy(sc)

    np.testing.assert_allclose(t_jx, t_np, rtol=0, atol=1e-6)   # identical outer grid
    assert np.all(np.isfinite(x_jx))
    for name in SPECIES:
        a, b = x_jx[:, IDX[name]], x_np[:, IDX[name]]
        scale = np.max(np.abs(b))
        tol = _TRACE_TOL.get(name, 2e-3)
        assert np.max(np.abs(a - b)) <= tol * scale + 1.0, name

    # both conserve sulfur, and the H2SO4 endpoints agree
    for x in (x_jx, x_np):
        S = x[:, IDX["SO2"]] + x[:, IDX["SO3"]] + x[:, IDX["H2SO4"]]
        assert abs(S[-1] / S[0] - 1.0) < 1e-6
    np.testing.assert_allclose(x_jx[-1, IDX["H2SO4"]], x_np[-1, IDX["H2SO4"]],
                               rtol=2e-3, atol=1.0)
