# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Phase 3.5: cross-backend parity of the coupled run WITH TOMAS microphysics active.

Both backends run the identical JAX TOMAS ``make_step`` and the same operator split; only the GAS
integrator differs (diffrax Kvaerno5 vs SciPy BDF). So bounded agreement isolates the gas-solver
difference even with the aerosol feedback in the loop. Slow (two real TOMAS coupled runs).
"""

import numpy as np
import pytest

from coupled import CoupledScenario
from coupled.driver import run_coupled
from coupled.reference_numpy import run_coupled_numpy
from config import IDX, SPECIES


@pytest.mark.slow
def test_tomas_on_jax_numpy_parity():
    sc = CoupledScenario(T=210.0, P=68.0, WTR=5.0, latitude=0.0, day_of_year=80, start_utc_hour=6.0,
                         days=1, DT=3600.0, dt_couple=3600.0, photolysis="sza",
                         concentrations={"O2": 2.1e11, "O3": 1.18e6, "SO2": 1.0e4, "OH": 0.5,
                                         "HO2": 3.0, "HCl": 777.0, "ClONO2": 127.0,
                                         "NO": 450.0, "NO2": 450.0})
    sc.switches.nucleation = True
    sc.switches.condensation = True
    sc.switches.coagulation = True

    t_jx, x_jx = run_coupled(sc)
    t_np, x_np = run_coupled_numpy(sc)

    np.testing.assert_allclose(t_jx, t_np, rtol=0, atol=1e-6)
    assert np.all(np.isfinite(x_jx)) and np.all(np.isfinite(x_np))
    # bulk species agree between the two gas integrators (aerosol feedback is identical TOMAS in both);
    # trace fast-cyclers get a looser bound, as in the Phase-2.5 parity test.
    trace = {"NO", "NO3", "O", "O1D", "H2SO4", "SO3"}
    for name in SPECIES:
        a, b = x_jx[:, IDX[name]], x_np[:, IDX[name]]
        scale = np.max(np.abs(b))
        tol = 5e-2 if name in trace else 5e-3
        assert np.max(np.abs(a - b)) <= tol * scale + 1.0, name
