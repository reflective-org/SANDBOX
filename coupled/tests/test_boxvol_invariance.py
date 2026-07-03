# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The TOMAS grid-cell volume must not affect intensive results.

The physical plume volume V0 enters the model ONLY as the dilution ratio V(t)/V0 and as a
conc <-> plume-integrated-mass conversion factor; the TOMAS ``boxvol`` is a normalization volume
(all process rates are intensive). This guards that: a full-physics-minus-tuvx run at 1 m^3 vs
the D1 plume volume (10m x 10m x 30km = 3e12 cm^3, a x3e6 change) must give the same
concentrations / SA / sulfate to well under 1%.

Not bit-identical: absolute fp guard constants (e.g. +1e-30 epsilons inside TOMAS) are not exactly
scale-equivariant, and the nucleation burst chaotically amplifies those seeds -- measured ~1e-3
worst-case for x3e6 (and ~7e-5 for an exact x2), with NO trend in volume. That is measurement
noise relative to every physics uncertainty in the system, and this test pins it.
"""

import numpy as np

from coupled import CoupledScenario
from coupled.coupled_scenario import Switches
import coupled.driver as cd
import coupled.tomas_bridge as tb


def _sc():
    return CoupledScenario(
        T=215.0, P=55.0, WTR=4.5, latitude=30.0, day_of_year=172, start_utc_hour=6.0,
        days=1, DT=600.0, dt_couple=600.0, photolysis="sza",   # sza: no TUV-x cost in the test
        dilution_regime="D1", dilution_zero_species=("SO2", "SO3", "H2SO4", "OH", "HO2"),
        dilution_background={"SO2": 15.0}, ion_pair_rate=30.0,
        switches=Switches(sulfur=True, nucleation=True, condensation=True, coagulation=True,
                          aerosol_to_j=False, heating_to_t=False, dilution=True),
        concentrations={"O2": 2.1e11, "O3": 1.18e6, "SO2": 2.9e9, "OH": 0.5, "HO2": 3.0,
                        "NO": 450.0, "NO2": 450.0, "HCl": 777.0, "ClONO2": 127.0, "HNO3": 5000.0})


def test_boxvol_does_not_change_intensive_results(monkeypatch):
    # first 5 outer intervals only (covers the t=0 nucleation burst, the most volume-sensitive part)
    orig_intervals = cd._outer_intervals
    monkeypatch.setattr(cd, "_outer_intervals",
                        lambda cfg, days, dt: orig_intervals(cfg, days, dt)[:5])
    out = {}
    for vol in (1.0e6, 3.0e12):     # 1 m^3 vs V0 = 10m x 10m x 30km
        monkeypatch.setattr(tb, "BOXVOL_CM3", vol)   # read by initial_tomas_state
        monkeypatch.setattr(cd, "BOXVOL_CM3", vol)   # imported name used by micro_consume/driver
        t, x, aero, _st, sd = cd.run_coupled(_sc(), return_aerosol=True, return_state=True,
                                             return_size_dist=True)
        out[vol] = (x, np.asarray(aero["SA"], float), np.asarray(aero["particulate_S"], float),
                    sd["n_cm3"].sum(axis=1))
    (x1, sa1, ps1, n1), (x2, sa2, ps2, n2) = out[1.0e6], out[3.0e12]
    rel = lambda p, q: np.max(np.abs(q - p) / np.maximum(np.abs(p), 1e-30))
    assert rel(x1, x2) < 1e-2, f"gas concentrations depend on boxvol: {rel(x1, x2):.3e}"
    assert rel(sa1, sa2) < 1e-2, f"surface area depends on boxvol: {rel(sa1, sa2):.3e}"
    assert rel(ps1, ps2) < 1e-2, f"particulate sulfate depends on boxvol: {rel(ps1, ps2):.3e}"
    assert rel(n1, n2) < 1e-2, f"total number depends on boxvol: {rel(n1, n2):.3e}"
