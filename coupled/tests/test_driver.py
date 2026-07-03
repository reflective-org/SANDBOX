# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Operator-split coupling driver (Phase 2.4b): outer-grid snapping + end-to-end runs."""

import numpy as np

import coupled.driver as cd
from coupled import CoupledScenario
from coupled.coupled_scenario import Switches

_COMP = {"O2": 2.1e11, "O3": 1.18e6, "CH4": 1.6e6, "SO2": 9.5e5, "ClO": 10.0,
         "ClONO2": 127.0, "HCl": 777.0, "N2O5": 20.0, "NO": 450.0, "NO2": 450.0,
         "OH": 0.5, "HO2": 3.0}

# gas-only switches: this file tests the gas-level driver mechanics (outer-grid snapping, gas sulfur
# conservation, the sulfur gate). TOMAS/aerosol/heating/dilution are tested in their own files, and
# leaving them at the all-on default would (a) break the gas-only-S invariant and (b) be slow.
_GAS_ONLY = dict(sulfur=True, nucleation=False, condensation=False, coagulation=False,
                 aerosol_to_j=False, heating_to_t=False, dilution=False)


def _scn(**kw):
    base = dict(P=68.0, T=210.0, WTR=5.0, latitude=0.0, longitude=0.0, day_of_year=80,
                start_utc_hour=0.0, days=1, DT=43200.0, dt_couple=43200.0, concentrations=_COMP,
                switches=Switches(**_GAS_ONLY))
    base.update(kw)
    return CoupledScenario(**base)


def _total_sulfur(x, cfg_M=None):
    from config import IDX
    return x[:, IDX["SO2"]] + x[:, IDX["SO3"]] + x[:, IDX["H2SO4"]]


def test_outer_intervals_snap_to_terminator():
    sc = _scn(days=2, dt_couple=3600.0)
    cfg = cd.to_model_config(sc)
    intervals = cd._outer_intervals(cfg, sc.days, sc.dt_couple)
    assert intervals[0][0] == 0.0 and intervals[-1][1] == 2 * 86400.0
    for t0, t1 in intervals:
        assert 0 < (t1 - t0) <= sc.dt_couple + 1e-6            # never longer than dt_couple
        c0, c1 = cd._cosz(cfg, t0), cd._cosz(cfg, t1)
        # no interior terminator: endpoints don't strictly straddle cos(SZA)=0
        assert not (c0 > 1e-9 and c1 < -1e-9) and not (c0 < -1e-9 and c1 > 1e-9)


def test_run_coupled_sza_conserves_sulfur():
    sc = _scn(photolysis="sza")           # non-reference -> sulfur chain ON (no TUV-x port needed)
    t, x = cd.run_coupled(sc)
    assert x.shape[0] == t.shape[0] and t[0] == 0.0 and t[-1] == 86400.0
    assert np.all(np.isfinite(x))
    S = _total_sulfur(x)
    assert abs(S[-1] / S[0] - 1.0) < 1e-6                       # sulfur atoms conserved (no sink)


def test_run_coupled_tuvx_wires_absolute_J(monkeypatch):
    # stub the adapter's uncached J+heating compute so no real TUV-x solve runs; give every photo
    # reaction a positive J -> the tuvx branch must produce H2SO4 and conserve sulfur.
    from reactions import MECHANISM
    photo = [r.equation for r in MECHANISM.active if r.kind == "photo"]
    monkeypatch.setattr(cd, "compute_j_and_heating",
                        lambda cfg, t, aerosol_props=None: ({eq: 1.0e-4 for eq in photo}, None))
    from config import IDX
    sc = _scn(photolysis="tuvx", start_utc_hour=6.0)          # daytime start so J is on
    t, x = cd.run_coupled(sc)
    assert np.all(np.isfinite(x))
    assert x[-1, IDX["H2SO4"]] > 0.0                           # H2SO4 produced via the chain
    S = _total_sulfur(x)
    assert abs(S[-1] / S[0] - 1.0) < 1e-6


def test_switches_sulfur_false_disables_chain(monkeypatch):
    # switches.sulfur=False must turn the chain OFF in the coupled driver: no H2SO4, SO2 ~flat.
    from reactions import MECHANISM
    photo = [r.equation for r in MECHANISM.active if r.kind == "photo"]
    monkeypatch.setattr(cd, "compute_j_and_heating",
                        lambda cfg, t, aerosol_props=None: ({eq: 1.0e-4 for eq in photo}, None))
    from config import IDX
    sc = _scn(photolysis="tuvx", start_utc_hour=6.0, switches=Switches(**{**_GAS_ONLY, "sulfur": False}))
    t, x = cd.run_coupled(sc)
    assert np.all(x[:, IDX["H2SO4"]] == 0.0)                   # chain off -> no H2SO4 ever
    assert np.all(x[:, IDX["SO3"]] == 0.0)
