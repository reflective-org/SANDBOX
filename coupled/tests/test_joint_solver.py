# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Joint stiff-ODE microphysics solver (coupled/joint_solver.py): guards.

The joint solver co-evolves gas chemistry + nucleation + coagulation + condensation's gas sink in one
Diffrax step, then a PPM remap deposits the condensed mass. These tests check the load-bearing
properties: sulfur conservation across the ODE+PPM, finiteness/positivity, and that a short coupled
run on the joint path stays sane and produces a nucleation burst.
"""

import numpy as np
import jax.numpy as jnp

from coupled import CoupledScenario
from coupled.coupled_scenario import Switches
import coupled.driver as cd
import coupled.tomas_bridge as tb
from coupled.aerosol_props import het_inputs
from coupled.units import conc_to_mass
from coupled.joint_solver import make_joint_step
from config import IDX
from reactions import photolysis_coeffs

_ALL_ON = dict(sulfur=True, nucleation=True, condensation=True, coagulation=True,
               aerosol_to_j=False, heating_to_t=False, dilution=False)


def _scn(**kw):
    base = dict(T=215.0, P=55.0, WTR=4.5, latitude=30.0, day_of_year=172, start_utc_hour=6.0,
                days=1, DT=600.0, dt_couple=600.0, photolysis="sza",
                switches=Switches(**_ALL_ON), ion_pair_rate=30.0,
                concentrations={"O2": 2.1e11, "O3": 1.18e6, "SO2": 2.9e9, "OH": 0.5, "HO2": 3.0,
                                "NO": 450.0, "NO2": 450.0, "HCl": 777.0, "ClONO2": 127.0,
                                "HNO3": 5000.0})
    base.update(kw)
    return CoupledScenario(**base)


def _gas_sulfur_kg(conc, boxvol):
    return sum(conc_to_mass(float(conc[IDX[s]]), boxvol, 98.0) for s in ("SO2", "SO3", "H2SO4"))


def test_joint_step_conserves_sulfur():
    """One joint step (ODE + PPM remap): gas S (SO2+SO3+H2SO4) + particulate SO4 conserved."""
    sc = _scn()
    cfg = cd.to_model_config(sc)
    y0 = jnp.asarray(cd.initial_state(sc))
    tst = tb.initial_tomas_state(sc)
    het = het_inputs(tst)
    j_scale = float(cd.photolysis_scale(cd._cosz(cfg, 3900.0)))
    override = jnp.asarray(photolysis_coeffs(cfg, j_scale, None))
    args = {"T": cfg.T, "M": cfg.M, "P": cfg.P, "SA": het["SA"], "WTR": cfg.WTR, "Yn2o5": cfg.Yn2o5,
            "j_scale": j_scale, "sulfur_chain": 1.0, "photo_override": override,
            "particle_radius": het["radius_cm"], "h2so4wp": het["h2so4wp"],
            "k_so2_ho2": float(sc.so2_ho2_rate)}
    step = make_joint_step(cfg.opt, 40, tst.temp, tst.pres, tst.boxvol,
                           ion_pair_rate=float(sc.ion_pair_rate), nuc_scale=1.0, rh=tst.rh)
    S0 = _gas_sulfur_kg(np.asarray(y0), tst.boxvol) + float(jnp.sum(tst.Mk[:, tb.SRTSO4]))
    conc1, Nk1, Mk1 = step(y0, tst.Nk, tst.Mk, 3600.0, 4200.0, args)
    conc1 = np.asarray(conc1)
    assert np.all(np.isfinite(conc1)) and np.all(np.isfinite(np.asarray(Mk1)))
    assert (conc1 >= 0).all() and (np.asarray(Nk1) >= 0).all() and (np.asarray(Mk1) >= 0).all()
    S1 = _gas_sulfur_kg(conc1, tst.boxvol) + float(jnp.sum(Mk1[:, tb.SRTSO4]))
    assert abs(S1 / S0 - 1.0) < 1e-9, f"sulfur not conserved: {abs(S1/S0-1):.3e}"


def test_joint_run_produces_burst_and_stays_sane(monkeypatch):
    """A few outer intervals of the joint path: finite, positive, and nucleation grows N."""
    orig = cd._outer_intervals
    monkeypatch.setattr(cd, "_outer_intervals", lambda cfg, days, dt: orig(cfg, days, dt)[:6])
    sc = _scn(microphysics_solver="joint")
    t, x, aero, st, sd = cd.run_coupled(sc, return_aerosol=True, return_state=True,
                                        return_size_dist=True)
    assert np.all(np.isfinite(x)) and (x >= 0).all()
    N = sd["n_cm3"].sum(axis=1)
    assert N[-1] > 10 * N[0]          # nucleation burst from the concentrated plume
    assert np.nanmax(aero["SA"]) > aero["SA"][0]


def test_joint_agrees_with_split(monkeypatch):
    """Cross-check: the joint stiff solve and the Fortran-validated operator-split path must agree on
    the physically meaningful aerosol integrals. A few intervals of the concentrated D1 burst; peak N
    and SA agree well within tolerance. This is the regression tripwire for the dry-vs-wet condensation
    sink + Neps-discontinuity bug that once gave a ~36% N / ~27% structural gap (see joint_solver.py)."""
    orig = cd._outer_intervals
    monkeypatch.setattr(cd, "_outer_intervals", lambda cfg, days, dt: orig(cfg, days, dt)[:6])
    out = {}
    for solver in ("split", "joint"):
        _, _, aero, _, sd = cd.run_coupled(_scn(microphysics_solver=solver), return_aerosol=True,
                                           return_state=True, return_size_dist=True)
        out[solver] = (float(sd["n_cm3"].sum(axis=1).max()), float(np.nanmax(aero["SA"])))
    (Ns, SAs), (Nj, SAj) = out["split"], out["joint"]
    assert abs(Nj / Ns - 1.0) < 0.15, f"peak N disagrees: split={Ns:.3e} joint={Nj:.3e}"
    assert abs(SAj / SAs - 1.0) < 0.15, f"peak SA disagrees: split={SAs:.2f} joint={SAj:.2f}"


def test_joint_requires_full_tomas_and_no_heating():
    import pytest
    with pytest.raises(NotImplementedError, match="joint"):
        cd.run_coupled(_scn(microphysics_solver="joint",
                            switches=Switches(**{**_ALL_ON, "coagulation": False})))
