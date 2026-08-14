# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Phase 3.1: the TOMAS bridge -- initial state from the Marianna distribution + the SO2-off step."""

import numpy as np
import pytest

from coupled import CoupledScenario
from coupled import tomas_bridge as tb


def _scenario(**kw):
    base = dict(T=210.0, P=68.0, WTR=5.0, days=1, DT=3600.0, dt_couple=3600.0, photolysis="tuvx")
    base.update(kw)
    return CoupledScenario(**base)


def test_initial_state_shapes_and_types():
    st = tb.initial_tomas_state(_scenario())
    assert st.Nk.shape == (tb.NBINS,)
    assert st.Mk.shape == (tb.NBINS, 44)
    assert st.Gc.shape[0] == 44
    assert str(st.Nk.dtype) == "float64" and str(st.Mk.dtype) == "float64"
    # Gc starts empty: gaseous H2SO4 is handed in by the driver each outer step.
    assert float(np.max(np.abs(st.Gc))) == 0.0


def test_initial_state_physical():
    st = tb.initial_tomas_state(_scenario())
    assert float(np.min(st.Nk)) >= 0.0
    assert float(np.min(st.Mk)) >= 0.0
    assert float(np.sum(st.Nk)) > 0.0          # the background distribution has particles
    assert float(np.sum(st.Mk[:, tb.SRTSO4])) > 0.0   # ... made of sulfate
    assert 1e-4 <= float(st.rh) <= 0.99
    assert float(st.pres) == 6800.0            # 68 mbar -> Pa
    assert float(st.temp) == 210.0


def test_rh_tracks_water_activity_and_is_clipped():
    # more water vapour -> higher RH; and the clip holds at the dry/wet extremes
    dry = tb.rh_from_scenario(_scenario(WTR=1.0))
    wet = tb.rh_from_scenario(_scenario(WTR=8.0))
    assert wet > dry
    assert 1e-4 <= dry <= 0.99 and 1e-4 <= wet <= 0.99


def test_active_processes_filters_and_never_runs_so2_chem():
    sc = _scenario()
    sc.switches.nucleation = True
    sc.switches.condensation = True
    sc.switches.coagulation = False
    procs = tb.active_processes(sc.switches)
    assert procs == ["nucleation", "condensation"]     # canonical order, coagulation dropped
    assert "so2_chemistry" not in procs                 # gas model owns sulfur


def test_make_step_none_when_all_off():
    sc = _scenario()
    sc.switches.nucleation = sc.switches.condensation = sc.switches.coagulation = False
    assert tb.make_microphysics_step(sc.switches) is None


def test_step_consumes_injected_h2so4():
    import jax.numpy as jnp
    from coupled.units import conc_to_mass
    sc = _scenario()
    sc.switches.nucleation = True
    sc.switches.condensation = True
    sc.switches.coagulation = True
    st = tb.initial_tomas_state(sc)
    step = tb.make_microphysics_step(sc.switches)
    # inject 1e8 molec/cm^3 of gaseous H2SO4
    h2so4_kg = conc_to_mass(1.0e8, tb.BOXVOL_CM3, tb.MW_H2SO4)
    Gc = st.Gc.at[tb.SRTSO4].add(h2so4_kg)
    Nk2, Mk2, Gc2 = step(st.Nk, st.Mk, Gc, st.xk, st.temp, st.pres, st.boxvol,
                         st.rh, st.alpha, 10.0)
    # gas H2SO4 is consumed (condensation + nucleation) and aerosol sulfate grows
    assert float(Gc2[tb.SRTSO4]) < float(Gc[tb.SRTSO4])
    assert float(jnp.sum(Mk2[:, tb.SRTSO4])) > float(jnp.sum(st.Mk[:, tb.SRTSO4]))
    assert float(jnp.min(Nk2)) >= 0.0


def test_80bin_initial_state_matches_40bin_totals():
    # tomas_nbins=80: same observation on the sqrt(2) grid -- same total number, finer bins.
    import jax.numpy as jnp
    st40 = tb.initial_tomas_state(_scenario())
    st80 = tb.initial_tomas_state(_scenario(tomas_nbins=80))
    assert st80.Nk.shape == (80,) and len(st80.xk) == 81
    n40, n80 = float(jnp.sum(st40.Nk)), float(jnp.sum(st80.Nk))
    assert abs(n80 / n40 - 1.0) < 0.05
    m40 = float(jnp.sum(st40.Mk[:, tb.SRTSO4]))
    m80 = float(jnp.sum(st80.Mk[:, tb.SRTSO4]))
    assert abs(m80 / m40 - 1.0) < 0.10
    # het inputs (SA / r_eff / wt%) agree across resolutions to a few percent
    from coupled.aerosol_props import het_inputs
    h40, h80 = het_inputs(st40), het_inputs(st80)
    assert abs(h80["SA"] / h40["SA"] - 1.0) < 0.05
    assert abs(h80["radius_cm"] / h40["radius_cm"] - 1.0) < 0.05


def test_user_modes_reproduce_the_equivalent_named_background_exactly():
    """A user-supplied mode list is seeded by the same code path as a named set.

    Passing the named set's own modes with its own basis must give a BIT-IDENTICAL initial state:
    the only difference between the two branches is where the tuples came from, so anything less
    than exact equality would mean the user path applies a different conversion.
    """
    import jax.numpy as jnp
    from coupled.backgrounds import BACKGROUND_MODES, AMBIENT_BACKGROUNDS
    for name in ("sabr_220", "cesm_g6", "aer_geo"):
        basis = "ambient" if name in AMBIENT_BACKGROUNDS else "stp"
        named = tb.initial_tomas_state(_scenario(background_dist=name))
        user = tb.initial_tomas_state(_scenario(background_dist=BACKGROUND_MODES[name],
                                                background_modes_basis=basis))
        assert jnp.array_equal(named.Nk, user.Nk), name
        assert jnp.array_equal(named.Mk, user.Mk), name


def test_user_modes_basis_changes_the_seeded_number():
    """stp vs ambient is not cosmetic: at 68 mbar / 210 K the conversion is ~0.09, so a scenario
    that omitted the basis and got a default would be wrong by an order of magnitude."""
    import jax.numpy as jnp
    modes = [(50.0, 0.10, 1.6)]
    n_stp = float(jnp.sum(tb.initial_tomas_state(
        _scenario(background_dist=modes, background_modes_basis="stp")).Nk))
    n_amb = float(jnp.sum(tb.initial_tomas_state(
        _scenario(background_dist=modes, background_modes_basis="ambient")).Nk))
    from background_aerosol_distribution import stp_to_ambient_factor
    f = stp_to_ambient_factor(210.0, 6800.0)
    assert f < 0.2                                    # ~0.09 at 68 mbar / 210 K
    assert n_stp == pytest.approx(n_amb * f, rel=1e-12)


def test_user_modes_multi_mode_number_is_physical():
    # two well-separated modes: total seeded number ~= sum of the mode N's (x the STP->ambient
    # factor), to a few percent -- the grid spans 1.7 nm - 17.5 um, so almost nothing falls outside.
    import jax.numpy as jnp
    from background_aerosol_distribution import stp_to_ambient_factor
    modes = [(50.0, 0.05, 1.6), (3.0, 0.4, 1.5)]
    st = tb.initial_tomas_state(_scenario(background_dist=modes, background_modes_basis="stp"))
    expected = (50.0 + 3.0) * stp_to_ambient_factor(210.0, 6800.0) * tb.BOXVOL_CM3
    assert float(jnp.sum(st.Nk)) == pytest.approx(expected, rel=0.02)
    assert float(jnp.sum(st.Mk[:, tb.SRTSO4])) > 0.0


def test_ion_pair_rate_scales_nucleation():
    # fion > 0 must strengthen nucleation (Dunne ion-induced channels) for the same H2SO4.
    import jax.numpy as jnp
    from coupled.units import conc_to_mass
    sc = _scenario()
    st = tb.initial_tomas_state(sc)
    h2so4_kg = conc_to_mass(1.0e7, tb.BOXVOL_CM3, tb.MW_H2SO4)
    Gc = st.Gc.at[tb.SRTSO4].add(h2so4_kg)
    out = {}
    for fion in (0.0, 30.0):
        step = tb.make_microphysics_step(sc.switches, ion_pair_rate=fion)
        Nk2, _Mk2, _Gc2 = step(st.Nk, st.Mk, Gc, st.xk, st.temp, st.pres, st.boxvol,
                               st.rh, st.alpha, 10.0)
        out[fion] = float(jnp.sum(Nk2))
    assert out[30.0] > out[0.0]
