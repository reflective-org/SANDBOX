# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Phase 3.2: aerosol diagnostics (surface area, wet radius, H2SO4 weight-percent)."""

import jax.numpy as jnp

from coupled import CoupledScenario
from coupled import tomas_bridge as tb
from coupled import aerosol_props as ap


def _state():
    sc = CoupledScenario(T=210.0, P=68.0, WTR=5.0, days=1, DT=3600.0, dt_couple=3600.0,
                         photolysis="tuvx")
    return tb.initial_tomas_state(sc)


def test_props_physical_ranges():
    st = _state()
    sa = ap.surface_area_um2_cm3(st)
    r = ap.mean_wet_radius_cm(st)
    wp = ap.h2so4_weight_pct(st)
    assert sa > 0.0                       # background distribution has surface area
    assert 1e-7 < r < 1e-2                # wet radius between ~1 nm and ~100 um, in cm
    assert 0.0 < wp < 100.0
    assert wp > 60.0                      # cold dry stratosphere -> concentrated sulfate


def test_het_inputs_bundle():
    st = _state()
    d = ap.het_inputs(st)
    assert set(d) == {"SA", "radius_cm", "h2so4wp"}
    assert d["SA"] == ap.surface_area_um2_cm3(st)
    assert d["radius_cm"] == ap.mean_wet_radius_cm(st)
    assert d["h2so4wp"] == ap.h2so4_weight_pct(st)


def test_surface_area_aggregation_and_units_single_bin():
    # verify the um^2/cm^3 aggregation directly against Nk*pi*Dp_wet^2 for a one-bin state
    # (does NOT assume how TOMAS reconstructs Dp -- reads Dp from the same helper the fn uses).
    import math
    st = _state()
    k = 22
    one = st._replace(Nk=jnp.zeros_like(st.Nk).at[k].set(st.Nk[k]),
                      Mk=jnp.zeros_like(st.Mk).at[k].set(st.Mk[k]))
    Dpk, _ = ap._wet_diameters_m(one)
    expect = float(one.Nk[k]) * math.pi * float(Dpk[k]) ** 2 * 1e12 / float(one.boxvol)
    assert abs(ap.surface_area_um2_cm3(one) - expect) < 1e-9 * expect


def test_weight_pct_is_rh_determined():
    # At fixed RH, equilibrium water tracks sulfate mass, so the H2SO4 weight-percent is set by RH
    # (an equilibrium quantity), NOT by how much sulfate is present -- matches the thermodynamic
    # h2so4wp_at, which depends only on T/P/H2O. Scaling sulfate leaves wt% invariant...
    st = _state()
    wp0 = ap.h2so4_weight_pct(st)
    more_so4 = st._replace(Mk=st.Mk.at[:, tb.SRTSO4].add(st.Mk[:, tb.SRTSO4]))  # 2x sulfate
    assert abs(ap.h2so4_weight_pct(more_so4) - wp0) < 1e-9
    # ... while a wetter aerosol (higher RH) dilutes it (lower wt%).
    assert ap.h2so4_weight_pct(st._replace(rh=0.5)) < wp0


def test_number_weighted_radius_single_bin():
    # put all number+mass in one bin -> mean wet radius equals that bin's wet radius
    st = _state()
    k = 20
    Nk = jnp.zeros_like(st.Nk).at[k].set(st.Nk[k])
    Mk = jnp.zeros_like(st.Mk).at[k].set(st.Mk[k])
    one = st._replace(Nk=Nk, Mk=Mk)
    Dpk, _ = ap._wet_diameters_m(one)
    expected_cm = float(0.5 * Dpk[k]) * 100.0
    assert abs(ap.mean_wet_radius_cm(one) - expected_cm) < 1e-12


def test_empty_state_limits():
    st = _state()
    empty = st._replace(Nk=jnp.zeros_like(st.Nk), Mk=jnp.zeros_like(st.Mk))
    assert ap.surface_area_um2_cm3(empty) == 0.0
    assert ap.mean_wet_radius_cm(empty) == 0.1e-4     # legacy 1-um fallback
    assert ap.h2so4_weight_pct(empty) == 0.0


def test_condensing_sulfate_raises_surface_area():
    st = _state()
    sa0 = ap.surface_area_um2_cm3(st)
    # add sulfate mass across bins (as condensation would) -> larger particles -> more area
    grown = st._replace(Mk=st.Mk.at[:, tb.SRTSO4].add(st.Mk[:, tb.SRTSO4] * 0.5))
    assert ap.surface_area_um2_cm3(grown) > sa0
