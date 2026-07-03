# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Phase 4.1: spectral aerosol optics (Mie table + per-layer OD/SSA/g) from a TomasState."""

import numpy as np

from coupled import CoupledScenario
from coupled import tomas_bridge as tb
from coupled import aerosol_optics as ao

_WL = np.linspace(200.0, 700.0, 25)              # nm (a small UV-vis grid for tests)
_EDGES = np.arange(0.0, 41.0, 1.0)               # km level edges -> 40 layers
_BAND = (18.0, 25.0)                             # aerosol slab


def _state(**kw):
    sc = CoupledScenario(T=210.0, P=68.0, WTR=5.0, days=1, DT=3600.0, dt_couple=3600.0,
                         photolysis="tuvx", **kw)
    return tb.initial_tomas_state(sc)


def test_mie_table_shapes_and_ranges():
    mie = ao.MieTable(_WL)
    n_wl, n_bin = _WL.size, mie.radii_m.size
    assert mie.Qext.shape == (n_wl, n_bin)
    assert np.all(np.isfinite(mie.Qext)) and np.all(np.isfinite(mie.Qsca))
    assert np.all(mie.Qsca <= mie.Qext + 1e-9)              # scattering <= extinction
    assert np.all(mie.Qsca >= -1e-12)
    assert np.all(mie.gsca <= 1.0 + 1e-9) and np.all(mie.gsca >= -1.0 - 1e-9)


def test_bulk_optics_physical():
    mie = ao.MieTable(_WL)
    b_ext, b_sca, ssa, g = ao.bulk_optics(_state(), mie)
    assert b_ext.shape == (_WL.size,)
    assert np.all(b_ext > 0.0) and np.all(b_sca >= 0.0)
    assert np.all(ssa >= 0.0) and np.all(ssa <= 1.0 + 1e-12)
    assert np.all(np.abs(g) <= 1.0 + 1e-9)
    # nearly non-absorbing sulfate (tiny imaginary index) -> SSA close to 1
    assert np.all(ssa > 0.9)


def test_extinction_scales_linearly_with_number():
    mie = ao.MieTable(_WL)
    st = _state()
    b0, _, ssa0, g0 = ao.bulk_optics(st, mie)
    b1, _, ssa1, g1 = ao.bulk_optics(st._replace(Nk=st.Nk * 3.0), mie)
    np.testing.assert_allclose(b1, 3.0 * b0, rtol=1e-9)
    np.testing.assert_allclose(ssa1, ssa0, rtol=1e-9)       # intensive
    np.testing.assert_allclose(g1, g0, rtol=1e-9)


def test_optical_depth_placement_and_scaling():
    st = _state()
    props = ao.aerosol_optical_props(st, _WL, _EDGES, _BAND)
    od = props.optical_depth
    n_layers = _EDGES.size - 1
    assert od.shape == (n_layers, _WL.size)
    z_center = 0.5 * (_EDGES[:-1] + _EDGES[1:])
    in_band = (z_center >= _BAND[0]) & (z_center <= _BAND[1])
    assert np.all(od[~in_band] == 0.0)                     # no aerosol outside the band
    assert np.all(od[in_band] > 0.0)
    # total column OD proportional to band thickness (per-layer OD = b_ext*dz)
    wide = ao.aerosol_optical_props(st, _WL, _EDGES, (15.0, 29.0))
    assert wide.optical_depth.sum() > od.sum()


def test_bulk_ssa_and_g_match_scattering_weighted_definition():
    # assert SSA and g equal their definitions against an independent per-bin recompute (not just ranges)
    mie = ao.MieTable(_WL)
    st = _state()
    b_ext, b_sca, ssa, g = ao.bulk_optics(st, mie)
    n_cm3 = np.asarray(st.Nk, dtype=float) / float(st.boxvol)
    geo = np.pi * (mie.radii_m * 100.0) ** 2
    be = (mie.Qext * (n_cm3 * geo)[None, :]).sum(axis=1)
    bs = (mie.Qsca * (n_cm3 * geo)[None, :]).sum(axis=1)
    g_ref = (mie.gsca * mie.Qsca * (n_cm3 * geo)[None, :]).sum(axis=1) / bs
    np.testing.assert_allclose(ssa, bs / be, rtol=1e-12)
    np.testing.assert_allclose(g, g_ref, rtol=1e-12)


def test_empty_aerosol_gives_zero_od():
    import jax.numpy as jnp
    st = _state()
    empty = st._replace(Nk=jnp.zeros_like(st.Nk))
    props = ao.aerosol_optical_props(empty, _WL, _EDGES, _BAND)
    assert np.all(props.optical_depth == 0.0)


def test_more_aerosol_more_optical_depth():
    st = _state()
    od1 = ao.aerosol_optical_props(st, _WL, _EDGES, _BAND).optical_depth.sum()
    od2 = ao.aerosol_optical_props(st._replace(Nk=st.Nk * 5.0), _WL, _EDGES, _BAND).optical_depth.sum()
    assert od2 > od1
    np.testing.assert_allclose(od2, 5.0 * od1, rtol=1e-9)
