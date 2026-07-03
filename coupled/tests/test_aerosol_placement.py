# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Pressure-anchored aerosol placement: a plume of thickness `aerosol_thickness_km` centered on the
box altitude (from the input pressure), with a grid-independent total column OD and a nearest-layer
guard so a thin plume lands in the box layer instead of silently vanishing."""

import numpy as np

from coupled import CoupledScenario
from coupled import tomas_bridge as tb
from coupled import aerosol_optics as ao

_WL = np.linspace(200.0, 700.0, 25)              # nm
_EDGES = np.arange(0.0, 41.0, 1.0)               # 1 km level grid -> centers 0.5, 1.5, ...


def _state(**kw):
    sc = CoupledScenario(T=210.0, P=68.0, WTR=5.0, days=1, DT=3600.0, dt_couple=3600.0,
                         photolysis="tuvx", **kw)
    return tb.initial_tomas_state(sc)


def test_placement_band_is_anchored_on_box_altitude():
    sc = CoupledScenario(aerosol_thickness_km=2.0)         # band None -> pressure-anchored
    assert ao.placement_band_km(20.0, sc) == (19.0, 21.0)  # thickness centered on the box altitude
    # an explicit absolute band overrides the anchoring (fallback)
    sc2 = CoupledScenario(aerosol_band_km=(12.0, 28.0))
    assert ao.placement_band_km(20.0, sc2) == (12.0, 28.0)


def test_thin_plume_lands_in_the_single_box_layer():
    # a band thinner than the 1 km grid, straddling no layer center -> exactly one (nearest) layer,
    # NOT silently zero.
    st = _state()
    box_alt, dz = 20.0, 0.4                                # band (19.8, 20.2) contains no center
    band = (box_alt - 0.5 * dz, box_alt + 0.5 * dz)
    od = ao.aerosol_optical_props(st, _WL, _EDGES, band).optical_depth
    lit = np.where(od.sum(axis=1) > 0.0)[0]
    assert lit.size == 1                                   # placed, in one layer
    zc = 0.5 * (_EDGES[:-1] + _EDGES[1:])
    assert abs(zc[lit[0]] - box_alt) <= 0.5               # the layer nearest the box altitude


def test_total_column_od_is_grid_independent_and_scales_with_thickness():
    # column OD == b_ext * thickness exactly, regardless of how many grid layers the band spans.
    st = _state()
    mie = ao.MieTable(_WL)
    b_ext, _b_sca, _ssa, _g = ao.bulk_optics(st, mie)
    for dz in (0.4, 1.0, 3.0, 7.0):
        band = (20.0 - 0.5 * dz, 20.0 + 0.5 * dz)
        col = ao.aerosol_optical_props(st, _WL, _EDGES, band, mie=mie).optical_depth.sum(axis=0)
        np.testing.assert_allclose(col, b_ext * dz * 1.0e5, rtol=1e-9)   # dz km -> cm


def test_thickness_must_be_positive():
    import pytest
    with pytest.raises(ValueError):
        CoupledScenario(aerosol_thickness_km=0.0)
