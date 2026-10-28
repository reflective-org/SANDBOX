# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Dilution-aware outer-interval refinement (dilution_max_dlnv) and the Gaussian-case scenario wiring."""

import numpy as np
import pytest

import coupled.driver as cd
from coupled import CoupledScenario
from coupled import dilution as dl
from coupled.coupled_scenario import GAUSSIAN_DILUTION_CASES, Switches

_MAX = 0.1


def _scn(**kw):
    base = dict(P=55.0, T=210.0, WTR=6.91, latitude=30.0, longitude=0.0, day_of_year=172,
                start_utc_hour=6.0, days=3, concentrations={"O2": 2.1e11, "O3": 1.18e6},
                switches=Switches(sulfur=True, nucleation=False, condensation=False, coagulation=False,
                                  aerosol_to_j=False, heating_to_t=False, dilution=True))
    base.update(kw)
    return CoupledScenario(**base)


def test_scenario_case_list_mirrors_dilution_registry():
    assert GAUSSIAN_DILUTION_CASES == tuple(dl.GAUSSIAN_CASES)


def test_scenario_accepts_gaussian_cases_and_validates_max_dlnv():
    for name in GAUSSIAN_DILUTION_CASES:
        assert _scn(dilution_regime=name).dilution_regime == name
    assert _scn().dilution_max_dlnv == 0.0                      # default off (v1 behaviour)
    with pytest.raises(ValueError):
        _scn(dilution_max_dlnv=-0.1)
    with pytest.raises(ValueError):
        _scn(dilution_regime="C6")


def _refined(regime, days=3):
    sc = _scn(dilution_regime=regime, days=days)
    cfg = cd.to_model_config(sc)
    gc = dl.GAUSSIAN_CASES[regime]
    edges = (gc.t_burst_start, gc.t_burst_start + gc.tau_burst) if gc.tau_burst is not None else ()
    plain = cd._outer_intervals(cfg, sc.days, sc.dt_couple)
    fine = cd._outer_intervals(cfg, sc.days, sc.dt_couple, vratio=lambda t: dl.volume_ratio(t, regime),
                               max_dlnv=_MAX, extra_edges=edges)
    return sc, plain, fine, edges


@pytest.mark.parametrize("regime", ["C3", "C5", "C1", "C4a"])
def test_refinement_bounds_per_interval_dilution(regime):
    sc, plain, fine, edges = _refined(regime)
    E = [fine[0][0]] + [b for _a, b in fine]
    assert all(b > a for a, b in fine) and np.all(np.diff(E) > 0)             # contiguous, sorted
    assert {b for _a, b in plain} <= set(E)                                   # only subdivides
    assert set(edges) <= set(E)                                               # burst edges kept
    for a, b in fine:
        dlnv = np.log(float(dl.volume_ratio(b, regime)) / float(dl.volume_ratio(a, regime)))
        assert dlnv <= _MAX + 1e-12 or (b - a) <= cd._MIN_REFINED_DT
        assert (b - a) <= sc.dt_couple + 1e-6


def test_refinement_is_local_and_cheap():
    _sc, plain, fine, _e = _refined("C3")
    extra = len(fine) - len(plain)
    assert 0 < extra < 400                                    # only t~0 and the burst onset refine
    late = [iv for iv in fine if iv[0] > 3 * 3600.0 and not (48 * 3600.0 <= iv[0] <= 52 * 3600.0)]
    assert all(abs((b - a) - 600.0) < 1e-6 or (b - a) < 600.0 for a, b in late)
    _sc, plain4a, fine4a, _e = _refined("C4a")
    assert len(fine4a) == len(plain4a)                        # slow case: nothing to refine
