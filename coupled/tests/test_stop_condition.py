# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""``run_coupled(stop_condition=...)``: the diagnostics dict, and the legacy two-argument shape.

The callback used to receive only ``(t1_seconds, wet_SA)``, so a criterion could not reference SO2
or particle number. It now receives one dict; the old shape is still accepted and dispatched by
declared arity, never by trying a call and catching the TypeError.
"""

import warnings

import numpy as np
import pytest

import coupled.driver as cd
from coupled import CoupledScenario
from coupled.coupled_scenario import Switches
from config import IDX

_COMP = {"O2": 2.1e11, "O3": 1.18e6, "CH4": 1.6e6, "SO2": 9.5e5, "ClO": 10.0,
         "ClONO2": 127.0, "HCl": 777.0, "N2O5": 20.0, "NO": 450.0, "NO2": 450.0,
         "OH": 0.5, "HO2": 3.0}

# Gas-only: this file tests the callback plumbing, not microphysics. TOMAS-inactive is also the case
# where the aerosol entries must be NaN, which is half of what is asserted below.
_GAS_ONLY = dict(sulfur=True, nucleation=False, condensation=False, coagulation=False,
                 aerosol_to_j=False, heating_to_t=False, dilution=False)


def _scn(**kw):
    base = dict(P=68.0, T=210.0, WTR=5.0, latitude=0.0, longitude=0.0, day_of_year=80,
                start_utc_hour=0.0, days=1, DT=14400.0, dt_couple=14400.0, photolysis="sza",
                concentrations=_COMP, switches=Switches(**_GAS_ONLY))
    base.update(kw)
    return CoupledScenario(**base)


# --------------------------------------------------------------------------------------------
# Arity dispatch, in isolation from a run.
# --------------------------------------------------------------------------------------------
def test_one_argument_callback_passes_through_unwrapped():
    def stop(diag):
        return False
    assert cd._adapt_stop_condition(stop) is stop


def test_none_stays_none():
    assert cd._adapt_stop_condition(None) is None


def test_two_argument_callback_is_adapted_and_deprecated():
    seen = []

    def legacy(t1, sa):
        seen.append((t1, sa))
        return t1 > 100.0

    with pytest.warns(DeprecationWarning, match="two-argument stop_condition"):
        adapted = cd._adapt_stop_condition(legacy)
    assert adapted is not legacy
    # the adapter must map exactly diag["t"] -> t1 and diag["SA"] -> sa, in that order
    assert adapted({"t": 500.0, "SA": 2.5, "gas": {}}) is True
    assert seen == [(500.0, 2.5)]


@pytest.mark.parametrize("fn", [
    lambda: False,                          # 0 positional -> no shape
    lambda a, b, c: False,                  # 3 positional -> no shape
    lambda *args: False,                    # matches BOTH shapes -> ambiguous, must not be guessed
])
def test_undispatchable_arities_raise(fn):
    with pytest.raises(TypeError, match="stop_condition"):
        cd._adapt_stop_condition(fn)


def test_non_callable_raises():
    with pytest.raises(TypeError, match="callable"):
        cd._adapt_stop_condition(3.0)


def test_dispatch_counts_all_positional_parameters_not_only_required_ones():
    # `def stop(t1, sa=nan)` declares two positionals and is therefore read as the LEGACY shape,
    # even though it is callable with one argument. Dispatch is on the declared signature, so it
    # does not depend on defaults -- and a one-argument callback must not be written that way.
    def stop(t1, sa=float("nan")):
        return False
    with pytest.warns(DeprecationWarning):
        adapted = cd._adapt_stop_condition(stop)
    assert adapted is not stop


# --------------------------------------------------------------------------------------------
# End to end, through a real (gas-only) run.
# --------------------------------------------------------------------------------------------
def test_diagnostics_dict_contents_and_early_stop():
    seen = []

    def stop(diag):
        seen.append(diag)
        # An SO2-based criterion -- not expressible with the old (t1, wet_SA) signature at all.
        # SO2 is a pure sink here (the chain has no source), so it is non-increasing.
        return diag["gas"]["SO2"] <= seen[0]["gas"]["SO2"] and diag["interval"] >= 2

    sc = _scn(days=1)
    t, x = cd.run_coupled(sc, stop_condition=stop)

    assert len(seen) == 2                       # stopped at the end of the 2nd outer interval
    assert t.shape[0] == 3                      # t=0 plus the two completed intervals
    assert x.shape[0] == t.shape[0]

    first, last = seen[0], seen[-1]
    assert set(first) == {"t", "interval", "T", "SA", "radius_cm", "h2so4wp",
                          "particulate_S", "N_total", "gas"}
    assert first["interval"] == 1 and last["interval"] == 2
    # `t` is the driver's own clock, matching the returned time vector exactly -- the outer grid
    # snaps to the terminator, so interval * DT would drift from it.
    assert first["t"] == pytest.approx(t[1]) and last["t"] == pytest.approx(t[2])
    assert last["t"] > first["t"]
    assert first["T"] == pytest.approx(sc.T)    # heating_to_t off -> box T unchanged
    # TOMAS inactive -> aerosol entries are NaN, never 0: a `SA < x` criterion must not read
    # "no aerosol model" as "the plume has relaxed".
    for key in ("SA", "radius_cm", "h2so4wp", "particulate_S", "N_total"):
        assert np.isnan(first[key]), key
    # gas is keyed by species NAME and matches the recorded state row exactly (same units,
    # molec/cm^3); indexing by name is what protects against a state-vector reordering.
    assert set(last["gas"]) == set(IDX)
    for name, idx in IDX.items():
        assert last["gas"][name] == pytest.approx(float(x[2, idx]), rel=0.0, abs=0.0)


def test_never_stopping_condition_runs_to_completion():
    calls = []
    sc = _scn(days=1)
    t, _x = cd.run_coupled(sc, stop_condition=lambda diag: bool(calls.append(diag["t"])))
    assert t[-1] == 86400.0                     # ran the full day
    assert calls == list(t[1:])                 # called once per completed outer interval


def test_legacy_two_argument_shape_stops_at_the_same_point_as_the_dict_form():
    """The old f(t1, wet_SA) contract, unchanged: identical stop point, identical time vector."""
    def legacy(t1, sa):
        assert np.isnan(sa)                     # gas-only run: wet SA is NaN, exactly as before
        return t1 >= 20000.0

    def modern(diag):
        assert np.isnan(diag["SA"])
        return diag["t"] >= 20000.0

    sc = _scn(days=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        t_legacy, _x = cd.run_coupled(sc, stop_condition=legacy)
    t_modern, _x2 = cd.run_coupled(sc, stop_condition=modern)
    assert np.array_equal(t_legacy, t_modern)
    assert t_legacy[-1] < 86400.0               # and it really did stop early


def test_mis_shaped_callback_raises_before_any_integration(monkeypatch):
    # a bad callback must be rejected at entry, not after the first (minutes-long) interval
    def boom(*_a, **_kw):
        raise AssertionError("run_coupled started integrating despite a bad stop_condition")
    monkeypatch.setattr(cd, "to_model_config", boom)
    with pytest.raises(TypeError, match="stop_condition"):
        cd.run_coupled(_scn(), stop_condition=lambda a, b, c: True)


@pytest.mark.slow
def test_tomas_active_run_reports_finite_aerosol_and_number():
    """With TOMAS on, the aerosol entries are real numbers -- including N_total, which is the
    quantity the old two-argument callback could not see at all."""
    sc = _scn(days=1, DT=3600.0, dt_couple=3600.0,
              concentrations={**_COMP, "SO2": 1.0e4},
              switches=Switches(**{**_GAS_ONLY, "nucleation": True, "condensation": True,
                                   "coagulation": True}))
    seen = []
    cd.run_coupled(sc, stop_condition=lambda diag: bool(seen.append(diag)) or diag["interval"] >= 2)
    diag = seen[-1]
    for key in ("SA", "radius_cm", "h2so4wp", "particulate_S", "N_total"):
        assert np.isfinite(diag[key]) and diag[key] > 0.0, key
    # N_total is a number density at ambient T,P: the stratospheric background is O(1-100) cm^-3,
    # so anything outside a very generous 1e-3..1e6 band means a units or per-cell/per-cm^3 mix-up.
    assert 1.0e-3 < diag["N_total"] < 1.0e6
