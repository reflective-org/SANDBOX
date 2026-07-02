# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Operator-split coupling driver (Phase 2.4b).

Runs the coupled box on the JAX gas backend with an outer loop of ``dt_couple`` steps. Each outer
step: compute the photolysis J from the TUV-x port at the step MIDPOINT (2nd-order splitting), freeze
it, and integrate the gas ODE over the interval with a fresh solve (never stepping across a J jump).
The outer grid is snapped to sunrise/sunset (so no interval straddles the terminator; J is off at
night). One SZA source (gas ``solar.py``, the same the adapter uses) drives J, the day/night snap, and
the fallback ``j_scale``. ``switches.sulfur`` drives the gas sulfur gate.

TOMAS microphysics / dilution / heating are NOT here yet -- they slot into this same outer loop in
Phases 3-6.
"""

from __future__ import annotations

import numpy as np

# model_bridge puts gas_phase_chemistry on sys.path (interim; see DEFERRED) -- import it first.
from .model_bridge import initial_state, to_model_config

from driver import _abstol                                   # noqa: E402  (gas model)
from reactions import photolysis_coeffs                      # noqa: E402
from solar import cos_solar_zenith, photolysis_scale         # noqa: E402  (single SZA source)
from tuvx_photolysis_adapter import _compute_j_values        # noqa: E402  (uncached J compute)
from jaxmodel.model import make_frozen_step                  # noqa: E402

import jax.numpy as jnp                                      # noqa: E402


def _cosz(cfg, t_seconds: float) -> float:
    """cos(SZA) at model time t (s), via the gas solar source the adapter also uses."""
    total_hours = cfg.start_utc_hour + t_seconds / 3600.0
    doy = cfg.day_of_year + total_hours / 24.0
    return cos_solar_zenith(cfg.latitude, cfg.longitude, doy, total_hours % 24.0)


def _outer_intervals(cfg, days: int, dt_couple: float) -> list[tuple[float, float]]:
    """(t0, t1) intervals of length <= dt_couple, SNAPPED so none straddles sunrise/sunset.

    Uniform dt_couple edges plus the day/night terminator crossings (found by a coarse cos(SZA) scan
    then bisection). Because terminators only subdivide uniform intervals, every interval is <=
    dt_couple AND lies entirely in day or entirely in night.
    """
    total = float(days) * 86400.0
    edges = set(np.arange(0.0, total, dt_couple).tolist()) | {0.0, total}
    scan = np.arange(0.0, total + 1.0, min(dt_couple, 300.0))
    cz = np.array([_cosz(cfg, t) for t in scan])
    for i in range(len(scan) - 1):
        if cz[i] != 0.0 and cz[i + 1] != 0.0 and np.sign(cz[i]) != np.sign(cz[i + 1]):
            a, b, sa = scan[i], scan[i + 1], np.sign(cz[i])   # bisection for the crossing time
            for _ in range(40):
                m = 0.5 * (a + b)
                a, b = (m, b) if np.sign(_cosz(cfg, m)) == sa else (a, m)
            edges.add(0.5 * (a + b))
    E = sorted(e for e in edges if 0.0 <= e <= total)
    return list(zip(E[:-1], E[1:]))


def _frozen_j_values(cfg, t_mid: float):
    """Absolute per-reaction J at the interval midpoint (uncached -> dt_couple drives the recompute,
    bypassing the adapter's 120 s cache). ``None`` in non-tuvx modes (photolysis_coeffs then uses
    j45*j_scale). The adapter returns zeros at night."""
    return _compute_j_values(cfg, t_mid) if cfg.photolysis == "tuvx" else None


def run_coupled(scenario):
    """Integrate a CoupledScenario with operator splitting. Returns ``(t [s], states [n_t, n_species])``."""
    cfg = to_model_config(scenario)
    y0 = initial_state(scenario)
    sulfur = float(scenario.switches.sulfur)                 # switches.sulfur -> the gas sulfur gate
    params = dict(T=cfg.T, M=cfg.M, P=cfg.P, SA=cfg.SA, WTR=cfg.WTR, Yn2o5=cfg.Yn2o5)
    step = make_frozen_step(cfg.opt, atol=jnp.asarray(_abstol(cfg.opt)))

    t_list, x_list = [0.0], [np.asarray(y0)]
    yc = jnp.asarray(y0)
    for t0, t1 in _outer_intervals(cfg, scenario.days, scenario.dt_couple):
        t_mid = 0.5 * (t0 + t1)
        j_scale = photolysis_scale(_cosz(cfg, t_mid))        # for fallbacks; 0 at night
        override = jnp.asarray(photolysis_coeffs(cfg, j_scale, _frozen_j_values(cfg, t_mid)))
        args = {**params, "j_scale": j_scale, "sulfur_chain": sulfur, "photo_override": override}
        yc = step(yc, t0, t1, args)
        t_list.append(t1)
        x_list.append(np.asarray(yc))
    return np.asarray(t_list), np.asarray(x_list)
