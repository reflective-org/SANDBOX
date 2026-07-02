# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Phase 3.3: TOMAS-derived radius + composition injected into the het chemistry.

Guards (a) NumPy<->JAX parity of the uptake gammas when the aerosol overrides are supplied, and
(b) that the defaults (no override) reproduce the pre-TOMAS behaviour exactly.
"""

import numpy as np

from coupled import CoupledScenario
from coupled.model_bridge import initial_state, to_model_config

from config import ModelConfig                 # gas model (on path via conftest)
from reactions import build_env
from jaxmodel.chem import build_params


def _conc_and_cfg(**cfg_kw):
    sc = CoupledScenario(T=210.0, P=68.0, WTR=5.0, days=1, DT=3600.0, dt_couple=3600.0,
                         photolysis="tuvx",
                         concentrations={"O2": 2.1e11, "O3": 1.18e6, "HCl": 777.0,
                                         "ClONO2": 127.0})   # H2O derived from WTR
    conc = initial_state(sc)
    cfg = to_model_config(sc)
    for k, v in cfg_kw.items():
        setattr(cfg, k, v)
    return conc, cfg


def _numpy_gammas(cfg, conc):
    env = build_env(cfg, conc, j_scale=1.0)
    return {k: env.gammas[k] for k in ("Yhocl", "Yclnh2o", "Yclnhcl")}


def _jax_gammas(cfg, conc, radius, wp):
    p = build_params(cfg.T, cfg.M, cfg.P, cfg.SA, cfg.WTR, cfg.Yn2o5, conc, 1.0,
                     particle_radius=radius, h2so4wp=wp)
    return {k: float(p[k]) for k in ("Yhocl", "Yclnh2o", "Yclnhcl")}


def test_numpy_jax_parity_with_tomas_overrides():
    radius, wp = 0.35e-4, 72.0        # TOMAS-like wet radius (cm) + weight percent
    conc, cfg = _conc_and_cfg(particle_radius=radius, h2so4wp=wp)
    npg = _numpy_gammas(cfg, conc)
    jxg = _jax_gammas(cfg, conc, radius, wp)
    for k in npg:
        assert abs(npg[k] - jxg[k]) <= 1e-10 * max(1.0, abs(npg[k])), k


def test_defaults_reproduce_legacy():
    # cfg with no overrides -> radius 0.1e-4 + thermodynamic wt%; must match explicitly passing those.
    conc, cfg = _conc_and_cfg()
    assert cfg.particle_radius is None and cfg.h2so4wp is None
    legacy = _numpy_gammas(cfg, conc)
    # JAX defaults (radius=0.1e-4, h2so4wp=None -> thermodynamic) must equal the NumPy legacy result.
    p = build_params(cfg.T, cfg.M, cfg.P, cfg.SA, cfg.WTR, cfg.Yn2o5, conc, 1.0)
    for k in legacy:
        assert abs(legacy[k] - float(p[k])) <= 1e-10 * max(1.0, abs(legacy[k])), k


def test_overrides_actually_change_gammas():
    conc, _ = _conc_and_cfg()
    _, cfg_def = _conc_and_cfg()
    _, cfg_ovr = _conc_and_cfg(particle_radius=0.5e-4, h2so4wp=60.0)
    g_def = _numpy_gammas(cfg_def, conc)
    g_ovr = _numpy_gammas(cfg_ovr, conc)
    # a different radius + composition must move at least one uptake coefficient measurably
    assert any(abs(g_def[k] - g_ovr[k]) > 1e-6 for k in g_def)
