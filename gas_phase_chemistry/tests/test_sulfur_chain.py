"""Phase 1.2: the gated gas-phase sulfur chain SO2 -> SO3 -> H2SO4.

The chain is active only in non-reference photolysis modes (Env.sulfur_chain). In reference mode
the legacy SO2 reactions run instead and SO3/H2SO4 stay zero, so the original 34 species are
unaffected. Sulfur atoms are conserved across SO2+SO3+H2SO4. Rates are JPL 19-5 (I4+I92 net, I34,
I79) -- see docs/jpl19-5-sulfur-crosscheck.md.
"""

import numpy as np

from config import IDX, ModelConfig
from driver import initial_concentrations
from rhs import concs_het
from reactions import MECHANISM, build_env, _k68


def _state(cfg):
    x = initial_concentrations(cfg.P, cfg.M, cfg.WTR)
    x[IDX["SO3"]] = 1.0e6  # seed SO3 so SO3 + H2O -> H2SO4 is exercised
    return x


def _cfg(photolysis):
    return ModelConfig(T=210.0, P=68.0, SA=2.0, WTR=5.0, Yn2o5=0.1, opt=1,
                       photolysis=photolysis, latitude=0.0, longitude=0.0,
                       day_of_year=80, start_utc_hour=12.0)


def test_reference_mode_gate_off():
    # reference mode: no sulfur chain -> SO3/H2SO4 have exactly zero tendency ...
    cfg = _cfg("reference")
    d = concs_het(0.0, _state(cfg), cfg)
    assert d[IDX["SO3"]] == 0.0
    assert d[IDX["H2SO4"]] == 0.0
    # ... but the legacy SO2 + OH loss still runs (SO2 is consumed)
    assert d[IDX["SO2"]] < 0.0


def test_nonreference_mode_produces_h2so4():
    # sza (non-reference) mode: chain on -> SO2 consumed, H2SO4 produced from SO3 + H2O
    cfg = _cfg("sza")
    d = concs_het(0.0, _state(cfg), cfg)
    assert d[IDX["SO2"]] < 0.0
    assert d[IDX["H2SO4"]] > 0.0


def test_sulfur_atoms_conserved():
    # every SO2 lost -> SO3, every SO3 lost -> H2SO4: d(SO2+SO3+H2SO4)/dt = 0 to round-off
    cfg = _cfg("sza")
    d = concs_het(0.0, _state(cfg), cfg)
    total = d[IDX["SO2"]] + d[IDX["SO3"]] + d[IDX["H2SO4"]]
    scale = abs(d[IDX["SO2"]]) + abs(d[IDX["SO3"]]) + abs(d[IDX["H2SO4"]]) + 1.0
    assert abs(total) <= 1e-10 * scale


def test_so3_h2o_rate_matches_jpl_i79():
    # coefficient folds one [H2O] so mass action gives kI = 8.5e-41 exp(6540/T) [H2O]^2 [SO3]
    cfg = _cfg("sza")
    x = _state(cfg)
    # isolate SO3 + H2O -> H2SO4 by zeroing OH/HO2 (kills SO2->SO3 production)
    x[IDX["OH"]] = 0.0
    x[IDX["HO2"]] = 0.0
    d = concs_het(0.0, x, cfg)
    H2O = x[IDX["H2O"]]
    expect = 8.5e-41 * np.exp(6540.0 / cfg.T) * H2O**2 * x[IDX["SO3"]]
    np.testing.assert_allclose(d[IDX["H2SO4"]], expect, rtol=1e-12)
    np.testing.assert_allclose(d[IDX["SO3"]], -expect, rtol=1e-12)  # SO3 only sink now


def test_so2_to_so3_production_rate_matches_k68():
    # Pin the SO2->SO3 production magnitude to _k68 (JPL 19-5 I4+I92 net), not just its sign.
    # Isolate it: zero HO2 (kills SO2+HO2->SO3+OH) and SO3 (kills SO3+H2O->H2SO4), so the ONLY
    # source/sink of SO3 is SO2+OH->SO3+HO2 -> d[SO3] == _k68 * [SO2] * [OH].
    cfg = _cfg("sza")
    x = _state(cfg)
    x[IDX["SO3"]] = 0.0
    x[IDX["HO2"]] = 0.0
    d = concs_het(0.0, x, cfg)
    env = build_env(cfg, x, 1.0)  # j_scale irrelevant to the SO2+OH gas rate
    expect = _k68(env) * x[IDX["SO2"]] * x[IDX["OH"]]
    np.testing.assert_allclose(d[IDX["SO3"]], expect, rtol=1e-12)


def test_chain_on_numpy_jax_parity():
    # dC/dt parity with the sulfur chain ON (the existing jax-dcdt test only covers chain-OFF).
    import jax.numpy as jnp
    from jaxmodel.chem import build_params, dCdt as jax_dCdt

    cfg = _cfg("sza")            # non-reference -> Env.sulfur_chain True
    x = _state(cfg)             # seeded SO3 so SO3+H2O is active too
    j_scale = 0.7               # fixed, identical on both sides
    env = build_env(cfg, x, j_scale)
    assert env.sulfur_chain is True
    d_np = MECHANISM.dCdt(env, x)

    p = build_params(cfg.T, cfg.M, cfg.P, cfg.SA, cfg.WTR, cfg.Yn2o5,
                     jnp.asarray(x), j_scale, sulfur_chain=1.0)
    d_jx = np.asarray(jax_dCdt(jnp.asarray(x), p, cfg.opt))
    np.testing.assert_allclose(d_jx, d_np, rtol=1e-9, atol=1e-6)
