# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Phase 7.1: sensitivity knobs (nucleation_rate_scale, condensation_alpha, coag_kernel_scale)."""

import numpy as np
import jax.numpy as jnp
import pytest

from coupled import CoupledScenario
from coupled import tomas_bridge as tb
from coupled.units import conc_to_mass


def test_knob_validation():
    with pytest.raises(ValueError):
        CoupledScenario(nucleation_rate_scale=-1.0)
    with pytest.raises(ValueError):
        CoupledScenario(condensation_alpha=0.0)      # must be in (0,1]
    with pytest.raises(ValueError):
        CoupledScenario(condensation_alpha=1.5)
    # coag_kernel_scale is NOT wired -> anything != 1.0 fails loudly (no silent no-op)
    with pytest.raises(NotImplementedError):
        CoupledScenario(coag_kernel_scale=2.0)
    # defaults are all 1.0 and allowed
    sc = CoupledScenario()
    assert (sc.nucleation_rate_scale, sc.condensation_alpha, sc.coag_kernel_scale) == (1.0, 1.0, 1.0)


def test_condensation_alpha_flows_to_tomas_state():
    sc = CoupledScenario(condensation_alpha=0.3, photolysis="tuvx", days=1, DT=3600.0, dt_couple=3600.0)
    st = tb.initial_tomas_state(sc)
    assert float(st.alpha) == 0.3


def _st_and_step(nuc_scale):
    sc = CoupledScenario(T=210.0, P=68.0, WTR=5.0, days=1, DT=3600.0, dt_couple=3600.0,
                         photolysis="tuvx", nucleation_rate_scale=nuc_scale)
    sc.switches.nucleation = True
    sc.switches.condensation = True
    sc.switches.coagulation = True
    st = tb.initial_tomas_state(sc)
    step = tb.make_microphysics_step(sc.switches)
    return st, step, sc


def test_nucleation_rate_scale_changes_particle_production():
    # more nucleation scale -> more new particles from the same H2SO4 slug
    st, step, _ = _st_and_step(1.0)
    _st2, step10, _ = _st_and_step(10.0)
    kg = conc_to_mass(1.0e8, tb.BOXVOL_CM3, tb.MW_H2SO4)
    Gc = st.Gc.at[tb.SRTSO4].set(kg)
    Nk_1, _, _ = step(st.Nk, st.Mk, Gc, st.xk, st.temp, st.pres, st.boxvol, st.rh, st.alpha, 60.0,
                      fn_scale=1.0)
    Nk_10, _, _ = step10(st.Nk, st.Mk, Gc, st.xk, st.temp, st.pres, st.boxvol, st.rh, st.alpha, 60.0,
                         fn_scale=10.0)
    assert float(jnp.sum(Nk_10)) > float(jnp.sum(Nk_1))   # 10x nucleation scale -> more particles
