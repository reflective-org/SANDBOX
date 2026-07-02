# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Bridge from the coupled config to the TOMAS sectional-aerosol model (Phase 3).

This is the ONE place the coupling layer reaches into ``tomas_jax``. It builds an initial
``TomasState`` from the scenario (using the Marianna 'redcircles' background distribution) and
constructs the microphysics step with TOMAS's own SO2 chemistry OMITTED -- the gas-phase model
(frank) owns sulfur and hands gaseous H2SO4 to TOMAS via ``Gc[SRTSO4]`` each outer step.

Packaging note (interim, AD-3.1): ``tomas_jax`` is not installed; we add ``../tomas-jax`` (and its
``experimental_case`` dir, for the Marianna distribution loader) to ``sys.path`` here, mirroring the
gas-model bridge. Pinning to branch ``feat/marianna-dilution`` (AD-3.2). Proper packaging: DEFERRED.
"""

from __future__ import annotations

import os
import sys

# gas model on path first (for the water-activity calc used to set RH) -- model_bridge does the insert.
from . import model_bridge  # noqa: F401  (side effect: puts gas_phase_chemistry on sys.path)
from aerosol import h2so4wp_at  # noqa: E402  (gas model: water activity a_W from T,P,H2O)

_TOMAS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tomas-jax"))
for _p in (_TOMAS, os.path.join(_TOMAS, "experimental_case")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)   # TOMAS is float64 throughout; match it

import jax.numpy as jnp  # noqa: E402

from tomas_jax.core import config as tcfg          # noqa: E402
from tomas_jax.core.state import TomasState        # noqa: E402
from tomas_jax.solvers.condensation import make_step  # noqa: E402
import background_aerosol_distribution as _bad      # noqa: E402  (Marianna dist loader)

# --- exposed constants the driver/props need (single source) ---
SRTSO4 = tcfg.SRTSO4          # sulfate index in Gc (gaseous H2SO4 handoff slot) and Mk
SRTH2O = tcfg.SRTH2O          # aerosol water index in Mk
NBINS = tcfg.NBINS
MW_H2SO4 = tcfg.MW_H2SO4      # g/mol (98.0), for the units bridge
BOXVOL_CM3 = 1.0e6            # canonical grid-cell volume = 1 m^3 (intensive results independent of it)
ALPHA_DEFAULT = 1.0           # condensation accommodation coefficient (Phase 8 sensitivity knob)

# TOMAS scheme choices matching the validated Marianna run (AD-3.5); documented in ASSUMPTIONS.md.
_COND_METHOD = "ppm_jit"
_NUCL_SCHEME = "ricco_dunne"
_WATER_SCHEME = "h2so4_tabazadeh"

#: canonical process order for make_step; the driver filters this by the scenario switches.
_PROCESS_ORDER = ("nucleation", "coagulation", "condensation")


def rh_from_scenario(scenario) -> float:
    """Relative humidity (fraction, over liquid water) the aerosol equilibrates to.

    Set to the gas-model water activity ``a_W = pH2O / p0_liquid`` (``aerosol.h2so4wp_at``) so TOMAS's
    water uptake sees the SAME water field as the gas heterogeneous chemistry (AD-3.5). Clipped to
    [1e-4, 0.99] because the TOMAS water-uptake fit is valid for RH in ~1-99%.
    """
    _wp, _ml, a_W = h2so4wp_at(scenario.T, scenario.P, scenario.WTR)
    return float(min(max(a_W, 1.0e-4), 0.99))


def initial_tomas_state(scenario) -> TomasState:
    """Build the initial ``TomasState`` from the scenario using the Marianna 'redcircles' distribution.

    Gc is all-zero: gaseous H2SO4 is handed in by the driver each outer step (the gas model owns it).
    Number/mass are per grid cell (``boxvol=BOXVOL_CM3``); T in K, P in Pa (scenario.P is mbar).
    """
    pres_pa = scenario.P * 100.0                       # mbar -> Pa (TOMAS uses Pa)
    Nk_np, Mk_np = _bad.get_initial_state(
        nbins=NBINS, boxvol=BOXVOL_CM3, dist="redcircles",
        to_ambient=True, temp=scenario.T, pres=pres_pa)
    Nk = jnp.asarray(Nk_np, dtype=jnp.float64)
    Mk = jnp.asarray(Mk_np, dtype=jnp.float64)
    xk = tcfg.xk_boundaries()
    Gc = jnp.zeros(tcfg.N_GAS_SPECIES, dtype=jnp.float64)
    return TomasState.create(Nk, Mk, xk, scenario.T, pres_pa, BOXVOL_CM3,
                             Gc=Gc, rh=rh_from_scenario(scenario), alpha=ALPHA_DEFAULT)


def active_processes(switches) -> list[str]:
    """The TOMAS process list for ``make_step`` given the scenario switches (SO2 chem always OMITTED).

    Order is fixed (``_PROCESS_ORDER``); a process is included only if its switch is on. 'so2_chemistry'
    is never included -- the gas model owns sulfur (DECISIONS.md: sulfur ownership).
    """
    flags = {"nucleation": switches.nucleation,
             "coagulation": switches.coagulation,
             "condensation": switches.condensation}
    return [p for p in _PROCESS_ORDER if flags[p]]


def make_microphysics_step(switches):
    """Build the TOMAS ``step_fn(Nk,Mk,Gc,xk,temp,pres,boxvol,rh,alpha,dt,**kw)->(Nk,Mk,Gc)``.

    Uses the Marianna-validated scheme choices; SO2 chemistry is omitted. Returns ``None`` if no
    microphysics process is switched on (the driver then skips the TOMAS step entirely).
    """
    procs = active_processes(switches)
    if not procs:
        return None
    return make_step(procs, cond_method=_COND_METHOD, nucl_scheme=_NUCL_SCHEME,
                     water_scheme=_WATER_SCHEME)
