# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Bridge from the coupled config to the TOMAS sectional-aerosol model (Phase 3).

This is the ONE place the coupling layer reaches into ``tomas_jax``. It builds an initial
``TomasState`` from the scenario (using the Marianna 'redcircles' background distribution) and
constructs the microphysics step with TOMAS's own SO2 chemistry OMITTED -- the gas-phase model
(frank) owns sulfur and hands gaseous H2SO4 to TOMAS via ``Gc[SRTSO4]`` each outer step.

Packaging note (interim, AD-3.1): ``tomas_jax`` is not installed; we add the ``tomas-jax``
git submodule at the repo root (SHA-pinned; ``git submodule update --init`` after clone) — or,
if the submodule is not checked out, a sibling ``../tomas-jax`` checkout — plus its
``experimental_case`` dir (Marianna distribution loader) to ``sys.path`` here, mirroring the
gas-model bridge. Proper packaging: DEFERRED.
"""

from __future__ import annotations

import os
import sys

# The background mode tables live in coupled.backgrounds -- a JAX-free module, so that scenario
# validation can reach them without importing this one. Re-exported here because the paper scripts and
# docs refer to ``tomas_bridge.BACKGROUND_MODES``; this module remains where they are USED.
from .backgrounds import (BACKGROUND_MODES, AMBIENT_BACKGROUNDS,  # noqa: F401  (re-export)
                          MODE_BASES, TABULATED_BACKGROUND, normalize_modes)

# gas model on path first (for the water-activity calc used to set RH) -- model_bridge does the insert.
from . import model_bridge  # noqa: F401  (side effect: puts gas_phase_chemistry on sys.path)
from aerosol import h2so4wp_at  # noqa: E402  (gas model: water activity a_W from T,P,H2O)

_HERE = os.path.dirname(__file__)
_TOMAS = os.path.abspath(os.path.join(_HERE, "..", "tomas-jax"))          # git submodule (preferred)
if not os.path.isdir(os.path.join(_TOMAS, "tomas_jax")):
    _TOMAS = os.path.abspath(os.path.join(_HERE, "..", "..", "tomas-jax"))  # sibling checkout fallback
for _p in (_TOMAS, os.path.join(_TOMAS, "experimental_case")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)   # TOMAS is float64 throughout; match it

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from tomas_jax.core import config as tcfg          # noqa: E402
from tomas_jax.core.state import TomasState        # noqa: E402
from tomas_jax.solvers.condensation import make_step  # noqa: E402
import background_aerosol_distribution as _bad      # noqa: E402  (Marianna dist loader)

# --- exposed constants the driver/props need (single source) ---
SRTSO4 = tcfg.SRTSO4          # sulfate index in Gc (gaseous H2SO4 handoff slot) and Mk
SRTSO2 = tcfg.SRTSO2          # SO2 index in the gas array Gc (TOMAS's own SO2 chem is OFF here)
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


def _grid_for(nbins):
    """Bin boundaries for a resolution study over the SAME diameter range: ratio = 2**(40/nbins),
    so 40 -> 2.0, 80 -> sqrt(2), 160 -> 2**0.25, and the top boundary XK0*2**40 is fixed for all."""
    if nbins == 40:
        return tcfg.xk_boundaries()
    if nbins == 80:
        return tcfg.make_grid_80bin()
    return tcfg.make_grid(nbins, tcfg.XK0, 2.0 ** (40.0 / nbins))


def _seed_lognormal(xk_np, boxvol, modes, temp, pres, ambient=False):
    """Nk [#/cell], Mk [kg/cell] from (multi-)lognormal modes -- mirrors _bad.map_to_grid: integrate
    dN/dlog10Dp over each bin [cm^-3 STP], x STP->ambient x boxvol, Mk = Nk x geometric-mean bin mass
    (sulfate column only, matching the redcircles seed)."""
    from scipy.integrate import quad
    dp_edges = _bad._xk_to_dp_um(np.asarray(xk_np)); logdp = np.log10(dp_edges)
    nbins = len(xk_np) - 1

    def dNdlogDp(x):
        t = 0.0
        for N, Dg, sg in modes:
            s = np.log10(sg)
            t += N / (np.sqrt(2.0 * np.pi) * s) * np.exp(-(x - np.log10(Dg)) ** 2 / (2.0 * s * s))
        return t

    Nk_cm3 = np.array([max(0.0, quad(dNdlogDp, logdp[k], logdp[k + 1], limit=50)[0])
                       for k in range(nbins)])
    # STP dN/dlogDp -> ambient (as redcircles); ambient-specified mode sets skip the factor
    f = 1.0 if ambient else _bad.stp_to_ambient_factor(temp, pres)
    Nk = Nk_cm3 * f * boxvol
    m_mid = np.sqrt(np.asarray(xk_np)[:-1] * np.asarray(xk_np)[1:])
    Mk = np.zeros((nbins, tcfg.ICOMP))
    Mk[:, SRTSO4] = Nk * m_mid
    return Nk, Mk


def initial_tomas_state(scenario) -> TomasState:
    """Build the initial ``TomasState`` from the scenario's background aerosol distribution.

    ``scenario.background_dist`` is either a NAME -- ``"redcircles"`` (Marianna, the tabulated
    loader) or a key of ``BACKGROUND_MODES`` -- or a USER-SUPPLIED list of ``(N, Dg, sigma_g)``
    lognormal modes, in which case ``scenario.background_modes_basis`` ("stp" / "ambient") states
    the number basis, since a bare mode list carries none (see coupled/backgrounds.py).

    Gc is all-zero: gaseous H2SO4 is handed in by the driver each outer step (the gas model owns it).
    Number/mass are per grid cell (``boxvol=BOXVOL_CM3``); T in K, P in Pa (scenario.P is mbar).
    ``scenario.tomas_nbins`` picks the size resolution over the SAME diameter range: 40 (mass-
    doubling), 80 (sqrt(2)), 160 (2^0.25) -- ratio = 2**(40/nbins), so the top boundary is fixed.
    downstream consumers (Mie table, optics, diagnostics) must use the state's own ``xk``.
    """
    nbins = int(getattr(scenario, "tomas_nbins", 40))
    if nbins not in (40, 80, 160):
        raise ValueError(f"tomas_nbins must be 40, 80, or 160, got {nbins}")
    pres_pa = scenario.P * 100.0                       # mbar -> Pa (TOMAS uses Pa)
    xk = _grid_for(nbins)
    bg = getattr(scenario, "background_dist", TABULATED_BACKGROUND)
    if not isinstance(bg, str):
        # user-supplied lognormal modes; the basis is explicit (CoupledScenario enforces it, and it
        # is re-checked here because initial_tomas_state accepts any scenario-shaped object)
        modes = normalize_modes(bg)
        basis = str(getattr(scenario, "background_modes_basis", ""))
        if basis not in MODE_BASES:
            raise ValueError(f"a user-supplied background_dist mode list needs "
                             f"background_modes_basis in {list(MODE_BASES)}, got {basis!r}")
        Nk_np, Mk_np = _seed_lognormal(np.asarray(xk), BOXVOL_CM3, modes,
                                       scenario.T, pres_pa, ambient=basis == "ambient")
    elif bg in BACKGROUND_MODES:
        # seed a (multi-)lognormal background (SABR / CESM) on our grid
        Nk_np, Mk_np = _seed_lognormal(np.asarray(xk), BOXVOL_CM3, BACKGROUND_MODES[bg],
                                       scenario.T, pres_pa, ambient=bg in AMBIENT_BACKGROUNDS)
    elif bg == TABULATED_BACKGROUND:
        if nbins in (40, 80):
            # validated paths: get_initial_state special-cases 80 (sqrt2); 40 uses ratio 2.0.
            Nk_np, Mk_np = _bad.get_initial_state(
                nbins=nbins, boxvol=BOXVOL_CM3, dist="redcircles",
                to_ambient=True, temp=scenario.T, pres=pres_pa)
        else:
            # general resolution: build the redcircles distribution on OUR same-range refined grid
            # (get_initial_state would use ratio 2.0 for nbins!=80 -> wrong diameter range).
            Nk_np, Mk_np, *_ = _bad.map_to_grid(np.asarray(xk), BOXVOL_CM3, dist="redcircles")
            f = _bad.stp_to_ambient_factor(scenario.T, pres_pa)   # STP dN/dlogDp -> ambient
            Nk_np, Mk_np = Nk_np * f, Mk_np * f
    else:
        raise ValueError(f"unknown background_dist {bg!r}; valid: {TABULATED_BACKGROUND!r}, "
                         f"{sorted(BACKGROUND_MODES)}, or a list of (N, Dg, sigma_g) modes")
    Nk = jnp.asarray(Nk_np, dtype=jnp.float64)
    Mk = jnp.asarray(Mk_np, dtype=jnp.float64)
    Gc = jnp.zeros(tcfg.N_GAS_SPECIES, dtype=jnp.float64)
    alpha = float(getattr(scenario, "condensation_alpha", ALPHA_DEFAULT))  # Phase-7 knob
    return TomasState.create(Nk, Mk, xk, scenario.T, pres_pa, BOXVOL_CM3,
                             Gc=Gc, rh=rh_from_scenario(scenario), alpha=alpha)


def active_processes(switches) -> list[str]:
    """The TOMAS process list for ``make_step`` given the scenario switches (SO2 chem always OMITTED).

    Order is fixed (``_PROCESS_ORDER``); a process is included only if its switch is on. 'so2_chemistry'
    is never included -- the gas model owns sulfur (DECISIONS.md: sulfur ownership).
    """
    flags = {"nucleation": switches.nucleation,
             "coagulation": switches.coagulation,
             "condensation": switches.condensation}
    return [p for p in _PROCESS_ORDER if flags[p]]


def make_microphysics_step(switches, ion_pair_rate=0.0, coag_kernel_scale=1.0):
    """Build the TOMAS ``step_fn(Nk,Mk,Gc,xk,temp,pres,boxvol,rh,alpha,dt,**kw)->(Nk,Mk,Gc)``.

    Uses the Marianna-validated scheme choices; SO2 chemistry is omitted. Returns ``None`` if no
    microphysics process is switched on (the driver then skips the TOMAS step entirely).
    ``ion_pair_rate`` [pairs/cm^3/s] is baked in as TOMAS's ``fion`` (Dunne-2016 ion-induced
    nucleation); make_step's own default is 0, which silently turns those channels off -- so the
    scenario value must be passed through here. ``coag_kernel_scale`` (default 1.0) is a free
    multiplier on the coagulation kernel (AD-7.2 sensitivity knob), passed to coag_euler_step."""
    procs = active_processes(switches)
    if not procs:
        return None
    step = make_step(procs, cond_method=_COND_METHOD, nucl_scheme=_NUCL_SCHEME,
                     water_scheme=_WATER_SCHEME)
    fion = float(ion_pair_rate)
    coag_scale = float(coag_kernel_scale)

    def step_fn(*args, **kwargs):
        return step(*args, fion=fion, coag_kernel_scale=coag_scale, **kwargs)
    return step_fn
