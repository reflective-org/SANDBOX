# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""LEGACY: the v1 Schumann et al. (1998) dilution regimes (D1/D2/D3/D5/burst).

Removed from the coupled model and the paper by PI decision L20 (2026-11-03); the paper uses the
sheared-Gaussian cases in ``coupled.dilution``. Kept here, numerically unchanged (bitwise identical
to ``coupled/dilution.py`` at tag ``paper-v1``), only so Plume Studio keeps working until its own
migration. A ``CoupledScenario`` accepts these regime names only with ``legacy_dilution=True``;
the driver then takes V(t)/V0 and k_dil from this module.

``volume_ratio`` / ``kdil_from_regime`` / ``dilution_case_metadata`` here keep the v1 dispatch: a
Schumann name is evaluated here, any other name is delegated to ``coupled.dilution`` (the
sheared-Gaussian cases), exactly as the v1 ``coupled.dilution`` functions did.
"""

from __future__ import annotations

import numpy as np

from . import dilution as _gaussian

# ---------------------------------------------------------------------------------------------
# Time-varying dilution regimes (ported from tomas-jax experimental_case/run_marianna_dilution.py):
# plume volume expansion V(t)/V0 -> k_dil(t) = d ln V / dt. Two-piece: early Schumann t^0.8 (clamped
# >=1) then a turbulent-growth exponential whose Kz coefficient sets the regime (Schumann et al. 1998).
# D1 'Low Kz' = low vertical diffusivity ~ high-altitude / low-latitude / clean stratosphere.
# ---------------------------------------------------------------------------------------------
_T_BREAK = 1.0e4
_BIG_T = 1.0e12


def _two_piece(k):
    return ((_T_BREAK, ("power", 0.8)), (_BIG_T, ("exp", 1585.0, k, _T_BREAK, 1.5)))


#: V(t)/V0 piecewise segments per regime (matches tomas-jax DILUTIONS).
# "burst" = burst-of-turbulence (Kz=10 m^2/s for 14.2 h): 4-piece Schumann + slow/fast/slow exponential
# growth; the prefactors (1585, 1906, 4.83e4) are chosen so V(t)/V0 is continuous at the breakpoints.
# "D5" (Low Lx, 5.33e-8) is the user's "Very high" regime (its own t_max=509,300 s cap is irrelevant
# for our 10-day runs -- the plume is fully diluted well before then; the exponential simply continues).
DILUTION_REGIMES = {
    "D1": ("Low Kz",  _two_piece(2.811e-9)),
    "D2": ("Med Kz",  _two_piece(8.89e-9)),
    "D3": ("High Kz", _two_piece(2.811e-8)),
    "D5": ("Very high (Low Lx)", _two_piece(5.33e-8)),
    "burst": ("Burst of turbulence", (
        (1.0e4,   ("power", 0.8)),
        (1.728e5, ("exp", 1585.0,  2.811e-9, 1.0e4,   1.5)),
        (2.238e5, ("exp", 1906.0,  2.811e-7, 1.728e5, 1.5)),
        (_BIG_T,  ("exp", 4.83e4,  2.811e-9, 2.238e5, 1.5)),
    )),
}


def _eval_segment(t, seg):
    if seg[0] == "power":
        return np.maximum(1.0, t ** seg[1])
    _kind, A, k, t0, q = seg
    # clamp t-t0 >= 0: np.select evaluates every segment everywhere, so this branch is computed (and
    # discarded) for t < t0 where (t-t0)**q would be NaN. Clamping avoids the spurious warning.
    return A * np.exp(k * np.maximum(t - t0, 0.0) ** q)


def _volume_ratio_schumann(t_seconds, regime):
    segs = DILUTION_REGIMES[regime][1]
    t = np.maximum(np.asarray(t_seconds, dtype=float), 0.0)
    conds, vals, t0 = [], [], 0.0
    for t_end, seg in segs:
        conds.append((t >= t0) & (t < t_end))
        vals.append(_eval_segment(t, seg))
        t0 = t_end
    return np.select(conds, vals, default=vals[-1])


SCHUMANN_REGIMES = tuple(DILUTION_REGIMES)            # ("D1", "D2", "D3", "D5", "burst")


def is_schumann_regime(name):
    """True if ``name`` is one of the legacy Schumann regimes (DILUTION_REGIMES)."""
    return name in DILUTION_REGIMES


def dilution_case_metadata(name):
    """Flat, JSON-serializable description of a regime (Schumann here, else coupled.dilution)."""
    if is_schumann_regime(name):
        label, _segs = DILUTION_REGIMES[name]
        return {"dilution_case": name, "family": "schumann", "label": label}
    return _gaussian.dilution_case_metadata(name)


def volume_ratio(t_seconds, regime):
    """V(t)/V0 for a named regime (scalar or array t): a Schumann regime (DILUTION_REGIMES, continuous
    piecewise), or a sheared-Gaussian case delegated to ``coupled.dilution``."""
    if is_schumann_regime(regime):
        return _volume_ratio_schumann(t_seconds, regime)
    return _gaussian.volume_ratio(t_seconds, regime)


def kdil_from_regime(regime, t0, t1):
    """Interval-average dilution rate [1/s] from the volume expansion: ln(V(t1)/V(t0)) / (t1 - t0)."""
    v0 = float(volume_ratio(t0, regime))
    v1 = float(volume_ratio(t1, regime))
    return max(0.0, np.log(max(v1, 1e-300) / max(v0, 1e-300)) / (t1 - t0))
