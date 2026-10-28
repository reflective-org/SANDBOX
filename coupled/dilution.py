# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Dilution: first-order relaxation of the box toward a background state (Phase 6).

Ports TOMAS's dilution formulation ``C_new = C_bg + (C - C_bg)*exp(-k_dil*dt)`` (exact for a constant
k_dil over the interval) and applies it to BOTH the gas species (frank's 36) and the aerosol
(``Nk``/``Mk``/``Gc`` via ``tomas_jax.physics.dilution.dilution_step``). The background is the initial
box state (AD-6.3); the rate is either a constant ``dilution_rate`` [1/s] (AD-6.2) or the interval
average ``k_dil = ln(V(t1)/V(t0))/(t1-t0)`` of a named plume-expansion schedule V(t)/V0. Applied once
per outer interval, operator-split (AD-6.1/6.4).

Two families of V(t)/V0 schedules, both reached through ``volume_ratio`` / ``kdil_from_regime``:

* ``DILUTION_REGIMES`` (D1/D2/D3/D5/burst) -- Schumann et al. (1998) two-piece scaling: early t^0.8
  then a Kz-controlled exponential (ported from tomas-jax ``run_marianna_dilution.py``).
* ``GAUSSIAN_CASES`` (C1..C5a) -- sheared-Gaussian plume cross-section, applied from t = 0. The
  second moments obey ``dszz/dt = 2 Kz``, ``dsyz/dt = S szz``, ``dsyy/dt = 2 S syz`` and
  ``V/V0 = sqrt(det Sigma(t) / det Sigma(0))``; integrated exactly for piecewise-constant Kz
  (``shear_moments``). C3/C3a add a Kz = 10 m^2/s burst at 48 h whose duration is solved so that
  V(7 d) matches C1/C2 respectively.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from . import tomas_bridge as tb   # puts tomas_jax on sys.path

from tomas_jax.physics.dilution import dilution_step  # noqa: E402

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


# ---------------------------------------------------------------------------------------------
# Sheared-Gaussian dilution (PI parameterization). Plume cross-section second moments (y, z) under
# vertical diffusion Kz and vertical wind shear S = du/dz:
#     dszz/dt = 2 Kz,   dsyz/dt = S szz,   dsyy/dt = 2 S syz,   syy(0)=sy0^2, szz(0)=sz0^2, syz(0)=0
# V(t)/V0 = sqrt(det/det0). For constant Kz this is the closed form
#     [1 + 2Kz t/sz0^2 + (2/3) S^2 Kz t^3/sy0^2 + (1/3) S^2 Kz^2 t^4/(sy0^2 sz0^2)]^(1/2).
# sz0 = 10 m tophat -> 10/sqrt(12); "across shear" sy0 = 30 m, "along shear" = 15 km tophat.
# ---------------------------------------------------------------------------------------------
_SHEAR = 2.0e-3                        # S [1/s], all cases
_SIGMA_Z0 = 10.0 / math.sqrt(12.0)     # [m]
_SIGMA_Y0_ACROSS = 30.0                # [m]
_SIGMA_Y0_ALONG = 15000.0 / math.sqrt(12.0)   # [m]
_T_BURST_START = 48 * 3600.0           # [s]
_KZ_BASE = 1.0e-3                      # [m^2/s] background Kz around the burst
_KZ_BURST = 10.0                       # [m^2/s]
_T_MATCH = 7 * 86400.0                 # [s] burst duration chosen so V(t_match) matches C1/C2


@dataclass(frozen=True)
class GaussianCase:
    """One sheared-Gaussian dilution case. ``kz_segments`` = ((t_end [s], Kz [m^2/s]), ...), last t_end inf."""
    label: str
    sigma_y0: float
    sigma_z0: float
    shear: float
    kz_segments: tuple
    t_burst_start: float | None = None
    kz_burst: float | None = None
    tau_burst: float | None = None
    kz_base: float | None = None


def shear_moments(t, sigma_y0, sigma_z0, shear, kz_segments):
    """Exact (syy, syz, szz) [m^2] at time(s) ``t`` [s] for piecewise-constant Kz (t < 0 treated as 0).

    Each segment of length dt at constant K maps (syy, syz, szz) analytically:
    szz += 2K dt; syz += S (szz dt + K dt^2); syy += 2S (syz dt + S (szz dt^2/2 + K dt^3/3)),
    all right-hand sides using the segment-start values. Vectorized over array ``t`` (float64).
    """
    t = np.maximum(np.asarray(t, dtype=np.float64), 0.0)
    S = float(shear)
    syy = np.full_like(t, float(sigma_y0) ** 2)
    syz = np.zeros_like(t)
    szz = np.full_like(t, float(sigma_z0) ** 2)
    t_start = 0.0
    for t_end, K in kz_segments:
        K = float(K)
        dt = np.clip(t - t_start, 0.0, t_end - t_start)   # 0 -> identity map, so apply to all t
        syy = syy + 2.0 * S * (syz * dt + S * (szz * dt ** 2 / 2.0 + K * dt ** 3 / 3.0))
        syz = syz + S * (szz * dt + K * dt ** 2)
        szz = szz + 2.0 * K * dt
        t_start = t_end
    if t_start != np.inf:
        raise ValueError("kz_segments must end with t_end = inf")
    return syy, syz, szz


def _resolve_case(case):
    return GAUSSIAN_CASES[case] if isinstance(case, str) else case


def volume_ratio_gaussian(t, case):
    """V(t)/V0 = sqrt(det Sigma(t)/det Sigma(0)) for a ``GaussianCase`` (or its GAUSSIAN_CASES name)."""
    c = _resolve_case(case)
    syy, syz, szz = shear_moments(t, c.sigma_y0, c.sigma_z0, c.shear, c.kz_segments)
    return np.sqrt((syy * szz - syz ** 2) / (c.sigma_y0 ** 2 * c.sigma_z0 ** 2))


def _burst_segments(t_start, tau, kz_base, kz_burst):
    return ((t_start, kz_base), (t_start + tau, kz_burst), (np.inf, kz_base))


def solve_burst_duration(sigma_y0, sigma_z0, shear, target_D, t_start=_T_BURST_START, kz_base=_KZ_BASE,
                         kz_burst=_KZ_BURST, t_match=_T_MATCH):
    """Burst duration tau [s] such that V(t_match)/V0 == target_D for Kz = kz_base except kz_burst on
    [t_start, t_start + tau]. D(t_match) increases monotonically in tau; raises ValueError if target_D
    lies outside [D(tau=0), D(tau=t_match-t_start)]."""
    def resid(tau):
        segs = _burst_segments(t_start, tau, kz_base, kz_burst)
        return math.log(float(np.sqrt(_det_ratio(t_match, sigma_y0, sigma_z0, shear, segs))) / target_D)

    tau_max = t_match - t_start
    if tau_max <= 0.0:
        raise ValueError(f"t_match ({t_match}) must be after t_start ({t_start})")
    lo, hi = resid(0.0), resid(tau_max)
    if not (lo <= 0.0 <= hi):
        raise ValueError(f"target D={target_D:.6g} unreachable: burst gives D in "
                         f"[{target_D * math.exp(lo):.6g}, {target_D * math.exp(hi):.6g}] at t={t_match} s")
    if lo == 0.0:
        return 0.0
    return brentq(resid, 0.0, tau_max, xtol=1e-12, rtol=4.0 * np.finfo(float).eps, maxiter=500)


def _det_ratio(t, sigma_y0, sigma_z0, shear, segs):
    syy, syz, szz = shear_moments(t, sigma_y0, sigma_z0, shear, segs)
    return (syy * szz - syz ** 2) / (sigma_y0 ** 2 * sigma_z0 ** 2)


def _constant_case(label, sigma_y0, kz):
    return GaussianCase(label, sigma_y0, _SIGMA_Z0, _SHEAR, ((np.inf, kz),))


def _burst_case(label, sigma_y0, matched):
    target = float(volume_ratio_gaussian(_T_MATCH, matched))
    tau = solve_burst_duration(sigma_y0, _SIGMA_Z0, _SHEAR, target)
    return GaussianCase(label, sigma_y0, _SIGMA_Z0, _SHEAR,
                        _burst_segments(_T_BURST_START, tau, _KZ_BASE, _KZ_BURST),
                        t_burst_start=_T_BURST_START, kz_burst=_KZ_BURST, tau_burst=tau, kz_base=_KZ_BASE)


def _build_gaussian_cases():
    c = {
        "C1":  _constant_case("Across shear, Kz=0.01", _SIGMA_Y0_ACROSS, 1.0e-2),
        "C2":  _constant_case("Along shear, Kz=0.01", _SIGMA_Y0_ALONG, 1.0e-2),
        "C4":  _constant_case("Across shear, Kz=0.001", _SIGMA_Y0_ACROSS, 1.0e-3),
        "C4a": _constant_case("Along shear, Kz=0.001", _SIGMA_Y0_ALONG, 1.0e-3),
        "C5":  _constant_case("Across shear, Kz=0.1", _SIGMA_Y0_ACROSS, 1.0e-1),
        "C5a": _constant_case("Along shear, Kz=0.1", _SIGMA_Y0_ALONG, 1.0e-1),
    }
    c["C3"] = _burst_case("Across shear, Kz burst (matches C1 at 7 d)", _SIGMA_Y0_ACROSS, c["C1"])
    c["C3a"] = _burst_case("Along shear, Kz burst (matches C2 at 7 d)", _SIGMA_Y0_ALONG, c["C2"])
    order = ("C1", "C2", "C3", "C3a", "C4", "C4a", "C5", "C5a")
    return {k: c[k] for k in order}


#: Sheared-Gaussian dilution cases (S = 0.002 1/s, sz0 = 10/sqrt(12) m). Kept separate from
#: DILUTION_REGIMES (whose structure studio code iterates over). C3/C3a tau solved at import.
GAUSSIAN_CASES = _build_gaussian_cases()


def is_gaussian_case(name):
    """True if ``name`` is a sheared-Gaussian case (GAUSSIAN_CASES) rather than a Schumann regime."""
    return name in GAUSSIAN_CASES


def _kz_schedule_str(segs):
    parts, t0 = [], 0.0
    for t_end, K in segs:
        parts.append(f"[{t0:.10g},{t_end:.10g}) s: Kz={K:g} m2/s")
        t0 = t_end
    return "; ".join(parts)


def dilution_case_metadata(name):
    """Flat, JSON-serializable description of a dilution regime/case (for run manifests)."""
    if is_gaussian_case(name):
        c = GAUSSIAN_CASES[name]
        return {"dilution_case": name, "family": "gaussian_shear", "label": c.label,
                "sigma_y0_m": c.sigma_y0, "sigma_z0_m": c.sigma_z0, "shear_s-1": c.shear,
                "kz_schedule": _kz_schedule_str(c.kz_segments), "tau_burst_s": c.tau_burst}
    label, _segs = DILUTION_REGIMES[name]
    return {"dilution_case": name, "family": "schumann", "label": label}


def volume_ratio(t_seconds, regime):
    """V(t)/V0 for a named regime (scalar or array t): a Schumann regime (DILUTION_REGIMES, continuous
    piecewise) or a sheared-Gaussian case (GAUSSIAN_CASES)."""
    if is_gaussian_case(regime):
        return volume_ratio_gaussian(t_seconds, regime)
    return _volume_ratio_schumann(t_seconds, regime)


def kdil_from_regime(regime, t0, t1):
    """Interval-average dilution rate [1/s] from the volume expansion: ln(V(t1)/V(t0)) / (t1 - t0)."""
    v0 = float(volume_ratio(t0, regime))
    v1 = float(volume_ratio(t1, regime))
    return max(0.0, np.log(max(v1, 1e-300) / max(v0, 1e-300)) / (t1 - t0))


def dilute_gas(conc, conc_bg, kdil, dt):
    """Relax a gas-species vector toward ``conc_bg`` at rate ``kdil`` [1/s] over ``dt`` [s].

    ``C_new = C_bg + (C - C_bg)*exp(-kdil*dt)`` -- the same analytic exponential as TOMAS's
    ``dilution_step`` (elementwise; works for numpy/JAX arrays).
    """
    decay = np.exp(-kdil * dt)
    return conc_bg + (conc - conc_bg) * decay


def dilute_aerosol(tstate, tstate_bg, kdil, dt):
    """Relax the aerosol (Nk, Mk, Gc) toward the background TomasState via TOMAS ``dilution_step``.

    Returns a new TomasState with diluted Nk/Mk/Gc (other fields unchanged).
    """
    Nk, Mk, Gc = dilution_step(tstate.Nk, tstate.Mk, tstate.Gc, dt, kdil,
                               tstate_bg.Nk, tstate_bg.Mk, tstate_bg.Gc)
    return tstate._replace(Nk=Nk, Mk=Mk, Gc=Gc)
