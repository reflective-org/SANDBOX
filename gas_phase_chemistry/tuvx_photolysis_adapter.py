"""Adapter: TUV-x (JAX) photolysis rate constants for the frank-model box.

Bridges the ``tuvx_photolysis`` package (the JAX port of NCAR TUV-x, validated to machine
precision against the Fortran) to frank-model's reaction table. For a given location/date/time it
computes absolute per-reaction J-values [1/s] at the box altitude and returns them keyed by
frank-model's reaction equation strings, ready to drop into ``Env.j_values``.

Install the port (editable) into the same environment, e.g.::

    pip install -e ..       # from gas_phase_chemistry/, installs this standalone package

With the Lyman-alpha/Schumann-Runge band parameterization ported, O2 photolysis and the deep-UV
reactions (HNO3, N2O5, HNO4) now validate against the Fortran. The only reactions that still fall
back to the reference ``j45 * j_scale`` scaling (see ``FALLBACK_REACTIONS``) are product *branches*
with no TUV-x counterpart -- ClOOCl -> 2 ClO and HNO4 -> NO3 + OH -- which require applying the JPL
branching quantum yields to the (covered) ClOOCl / HNO4 cross sections.
"""

from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path

# make the sibling solar.py importable regardless of the caller's working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

# frank-model reaction equation -> TUV-x reaction name
REACTION_MAP = {
    "ClONO2 -> Cl + NO3": "ClONO2+hv->Cl+NO3",
    "ClONO2 -> ClO + NO2": "ClONO2+hv->ClO+NO2",
    "ClOOCl -> 2 Cl + O2": "ClOOCl+hv->Cl+ClOO",
    "Cl2 -> 2 Cl": "Cl2+hv->Cl+Cl",
    "HOCl -> OH + Cl": "HOCl+hv->HO+Cl",
    "HNO3 -> OH + NO2": "HNO3+hv->OH+NO2",
    "NO3 -> NO2 + O": "NO3+hv->NO2+O(3P)",
    "NO3 -> NO + O2": "NO3+hv->NO+O2",
    "NO2 -> NO + O": "NO2+hv->NO+O(3P)",
    "N2O5 -> NO2 + NO3": "N2O5+hv->NO2+NO3",
    "HNO4 -> NO2 + HO2": "HNO4+hv->HO2+NO2",
    "OClO -> O + ClO": "OClO+hv->Products",
    "BrO -> Br + O": "BrO+hv->Br+O",
    "BrONO2 -> Br + NO3": "BrONO2+hv->Br+NO3",
    "BrONO2 -> BrO + NO2": "BrONO2+hv->BrO+NO2",
    "BrCl -> Br + Cl": "BrCl+hv->Br+Cl",
    "HONO -> OH + NO": "HNO2+hv->OH+NO",
    "HOBr -> OH + Br": "HOBr+hv->OH+Br",
    # O2 photolysis is handled via the Lyman-alpha/Schumann-Runge band parameterization
    "O2 -> 2 O": "O2+hv->O+O",
    # O3 photolysis: the model treats O3 as photolyzing 100% to O(1D) (concs_het.m L145-152),
    # so its single O3 rate maps to the TUV-x O(1D) *channel* J (Matsumi-2002 quantum yield).
    # reactions.py's _k20a/_k20b read this key and apply the opt-mode water split themselves;
    # the direct O(3P) channel is intentionally omitted (a null O3<->O cycle).
    "O3 -> O2 + O1D": "O3+hv->O2+O(1D)",
    # HNO4 and ClOOCl product channels, resolved via JPL branching quantum yields (branching=True):
    # the primary channels are corrected (scaled by ~0.8) and the secondary channels are added.
    "HNO4 -> NO3 + OH": "HNO4+hv->OH+NO3",       # secondary branch (Phi ~0.2)
    "ClOOCl -> 2 ClO": "ClOOCl+hv->ClO+ClO",     # secondary branch (Phi ~0.2)
}

# Every frank-model photolysis reaction now maps to a covered TUV-x channel; nothing falls back to
# the reference j45 * j_scale scaling.
FALLBACK_REACTIONS = set()

# All covered reactions validate against the Fortran (LA/SR bands included); the HNO4/ClOOCl
# branches rest on the JPL branching quantum yields (not Fortran-comparable) times the validated
# cross sections.
APPROXIMATE_REACTIONS = set()

# the no-aerosol config validates 1:1 against the Fortran; the full tuv_5_4 config (with aerosol)
# can be used instead once the aerosol radiator is ported exactly. Paths resolve to the data and
# examples bundled in this standalone package (the parent of gas_phase_chemistry/).
_STANDALONE_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _STANDALONE_ROOT / "examples" / "tuv_5_4_no_aerosol.json"
_TUVX_ROOT = _STANDALONE_ROOT  # bundled data lives at <root>/data


@lru_cache(maxsize=4)
def _calculator(config_path: str, data_root: str):
    from tuvx_photolysis import PhotolysisCalculator

    return PhotolysisCalculator.from_tuvx_json(config_path, data_root=data_root)


@lru_cache(maxsize=1)
def _box_altitude_km(pressure_mbar: float, data_root: str) -> float:
    from tuvx_photolysis import data, geometry

    root = Path(data_root)
    dens = data.load_profile_csv(root / "data/profiles/atmosphere/ussa.dens")
    temp = data.load_profile_csv(root / "data/profiles/atmosphere/ussa.temp")
    return geometry.pressure_to_altitude_km(pressure_mbar, dens, temp)


def _earth_sun_distance(day_of_year: float) -> float:
    """Earth-Sun distance [AU] from day-of-year (TUV-x mean-anomaly formula)."""
    # matches tuvx_photolysis.geometry: mean anomaly ~ proportional to day number
    m = math.radians((357.528 + 0.9856003 * (day_of_year - 1)) % 360.0)
    return 1.00014 - 0.01671 * math.cos(m) - 0.00014 * math.cos(2.0 * m)


# J-values depend only on time (not the chemical state), and a stiff solver evaluates the RHS many
# times at clustered t. Rather than solve the radiation field on every call, solve it on a fixed
# time grid (nodes every TIME_QUANTUM_S) and LINEARLY INTERPOLATE between nodes. Interpolation
# matters: a piecewise-*constant* cache makes J(t) jump every quantum, and the stiff BDF solver's
# step control collapses on those discontinuities (it fails a multi-day run); a continuous
# (piecewise-linear) J(t) integrates cleanly while still costing only ~one solve per quantum.
TIME_QUANTUM_S = 120.0
_J_CACHE: dict = {}


def _compute_j_values(cfg, t_seconds: float) -> dict:
    from solar import solar_zenith_angle

    config_path = str(getattr(cfg, "tuvx_config", _DEFAULT_CONFIG))
    data_root = str(getattr(cfg, "tuvx_data_root", _TUVX_ROOT))

    total_hours = cfg.start_utc_hour + t_seconds / 3600.0
    day_of_year = cfg.day_of_year + total_hours / 24.0
    utc_hour = total_hours % 24.0

    sza = solar_zenith_angle(cfg.latitude, cfg.longitude, day_of_year, utc_hour)
    if sza >= 90.0:
        return {eq: 0.0 for eq in REACTION_MAP}  # night: all mapped reactions off

    calc = _calculator(config_path, data_root)
    altitude = _box_altitude_km(float(cfg.P), data_root)
    esd = _earth_sun_distance(day_of_year)

    # branching=True applies the JPL product-branching quantum yields for HNO4 and ClOOCl
    profile = calc.rate_constants_profile(sza, esd, branching=True)  # {tuvx_name: J[n_levels]}
    import numpy as np

    out = {}
    for eq, tuvx_name in REACTION_MAP.items():
        if tuvx_name in profile:
            out[eq] = float(np.interp(altitude, calc.height_edges_km, profile[tuvx_name]))
    return out


def _j_at_node(cfg, node: int) -> dict:
    """J-values at grid node ``node`` (absolute time = node * TIME_QUANTUM_S), cached."""
    key = (node, round(float(cfg.latitude), 4), round(float(cfg.longitude), 4),
           round(float(cfg.day_of_year), 6), round(float(cfg.P), 4))
    cached = _J_CACHE.get(key)
    if cached is None:
        # model time whose absolute time is node*Q:  t = node*Q - start_utc_hour*3600
        t_model = node * TIME_QUANTUM_S - cfg.start_utc_hour * 3600.0
        cached = _compute_j_values(cfg, t_model)
        _J_CACHE[key] = cached
    return cached


def j_values_for(cfg, t_seconds: float) -> dict:
    """Absolute J-values [1/s] keyed by frank-model equation, for scenario ``cfg`` at time ``t``.

    Solar zenith angle uses frank-model's own ``solar.py`` (keeping the box model self-consistent);
    the radiation field / cross sections / quantum yields come from the validated TUV-x port. Only
    reactions in :data:`REACTION_MAP` are returned; the rest fall through to ``j45 * j_scale``.
    J is solved on a fixed time grid (nodes every :data:`TIME_QUANTUM_S`) and linearly interpolated
    between the two bracketing nodes, so J(t) is continuous (stiff-solver friendly) and cheap.
    """
    abs_t = cfg.start_utc_hour * 3600.0 + t_seconds
    n0 = int(math.floor(abs_t / TIME_QUANTUM_S))
    w = (abs_t - n0 * TIME_QUANTUM_S) / TIME_QUANTUM_S  # in [0, 1)
    j0 = _j_at_node(cfg, n0)
    if w == 0.0:
        return j0
    j1 = _j_at_node(cfg, n0 + 1)
    keys = j0.keys() | j1.keys()
    return {k: (1.0 - w) * j0.get(k, 0.0) + w * j1.get(k, 0.0) for k in keys}
