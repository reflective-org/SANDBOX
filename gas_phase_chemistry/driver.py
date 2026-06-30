"""Driver: set up a scenario, integrate the chemistry over day/night cycles.

Python port of the numeric core of ``src-matlab/runconcs_het.m`` (initial conditions,
the day/night integration loop, and unit conversions). Plotting lives in ``plotting.py``.

The integration mirrors the MATLAB/Octave reference:
  * stiff solver (SciPy ``BDF``, the analogue of MATLAB ``ode15s``);
  * RelTol = 1e-3, AbsTol = 1e-6 per species (with the O and O1D entries following the
    ``opt`` flag, exactly as the MATLAB ``odeset`` call);
  * a tiny initial step so the solver clears the fast O/O1D transient at t=0;
  * each day/night segment integrated on a fixed time grid, so results can be compared
    point-for-point against the Octave fixture.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from config import IDX, N_SPECIES, SPECIES, ModelConfig, air_number_density
from rhs import concs_het

# ---------------------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------------------
# Per-pressure-level mixing ratios (pptv) for the species that vary with altitude,
# transcribed from the P==... blocks in runconcs_het.m. O3 is given in pptv too
# (e.g. 0.43e6 pptv = 430 ppbv).
#
# NOTE: at P=68 the MATLAB source multiplies HNO3 and BrO by 0 (lines 69-70 of
#       runconcs_het.m). That is preserved here as-written -- it zeroes out all bromine
#       in the default scenario. Flagged in the porting plan for confirmation.
_PRESETS_PPT = {
    #        ClO  ClONO2   HCl    NO    NO2     O3        HNO3     BrO
    100: dict(ClO=10, ClONO2=42,  HCl=294, NO=300, NO2=300, O3=0.43e6, HNO3=1100, BrO=2),
    90:  dict(ClO=10, ClONO2=70,  HCl=450, NO=350, NO2=350, O3=0.65e6, HNO3=1840, BrO=3),
    83:  dict(ClO=10, ClONO2=87,  HCl=551, NO=400, NO2=400, O3=0.79e6, HNO3=2360, BrO=3),
    68:  dict(ClO=10, ClONO2=127, HCl=777, NO=450, NO2=450, O3=1.18e6, HNO3=3470 * 0, BrO=3 * 0),
    60:  dict(ClO=10, ClONO2=165, HCl=997, NO=500, NO2=500, O3=1.67e6, HNO3=4550, BrO=6),
}


def initial_concentrations(P: float, M: float, WTR: float) -> np.ndarray:
    """Build the 34-species initial state (molec/cm^3) for pressure level ``P``.

    ``WTR`` is water vapour in ppmv; ``M`` is the air number density.
    """
    if P not in _PRESETS_PPT:
        raise ValueError(f"No initial-condition preset for P={P} mbar; "
                         f"available: {sorted(_PRESETS_PPT)}")
    preset = _PRESETS_PPT[P]

    ppt = 1e-12 * M  # pptv -> molec/cm^3

    # Species common to all pressure levels (pptv unless noted). OH/HO2 gradients are
    # small over 16.5-19 km and the chemistry quickly sets their concentrations.
    conc = {name: 0.0 for name in SPECIES}
    conc.update({name: val * ppt for name, val in preset.items()})
    conc["O2"] = 0.21 * M                 # 21% O2 (not pptv)
    conc["CH4"] = 1.6e6 * ppt             # 1.6 ppmv
    conc["H2O"] = WTR * 1e6 * ppt         # ppmv -> pptv -> molec/cm^3
    conc["N2O5"] = 20 * ppt
    conc["OH"] = 0.5 * ppt
    conc["HO2"] = 3 * ppt
    conc["C2H6"] = 200 * ppt
    conc["SO2"] = (2.4e9 / 2516) * ppt

    return np.array([conc[name] for name in SPECIES])


# ---------------------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------------------
def _abstol(opt: int) -> np.ndarray:
    """Per-species absolute tolerance vector, matching the odeset call in runconcs_het.m.

    For opt==1 (the default workaround) the O and O1D entries are also 1e-6; for opt==0
    they are loosened to 1e-3.
    """
    tol = 1e-3 if opt == 0 else 1e-6
    atol = np.full(N_SPECIES, 1e-6)
    atol[IDX["O"]] = tol
    atol[IDX["O1D"]] = tol
    return atol


def _segment(cfg, t_start, t_end, x0, DT, atol):
    """Integrate one day or night segment on a fixed time grid; return (times, states)."""
    grid = np.arange(t_start, t_end, DT, dtype=float)
    if grid.size == 0 or grid[-1] != t_end:
        grid = np.append(grid, t_end)  # always include the segment endpoint

    sol = solve_ivp(
        concs_het, (t_start, t_end), x0,
        method="BDF", t_eval=grid, args=(cfg,),
        rtol=1e-3, atol=atol, first_step=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed on segment "
                          f"[{t_start}, {t_end}]: {sol.message}")
    return sol.t, sol.y.T  # states as (n_times, n_species)


def integrate(cfg: ModelConfig, x0: np.ndarray, td: float = 14.0, tn: float = 10.0,
              days: int = 5, DT: float = 600.0):
    """Integrate the day/night cycle from initial state ``x0``. Returns ``(time, states)``.

    The first daytime period initialises the run; thereafter the model alternates nighttime
    and daytime segments, carrying the end state of each segment into the next.
    """
    atol = _abstol(cfg.opt)

    # First daytime period.
    cfg.SZA = 1
    t_all, x_all = _segment(cfg, 0.0, td * 3600.0, x0, DT, atol)

    for _ in range(days):
        # nighttime
        cfg.SZA = 0
        t0 = t_all[-1]
        tt, xx = _segment(cfg, t0, t0 + tn * 3600.0, x_all[-1], DT, atol)
        t_all = np.concatenate([t_all, tt[1:]])        # drop duplicated boundary point
        x_all = np.concatenate([x_all, xx[1:]], axis=0)

        # daytime
        cfg.SZA = 1
        t0 = t_all[-1]
        tt, xx = _segment(cfg, t0, t0 + td * 3600.0, x_all[-1], DT, atol)
        t_all = np.concatenate([t_all, tt[1:]])
        x_all = np.concatenate([x_all, xx[1:]], axis=0)

    return t_all, x_all


def integrate_sza(cfg: ModelConfig, x0: np.ndarray, days: int = 5, DT: float = 600.0):
    """Integrate continuously for ``days`` x 24 h with SZA-driven photolysis.

    Used when ``cfg.photolysis == "sza"``: there is no prescribed day/night schedule -- the
    real sun (via the absolute time t in the right-hand side) turns photolysis on and off.
    Integrated in daily chunks for solver robustness; state is carried across chunks.
    """
    atol = _abstol(cfg.opt)
    total = days * 24.0 * 3600.0
    t_all, x_all = None, None
    cur = np.asarray(x0, dtype=float)
    t0 = 0.0
    while t0 < total - 1e-6:
        t1 = min(t0 + 24.0 * 3600.0, total)
        tt, xx = _segment(cfg, t0, t1, cur, DT, atol)
        if t_all is None:
            t_all, x_all = tt, xx
        else:
            t_all = np.concatenate([t_all, tt[1:]])
            x_all = np.concatenate([x_all, xx[1:]], axis=0)
        cur = x_all[-1]
        t0 = t1
    return t_all, x_all


def run(cfg: ModelConfig | None = None, td: float = 14.0, tn: float = 10.0,
        days: int = 5, DT: float = 600.0):
    """Run the model from the pressure-level preset initial conditions.

    ``states`` is (n_times, 34) in molec/cm^3, in ``config.SPECIES`` order. Set
    ``days=2, DT=600`` to reproduce the committed fixture.
    """
    if cfg is None:
        cfg = ModelConfig()
    x0 = initial_concentrations(cfg.P, cfg.M, cfg.WTR)
    return integrate(cfg, x0, td=td, tn=tn, days=days, DT=DT)


def run_scenario(scenario):
    """Run the model described by a ``Scenario`` (see scenario.py). Returns ``(time, states)``.

    Dispatches on the scenario's photolysis mode: "reference" uses the prescribed td/tn
    day/night loop; "sza" integrates continuously with sun-driven photolysis.
    """
    cfg = scenario.to_config()
    x0 = scenario.initial_state()
    if scenario.photolysis == "sza":
        return integrate_sza(cfg, x0, days=scenario.days, DT=scenario.DT)
    return integrate(cfg, x0, td=scenario.td, tn=scenario.tn,
                     days=scenario.days, DT=scenario.DT)


def to_pptv(states: np.ndarray, M: float) -> dict:
    """Convert a (n_times, 34) state array to a dict of pptv time series by species name."""
    return {name: states[:, IDX[name]] / M * 1e12 for name in SPECIES}
