"""The chemistry right-hand side dC/dt.

This delegates to the declarative mechanism in ``reactions.py``: the reactions, their rates,
and the stoichiometry all live there as a readable table, and ``Mechanism.dCdt`` assembles
dC/dt = S @ rates. This module wires the scenario (``cfg``) and the photolysis state into
that machinery.

Photolysis enters through ``Env.j_scale``:
  * ``cfg.photolysis == "reference"`` -- the validated MATLAB behaviour: daytime
    (cfg.SZA == 1) uses the tabulated 45-deg J-values (j_scale = 1); night switches them off
    (j_scale = 0).
  * ``cfg.photolysis == "sza"`` -- j_scale follows the real solar zenith angle computed from
    the location/date and the absolute time t (normalized-cosine scaling, -> 0 below horizon).
  * ``cfg.photolysis == "tuvx"`` -- absolute per-reaction J-values from the TUV-x (JAX) port at
    the box altitude (see ``tuvx_photolysis_adapter``); reactions the port does not cover fall
    back to the SZA-scaled reference values via j_scale.
"""

from __future__ import annotations

import numpy as np

from reactions import MECHANISM, build_env
from solar import cos_solar_zenith, photolysis_scale


def _j_scale(t: float, cfg) -> float:
    """Photolysis scaling factor at model time ``t`` (seconds since the run start)."""
    if cfg.photolysis in ("sza", "tuvx"):
        total_hours = cfg.start_utc_hour + t / 3600.0
        day_of_year = cfg.day_of_year + total_hours / 24.0   # advances over a multi-day run
        utc_hour = total_hours % 24.0
        cosz = cos_solar_zenith(cfg.latitude, cfg.longitude, day_of_year, utc_hour)
        return photolysis_scale(cosz)
    # reference mode: fixed 45-deg J-values, on by day / off by night
    return 1.0 if cfg.SZA == 1 else 0.0


def concs_het(t: float, x, cfg) -> np.ndarray:
    """Return dC/dt (length-34 array) for state ``x`` under scenario ``cfg``."""
    j_values = None
    if cfg.photolysis == "tuvx":
        from tuvx_photolysis_adapter import j_values_for

        j_values = j_values_for(cfg, t)  # absolute J for the covered reactions; rest use j_scale
    env = build_env(cfg, x, _j_scale(t, cfg), j_values=j_values)
    cfg.count += 1
    return MECHANISM.dCdt(env, np.asarray(x, dtype=float))
