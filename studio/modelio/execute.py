# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Running the model and writing what it produced.

Lives at the seam because it calls ``run_coupled``. The runner (``studio/runner``) knows about
processes, limits and lifecycle; this module knows about the model. Neither imports the other's
concerns.

Two things about the model shape the design here, both from BLOCKING-3:

* ``run_coupled`` is a **library call with no ``__main__``**. It returns arrays in memory and writes
  nothing. So the caller writes the npz, and ``studio/cli/run.py`` exists to be the process.
* It **prints** its diagnostics rather than logging them, so stdout is the run's log stream and the
  runner captures it as one.

The npz key set mirrors ``run_ensemble.py:150-156``, which is the canonical one every existing
figure script reads. The size-distribution reduction, though, goes through ``studio.science`` rather
than being inlined a fifth time -- that consolidation is exactly what task 0.5 was for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from studio.modelio.scenario import to_scenario
from studio.modelio.summary import RunSummary, TerminationReason, summarise_state_npz
from studio.resolve import ResolvedConfig
from studio.science import bin_midpoints_um, dn_dlogdp

#: Tolerance for "the run reached the end of its requested duration", in seconds. One output step is
#: 600 s nominal, so half a step is comfortably inside the noise of a completed run and far outside
#: an early stop.
_COMPLETION_TOLERANCE_S = 300.0


def run_and_write(config: ResolvedConfig, out_dir: Path) -> dict[str, Path]:
    """Run the model and write ``state.npz`` and ``summary.json`` into ``out_dir``.

    Returns the paths written. Raises whatever the model raises: a failed run is a failed run, and
    the runner records the traceback from stderr rather than this function inventing a result.
    """
    from coupled.driver import run_coupled
    from coupled.tomas_bridge import _bad

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario = to_scenario(config)

    requested_s = float(config.config.schedule.duration_days) * 86400.0
    stop_condition = _max_sim_time_stop(config)

    times, states, aerosol, final_state, size_dist, photolysis = run_coupled(
        scenario,
        return_aerosol=True,
        return_state=True,
        return_size_dist=True,
        return_photolysis=True,
        stop_condition=stop_condition,
    )

    from config import IDX, air_number_density  # the model's flat modules, via the seam

    edges_um = _bad._xk_to_dp_um(np.asarray(final_state.xk))
    state_path = out_dir / "state.npz"
    np.savez(
        state_path,
        t=times,
        x=states,
        species=list(IDX),
        M=air_number_density(scenario.P, scenario.T),
        SA=aerosol["SA"],
        radius_cm=aerosol["radius_cm"],
        h2so4wp=aerosol["h2so4wp"],
        particulate_S=aerosol["particulate_S"],
        T=aerosol["T"],
        n_cm3=size_dist["n_cm3"],
        Dp_m=size_dist["Dp_m"],
        dp_mid_um=bin_midpoints_um(edges_um),
        dNdlogDp=dn_dlogdp(size_dist["n_cm3"], edges_um),
        V_ratio=_volume_ratio(times, scenario),
        total_n=size_dist["n_cm3"].sum(axis=1),
        J_tmid=photolysis["t_mid"],
        J=photolysis["J"],
        J_equations=photolysis["equations"],
    )

    termination = (
        TerminationReason.COMPLETED
        if float(times[-1]) >= requested_s - _COMPLETION_TOLERANCE_S
        else TerminationReason.TERMINATED_ON_LIMIT
    )
    summary = summarise_state_npz(
        state_path,
        label=config.config.config_hash()[:12],
        config_hash=config.config.config_hash(),
        termination=termination,
    )
    summary_path = summary.write(out_dir / "summary.json")
    return {"state": state_path, "summary": summary_path}


def _max_sim_time_stop(config: ResolvedConfig) -> Any:
    """A ``stop_condition`` enforcing ``termination.max_sim_time_days``, or ``None``.

    **The two-argument shape is deliberate and load-bearing.** Task 0.8 (PR #73) widens the model's
    callback to take a diagnostics dict and dispatches on the callback's DECLARED ARITY:

    ===========================  ==========================================
    ``def stop(t1, wet_SA)``     accepted as legacy, with a DeprecationWarning
    ``def stop(diag)``           accepted as the new dict shape
    ``def stop(*args)``          **TypeError** -- it matches both, so it is ambiguous
    ===========================  ==========================================

    Raising on ``*args`` is the right call by the model: guessing which shape a variadic callback
    wanted would be a silent wrong answer. But it means the obvious "works with either" spelling is
    the one thing that does not, so this stays two-positional -- which works against the model both
    before and after that change.

    Migrating to the dict shape is worth doing once #73 is in ``studio/dev``: it is what makes
    termination criteria on SO2 or particle number possible, which is the reason the callback was
    widened at all. Until then this only needs the simulated time, which is the first argument in
    both shapes.
    """
    limit_days = config.config.termination.max_sim_time_days
    if limit_days is None:
        return None
    limit_s = float(limit_days) * 86400.0

    def stop(t1_seconds: float, wet_surface_area: float) -> bool:
        return float(t1_seconds) >= limit_s

    return stop


def _volume_ratio(times: np.ndarray, scenario: Any) -> np.ndarray:
    """V(t)/V0 for the run, from the model's own tested implementation.

    Reused rather than reimplemented: ``coupled.dilution.volume_ratio`` is tested, and a second copy
    in ``studio/science`` is exactly what task 0.5 removed elsewhere.
    """
    from coupled import dilution

    return dilution.volume_ratio(times, scenario.dilution_regime)


__all__ = ["RunSummary", "run_and_write"]
