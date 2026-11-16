# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Isolate WHY Kvaerno5 stalls at day 12.139 while (a) SciPy BDF and (b) the standalone
gas model integrate the same state fine.

Stage CAPTURE: rerun the 14-day case with the BDF fallback stubbed to dump the EXACT
solver inputs of the failing interval (y0, t0, t1, the full args dict, the save_ts grid,
atol) to coupled/analyses/day12_wall/interval_capture.npz, then abort.

Stage TEST: integrate that single 600 s interval under a matrix of configurations, toggling
each coupled-specific ingredient (fail-fast max_steps=20000; a healthy solve needs <<1000):
  exact        -- coupled config verbatim (jit_grid + LU forward_only + save_ts + atol vec)
  rootfinder   -- gradient-safe root finder instead of LU (forward_only=False)
  endpoint     -- no save_ts grid (endpoint only), LU kept
  no_het       -- default aerosol args (SA tiny, default radius/wt%) -> 'gas chemistry alone'
  loose_atol   -- atol 1e-4 (vs 1e-6)
  loose_rtol   -- rtol 1e-2 (vs 1e-3)
  bdf          -- SciPy BDF reference (expected OK)

Run (from SANDBOX/):
  python -m coupled.paper_ensemble.debug_day12_isolation capture
  python -m coupled.paper_ensemble.debug_day12_isolation test
"""
import dataclasses
import os
import sys

import numpy as np

_OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "analyses", "day12_wall"))
_CAP = os.path.join(_OUT, "interval_capture.npz")
_CASE = "30N_20km__sabr220__D2med__a1p0__nuc1__cg1"


class _Stop(RuntimeError):
    pass


def capture():
    from coupled.paper_ensemble import run_ensemble as re_mod
    from coupled import driver
    from coupled.driver import run_coupled, _abstol

    def _dump(gas_vf, y0, t0, t1, args, ts_grid, atol, h2so4_idx):
        np.savez(_CAP, y0=np.asarray(y0), t0=float(t0), t1=float(t1),
                 ts_grid=np.asarray(ts_grid), atol=np.asarray(atol),
                 **{f"arg_{k}": np.asarray(v) for k, v in args.items()})
        raise _Stop(f"captured failing interval [{t0:.0f},{t1:.0f}]s -> {_CAP}")

    driver._bdf_gas_solve = _dump
    sc = dataclasses.replace(re_mod.build_scenario(dict(re_mod.all_cases())[_CASE]), days=14)
    try:
        run_coupled(sc, return_aerosol=True, return_state=True, return_size_dist=True)
        print("NO STALL in 14 days -- nothing captured")
    except _Stop as e:
        print(e)


def test():
    from coupled import driver  # sys.path side effects
    import jax.numpy as jnp
    from jaxmodel.model import make_frozen_step
    from scipy.integrate import solve_ivp
    from jaxmodel.model import make_frozen_vf

    z = np.load(_CAP, allow_pickle=True)
    y0 = jnp.asarray(z["y0"])
    t0, t1 = float(z["t0"]), float(z["t1"])
    ts_grid = jnp.asarray(z["ts_grid"])
    atol_vec = jnp.asarray(z["atol"])
    args = {k[4:]: (float(z[k]) if z[k].ndim == 0 else jnp.asarray(z[k]))
            for k in z.files if k.startswith("arg_")}
    OPT = 1
    MS = 20000                     # fail-fast cap; healthy solves need only ~1e2-1e3 steps

    def attempt(tag, *, forward_only, grid, use_args, atol, rtol=1e-3):
        try:
            step = make_frozen_step(OPT, atol=atol, rtol=rtol, max_steps=MS,
                                    jit_grid=grid, forward_only=forward_only)
            if grid:
                sol = step(y0, t0, t1, use_args, save_ts=ts_grid)
                y_end = np.asarray(sol.ys[-1])
            else:
                y_end = np.asarray(step(y0, t0, t1, use_args))
            ok = np.all(np.isfinite(y_end))
            print(f"  {tag:12s} {'OK' if ok else 'NONFINITE'}   H2SO4(t1)={y_end[35]:.4g}",
                  flush=True)
        except Exception as e:
            print(f"  {tag:12s} STALL ({type(e).__name__}: {str(e)[:80]})", flush=True)

    print(f"failing interval [{t0:.0f},{t1:.0f}] s = day {t0/86400:.4f}; "
          f"SA={args['SA']:.4g} r={args['particle_radius']:.4g} wt%={args['h2so4wp']:.4g}")
    attempt("exact",      forward_only=True,  grid=True,  use_args=args, atol=atol_vec)
    attempt("rootfinder", forward_only=False, grid=True,  use_args=args, atol=atol_vec)
    attempt("endpoint",   forward_only=True,  grid=False, use_args=args, atol=atol_vec)
    args_nohet = {**args, "SA": 1e-12, "particle_radius": 0.1e-4, "h2so4wp": 65.0}
    attempt("no_het",     forward_only=True,  grid=True,  use_args=args_nohet, atol=atol_vec)
    attempt("loose_atol", forward_only=True,  grid=True,  use_args=args, atol=1e-4)
    attempt("loose_rtol", forward_only=True,  grid=True,  use_args=args, atol=atol_vec,
            rtol=1e-2)

    vf = make_frozen_vf(OPT)
    rhs = lambda t, y: np.asarray(vf(t, jnp.asarray(y), args))
    sol = solve_ivp(rhs, (t0, t1), np.asarray(y0), method="BDF", rtol=1e-3,
                    atol=np.asarray(atol_vec), first_step=1e-10)
    print(f"  {'bdf':12s} {'OK' if sol.success else 'FAIL'}   nsteps={sol.t.size}")


if __name__ == "__main__":
    (capture if "capture" in sys.argv else test)()
