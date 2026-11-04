# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Run one v2 ensemble member for a few days and time it, on whatever JAX backend is active.

Portable version of the time-step convergence runner (RUNS_V2.md, L28). It changes nothing in the
model: it builds the real ``ens`` member through ``run_v2.build_scenario`` (so the scenario, IC and
micro-step settings are exactly the production ones) and calls ``run_coupled``.

    JAX_PLATFORMS=cpu  python -m coupled.paper_ensemble.gpu_probe.bench_member M1 --days 1 --out cpu
    JAX_PLATFORMS=cuda python -m coupled.paper_ensemble.gpu_probe.bench_member M1 --days 1 --out gpu

Members (all ``ens`` members, peak V0):
    M1  C1, default microphysics                    (fast case)
    M2  C4a, default microphysics                   (slow dilution, most micro-steps per day late on)
    M3  C1, nucleation rate x100                    (nucleation-heavy)

``--out`` is a folder under ``coupled/paper_ensemble/runs_gpu/`` (git-ignored by ``runs*/``).
``--setting`` defaults to ``prod`` = run_v2.MICRO_SETTINGS (the L28 production settings). Writes
``<out>/<member>_<setting>_<days>d.npz`` and prints one ``RESULT {json}`` line with the timings.
"""
import argparse
import dataclasses
import json
import os
import platform
import sys
import time

import jax

if os.environ.get("JAX_COMPILATION_CACHE_DIR", "") != "off":
    jax.config.update("jax_compilation_cache_dir",
                      os.environ.get("JAX_COMPILATION_CACHE_DIR", os.path.expanduser("~/.cache/jax_coupled")))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

import numpy as np  # noqa: E402

from coupled.driver import run_coupled  # noqa: E402
from coupled.paper_ensemble import run_v2 as rv  # noqa: E402
from coupled.tomas_bridge import _bad  # noqa: E402
from config import IDX  # noqa: E402  (stratchem-jax, put on sys.path by coupled.model_bridge)

RUNS_GPU = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs_gpu")

MEMBERS = {
    "M1": dict(dil="C1", v0="peak"),
    "M2": dict(dil="C4a", v0="peak"),
    "M3": dict(dil="C1", v0="peak", nucleation_rate_scale=100.0),
}
SETTINGS = {
    "prod": dict(rv.MICRO_SETTINGS),                                              # L28: eps 0.01
    "Tm02": dict(rv.MICRO_SETTINGS, micro_eps=0.02),                              # cheaper, ~1% off
    "S0": dict(micro_eps=0.1, micro_cap_s=20.0, micro_floor_s=1e-4, micro_nuc_eps=0.0,
               micro_prod_ref="envelope", micro_handoff="end"),                   # v1 (NOT converged)
}


def member_spec(name):
    mk = dict(MEMBERS[name])
    dil, v0 = mk.pop("dil"), mk.pop("v0")
    matches = [c for c in rv.cases_for("ens") if c.dil == dil and c.v0 == v0 and c.kind == "plume"
               and c.microphysics == {**rv.DEFAULT_MICROPHYSICS, **mk}]
    assert len(matches) == 1, matches
    return matches[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("member", choices=sorted(MEMBERS))
    ap.add_argument("--setting", default="prod", choices=sorted(SETTINGS))
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--out", default="run", help="subfolder of coupled/paper_ensemble/runs_gpu/")
    a = ap.parse_args(argv)

    spec = member_spec(a.member)
    sc = dataclasses.replace(rv.build_scenario(spec, days=a.days), **SETTINGS[a.setting])
    dev = jax.devices()[0]
    print(f"backend={jax.default_backend()} device={dev} x64={jax.config.jax_enable_x64} "
          f"jax={jax.__version__} case={spec.case_id}", flush=True)
    assert jax.config.jax_enable_x64, "the model must run in float64"

    rec, nxt = [], [3600.0]
    t0 = time.perf_counter()

    def hook(*r):
        rec.append(r)
        if r[0] >= nxt[0]:
            print(f"progress t={r[0] / 3600:.2f} h wall={time.perf_counter() - t0:.0f}s n={len(rec)}",
                  flush=True)
            nxt[0] += 3600.0
    t, x, aero, st, sd = run_coupled(sc, return_aerosol=True, return_state=True, return_size_dist=True,
                                     micro_diagnostics=hook)
    wall = time.perf_counter() - t0

    logdp = np.log10(_bad._xk_to_dp_um(np.asarray(st.xk)))
    dp_mid = 10 ** (0.5 * (logdp[:-1] + logdp[1:]))
    out = os.path.join(RUNS_GPU, a.out)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{a.member}_{a.setting}_{a.days}d.npz")
    np.savez_compressed(path, t=t, x=x, h2so4=x[:, IDX["H2SO4"]], SA=aero["SA"],
                        radius_cm=aero["radius_cm"], particulate_S=aero["particulate_S"],
                        V_ratio_applied=aero["V_ratio_applied"], n_cm3=sd["n_cm3"], Dp_m=sd["Dp_m"],
                        dp_mid_um=dp_mid, micro=np.asarray(rec, dtype=float), wall=wall,
                        case_id=spec.case_id, settings=json.dumps(SETTINGS[a.setting]),
                        backend=jax.default_backend(), device=str(dev))
    n = len(rec)
    print("RESULT " + json.dumps(dict(
        member=a.member, setting=a.setting, days=a.days, backend=jax.default_backend(), device=str(dev),
        host=platform.node(), wall_s=round(wall, 1), n_micro=n,
        ms_per_micro_step=round(1e3 * wall / max(n, 1), 3), out=path)), flush=True)


if __name__ == "__main__":
    sys.exit(main())
