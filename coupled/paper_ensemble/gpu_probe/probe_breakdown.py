# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""P2 probe: where does one coupled run spend its time, per outer interval, on the active backend?

Runs a real ``ens`` member (production settings, as bench_member.py) for ``--hours`` of model time and
times, per outer interval: the gas solve, the TUV-x photolysis solve, the TOMAS micro-loop, and inside
the micro-loop the jitted TOMAS step itself (each call followed by ``block_until_ready``; the driver
syncs right after every call anyway, so this adds no extra sync). The rest of the micro-loop is host
overhead (envelope interpolation, the scalar transfer, Python control flow).

The model is NOT edited: the timers are wrappers put on ``coupled.driver``'s module attributes
(``make_frozen_step``, ``_frozen_j_and_heating``, ``micro_consume``) before ``run_coupled`` is called;
``run_coupled`` looks them up at call time. The stepping is unchanged.

Then a micro-benchmark on the LAST real TOMAS call of the run (its exact arguments): N back-to-back
calls of the jitted step, chaining Nk/Mk and re-setting Gc to the recorded input each time,
(a) with no host sync until the end, (b) with the driver's per-step host sync
(``np.asarray(jnp.stack([finite flag, Gc[SO4], sum(Nk)]))``) after each call.

Also saves that call's arguments (the representative mid-run state for the P3 vmap probe) to
``runs_gpu/<out>/<member>_<hours>h_state.npz``.

    JAX_PLATFORMS=cuda python -m coupled.paper_ensemble.gpu_probe.probe_breakdown M1 --hours 6 --out brk_gpu
"""
import argparse
import dataclasses
import json
import os
import sys
import time

import jax

if os.environ.get("JAX_COMPILATION_CACHE_DIR", "") != "off":
    jax.config.update("jax_compilation_cache_dir",
                      os.environ.get("JAX_COMPILATION_CACHE_DIR", os.path.expanduser("~/.cache/jax_coupled")))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import coupled.driver as drv  # noqa: E402
from coupled.paper_ensemble import run_v2 as rv  # noqa: E402
from coupled.paper_ensemble.gpu_probe.bench_member import RUNS_GPU, SETTINGS, member_spec  # noqa: E402
from tomas_jax.core.config import SRTSO4  # noqa: E402

REC = []          # one dict per outer interval
_cur = {}
LAST_CALL = {}    # args of the most recent real TOMAS call


def _new_interval():
    if _cur:
        REC.append(dict(_cur))
    _cur.clear()
    _cur.update(gas_s=0.0, tuvx_s=0.0, micro_s=0.0, tomas_s=0.0, n_calls=0, n_micro=0, t0=np.nan)


def install_timers():
    orig_make = drv.make_frozen_step
    orig_j = drv._frozen_j_and_heating
    orig_micro = drv.micro_consume

    def make_frozen_step(*a, **k):
        step = orig_make(*a, **k)

        def timed(yc, t0, t1, args, **kw):
            t = time.perf_counter()
            out = step(yc, t0, t1, args, **kw)
            jax.block_until_ready(out)
            _cur["gas_s"] += time.perf_counter() - t
            _cur["t0"] = float(t0)
            return out
        return timed

    def frozen_j(*a, **k):
        _new_interval()             # TUV-x is the first thing run_coupled does in every interval
        t = time.perf_counter()
        out = orig_j(*a, **k)
        jax.block_until_ready(out)
        _cur["tuvx_s"] += time.perf_counter() - t
        return out

    def micro_consume(env, t0, t1, tstate, tomas_step, *a, **k):
        def timed_step(*sa, **sk):
            t = time.perf_counter()
            out = tomas_step(*sa, **sk)
            jax.block_until_ready(out)
            _cur["tomas_s"] += time.perf_counter() - t
            _cur["n_calls"] += 1
            LAST_CALL.update(args=sa, kwargs=sk)
            return out
        t = time.perf_counter()
        out = orig_micro(env, t0, t1, tstate, timed_step, *a, **k)
        _cur["micro_s"] += time.perf_counter() - t
        _cur["n_micro"] += out[3]
        return out

    drv.make_frozen_step = make_frozen_step
    drv._frozen_j_and_heating = frozen_j
    drv.micro_consume = micro_consume


def micro_bench(step_fn, n, sync):
    Nk, Mk, Gc_in, xk, temp, pres, boxvol, rh, alpha, dt = LAST_CALL["args"]
    kw = LAST_CALL["kwargs"]
    out = step_fn(Nk, Mk, Gc_in, xk, temp, pres, boxvol, rh, alpha, dt, **kw)   # warm
    jax.block_until_ready(out)
    t = time.perf_counter()
    for _ in range(n):
        Nk, Mk, _gc = step_fn(Nk, Mk, Gc_in, xk, temp, pres, boxvol, rh, alpha, dt, **kw)
        if sync:   # the driver's one transfer per step
            np.asarray(jnp.stack([(jnp.all(jnp.isfinite(Nk)) & jnp.all(jnp.isfinite(Mk))
                                   & jnp.all(jnp.isfinite(_gc))).astype(jnp.float64),
                                  _gc[SRTSO4], jnp.sum(Nk)]))
    jax.block_until_ready((Nk, Mk))
    return (time.perf_counter() - t) / n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("member", choices=["M1", "M2", "M3"])
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--setting", default="prod", choices=sorted(SETTINGS))
    ap.add_argument("--nbench", type=int, default=1000)
    ap.add_argument("--out", default="brk")
    a = ap.parse_args(argv)

    spec = member_spec(a.member)
    days = max(1, int(np.ceil(a.hours / 24.0)))
    sc = dataclasses.replace(rv.build_scenario(spec, days=days), **SETTINGS[a.setting])
    dev = jax.devices()[0]
    print(f"backend={jax.default_backend()} device={dev} x64={jax.config.jax_enable_x64} "
          f"case={spec.case_id} hours={a.hours}", flush=True)
    assert jax.config.jax_enable_x64

    install_timers()
    _new_interval()
    t_stop = a.hours * 3600.0
    n_micro_seen = [0]

    def hook(*r):
        n_micro_seen[0] += 1
    t0 = time.perf_counter()
    res = drv.run_coupled(sc, return_state=True, micro_diagnostics=hook,
                          stop_condition=lambda d: d["t"] >= t_stop - 1e-6)
    wall = time.perf_counter() - t0
    REC.append(dict(_cur))
    tstate = res[-1]
    REC[:] = [r for r in REC if r.get("n_calls", 0) or r.get("gas_s", 0)]

    tot = {k: float(sum(r[k] for r in REC)) for k in ("gas_s", "tuvx_s", "micro_s", "tomas_s", "n_calls",
                                                      "n_micro")}
    n_int = len(REC)
    host_micro = tot["micro_s"] - tot["tomas_s"]
    other = wall - tot["gas_s"] - tot["tuvx_s"] - tot["micro_s"]
    first = REC[0]   # first interval carries the compiles if the cache is cold
    summary = dict(member=a.member, hours=a.hours, backend=jax.default_backend(), device=str(dev),
                   wall_s=round(wall, 2), n_intervals=n_int, n_micro=int(tot["n_micro"]),
                   n_tomas_calls=int(tot["n_calls"]),
                   per_interval_ms=dict(gas=1e3 * tot["gas_s"] / n_int, tuvx=1e3 * tot["tuvx_s"] / n_int,
                                        micro_loop=1e3 * tot["micro_s"] / n_int, other=1e3 * other / n_int),
                   per_tomas_call_ms=dict(step_with_sync=1e3 * tot["tomas_s"] / tot["n_calls"],
                                          host_overhead=1e3 * host_micro / tot["n_calls"]),
                   ms_per_accepted_micro_step=1e3 * wall / max(tot["n_micro"], 1),
                   share=dict(gas=tot["gas_s"] / wall, tuvx=tot["tuvx_s"] / wall,
                              tomas_step=tot["tomas_s"] / wall, micro_host=host_micro / wall,
                              other=other / wall),
                   first_interval_s=dict(gas=first["gas_s"], tuvx=first["tuvx_s"], micro=first["micro_s"]))

    step_fn = drv.make_microphysics_step(sc.switches, ion_pair_rate=float(sc.ion_pair_rate),
                                         coag_kernel_scale=float(sc.coag_kernel_scale),
                                         water_every_process=bool(sc.water_every_process),
                                         nuc_org_mass_frac=float(sc.nuc_org_mass_frac),
                                         molar_so4_transfer=bool(sc.molar_so4_transfer))
    nosync = micro_bench(step_fn, a.nbench, sync=False)
    withsync = micro_bench(step_fn, a.nbench, sync=True)
    summary["micro_bench_ms"] = dict(n=a.nbench, dt_s=float(LAST_CALL["args"][9]), no_sync=1e3 * nosync,
                                     host_sync_each=1e3 * withsync)

    out = os.path.join(RUNS_GPU, a.out)
    os.makedirs(out, exist_ok=True)
    Nk, Mk, Gc_in, xk, temp, pres, boxvol, rh, alpha, dt = (np.asarray(v) for v in LAST_CALL["args"])
    tag = f"{a.member}_{a.hours:g}h"
    np.savez(os.path.join(out, f"{tag}_state.npz"), Nk=Nk, Mk=Mk, Gc=Gc_in, xk=xk, temp=temp, pres=pres,
             boxvol=boxvol, rh=rh, alpha=alpha, dt=dt, fn_scale=float(sc.nucleation_rate_scale),
             end_Nk=np.asarray(tstate.Nk), end_Mk=np.asarray(tstate.Mk))
    np.savez(os.path.join(out, f"{tag}_intervals.npz"),
             **{k: np.array([r[k] for r in REC]) for k in REC[0]})
    print("BREAKDOWN " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    sys.exit(main())
