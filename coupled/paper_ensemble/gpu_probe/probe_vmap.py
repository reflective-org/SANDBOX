# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""P3 probe: throughput of the TOMAS step vmapped over a batch of B ensemble members.

Loads a representative state (the exact arguments of a real TOMAS call, saved by probe_breakdown.py),
makes B copies with Nk perturbed by U(-10%, +10%) per bin and member (Mk scaled with it, so the
per-particle masses are unchanged), dt drawn per member from U(0.5, 1.5) x the recorded dt, and
fn_scale per member from U(0.5, 2) x the scenario's value, so XLA cannot fold members together.

The step is the production one (``tomas_bridge.make_microphysics_step``, same switches/kwargs as the
member's scenario), vmapped over (Nk, Mk, Gc, dt, fn_scale) with the grid/T/P/V/RH/alpha shared,
then ``jax.jit``-ed. Each timed iteration chains Nk/Mk into the next call and re-sets Gc to the
recorded input (the driver hands TOMAS fresh H2SO4 every micro-step). Reports member-steps/s.

    JAX_PLATFORMS=cuda python -m coupled.paper_ensemble.gpu_probe.probe_vmap \
        coupled/paper_ensemble/runs_gpu/brk_cpu/M1_6h_state.npz --member M1 --batches 1 8 64 256 1024 4096
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

from coupled.tomas_bridge import make_microphysics_step  # noqa: E402
from coupled.paper_ensemble import run_v2 as rv  # noqa: E402
from coupled.paper_ensemble.gpu_probe.bench_member import member_spec  # noqa: E402


def batch_inputs(z, B, seed=0):
    rng = np.random.default_rng(seed)
    f = 1.0 + rng.uniform(-0.1, 0.1, size=(B, z["Nk"].shape[0]))
    Nk = z["Nk"][None, :] * f
    Mk = z["Mk"][None, :, :] * f[:, :, None]
    Gc = np.broadcast_to(z["Gc"], (B,) + z["Gc"].shape).copy()
    dt = float(z["dt"]) * rng.uniform(0.5, 1.5, size=B)
    fns = float(z["fn_scale"]) * rng.uniform(0.5, 2.0, size=B)
    return [jnp.asarray(v, dtype=jnp.float64) for v in (Nk, Mk, Gc, dt, fns)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("state")
    ap.add_argument("--member", default="M1")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 64, 256, 1024, 4096])
    ap.add_argument("--min-time", type=float, default=3.0, help="min timed seconds per batch size")
    ap.add_argument("--min-iters", type=int, default=5)
    a = ap.parse_args(argv)

    sc = rv.build_scenario(member_spec(a.member), days=1)
    step = make_microphysics_step(sc.switches, ion_pair_rate=float(sc.ion_pair_rate),
                                  coag_kernel_scale=float(sc.coag_kernel_scale),
                                  water_every_process=bool(sc.water_every_process),
                                  nuc_org_mass_frac=float(sc.nuc_org_mass_frac),
                                  molar_so4_transfer=bool(sc.molar_so4_transfer))
    z = dict(np.load(a.state))
    shared = [jnp.asarray(z[k], dtype=jnp.float64) for k in ("xk", "temp", "pres", "boxvol", "rh", "alpha")]
    dev = jax.devices()[0]
    print(f"backend={jax.default_backend()} device={dev} state={a.state} dt={float(z['dt']):.4g} "
          f"fn_scale={float(z['fn_scale'])}", flush=True)

    def one(Nk, Mk, Gc, dt, fns):
        return step(Nk, Mk, Gc, *shared, dt, fn_scale=fns)
    batched = jax.jit(jax.vmap(one))

    rows = []
    for B in a.batches:
        try:
            Nk, Mk, Gc, dt, fns = batch_inputs(z, B)
            assert Nk.dtype == jnp.float64
            tc = time.perf_counter()
            out = batched(Nk, Mk, Gc, dt, fns)
            jax.block_until_ready(out)
            compile_s = time.perf_counter() - tc
            assert out[0].dtype == jnp.float64 and out[1].dtype == jnp.float64
            finite = bool(jnp.all(jnp.isfinite(out[0])) & jnp.all(jnp.isfinite(out[1])))
            jax.block_until_ready(batched(Nk, Mk, Gc, dt, fns))   # 2nd call (post-compile warm)
            n, t0 = 0, time.perf_counter()
            while True:
                Nk, Mk, _ = batched(Nk, Mk, Gc, dt, fns)
                n += 1
                if n >= a.min_iters:
                    jax.block_until_ready(Nk)
                    if time.perf_counter() - t0 >= a.min_time:
                        break
            el = time.perf_counter() - t0
            row = dict(B=B, iters=n, s_per_call=el / n, member_steps_per_s=B * n / el,
                       us_per_member_step=1e6 * el / (B * n), first_call_s=compile_s, finite=finite)
        except Exception as e:   # report where batching fails, don't hide it
            row = dict(B=B, error=f"{type(e).__name__}: {e}"[:2000])
        rows.append(row)
        print("VMAP " + json.dumps(row), flush=True)
    print("VMAP_SUMMARY " + json.dumps(dict(backend=jax.default_backend(), device=str(dev), rows=rows)),
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
