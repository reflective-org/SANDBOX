# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Per-process latency of the TOMAS step on the active backend (nucleation / coagulation /
condensation alone, and all three as in production), on a saved representative state.

Uses tomas_jax ``make_step`` with exactly the bridge's scheme choices (read from tomas_bridge, not
changed). Prints ms per call for B=1 and the given batch size (vmapped), so we can see which process
carries the GPU's per-call latency.

    JAX_PLATFORMS=cuda python -m coupled.paper_ensemble.gpu_probe.probe_process_latency STATE.npz
"""
import argparse
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from coupled import tomas_bridge as tb
from coupled.paper_ensemble import run_v2 as rv
from coupled.paper_ensemble.gpu_probe.bench_member import member_spec
from coupled.paper_ensemble.gpu_probe.probe_vmap import batch_inputs
from tomas_jax.solvers.condensation import make_step


def timeit(f, args, min_time=2.0):
    jax.block_until_ready(f(*args))
    jax.block_until_ready(f(*args))
    n, t0 = 0, time.perf_counter()
    while True:
        jax.block_until_ready(f(*args))
        n += 1
        if time.perf_counter() - t0 > min_time and n >= 3:
            return (time.perf_counter() - t0) / n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--B", type=int, default=128)
    a = ap.parse_args(argv)
    sc = rv.build_scenario(member_spec("M1"), days=1)
    z = dict(np.load(a.state))
    shared = [jnp.asarray(z[k], dtype=jnp.float64) for k in ("xk", "temp", "pres", "boxvol", "rh", "alpha")]
    kw = dict(fion=float(sc.ion_pair_rate), coag_kernel_scale=float(sc.coag_kernel_scale),
              nuc_org_mass_frac=float(sc.nuc_org_mass_frac))
    for procs in (["nucleation"], ["coagulation"], ["condensation"], tb.active_processes(sc.switches)):
        step = make_step(procs, cond_method=tb._COND_METHOD, nucl_scheme=tb._NUCL_SCHEME,
                         water_scheme=tb._WATER_SCHEME, water_every_process=bool(sc.water_every_process),
                         molar_so4_transfer=bool(sc.molar_so4_transfer))

        def one(Nk, Mk, Gc, dt, fns, step=step):
            return step(Nk, Mk, Gc, *shared, dt, fn_scale=fns, **kw)
        f = jax.jit(jax.vmap(one))
        row = {"procs": "+".join(procs)}
        for B in (1, a.B):
            row[f"ms_B{B}"] = 1e3 * timeit(f, batch_inputs(z, B))
        print("PROC " + json.dumps(row), flush=True)


if __name__ == "__main__":
    sys.exit(main())
