# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Bitwise check: fast_mnfix == tomas_jax mnfix_jax, alone and inside the full production TOMAS step.

    JAX_PLATFORMS=cuda python -m coupled.paper_ensemble.gpu_probe.test_fast_mnfix STATE.npz [STATE2.npz ...]
    JAX_PLATFORMS=cpu  python -m coupled.paper_ensemble.gpu_probe.test_fast_mnfix STATE.npz

States: the saved real TOMAS inputs (Nk/Mk and end_Nk/end_Mk of probe_breakdown's state files) plus
stress states made from them: per-bin mass scaled by 10^U(-1.5, 1.5) (avg mass far outside the bin, so
phase 2/3 shift a lot), some bins emptied (phase 1), N=0 & M=0 bins (phase-3 edge case), mass above the
top edge and below the bottom edge (phase 2). Exits non-zero on any mismatch.
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np

from coupled import tomas_bridge as tb
from coupled.paper_ensemble import run_v2 as rv
from coupled.paper_ensemble.gpu_probe import fast_mnfix as fm
from coupled.paper_ensemble.gpu_probe.bench_member import member_spec


def stress_states(Nk, Mk, xk, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    nb = Nk.shape[0]
    for i in range(n):
        N = Nk * 10 ** rng.uniform(-0.5, 0.5, nb)
        M = Mk * (10 ** rng.uniform(-1.5, 1.5, nb))[:, None]
        if i % 3 == 0:
            idx = rng.choice(nb, 8, replace=False)
            N[idx] = rng.uniform(0, 2e-5, 8)
        if i % 4 == 1:
            idx = rng.choice(nb, 4, replace=False)
            N[idx] = 0.0
            M[idx] = 0.0
        if i % 5 == 2:
            M[-1] = M[-1] * 1e4
            M[0] = M[0] * 1e-4
        out.append((N, M))
    return out


def main(paths):
    print(f"backend={jax.default_backend()}", flush=True)
    orig = jax.jit(fm.ORIGINAL)
    fast = jax.jit(fm.fast_mnfix)
    bad = 0
    n_states = 0
    for p in paths:
        z = dict(np.load(p))
        xk = jnp.asarray(z["xk"])
        cases = [(z["Nk"], z["Mk"]), (z["end_Nk"], z["end_Mk"])]
        cases += stress_states(z["Nk"], z["Mk"], z["xk"], 300, 0) + stress_states(z["end_Nk"], z["end_Mk"], z["xk"], 300, 1)
        for N, M in cases:
            a = orig(jnp.asarray(N), jnp.asarray(M), xk)
            b = fast(jnp.asarray(N), jnp.asarray(M), xk)
            same = bool(jnp.array_equal(a[0], b[0]) & jnp.array_equal(a[1], b[1]))
            n_states += 1
            if not same:
                bad += 1
                if bad <= 5:
                    print("MISMATCH mnfix:", float(jnp.max(jnp.abs(a[0] - b[0]))), float(jnp.max(jnp.abs(a[1] - b[1]))))
    print(f"mnfix alone: {n_states - bad}/{n_states} bitwise identical", flush=True)

    # full production TOMAS step with original vs fast mnfix (two separately built + traced steps)
    sc = rv.build_scenario(member_spec("M3"), days=1)

    def build():
        return tb.make_microphysics_step(sc.switches, ion_pair_rate=float(sc.ion_pair_rate),
                                         coag_kernel_scale=float(sc.coag_kernel_scale),
                                         water_every_process=bool(sc.water_every_process),
                                         nuc_org_mass_frac=float(sc.nuc_org_mass_frac),
                                         molar_so4_transfer=bool(sc.molar_so4_transfer))
    s_orig = build()
    z = dict(np.load(paths[0]))
    args = [jnp.asarray(z[k]) for k in ("Nk", "Mk", "Gc", "xk", "temp", "pres", "boxvol", "rh", "alpha")]
    ref = []
    dts = [0.01, 0.3, 2.0, 20.0]
    for dt in dts:
        for fns in (0.01, 1.0, 100.0):
            ref.append(s_orig(*args, dt, fn_scale=fns))
    jax.block_until_ready(ref)
    patched = fm.install()
    jax.clear_caches()
    s_fast = build()
    bad_step = 0
    i = 0
    for dt in dts:
        for fns in (0.01, 1.0, 100.0):
            o = s_fast(*args, dt, fn_scale=fns)
            same = all(bool(jnp.array_equal(x, y)) for x, y in zip(ref[i], o))
            if not same:
                bad_step += 1
                print(f"MISMATCH step dt={dt} fns={fns}:",
                      [float(jnp.max(jnp.abs(x - y) / jnp.maximum(jnp.abs(x), 1e-300))) for x, y in zip(ref[i], o)])
            i += 1
    print(f"full TOMAS step: {i - bad_step}/{i} bitwise identical (patched: {patched})", flush=True)
    ok = bad == 0 and bad_step == 0
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
