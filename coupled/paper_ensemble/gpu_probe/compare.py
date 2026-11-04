# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Compare two bench_member.py outputs (e.g. GPU vs CPU) on the quantities the paper reports.

    python -m coupled.paper_ensemble.gpu_probe.compare \
        coupled/paper_ensemble/runs_gpu/cpu/M1_prod_1d.npz coupled/paper_ensemble/runs_gpu/gpu/M1_prod_1d.npz

Prints the signed relative difference (B - A) / A at each whole day and the max |diff| over the run
for N>3 nm, N>10 nm, N>100 nm, wet surface area, effective radius, particulate S x V/V0 and gas
H2SO4 (same quantities as the L28 convergence test; the accuracy budget there is 1%).
"""
import sys

import numpy as np

KEYS = ["N3", "N10", "N100", "SA", "reff", "pS", "H2SO4"]


def series(z):
    n, dp, v = z["n_cm3"], z["dp_mid_um"], z["V_ratio_applied"]
    return dict(t=z["t"], N3=n[:, dp >= 0.003].sum(1), N10=n[:, dp >= 0.01].sum(1),
                N100=n[:, dp >= 0.1].sum(1), SA=z["SA"], reff=z["radius_cm"] * 1e4,
                pS=z["particulate_S"] * v, H2SO4=z["h2so4"])


def main(a_path, b_path):
    za, zb = np.load(a_path), np.load(b_path)
    a, b = series(za), series(zb)
    t_end = min(a["t"][-1], b["t"][-1])
    hours = [h for h in range(24, int(t_end / 3600) + 1, 24)] or [int(t_end / 3600)]
    print(f"A = {a_path} ({za['backend']}, {float(za['wall']):.0f} s, {za['micro'].shape[0]} micro-steps)")
    print(f"B = {b_path} ({zb['backend']}, {float(zb['wall']):.0f} s, {zb['micro'].shape[0]} micro-steps)")
    print("| hour | " + " | ".join(k + " %" for k in KEYS) + " |")
    print("|---|" + "---|" * len(KEYS))
    for h in hours:
        i = np.argmin(abs(a["t"] - h * 3600))
        j = np.argmin(abs(b["t"] - h * 3600))
        print(f"| {h} | " + " | ".join(f"{100 * (b[k][j] - a[k][i]) / a[k][i]:+.2e}" for k in KEYS) + " |")
    # max over the common output times (both runs save on the same outer-interval grid)
    n = min(len(a["t"]), len(b["t"]))
    if np.allclose(a["t"][:n], b["t"][:n]):
        # skip output times where A is exactly 0 (e.g. gas H2SO4 at t = 0); near-zero night-time
        # H2SO4 can still give large relative values: check the per-day rows before worrying
        worst = {k: float(np.nanmax(np.abs((b[k][:n] - a[k][:n]) / np.where(a[k][:n] != 0, a[k][:n], np.nan))))
                 for k in KEYS}
        print("max |rel diff| over all output times (A != 0): "
              + ", ".join(f"{k} {100 * v:.2e}%" for k, v in worst.items()))
    else:
        print("output time grids differ; per-day rows only")


if __name__ == "__main__":
    main(*sys.argv[1:3])
