# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Measure the per-quantity deviation between a freshly-run case and the archived ``state.npz``.

**This is a measurement tool, not a test.** It contains no assertions and no tolerances of its own;
it prints what the deviation *is*. The numbers it produced are recorded in
[`REFERENCE_TOLERANCES.md`](REFERENCE_TOLERANCES.md) together with the SHAs they were taken at
(ADR-009: measure before asserting). Tier-B assertions, when written, cite that file.

Usage (from the SANDBOX root, with the studio venv):

    python -m coupled.paper_ensemble.run_ensemble one 121      # produce the fresh run
    python studio/tests/golden/measure_deviation.py 30N_20km__sabr220__D2med__a1p0__nuc1__cg1 \\
        --archive <path-to-archived-runs> [--fresh <path-to-fresh-runs>]

``--archive`` must point at a **read-only** copy of the archived ensemble: those outputs are not
regenerable at their original provenance (ADR-006) and this script never writes to that tree.

Relative-error floor
--------------------
Relative error is evaluated only where the archived series exceeds ``FLOOR_FRAC`` times its own
peak. Without a floor the metric is dominated by night-time O(1e-35) O1D/O values that oscillate
about zero — solver noise on a species whose peak is ~3 molec/cm3. See REFERENCE_TOLERANCES.md,
"The one trap".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

#: Samples below this fraction of the archived series peak are excluded from the relative metric.
FLOOR_FRAC = 1e-6


def _rel_series(fresh: np.ndarray, arch: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """Max relative deviation over the time axis, and the day at which it occurs."""
    peak = np.nanmax(np.abs(arch))
    mask = np.abs(arch) > FLOOR_FRAC * peak
    rel = np.full(arch.shape, np.nan)
    rel[mask] = np.abs(fresh[mask] - arch[mask]) / np.abs(arch[mask])
    i = int(np.nanargmax(rel))
    return float(np.nanmax(rel)), float(t[i] / 86400.0)


def _rel_scalar(fresh: float, arch: float) -> float:
    if arch == 0.0:
        return 0.0 if fresh == 0.0 else float("nan")
    return abs(fresh - arch) / abs(arch)


def compare(arch_npz: Path, fresh_npz: Path) -> None:
    a = np.load(arch_npz, allow_pickle=True)
    f = np.load(fresh_npz, allow_pickle=True)

    if a["t"].shape != f["t"].shape:
        raise ValueError(
            f"time-grid length differs: archived {a['t'].shape} vs fresh {f['t'].shape}"
        )
    t = a["t"]
    print(f"time axis identical: {np.array_equal(a['t'], f['t'])}")
    print(f"species list identical: {list(a['species']) == list(f['species'])}")

    print("\n-- bit-for-bit per stored array --")
    for k in a.files:
        A, F = a[k], f[k]
        eq = A.shape == F.shape and np.array_equal(A, F)
        if eq or A.dtype.kind in "US":
            print(f"  {k}: equal={eq}")
        else:
            frac = 100.0 * (A == F).sum() / A.size
            print(
                f"  {k}: equal=False  bit-identical {frac:.2f}%  "
                f"max|abs diff|={np.nanmax(np.abs(F.astype(float) - A.astype(float))):.3e}"
            )

    M = float(a["M"])
    ix = {s: i for i, s in enumerate(list(a["species"]))}
    so2_a, so2_f = a["x"][:, ix["SO2"]] / M * 1e12, f["x"][:, ix["SO2"]] / M * 1e12
    h_a, h_f = a["x"][:, ix["H2SO4"]] / M * 1e12, f["x"][:, ix["H2SO4"]] / M * 1e12

    rows: list[tuple[str, float, float, float]] = []

    def add(name: str, fv: float, av: float, series: tuple[np.ndarray, np.ndarray]) -> None:
        r, day = _rel_series(series[0], series[1], t)
        rows.append((name, _rel_scalar(fv, av), r, day))

    add("SO2, final [pptv]", float(so2_f[-1]), float(so2_a[-1]), (so2_f, so2_a))
    add("H2SO4, peak [pptv]", float(h_f.max()), float(h_a.max()), (h_f, h_a))
    add("H2SO4, final [pptv]", float(h_f[-1]), float(h_a[-1]), (h_f, h_a))
    add(
        "total N, peak [cm-3]",
        float(f["total_n"].max()),
        float(a["total_n"].max()),
        (f["total_n"], a["total_n"]),
    )
    add(
        "total N, final [cm-3]",
        float(f["total_n"][-1]),
        float(a["total_n"][-1]),
        (f["total_n"], a["total_n"]),
    )
    add(
        "wet SA, peak [um2 cm-3]",
        float(np.nanmax(f["SA"])),
        float(np.nanmax(a["SA"])),
        (f["SA"], a["SA"]),
    )
    add("wet SA, final [um2 cm-3]", float(f["SA"][-1]), float(a["SA"][-1]), (f["SA"], a["SA"]))
    add(
        "particulate S, peak",
        float(np.nanmax(f["particulate_S"])),
        float(np.nanmax(a["particulate_S"])),
        (f["particulate_S"], a["particulate_S"]),
    )
    add(
        "particulate S, final",
        float(f["particulate_S"][-1]),
        float(a["particulate_S"][-1]),
        (f["particulate_S"], a["particulate_S"]),
    )
    add(
        "wet radius, final [cm]",
        float(f["radius_cm"][-1]),
        float(a["radius_cm"][-1]),
        (f["radius_cm"], a["radius_cm"]),
    )
    add(
        "H2SO4 wt%, final",
        float(f["h2so4wp"][-1]),
        float(a["h2so4wp"][-1]),
        (f["h2so4wp"], a["h2so4wp"]),
    )

    print("\n-- per-quantity relative deviation --")
    print(f"{'quantity':30s} {'endpoint':>10s} {'max over t':>12s} {'at day':>8s}")
    for nm, sc, se, day in rows:
        print(f"{nm:30s} {sc:10.2e} {se:12.2e} {day:8.2f}")

    print("\n-- final size distribution (dry bins) --")
    for key in ("n_cm3", "dNdlogDp"):
        av, fv = a[key][-1], f[key][-1]
        mask = np.abs(av) > FLOOR_FRAC * np.abs(av).max()
        rel = np.abs(fv[mask] - av[mask]) / np.abs(av[mask])
        j = int(np.argmax(rel))
        bins = np.where(mask)[0]
        print(
            f"  {key}: max per-bin rel dev {rel.max():.2e} at Dp_dry="
            f"{a['dp_mid_um'][bins[j]]:.4g} um ({mask.sum()}/{len(av)} bins above floor); "
            f"L2 rel {np.linalg.norm(fv - av) / np.linalg.norm(av):.2e}"
        )

    print("\n-- photolysis J --")
    Ja, Jf = a["J"], f["J"]
    mask = np.abs(Ja) > FLOOR_FRAC * np.abs(Ja).max()
    rel = np.abs(Jf[mask] - Ja[mask]) / np.abs(Ja[mask])
    print(f"  max rel dev {rel.max():.2e} over {mask.sum()} entries above floor")

    print("\n-- gas species, worst 5 (floored) --")
    worst = []
    for s, j in ix.items():
        av, fv = a["x"][:, j], f["x"][:, j]
        if np.abs(av).max() == 0.0:
            continue
        r, day = _rel_series(fv, av, t)
        worst.append((r, s, day, float(np.abs(av).max())))
    worst.sort(reverse=True)
    for r, s, day, pk in worst[:5]:
        print(f"  {s:8s} {r:10.2e}  at day {day:6.3f}  (series peak {pk:.3e} molec/cm3)")

    print("\n-- unguarded worst over the whole gas state vector (shows why the floor exists) --")
    xa, xf = a["x"], f["x"]
    nz = np.abs(xa) > 0
    ru = np.zeros_like(xa)
    ru[nz] = np.abs(xf[nz] - xa[nz]) / np.abs(xa[nz])
    i, j = np.unravel_index(int(ru.argmax()), ru.shape)
    sp = list(a["species"])[j]
    print(
        f"  {ru.max():.3e} in {sp} at day {t[i] / 86400.0:.3f}: "
        f"archived {xa[i, j]:.3e} vs fresh {xf[i, j]:.3e} "
        f"(series peak {np.abs(xa[:, j]).max():.3e} molec/cm3)"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("case_id")
    p.add_argument(
        "--archive",
        required=True,
        type=Path,
        help="read-only root holding <case_id>/state.npz for the archived ensemble",
    )
    p.add_argument(
        "--fresh",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "coupled/paper_ensemble/runs",
        help="root holding the freshly-produced <case_id>/state.npz",
    )
    args = p.parse_args()
    print(f"=== {args.case_id} ===")
    compare(args.archive / args.case_id / "state.npz", args.fresh / args.case_id / "state.npz")


if __name__ == "__main__":
    main()
