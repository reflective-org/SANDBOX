"""Compare the values inverse_lab.html reports against the source state.npz.

Run harness.js first (it dumps probe.json by driving the page's own JavaScript):

    node coupled/viz/tests_headless/harness.js coupled/viz/inverse_lab.html probe.json
    python coupled/viz/tests_headless/check_page.py [probe.json]

Checks: every sampled value matches the model output at that exact time and lies between its
bracketing model steps; the tracer identity corr = obs x V; unit conversions; the size-distribution
raster within its quantization bound; CSV shape and stability; noise reproducibility.
"""
import json
import os
import sys

import numpy as np

_T = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.abspath(os.path.join(_T, "..", "..", ".."))
SCRATCH = os.path.dirname(os.path.abspath(sys.argv[1])) if len(sys.argv) > 1 else _T
sys.path.insert(0, os.path.join(HERE, "coupled", "viz"))
import bake_plume_dynamics as bp  # noqa: E402

P = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRATCH, "probe.json")))
GPD = P["gpd"]
AVOG, V0 = 6.02214076e23, 1.5e12
fails = []
print(f"cases in page: {len(P['cases'])}  sampling grid: {GPD}/day "
      f"({24 * 60 / GPD:.0f} min)\n")

for key, rec in P["probe"].items():
    site, bg, reg = key.split("|")
    z = np.load(bp._npz_path(os.path.join(HERE, "coupled/paper_ensemble"), site, bg, reg),
                allow_pickle=True)
    ct = np.load(bp._ctrl_npz_path(os.path.join(HERE, "coupled/paper_ensemble"), site, bg, reg),
                 allow_pickle=True)
    sp = [str(s) for s in z["species"]]
    cs = [str(s) for s in ct["species"]]
    n = np.searchsorted(z["t"], rec["daysTotal"] * 86400.0 + 1.0)
    t = np.asarray(z["t"], float)[:n]
    x = np.asarray(z["x"], float)[:n]
    cx = np.asarray(ct["x"], float)[:n]
    M = float(z["M"])
    V = np.asarray(z["V_ratio"], float)[:n]
    partS = np.asarray(z["particulate_S"], float)[:n]
    to_t = V0 * V / AVOG * 64.0 / 1e6
    truth = {
        "V": V,
        "so2_ppt": x[:, sp.index("SO2")] / M * 1e12,
        "oh": x[:, sp.index("OH")],
        "h2so4": x[:, sp.index("H2SO4")],
        "part_s": partS,
        "total_n": np.asarray(z["total_n"], float)[:n],
        "sa": np.asarray(z["SA"], float)[:n],
        "reff_um": np.asarray(z["radius_cm"], float)[:n] * 1e4,
        "so2_t": np.maximum((x[:, sp.index("SO2")] - cx[:, cs.index("SO2")]) * to_t, 0.0),
        "sulf_t": np.maximum(
            (partS - np.asarray(ct["particulate_S"], float)[:n]) * to_t
            + (x[:, sp.index("H2SO4")] - cx[:, cs.index("H2SO4")]) * to_t, 0.0),
    }
    ks = rec["ks"]
    marks = np.array(ks, float) / GPD * 86400.0
    marks = np.minimum(marks, t[-1])
    print(f"=== {key}   lifetime {rec['daysTotal']} d, {rec['nt'] + 1} samples")
    worst = 0.0
    for name, ref in truth.items():
        got = np.array(rec["clean"][name], float)
        # the page must report a value between the bracketing model steps at that exact time
        i = np.clip(np.searchsorted(t, marks), 1, n - 1)
        lo = np.minimum(ref[i - 1], ref[i])
        hi = np.maximum(ref[i - 1], ref[i])
        want = np.interp(marks, t, ref)
        e = np.abs(got - want) / np.maximum(np.abs(want), 1e-300)
        # excursion outside the bracketing model steps, relative to the VALUE (a span-relative
        # measure explodes on float32 round-off wherever two steps are nearly equal)
        outside = np.maximum(lo - got, got - hi) / np.maximum(np.abs(want), 1e-300)
        worst = max(worst, e.max())
        bad = e.max() > 2e-6 or outside.max() > 1e-5
        if bad:
            fails.append(f"{key}/{name}: relerr {e.max():.2e}")
        print(f"  {name:9s} max relerr vs npz {e.max():.2e} | "
              f"outside bracketing steps {outside.max():.1e}{'  <-- FAIL' if bad else ''}")

    # derived quantities the page computes itself
    s0 = rec["snapshot"][str(ks[0])]
    print(f"  t=0: V={s0['V']:.4f} tracer={s0['tracer']:.4f} corr={s0['tracer_corr']:.6f} "
          f"SO2={s0['so2_ppt']:.1f} ppt = {s0['so2_cm3']:.4e} cm-3 "
          f"excess={s0['so2_ex_t']:.4f} t (model {s0['so2_model_t']:.4f} t) side={s0['side']:.1f} m")
    if abs(s0["so2_ex_t"] - 1.0) > 0.02:
        fails.append(f"{key}: SO2 excess at t=0 is {s0['so2_ex_t']:.4f} t, expected ~1")
    if abs(s0["so2_model_t"] - 1.0) > 0.02:
        fails.append(f"{key}: model SO2 budget at t=0 is {s0['so2_model_t']:.4f} t")
    print(f"  tracer identity |corr-1| worst over all {rec['nt'] + 1} samples: "
          f"{rec['tracer_identity_worst']:.2e}")
    if rec["tracer_identity_worst"] > 1e-6:
        fails.append(f"{key}: tracer identity off by {rec['tracer_identity_worst']:.2e}")

    # unit conversions
    for k in ks:
        s = rec["snapshot"][str(k)]
        assert abs(s["so2_cm3"] - s["so2_ppt"] * 1e-12 * M) / max(s["so2_cm3"], 1e-30) < 1e-9, \
            f"{key} k={k}: ppt->cm-3 conversion drifts (baked M precision?)"
        if s["h2so4"] > 0:
            assert abs(s["h2so4_ppt"] - s["h2so4"] / M * 1e12) / (s["h2so4_ppt"]) < 1e-6
        if s["part"] > 0:
            want_ug = s["part"] * 98.0 / AVOG * 1e12
            assert abs(s["part_ug"] - want_ug) / want_ug < 1e-6

    # size distribution: 8-bit raster round-trip
    dN = np.asarray(z["dNdlogDp"], float)[:n]
    errs, ints = [], []
    dlog = np.log10(0.0)  # replaced below
    dp = None
    for k in ks:
        got = np.array(rec["dist"][str(k)], float)
        # log-space interpolation, matching how the bake samples this log-scaled quantity, so
        # what is left is the raster's 8-bit quantization and nothing else
        want = np.array([10 ** np.interp(min(k / GPD * 86400.0, t[-1]), t,
                                        np.log10(np.maximum(dN[:, b], 1e-30)))
                         for b in range(dN.shape[1])])
        m = (want > 1.0) & (got > 0)
        if m.any():
            errs.append(np.abs(got[m] - want[m]) / want[m])
        ints.append((got.sum(), np.interp(min(k / GPD * 86400.0, t[-1]), t,
                                          np.asarray(z["total_n"], float)[:n])))
    e = np.concatenate(errs)
    dlog = np.log10(float(z["dp_mid_um"][1]) / float(z["dp_mid_um"][0]))
    ratio = [a * dlog / b for a, b in ints if b > 0]
    print(f"  size dist: per-bin relerr median {np.median(e):.2%} p99 {np.percentile(e, 99):.2%} "
          f"| integrated N / exact N in [{min(ratio):.3f}, {max(ratio):.3f}]")
    # bound: half a quantization step of the 8-bit log raster over its baked range
    bound = 10 ** ((8.0 - (-1.0)) / 254 / 2) - 1
    if e.max() > bound * 1.05:
        fails.append(f"{key}: dist max relerr {e.max():.1%} exceeds the "
                     f"{bound:.1%} quantization bound")

    # CSV + noise
    ncols = rec["csv_cols"]
    print(f"  CSV: {rec['csv_rows'] - 1} data rows (expect {rec['nt'] + 1}), {ncols} columns, "
          f"stable across calls: {rec['csv_same_twice']}")
    if rec["csv_rows"] - 1 != rec["nt"] + 1:
        fails.append(f"{key}: CSV has {rec['csv_rows']-1} rows, expected {rec['nt']+1}")
    if not rec["csv_same_twice"]:
        fails.append(f"{key}: CSV not stable across calls")
    nz = rec["noise"]
    print(f"  noise: same seed stable {nz['same_seed_stable']}, different seed differs "
          f"{nz['diff_seed_differs']}, obs/truth ratios "
          f"{[round(r, 4) for r in nz['ratios']]}")
    for flag in ("same_seed_stable", "diff_seed_differs", "csv_stable"):
        if not nz[flag]:
            fails.append(f"{key}: noise check '{flag}' failed")
    if not all(0.5 < r < 2.0 and r != 1.0 for r in nz["ratios"]):
        fails.append(f"{key}: noise ratios implausible {nz['ratios']}")
    print()

print("COLUMNS:", P["probe"]["30N_20km|sabr220|D2med"]["csv_header"])
print()
if fails:
    print("FAILURES:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL CHECKS PASSED")
