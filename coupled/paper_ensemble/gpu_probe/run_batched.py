# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Run v2 ensemble members through the batched GPU driver and compare them with CPU runs.

    # M1 + M3 (both C1) for 1 day; compare with the CPU bench runs runs_gpu/warm_cpu/M{1,3}_prod_1d.npz
    python -m coupled.paper_ensemble.gpu_probe.run_batched m13 --days 1 --out batched_m13

    # every ens + ens_ctrl member of one dilution case (90 plume + 18 ctrl), 10 days; compare each with
    # its finished CPU run in runs_v2/ (if any)
    python -m coupled.paper_ensemble.gpu_probe.run_batched case C2 --out batched_C2

Outputs go to runs_gpu/<out>/<case_id>/state.npz (run_v2.run_one's keys, plus n_micro / backend /
batch info), runs_gpu/<out>/compare.csv and runs_gpu/<out>/run.json. Nothing is written to runs_v2/.
"""
import argparse
import csv
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

from coupled.paper_ensemble.gpu_probe import batched_driver as bd  # noqa: E402
from coupled.paper_ensemble import run_v2 as rv  # noqa: E402
from coupled.paper_ensemble.gpu_probe.bench_member import RUNS_GPU, member_spec  # noqa: E402
from coupled import dilution as dl  # noqa: E402
from coupled.tomas_bridge import _bad  # noqa: E402
from config import IDX, air_number_density  # noqa: E402

KEYS = ["N3", "N10", "N100", "SA", "reff", "pS", "H2SO4", "SO2"]


def series(z):
    n, dp = z["n_cm3"], z["dp_mid_um"]
    v = z["V_ratio_applied"]
    x = z["x"]
    return dict(t=z["t"], N3=n[:, dp >= 0.003].sum(1), N10=n[:, dp >= 0.01].sum(1), N100=n[:, dp >= 0.1].sum(1),
                SA=z["SA"], reff=z["radius_cm"] * 1e4, pS=z["particulate_S"] * v,
                H2SO4=x[:, IDX["H2SO4"]], SO2=x[:, IDX["SO2"]])


def compare(ref_path, new_path):
    """max |rel diff| (%) per quantity over the common output times, and at the last time. Values where
    the reference is below 1e-6 of its own max are skipped (night-time H2SO4 ~ 0)."""
    a, b = series(dict(np.load(ref_path, allow_pickle=True))), series(dict(np.load(new_path, allow_pickle=True)))
    n = min(len(a["t"]), len(b["t"]))
    if not np.allclose(a["t"][:n], b["t"][:n]):
        return {"error": "time grids differ"}
    out = {}
    for k in KEYS:
        ra, rb = a[k][:n], b[k][:n]
        mask = np.abs(ra) > 1e-6 * np.nanmax(np.abs(ra))
        rel = np.abs(rb[mask] - ra[mask]) / np.abs(ra[mask])
        out[k + "_max_pct"] = float(100 * np.nanmax(rel)) if rel.size else 0.0
        out[k + "_end_pct"] = float(100 * (rb[-1] - ra[-1]) / ra[-1]) if ra[-1] != 0 else 0.0
    out["n_times"] = int(n)
    return out


def save_member(path_dir, spec, sc, r, info):
    os.makedirs(path_dir, exist_ok=True)
    t, x, aero, st, sd, jrec = r["t"], r["x"], r["aero"], r["state"], r["sd"], r["jrec"]
    M = air_number_density(sc.P, sc.T)
    V = dl.volume_ratio(t, sc.dilution_regime)
    logdp = np.log10(_bad._xk_to_dp_um(np.asarray(st.xk)))
    dp_mid = 10 ** (0.5 * (logdp[:-1] + logdp[1:]))
    np.savez(os.path.join(path_dir, "state.npz"),
             t=t, x=x, species=list(IDX), M=M,
             SA=aero["SA"], radius_cm=aero["radius_cm"], h2so4wp=aero["h2so4wp"],
             particulate_S=aero["particulate_S"], T=aero["T"],
             n_cm3=sd["n_cm3"], Dp_m=sd["Dp_m"], dp_mid_um=dp_mid,
             dNdlogDp=sd["n_cm3"] / np.diff(logdp)[None, :],
             V_ratio=V, total_n=sd["n_cm3"].sum(axis=1),
             so2_chem_loss_cum=aero["so2_chem_loss_cum"], V_ratio_applied=aero["V_ratio_applied"],
             J_tmid=jrec["t_mid"], J=jrec["J"], J_equations=jrec["equations"],
             ic_file=os.path.relpath(rv.ic_path(spec.site), rv._SANDBOX), ic_sha256=rv.ic_sha256(spec.site),
             micro_settings=rv.micro_settings_json(sc),
             h2so4=x[:, IDX["H2SO4"]], n_micro=r["n_micro"], n_tomas_calls=r["n_tomas_calls"],
             floor_notes=r["floor_notes"], gas_retries=np.asarray(r["gas_retries"]),
             backend=info["backend"], device=info["device"], batch_B=info["B"], wall=info["wall_s"],
             driver="gpu_probe.batched_driver")


def specs_for(mode, dil):
    if mode == "m13":
        return [member_spec("M1"), member_spec("M3")]
    ens = [c for c in rv.cases_for("ens") if c.dil == dil]
    ctrl = [c for c in rv.cases_for("ens_ctrl") if c.dil == dil]
    return ens + ctrl


def cpu_reference(mode, spec, days):
    if mode == "m13":
        name = {member_spec("M1").case_id: "M1", member_spec("M3").case_id: "M3"}[spec.case_id]
        p = os.path.join(RUNS_GPU, "warm_cpu", f"{name}_prod_{days}d.npz")
        return p if os.path.exists(p) else None
    for s in ("ens", "ens_ctrl"):
        try:
            p = os.path.join(rv.case_dir(s, spec), "state.npz")
        except Exception:
            continue
        if os.path.exists(p):
            return p
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["m13", "case"])
    ap.add_argument("dil", nargs="?", default=None)
    ap.add_argument("--days", type=int, default=rv.MAX_DAYS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only the first N members (testing)")
    ap.add_argument("--no-fast-mnfix", action="store_true")
    ap.add_argument("--progress-hours", type=float, default=1.0, help="log every N model hours")
    a = ap.parse_args(argv)

    specs = specs_for(a.mode, a.dil)
    if a.limit:
        specs = specs[:a.limit]
    days = a.days
    scs = [rv.build_scenario(s, days=days) for s in specs]
    print(f"backend={jax.default_backend()} device={jax.devices()[0]} x64={jax.config.jax_enable_x64} "
          f"B={len(specs)} days={days} mode={a.mode} dil={a.dil}", flush=True)
    assert jax.config.jax_enable_x64
    t0 = time.time()
    for i, s in enumerate(specs):
        print(f"MEMBER_INDEX {i} {s.case_id}", flush=True)
    res, info = bd.run_batched(scs, use_fast_mnfix=not a.no_fast_mnfix, progress_every_h=a.progress_hours,
                               log=lambda m: print(m, flush=True))
    print(f"RUN wall={info['wall_s']:.1f}s timing={json.dumps({k: round(v, 1) for k, v in info['timing'].items()})}",
          flush=True)
    out = os.path.join(RUNS_GPU, a.out)
    rows = []
    for spec, sc, r in zip(specs, scs, res):
        d = os.path.join(out, spec.case_id)
        save_member(d, spec, sc, r, info)
        row = dict(case_id=spec.case_id, kind=spec.kind, n_micro=r["n_micro"], floor_notes=r["floor_notes"],
                   gas_retries=len(r["gas_retries"]))
        ref = cpu_reference(a.mode, spec, days)
        if ref is not None:
            row["cpu_ref"] = os.path.relpath(ref, rv._SANDBOX)
            z = np.load(ref, allow_pickle=True)
            if "micro" in z.files:
                row["cpu_n_micro"] = int(z["micro"].shape[0])
            row.update(compare(ref, os.path.join(d, "state.npz")))
        rows.append(row)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(out, "compare.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, "run.json"), "w") as fh:
        json.dump(dict(info, mode=a.mode, dil=a.dil, days=days, total_wall_s=time.time() - t0,
                       members=[s.case_id for s in specs]), fh, indent=1)
    cmp_rows = [r for r in rows if "SA_max_pct" in r]
    if cmp_rows:
        worst = {k: max(r[k] for r in cmp_rows) for k in cmp_rows[0] if k.endswith("_max_pct")}
        print(f"COMPARE {len(cmp_rows)} members vs CPU, worst max|rel diff| %: " +
              ", ".join(f"{k[:-8]} {v:.2e}" for k, v in worst.items()), flush=True)
    for r in rows[:4]:
        print("MEMBER " + json.dumps(r), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
