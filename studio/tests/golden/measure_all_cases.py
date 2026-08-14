"""Re-run the curated Tier-B cases and dump every deviation to JSON, incrementally.

    python -m studio.tests.golden.measure_all_cases <work-dir>

This is the tool behind the numbers in ``REFERENCE_TOLERANCES.md``. It is not a test: it asserts
nothing, so re-measuring never "fails" -- it reports, and a human decides whether the new numbers
are acceptable.

**Incremental on purpose.** Six 10-day cases is ~28 minutes and an interrupted run used to leave
nothing behind; each case now writes its results the moment it finishes, and an already-present run
is reused rather than repeated. Outputs live in the work directory you name, not a temp dir, so they
survive to be re-analysed.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from studio.modelio.execute import run_and_write
from studio.tests.golden.paper_cases import TIER_B_CASES, config_for_case
from studio.tests.golden.tolerances import (
    endpoint_deviations,
    worst_relative_deviation,
)

ARCHIVE = Path(
    "/Users/ali/Documents/GitHub/gas-phase-chemistry/SANDBOX/coupled/paper_ensemble/runs"
)
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = OUT / "deviations.json"
GAS = ("SO2", "SO3", "H2SO4", "OH")
SERIES = ("SA", "radius_cm", "h2so4wp", "particulate_S", "total_n")

results = json.loads(RESULTS.read_text()) if RESULTS.is_file() else {}
for case in TIER_B_CASES:
    if case in results:
        print(f"[skip] {case} already measured", flush=True)
        continue
    state = OUT / case / "state.npz"
    t0 = time.perf_counter()
    if not state.is_file():
        print(f"[run ] {case} ...", flush=True)
        state = run_and_write(config_for_case(case), OUT / case)["state"]
    wall = time.perf_counter() - t0

    with (
        np.load(state, allow_pickle=True) as f,
        np.load(ARCHIVE / case / "state.npz", allow_pickle=True) as a,
    ):
        fresh = {k: f[k] for k in f.files}
        arch = {k: a[k] for k in a.files}
    species = [str(s) for s in arch["species"]]
    entry = {"wall_s": wall, "endpoints": {}, "series": {}, "time_days": arch["t"].tolist()}

    for name in GAS:
        i = species.index(name)
        entry["endpoints"][name] = endpoint_deviations(fresh["x"][:, i], arch["x"][:, i])
        entry["series"][name] = worst_relative_deviation(fresh["x"][:, i], arch["x"][:, i])
        # deviation vs time, floored the same way the harness floors it
        ref = np.abs(arch["x"][:, i])
        peak = float(np.nanmax(ref))
        ok = ref > peak * 1e-6
        dev = np.where(
            ok, np.abs(fresh["x"][:, i] - arch["x"][:, i]) / np.where(ok, ref, 1), np.nan
        )
        entry.setdefault("dev_vs_time", {})[name] = dev.tolist()
    for key in SERIES:
        entry["endpoints"][key] = endpoint_deviations(fresh[key], arch[key])
        entry["series"][key] = worst_relative_deviation(fresh[key], arch[key])
    entry["series"]["dNdlogDp"] = worst_relative_deviation(fresh["dNdlogDp"], arch["dNdlogDp"])
    entry["exact"] = {k: bool(np.array_equal(fresh[k], arch[k])) for k in ("t", "V_ratio", "T")}
    # the near-zero trap, as data: unfloored relative error against series magnitude
    unfloored = {}
    for name in ("O1D", "O", "SO2", "H2SO4"):
        if name in species:
            i = species.index(name)
            ref = np.abs(arch["x"][:, i])
            peak = float(np.nanmax(ref))
            nz = ref > 0
            rel = np.abs(fresh["x"][:, i] - arch["x"][:, i])[nz] / ref[nz]
            unfloored[name] = {
                "peak": peak,
                "worst_unfloored": float(np.max(rel)) if rel.size else 0.0,
                "worst_floored": worst_relative_deviation(fresh["x"][:, i], arch["x"][:, i]),
            }
    entry["near_zero"] = unfloored

    results[case] = entry
    RESULTS.write_text(json.dumps(results, indent=1))
    print(
        f"[done] {case}  {wall:6.1f}s  worst series {max(entry['series'].values()):.2e}", flush=True
    )

print(f"\n{len(results)}/{len(TIER_B_CASES)} cases measured -> {RESULTS}")
