# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Launch the 120-case start-time ensemble (run_start_time.py) across N thread-pinned workers.
Same scheme as launch_parallel.py / launch_geo.py.

Run (from SANDBOX/):  python -m coupled.paper_ensemble.launch_start_time [n_proc]   (default 10)
"""
import csv
import glob
import math
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_SANDBOX = os.path.abspath(os.path.join(_HERE, "..", ".."))
_OUT = os.path.join(_HERE, "runs_start_time")
_TOTAL = 120

_PIN = {
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
}


def main(n_proc=10):
    os.makedirs(_OUT, exist_ok=True)
    env = dict(os.environ, **_PIN)
    per = math.ceil(_TOTAL / n_proc)
    t0 = time.time()
    procs = []
    for k in range(n_proc):
        lo, hi = k * per, min((k + 1) * per, _TOTAL)
        if lo >= hi:
            break
        logf = open(os.path.join(_OUT, f"worker_{lo}_{hi}.log"), "w")
        p = subprocess.Popen(
            [sys.executable, "-m", "coupled.paper_ensemble.run_start_time",
             "run", str(lo), str(hi)],
            cwd=_SANDBOX, env=env, stdout=logf, stderr=subprocess.STDOUT)
        procs.append((p, lo, hi, logf))
        print(f"worker {k}: cases [{lo},{hi}) pid={p.pid}", flush=True)
    print(f"launched {len(procs)} workers x ~{per} cases each; pinned 1 thread/proc", flush=True)

    for p, lo, hi, logf in procs:
        p.wait(); logf.close()
        print(f"worker [{lo},{hi}) exited code={p.returncode} at +{(time.time()-t0)/60:.1f} min",
              flush=True)

    rows = []
    for f in sorted(glob.glob(os.path.join(_OUT, "summary_*.csv"))):
        rows += list(csv.DictReader(open(f)))
    if rows:
        with open(os.path.join(_OUT, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    n_ok = sum(1 for r in rows if int(r.get("steps", -1)) > 0)
    print(f"ALL DONE in {(time.time()-t0)/60:.1f} min: {len(rows)} cases logged, {n_ok} succeeded "
          f"-> {os.path.join(_OUT, 'summary.csv')}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
