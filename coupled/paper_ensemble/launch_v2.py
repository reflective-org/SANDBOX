# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Launch a paper-runs-v2 set across N thread-pinned single-core worker processes.

Same fan-out as ``launch_parallel.py``: each worker runs a contiguous slice of the set's case
indices via ``run_v2 run <set> <lo> <hi>`` (skip-if-done, so re-launching resumes), pinned to ONE
thread (OMP/OPENBLAS/MKL/XLA) so N workers do not oversubscribe the cores. Per-worker logs go to
``runs_v2/<set>/worker_<lo>_<hi>.log``; the per-slice ``summary_*.csv`` files are merged into
``runs_v2/<set>/summary.csv`` at the end, and the manifest is refreshed.

Run (from SANDBOX/):  python -m coupled.paper_ensemble.launch_v2 <set> [n_proc]   (default n_proc 8)
"""
import csv
import glob
import math
import os
import subprocess
import sys
import time

from coupled.paper_ensemble import run_v2

_SANDBOX = run_v2._SANDBOX

_PIN = {  # single-thread pinning so N workers don't fight over cores (as launch_parallel)
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
}


def merge_summaries(set_name):
    """Merge summary_<lo>_<hi>.csv -> summary.csv (union of columns); returns the rows."""
    sdir = run_v2.set_dir(set_name)
    rows = []
    for f in sorted(glob.glob(os.path.join(sdir, "summary_*.csv"))):
        with open(f, newline="") as fh:
            rows += list(csv.DictReader(fh))
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(sdir, "summary.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    return rows


def main(set_name, n_proc=8):
    cases = run_v2.cases_for(set_name)           # raises NotImplementedError for blocked sets
    total = len(cases)
    sdir = run_v2.set_dir(set_name)
    os.makedirs(sdir, exist_ok=True)
    if any(c.kind == "ctrl" for c in cases):
        missing = [c.dil for c in cases
                   if not run_v2.is_current(run_v2.paired_base_npz(c), c.site)]
        if missing:
            print(f"WARNING: base runs not finished (or made on another IC) for {missing}; "
                  f"those controls will run the "
                  f"full {run_v2.MAX_DAYS} d instead of the paired length", flush=True)
    path, _ = run_v2.write_manifest(set_name, cases)
    env = dict(os.environ, **_PIN)
    n_proc = max(1, min(int(n_proc), total))
    per = math.ceil(total / n_proc)
    t0 = time.time()
    procs = []
    for k in range(n_proc):
        lo, hi = k * per, min((k + 1) * per, total)
        if lo >= hi:
            break
        logf = open(os.path.join(sdir, f"worker_{lo}_{hi}.log"), "w")
        p = subprocess.Popen(
            [sys.executable, "-m", "coupled.paper_ensemble.run_v2", "run", set_name,
             str(lo), str(hi)],
            cwd=_SANDBOX, env=env, stdout=logf, stderr=subprocess.STDOUT)
        procs.append((p, lo, hi, logf))
        print(f"worker {k}: cases [{lo},{hi}) pid={p.pid}", flush=True)
    print(f"[{set_name}] launched {len(procs)} workers x ~{per} cases each; pinned 1 thread/proc; "
          f"manifest {path}", flush=True)

    for p, lo, hi, logf in procs:
        p.wait()
        logf.close()
        print(f"worker [{lo},{hi}) exited code={p.returncode} at +{(time.time()-t0)/60:.1f} min",
              flush=True)

    run_v2.write_manifest(set_name, cases)      # refresh (ctrl days follow finished base runs)
    rows = merge_summaries(set_name)
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"ALL DONE in {(time.time()-t0)/60:.1f} min: {len(rows)} cases logged, {n_ok} succeeded "
          f"-> {os.path.join(sdir, 'summary.csv')}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8)
