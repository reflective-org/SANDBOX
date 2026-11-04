# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Launch a paper-runs-v2 set across N thread-pinned single-core worker processes (work queue).

The launcher lists the set's cases that still need running (``run_v2.case_status`` == 'todo': no
state.npz in the case's output dir; 'done' and 'stale' cases are reported and skipped, the same
stale-IC rule as ``run_v2``), orders them longest-expected-first (``expected_cost``) and hands them
out one at a time: N long-lived workers (``run_v2 serve <set> <k>``, pinned to ONE thread for
OMP/OPENBLAS/MKL/XLA so N workers do not oversubscribe the cores) each pull the next case as soon as
they finish one, so a few slow cases no longer leave the other cores idle at the end. A worker
process serves many cases (JAX compiles once per worker); if it dies, its case is reported and a
fresh worker takes the next case. Re-launching resumes: finished cases are skipped.

Per-worker logs go to ``runs_v2/<set>/worker_<k>.log`` (appended, one header per launch); each
worker appends to ``summary_w<k>.csv``; all ``summary_*.csv`` files are merged into
``runs_v2/<set>/summary.csv`` at the end, and the manifest is written before and after.

Do not launch two sets whose cases overlap (``run_v2.SHARED_OUTPUT``: ens contains base, ens_ctrl
contains ctrl) at the same time: a case that one launch is still running has no state.npz yet, so
the other would start it as well.

Run (from SANDBOX/):  python -m coupled.paper_ensemble.launch_v2 <set> [n_proc]   (default n_proc 8)
"""
import csv
import glob
import math
import os
import queue
import subprocess
import sys
import threading
import time

from coupled.paper_ensemble import run_v2

_SANDBOX = run_v2._SANDBOX

_PIN = {  # one thread per worker so N workers don't fight over cores (as legacy_v1/launch_parallel)
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
}

#: time [s] at which the plume dilution enters the cost proxy (mid-run of the fixed 10-d runs)
_COST_T = 0.5 * run_v2.MAX_DAYS * 86400.0


def expected_cost(spec):
    """Static run-time proxy (larger = slower): log of the mid-run box SO2, i.e. the initial SO2
    (1 t / V0, or the background for a ctrl twin) over V(t)/V0 at mid-run. Slow dilution (along
    shear, low Kz) and a small V0 (peak) keep the box concentrated and the solves stiff; no-injection
    controls are cheapest. Ties: higher nucleation scale first (more particles to evolve)."""
    from config import air_number_density
    from coupled import dilution as dl
    site = spec.site
    if spec.kind == "ctrl":
        so2 = site.bg_so2_ppt * 1e-12 * air_number_density(site.P, site.T)
    else:
        so2 = run_v2.SO2_SPIKE_MOLEC_CM3[spec.v0]
    return (math.log(so2) - math.log(float(dl.volume_ratio(_COST_T, spec.dil))),
            spec.nucleation_rate_scale)


def pending_cases(set_name, cases):
    """[(index, spec)] of the cases still to run, longest-expected-first; 'done' and 'stale' cases
    are reported (as ``run_v2.run_slice`` does) and left out."""
    todo = []
    for i, spec in enumerate(cases):
        status = run_v2.case_status(set_name, spec)
        if status == "todo":
            todo.append((i, spec))
        else:
            run_v2.report_not_todo(i, set_name, spec, status)
    return sorted(todo, key=lambda item: expected_cost(item[1]), reverse=True)


def run_queue(items, n_workers, make_worker):
    """Hand ``items`` out, in order, to ``n_workers`` threads; each thread owns one worker made by
    ``make_worker(k)`` (an object with ``process(item)`` and ``close()``) and pulls the next item as
    soon as it is free. Every item is processed exactly once. Returns [(k, item)] in completion
    order."""
    q = queue.Queue()
    for item in items:
        q.put(item)
    done, lock = [], threading.Lock()

    def loop(k):
        worker = make_worker(k)
        try:
            while True:
                try:
                    item = q.get_nowait()
                except queue.Empty:
                    return
                worker.process(item)
                with lock:
                    done.append((k, item))
        finally:
            worker.close()

    threads = [threading.Thread(target=loop, args=(k,), daemon=True)
               for k in range(max(1, min(int(n_workers), len(items))))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return done


class ServeWorker:
    """One ``run_v2 serve <set> <k>`` subprocess fed case indices on stdin; its output goes to
    worker_<k>.log. Started lazily and restarted if it dies (that case is reported, not retried)."""

    def __init__(self, set_name, k, env, t0, cmd=None):
        self.set_name, self.k, self.env, self.t0 = set_name, k, env, t0
        self.cmd = cmd or [sys.executable, "-m", "coupled.paper_ensemble.run_v2", "serve", set_name,
                           str(k)]
        self.proc = None
        self.log = open(os.path.join(run_v2.set_dir(set_name), f"worker_{k}.log"), "a")
        self.log.write(f"=== launch {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        self.log.flush()

    def _start(self):
        self.proc = subprocess.Popen(
            self.cmd, cwd=_SANDBOX, env=self.env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        print(f"worker {self.k}: pid={self.proc.pid}", flush=True)

    def process(self, item):
        i, spec = item
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        print(f"worker {self.k}: [{i}] {spec.case_id} at +{(time.time() - self.t0) / 60:.1f} min",
              flush=True)
        try:
            self.proc.stdin.write(f"{i}\n")
            self.proc.stdin.flush()
            for line in self.proc.stdout:
                if line.startswith(run_v2.SERVE_DONE):
                    return
                self.log.write(line)
                self.log.flush()
        except BrokenPipeError:
            pass
        code = self.proc.wait()
        self.proc = None
        msg = f"worker {self.k}: process died (exit {code}) during [{i}] {spec.case_id}"
        self.log.write(msg + "\n")
        self.log.flush()
        print(msg + " -- not retried; re-launch to resume it", flush=True)

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            for line in self.proc.stdout:          # drain anything printed after the last case
                self.log.write(line)
            self.proc.wait()
        self.log.close()


def merge_summaries(set_name):
    """Merge summary_*.csv -> summary.csv (union of columns); returns the rows."""
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
    sdir = run_v2.set_dir(set_name)
    os.makedirs(sdir, exist_ok=True)
    if run_v2.EARLY_STOP and any(c.kind == "ctrl" for c in cases):
        missing = [c.case_id for c in cases if c.kind == "ctrl"
                   and not run_v2.is_current(run_v2.paired_base_npz(c), c.site)]
        if missing:
            print(f"WARNING: paired plume runs not finished (or made on another IC) for {missing}; "
                  f"those controls will run the full {run_v2.MAX_DAYS} d instead of the paired "
                  f"length", flush=True)
    path, _ = run_v2.write_manifest(set_name, cases)
    todo = pending_cases(set_name, cases)
    env = dict(os.environ, **_PIN)
    n_proc = max(1, min(int(n_proc), len(todo) or 1))
    t0 = time.time()
    print(f"[{set_name}] {len(todo)} of {len(cases)} cases to run on {n_proc} workers "
          f"(work queue, longest-expected-first); pinned 1 thread/proc; manifest {path}",
          flush=True)
    if todo:
        run_queue(todo, n_proc, lambda k: ServeWorker(set_name, k, env, t0))

    run_v2.write_manifest(set_name, cases)      # refresh
    rows = merge_summaries(set_name)
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"ALL DONE in {(time.time()-t0)/60:.1f} min: {len(rows)} cases logged, {n_ok} succeeded "
          f"-> {os.path.join(sdir, 'summary.csv')}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8)
