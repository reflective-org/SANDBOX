# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""launch_v2 work queue: pending-case selection, priority and dispatch (no simulation spawned)."""

import io
import os
import threading
import time

import numpy as np
import pytest

from coupled.paper_ensemble import launch_v2 as lv
from coupled.paper_ensemble import run_v2 as rv


@pytest.fixture(autouse=True)
def _tmp_out(tmp_path, monkeypatch):
    monkeypatch.setattr(rv, "_OUT", str(tmp_path / "runs_v2"))   # never touch real run outputs


def _need_ic():
    if not os.path.exists(rv.ic_path(rv.SITE_30N)):
        pytest.skip("30N IC json missing")


def _write_npz(set_name, spec, sha):
    npz = os.path.join(rv.case_dir(set_name, spec), "state.npz")
    os.makedirs(os.path.dirname(npz), exist_ok=True)
    if sha is None:
        np.savez(npz, t=np.array([0.0]))                       # stale: no IC hash recorded
    else:
        np.savez(npz, t=np.array([0.0]), ic_sha256=sha,
                 so2_chem_loss_cum=np.array([0.0]), V_ratio_applied=np.array([1.0]), micro_settings=rv.micro_settings_json())


class _FakeWorker:
    def __init__(self, k, log, delay=0.0):
        self.k, self.log, self.delay, self.closed = k, log, delay, False

    def process(self, item):
        time.sleep(self.delay)
        self.log.append((self.k, item))

    def close(self):
        self.closed = True


def test_run_queue_hands_out_every_item_exactly_once():
    items = list(range(50))
    log, workers, lock = [], [], threading.Lock()

    def make(k):
        w = _FakeWorker(k, log, delay=0.001 * (k + 1))          # uneven workers
        with lock:
            workers.append(w)
        return w

    done = lv.run_queue(items, 4, make)
    assert sorted(i for _k, i in log) == items                   # each item once
    assert sorted(i for _k, i in done) == items
    assert len(workers) == 4 and all(w.closed for w in workers)
    assert len({k for k, _i in log}) == 4                        # all workers pulled work
    # a worker takes the next item in queue order: per worker, items arrive increasing
    for k in range(4):
        mine = [i for kk, i in log if kk == k]
        assert mine == sorted(mine)


def test_run_queue_single_worker_keeps_order_and_caps_workers():
    log = []
    lv.run_queue(["a", "b", "c"], 1, lambda k: _FakeWorker(k, log))
    assert [i for _k, i in log] == ["a", "b", "c"]
    made = []
    lv.run_queue([1, 2], 8, lambda k: made.append(k) or _FakeWorker(k, []))
    assert sorted(made) == [0, 1]                                # no more workers than items


def test_pending_cases_skips_done_and_stale_and_queues_the_rest_once(capsys):
    _need_ic()
    cases = rv.cases_for("ens")
    sha = rv.ic_sha256(rv.SITE_30N)
    done = {0, 7, 300, 719}
    stale = {5, 400}
    for i in done:
        _write_npz("ens", cases[i], sha)
    for i in stale:
        _write_npz("ens", cases[i], None)
    base0 = rv.cases_for("base")[1]                              # finished in the base set
    _write_npz("base", base0, sha)
    i_base0 = [c.case_id for c in cases].index(base0.case_id)
    todo = lv.pending_cases("ens", cases)
    idx = [i for i, _s in todo]
    expect = set(range(720)) - done - stale - {i_base0}
    assert sorted(idx) == sorted(expect) and len(idx) == len(set(idx))
    out = capsys.readouterr().out
    assert out.count("STALE") == len(stale) and out.count("SKIP") == len(done) + 1
    # dispatch them: workers receive exactly the not-done cases, once each
    log = []
    lv.run_queue(todo, 6, lambda k: _FakeWorker(k, log))
    assert sorted(i for _k, (i, _s) in log) == sorted(expect)


def test_priority_longest_expected_first():
    cost = lv.expected_cost
    s = rv.CaseSpec
    assert cost(s("C4a")) > cost(s("C4")) > cost(s("C5"))       # along-shear / low Kz first
    assert cost(s("C2", v0="peak")) > cost(s("C2")) > cost(s("C2", v0="c2sig"))
    assert cost(s("C1")) > cost(s("C1", kind="ctrl"))            # injected before controls
    assert cost(s("C1", nucleation_rate_scale=100.0)) > cost(s("C1"))
    order = sorted(rv.cases_for("ens") + rv.cases_for("ens_ctrl"), key=cost, reverse=True)
    assert order[0].dil == "C4a" and order[0].v0 == "peak"
    assert all(c.kind == "ctrl" for c in order[720:])


def test_serve_runs_each_index_and_signals_done(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(rv, "run_slice", lambda name, cases, lo, hi, summary_name=None:
                        calls.append((name, lo, hi, summary_name)))
    rv.serve("ens", [], 3, stdin=io.StringIO("4\n\n9\n"))
    assert calls == [("ens", 4, 5, "summary_w3.csv"), ("ens", 9, 10, "summary_w3.csv")]
    out = capsys.readouterr().out.splitlines()
    assert out == [f"{rv.SERVE_DONE} 4", f"{rv.SERVE_DONE} 9"]


_FAKE_SERVE = r"""
import sys
for line in sys.stdin:
    i = int(line)
    print(f"running {i}", flush=True)
    if i == 13:
        sys.exit(3)                                    # dies mid-case
    print("@@run_v2-done", i, flush=True)
print("bye", flush=True)
"""


def test_serve_worker_protocol_logs_and_restarts_after_death(tmp_path):
    import sys
    os.makedirs(rv.set_dir("fake"))
    script = tmp_path / "fake_serve.py"
    script.write_text(_FAKE_SERVE)
    assert rv.SERVE_DONE == "@@run_v2-done"
    w = lv.ServeWorker("fake", 0, dict(os.environ), time.time(), cmd=[sys.executable, str(script)])
    spec = rv.CaseSpec("C1")
    w.process((1, spec))
    pid1 = w.proc.pid
    w.process((2, spec))
    assert w.proc.pid == pid1                          # one process serves many cases
    w.process((13, spec))                              # dies -> reported, not retried
    assert w.proc is None
    w.process((4, spec))                               # fresh process takes the next case
    assert w.proc.pid != pid1
    w.close()
    log = open(os.path.join(rv.set_dir("fake"), "worker_0.log")).read()
    for i in (1, 2, 13, 4):
        assert f"running {i}\n" in log
    assert "process died (exit 3) during [13]" in log and "bye" in log
    assert rv.SERVE_DONE not in log
