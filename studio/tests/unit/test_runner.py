# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The job lifecycle and the local subprocess runner.

These launch **real subprocesses** -- the runner's whole job is process management, and a mocked
``Popen`` would test the mock. What they do not launch is the model: ``entry_module`` points at a
small fixture module that exits, fails, or sleeps on command. That is a parameter of the runner
rather than a test hook: nothing in ``LocalSubprocessRunner`` branches on its value, and the default
is the real entry point.

The tests that matter most are the ones about **what survives a failure**. A run that dies at minute
three of four must leave enough behind to diagnose it without paying those three minutes again.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from studio.resolve import ResolvedConfig, apply_change, resolve, set_override
from studio.runner import (
    THREAD_PINNING,
    InvalidTransitionError,
    JobRecord,
    JobState,
    LocalSubprocessRunner,
    SlurmRunner,
)
from studio.schema import RunConfig

#: A stand-in for ``studio.cli.run``: same argv shape, controllable outcome, no model.
FIXTURE_MODULE = "studio.tests.fixtures.fake_run"


@pytest.fixture
def resolved() -> ResolvedConfig:
    return resolve(RunConfig())


@pytest.fixture
def runner(tmp_path: Path) -> LocalSubprocessRunner:
    made = LocalSubprocessRunner(tmp_path / "jobs", max_workers=2, entry_module=FIXTURE_MODULE)
    yield made
    made.shutdown(cancel_running=True)


class TestLifecycle:
    """Transitions are data, and illegal ones raise."""

    @pytest.mark.tier_a
    def test_the_happy_path_records_every_step(self) -> None:
        record = (
            JobRecord(job_id="j", config_hash="h")
            .transition_to(JobState.QUEUED)
            .transition_to(JobState.RUNNING)
            .transition_to(JobState.SUCCEEDED)
        )
        assert [t.state for t in record.transitions] == [
            JobState.QUEUED,
            JobState.RUNNING,
            JobState.SUCCEEDED,
        ]
        assert record.is_terminal
        assert record.submitted_at and record.started_at and record.ended_at
        assert record.duration_s is not None and record.duration_s >= 0.0

    @pytest.mark.tier_a
    @pytest.mark.parametrize(
        ("from_state", "to_state"),
        [
            (JobState.DRAFT, JobState.RUNNING),  # never ran the queue
            (JobState.RUNNING, JobState.QUEUED),  # backwards
            (JobState.SUCCEEDED, JobState.RUNNING),  # terminal
            (JobState.FAILED, JobState.SUCCEEDED),  # rewriting history
        ],
    )
    def test_illegal_transitions_raise(self, from_state: JobState, to_state: JobState) -> None:
        """A job that appears to move backwards means the runner lost track of a process.

        Accepting it silently would turn the record from a log into a story.
        """
        record = JobRecord(job_id="j", config_hash="h").model_copy(update={"state": from_state})
        with pytest.raises(InvalidTransitionError):
            record.transition_to(to_state)

    @pytest.mark.tier_a
    def test_a_record_is_immutable(self) -> None:
        """It is an audit trail; one that can be edited in place can disagree with what happened."""
        record = JobRecord(job_id="j", config_hash="h")
        with pytest.raises(ValueError, match="frozen"):
            record.state = JobState.RUNNING  # type: ignore[misc]
        assert record.transition_to(JobState.QUEUED) is not record

    @pytest.mark.tier_a
    def test_terminated_on_limit_is_not_failed(self) -> None:
        """Two different things: "could not produce a result" vs "we stopped it mid-flight".

        Collapsing them would let a partial run be read as a converged one.
        """
        assert JobState.TERMINATED_ON_LIMIT != JobState.FAILED
        record = (
            JobRecord(job_id="j", config_hash="h")
            .transition_to(JobState.QUEUED)
            .transition_to(JobState.RUNNING)
            .transition_to(JobState.TERMINATED_ON_LIMIT, detail="exceeded max_wall_time_s")
        )
        assert record.is_terminal
        assert "max_wall_time" in record.detail


class TestLocalSubprocessRunner:
    @pytest.mark.tier_a
    def test_a_successful_run_reaches_succeeded(
        self, runner: LocalSubprocessRunner, resolved: ResolvedConfig
    ) -> None:
        record = runner.submit(resolved, label="ok")
        assert record.state is JobState.QUEUED
        final = runner.wait(record.job_id, timeout=30)
        assert final.state is JobState.SUCCEEDED
        assert final.exit_code == 0
        assert final.config_hash == resolved.config.config_hash()

    @pytest.mark.tier_a
    def test_the_resolved_input_is_written_before_the_run(
        self, runner: LocalSubprocessRunner, resolved: ResolvedConfig
    ) -> None:
        """Written at submit, not completion, so a job that dies at once still has its input."""
        record = runner.submit(resolved)
        assert record.input_path is not None and record.input_path.is_file()
        restored = ResolvedConfig.model_validate_json(record.input_path.read_text())
        assert restored.config.config_hash() == resolved.config.config_hash()
        runner.wait(record.job_id, timeout=30)

    @pytest.mark.tier_a
    def test_a_failed_run_keeps_everything_needed_to_diagnose_it(
        self,
        runner: LocalSubprocessRunner,
        resolved: ResolvedConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The point of the whole design: no re-run required.

        A four-minute case that fails intermittently must not have to be reproduced to be
        understood, so the exit code, both log streams and the exact input are all on disk.
        """
        _directive(monkeypatch, mode="fail", message="synthetic failure")
        record = runner.submit(resolved, label="boom")
        final = runner.wait(record.job_id, timeout=30)

        assert final.state is JobState.FAILED
        assert final.exit_code == 1
        assert final.stderr_path is not None
        assert "synthetic failure" in final.stderr_path.read_text()
        assert final.input_path is not None and final.input_path.is_file()
        names = {path.name for path in runner.artifacts(final.job_id)}
        assert {"input.json", "stdout.log", "stderr.log"} <= names

    @pytest.mark.tier_a
    def test_stdout_is_captured_as_the_log_stream(
        self,
        runner: LocalSubprocessRunner,
        resolved: ResolvedConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``run_coupled`` prints rather than logs, so stdout IS the run log."""
        _directive(monkeypatch, mode="ok", message="[coupled] NOTE: something happened")
        record = runner.submit(resolved)
        final = runner.wait(record.job_id, timeout=30)
        assert final.stdout_path is not None
        assert "[coupled] NOTE: something happened" in final.stdout_path.read_text()

    @pytest.mark.tier_a
    def test_exceeding_the_wall_clock_limit_is_terminated_not_failed(
        self, runner: LocalSubprocessRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """And the record says so, so nothing downstream reads the partial output as converged."""
        config = resolve(
            RunConfig.model_validate(
                {**RunConfig().model_dump(), "termination": {"max_wall_time_s": 1.0}}
            )
        )
        _directive(monkeypatch, mode="sleep", seconds=30)
        record = runner.submit(config, label="slow")
        final = runner.wait(record.job_id, timeout=60)

        assert final.state is JobState.TERMINATED_ON_LIMIT
        assert "max_wall_time_s" in final.detail
        assert "NOT a converged result" in final.detail

    @pytest.mark.tier_a
    def test_cancelling_a_running_job_stops_it(
        self,
        runner: LocalSubprocessRunner,
        resolved: ResolvedConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _directive(monkeypatch, mode="sleep", seconds=30)
        record = runner.submit(resolved)
        _wait_for_state(runner, record.job_id, JobState.RUNNING)
        cancelled = runner.cancel(record.job_id)
        assert cancelled.state is JobState.CANCELLED
        assert runner.poll(record.job_id).state is JobState.CANCELLED

    @pytest.mark.tier_a
    def test_cancelling_a_finished_job_is_not_an_error(
        self, runner: LocalSubprocessRunner, resolved: ResolvedConfig
    ) -> None:
        """Cancelling something that already finished is a race, not a mistake by the caller."""
        record = runner.submit(resolved)
        final = runner.wait(record.job_id, timeout=30)
        assert runner.cancel(final.job_id).state is final.state

    @pytest.mark.tier_a
    def test_a_stale_config_is_refused_at_submission(self, runner: LocalSubprocessRunner) -> None:
        """Nothing downstream could tell that the numbers did not follow from each other."""
        from studio.resolve import InconsistentConfigError

        stale = apply_change(
            set_override(resolve(RunConfig()), "injection.so2_initial_pptv", 5.0e9),
            "site.temperature_k",
            213.0,
        )
        with pytest.raises(InconsistentConfigError):
            runner.submit(stale)

    @pytest.mark.tier_a
    def test_polling_an_unknown_job_raises(self, runner: LocalSubprocessRunner) -> None:
        with pytest.raises(KeyError, match="unknown job id"):
            runner.poll("nope")

    @pytest.mark.tier_a
    def test_jobs_get_separate_work_directories(
        self, runner: LocalSubprocessRunner, resolved: ResolvedConfig
    ) -> None:
        """Two runs of the SAME config must not share a directory and overwrite each other."""
        first = runner.submit(resolved)
        second = runner.submit(resolved)
        assert first.work_dir != second.work_dir
        assert first.config_hash == second.config_hash, "same config, same identity"
        for job in (first, second):
            runner.wait(job.job_id, timeout=30)

    @pytest.mark.tier_a
    def test_the_subprocess_gets_the_thread_pinning_environment(
        self,
        runner: LocalSubprocessRunner,
        resolved: ResolvedConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """This pinning -- not vmap -- is what gives ~N times the throughput for N workers.

        Asserted by having the fixture module dump its own environment, because a runner that
        *intends* to pin threads and does not would only show up as everything being slow.
        """
        _directive(monkeypatch, mode="dump_env")
        record = runner.submit(resolved)
        final = runner.wait(record.job_id, timeout=30)
        assert final.stdout_path is not None
        reported = json.loads(final.stdout_path.read_text())
        assert {key: reported.get(key) for key in THREAD_PINNING} == THREAD_PINNING

    @pytest.mark.tier_a
    def test_a_bad_input_file_exits_distinctly_from_a_model_failure(
        self, tmp_path: Path, resolved: ResolvedConfig
    ) -> None:
        """Exit code 2 means "never started"; 1 means "the model raised". The runner needs both.

        Runs the REAL entry point (``studio.cli.run``), because this is its contract, and a bad
        input is rejected before any model import -- so it costs milliseconds, not a JAX load.
        """
        import subprocess

        bad = tmp_path / "bad.json"
        bad.write_text('{"config": {"site": {"temperature_k": -5}}}', encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "-m", "studio.cli.run", str(bad), str(tmp_path / "out")],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[3]),
        )
        assert proc.returncode == 2, proc.stderr
        assert "cannot run" in proc.stderr


class TestUnimplementedBackends:
    @pytest.mark.tier_a
    def test_slurm_raises_rather_than_falling_back_to_local(self) -> None:
        """A job running somewhere other than where it was sent is worse than an error (ADR-008)."""
        with pytest.raises(NotImplementedError, match="Slurm"):
            SlurmRunner().submit(None)
        with pytest.raises(NotImplementedError, match="not implemented"):
            SlurmRunner().poll("x")


def _directive(monkeypatch: pytest.MonkeyPatch, **directive: object) -> None:
    """Tell the fixture module what to do, BEFORE the job is submitted.

    Via the environment rather than a file in the work directory: the subprocess can start before a
    file written after ``submit()`` lands, which is exactly how process tests become flaky.
    """
    monkeypatch.setenv("STUDIO_FAKE_RUN", json.dumps(directive))


def _wait_for_state(
    runner: LocalSubprocessRunner, job_id: str, state: JobState, timeout: float = 30.0
) -> None:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if runner.poll(job_id).state is state:
            return
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached {state}")


class TestMaxSimTimeStopCondition:
    """The ``stop_condition`` this layer hands the model, and the shape it must have.

    Pure-function tests: no model run, but they pin the coupling that PR #73 exposed. The model
    dispatches on the callback's DECLARED ARITY and raises ``TypeError`` on ``*args`` -- correctly,
    since a variadic callback matches both shapes and guessing would be a silent wrong answer. That
    makes the arity part of this function's contract rather than an implementation detail, so it is
    asserted here where a change is cheap to notice.
    """

    @pytest.mark.tier_a
    def test_no_limit_means_no_stop_condition(self) -> None:
        """``None`` is not a callback that never fires; it is no callback at all."""
        from studio.modelio.execute import _max_sim_time_stop

        assert _max_sim_time_stop(resolve(RunConfig())) is None

    @pytest.mark.tier_a
    def test_the_callback_takes_exactly_one_parameter(self) -> None:
        """The diagnostics-dict shape. Not ``*args``, which the model rejects as ambiguous.

        Arity is part of the contract, not an implementation detail: two parameters still work but
        emit a ``DeprecationWarning``, and a variadic callback raises ``TypeError``. Asserted here
        because it is cheap to notice and expensive to discover from a run.
        """
        import inspect

        from studio.modelio.execute import _max_sim_time_stop

        stop = _max_sim_time_stop(_with_sim_limit(2.0))
        assert stop is not None
        parameters = list(inspect.signature(stop).parameters.values())
        assert len(parameters) == 1
        assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD

    @pytest.mark.tier_a
    def test_it_fires_exactly_at_the_limit(self) -> None:
        """Boundary included: at the limit the run has reached its cap, not almost reached it."""
        from studio.modelio.execute import _max_sim_time_stop

        stop = _max_sim_time_stop(_with_sim_limit(2.0))
        assert stop is not None
        two_days_s = 2.0 * 86400.0
        assert stop({"t": two_days_s - 1.0, "SA": 12.0}) is False
        assert stop({"t": two_days_s, "SA": 12.0}) is True
        assert stop({"t": two_days_s + 1.0, "SA": 12.0}) is True


def _with_sim_limit(days: float) -> ResolvedConfig:
    payload = RunConfig().model_dump()
    payload["termination"]["max_sim_time_days"] = days
    return resolve(RunConfig.model_validate(payload))
