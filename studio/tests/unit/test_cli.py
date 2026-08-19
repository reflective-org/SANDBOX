# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The CLI: the scripted path to everything the UI can do (ADR-002).

Most of these need neither the model nor a long run -- ``--plan`` and ``--dry-run`` exist precisely
so that deciding to spend compute is a separate act from spending it, and that makes them cheap to
test. The one end-to-end test runs the real model for one simulated day and skips without the
submodules, like every other model-touching test.

No test hooks: the CLI is exercised through its real arguments, and the seams it uses
(``--database``, ``--out``) are ones a real deployment uses too.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from studio.cli.main import app
from studio.schema import RunConfig

runner = CliRunner()


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"schedule": {"duration_days": 1}, "microphysics": {"n_bins": 40}}),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def sweep_file(tmp_path: Path) -> Path:
    path = tmp_path / "sweep.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "base": {"schedule": {"duration_days": 1}, "microphysics": {"n_bins": 40}},
                "axes": [
                    {
                        "name": "nucleation",
                        "kind": "grid",
                        "points": [
                            {
                                "label": "nuc1",
                                "assignments": {"microphysics.nucleation_rate_scale": 1.0},
                            },
                            {
                                "label": "nuc100",
                                "assignments": {"microphysics.nucleation_rate_scale": 100.0},
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.tier_a
def test_a_dry_run_prints_the_identity_and_touches_nothing(
    config_file: Path, tmp_path: Path
) -> None:
    """Identity before compute: the hash is knowable without running anything."""
    result = runner.invoke(
        app, ["run", str(config_file), "--dry-run", "--out", str(tmp_path / "o")]
    )
    assert result.exit_code == 0, result.output
    assert "config hash" in result.output
    assert "nothing written, nothing submitted" in result.output
    assert not (tmp_path / "o").exists(), "--dry-run must not create the output tree"


@pytest.mark.tier_a
def test_the_dry_run_hash_is_the_schema_hash(config_file: Path) -> None:
    """The CLI must not have its own idea of identity."""
    from studio.resolve import resolve

    expected = resolve(
        RunConfig.model_validate({"schedule": {"duration_days": 1}, "microphysics": {"n_bins": 40}})
    ).config.config_hash()
    result = runner.invoke(app, ["run", str(config_file), "--dry-run"])
    assert expected in result.output


@pytest.mark.tier_a
def test_plan_lists_the_runs_and_submits_nothing(sweep_file: Path, tmp_path: Path) -> None:
    """Mirrors ``run_ensemble.py``'s ``plan`` verb: deciding to spend compute is its own command."""
    result = runner.invoke(app, ["sweep", str(sweep_file), "--plan", "--out", str(tmp_path / "o")])
    assert result.exit_code == 0, result.output
    assert "2 run(s) from 1 axis/axes" in result.output
    assert "nuc1" in result.output and "nuc100" in result.output
    assert "nothing submitted" in result.output
    assert not (tmp_path / "o").exists()


@pytest.mark.tier_a
def test_planned_runs_have_distinct_identities(sweep_file: Path) -> None:
    """A sweep whose points collided would silently run the same case twice."""
    result = runner.invoke(app, ["sweep", str(sweep_file), "--plan"])
    hashes = [line.split()[-1] for line in result.output.splitlines() if line.startswith("  [")]
    assert len(hashes) == len(set(hashes)) == 2


@pytest.mark.tier_a
@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("not: [valid", "not valid"),
        ("- a\n- b\n", "must contain a mapping"),
        ("site:\n  given_temperature_k: -5\n", "not a valid configuration"),
    ],
)
def test_a_bad_config_file_exits_two_with_a_message(
    tmp_path: Path, contents: str, message: str
) -> None:
    """Exit 2 is "we never started", distinct from a model failure (exit 1)."""
    path = tmp_path / "bad.yaml"
    path.write_text(contents, encoding="utf-8")
    result = runner.invoke(app, ["run", str(path), "--dry-run"])
    assert result.exit_code == 2
    assert message in result.output


@pytest.mark.tier_a
def test_a_missing_file_names_itself(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", str(tmp_path / "absent.yaml"), "--dry-run"])
    assert result.exit_code == 2
    assert "no such file" in result.output


@pytest.mark.tier_a
def test_status_of_an_unknown_run_exits_two(tmp_path: Path) -> None:
    """Distinguishes "no such run" from "a run with nothing to report"."""
    from studio.store import upgrade_to_head

    url = f"sqlite:///{tmp_path / 'studio.db'}"
    upgrade_to_head(url)
    result = runner.invoke(app, ["status", "nope", "--database", url])
    assert result.exit_code == 2
    assert "no run" in result.output


@pytest.mark.tier_a
def test_a_run_is_persisted_and_readable_from_a_new_process(
    config_file: Path, tmp_path: Path, repo_root: Path
) -> None:
    """The vertical slice's CLI half, end to end, with the real model. ~20 s.

    ``status`` runs against a session opened after the run finished -- the point being that job
    state lives in the database rather than in the submitting process's memory (ADR-007), so a
    crashed or exited CLI leaves a readable record rather than an orphan.
    """
    if not (repo_root / "stratchem-jax" / "config.py").is_file():
        pytest.skip("model submodules not checked out (`git submodule update --init`)")

    url = f"sqlite:///{tmp_path / 'studio.db'}"
    out = tmp_path / "runs"
    result = runner.invoke(
        app, ["run", str(config_file), "--out", str(out), "--database", url, "--label", "slice"]
    )
    assert result.exit_code == 0, result.output
    assert "succeeded" in result.output

    from sqlalchemy import select

    from studio.store import create_db_engine, session_factory, session_scope
    from studio.store.models import RunRow

    factory = session_factory(create_db_engine(url))
    with session_scope(factory) as session:
        run = session.scalars(select(RunRow)).one()
        run_id = run.id
        assert run.label == "slice"
        assert run.provenance is not None, "provenance is recorded at submit (ADR-006)"
        assert set(run.provenance["submodules"]), "the model is pinned, not just the app"
        kinds = {artifact.kind for artifact in run.artifacts}
        assert {"input", "provenance", "state", "summary", "stdout"} <= kinds
        (job,) = run.jobs
        assert job.state == "succeeded"
        assert [t.state for t in job.transitions] == [
            "queued",
            "running",
            "succeeded",
        ], "the persisted trail must be as complete as the runner's, or persisting it is pointless"

    status = runner.invoke(app, ["status", run_id, "--database", url])
    assert status.exit_code == 0, status.output
    assert run_id in status.output
    assert "succeeded" in status.output
    assert "state" in status.output and "summary" in status.output

    summary_row = json.loads((out / "artifacts" / run_id / "summary.json").read_text())
    assert summary_row["config_hash"] == run.config_hash, "the summary carries the same identity"
