# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""``plume-studio`` -- the scripted path to everything the UI can do.

**Scripted ensembles must not require the browser** (ADR-002). This is not a convenience wrapper
over the API: it goes through the same schema, resolver, store and runner, so a sweep launched from
a terminal and one launched from the web produce identical rows and identical provenance.

Verbs stay recognisable to anyone who has used the existing runners, which share a
``plan | one <i> | run <lo> <hi>`` shape (``run_ensemble.py:174``):

    plume-studio run config.yaml --out runs/          # one run, end to end
    plume-studio sweep sweep.yaml --plan              # expand axes, print N, submit NOTHING
    plume-studio sweep sweep.yaml --out runs/         # expand and submit
    plume-studio status <run-id>                      # persisted state, from any process

``--plan`` exists because deciding to spend 810 x 4.6 minutes should take a second command. It is
the same reflex the existing runners encode, and it prints what would run without creating a row.

``status`` reads from the database rather than from memory, which is the point: job state outlives
the process that submitted it (ADR-007), so a crashed CLI does not orphan a running job.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer
import yaml

from studio.resolve import ResolvedConfig, resolve
from studio.schema import RunConfig, RunSet
from studio.service import prepare, submit_and_record
from studio.store import (
    create_db_engine,
    create_run_set,
    database_url,
    session_factory,
    session_scope,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Configure, run and inspect coupled SAI plume box-model simulations.",
)


def _load_mapping(path: Path) -> dict[str, Any]:
    """Read a YAML or JSON document, or exit with a message naming the file.

    YAML by suffix, not by sniffing: a file called ``.json`` that happens to parse as YAML is still
    a mistake worth reporting.
    """
    if not path.is_file():
        typer.echo(f"[studio] no such file: {path}", err=True)
        raise typer.Exit(code=2)
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        typer.echo(
            f"[studio] {path} is not valid {path.suffix.lstrip('.') or 'YAML'}: {exc}", err=True
        )
        raise typer.Exit(code=2) from exc
    if not isinstance(data, dict):
        typer.echo(f"[studio] {path} must contain a mapping, got {type(data).__name__}", err=True)
        raise typer.Exit(code=2)
    return data


def _resolved_config(path: Path) -> ResolvedConfig:
    """Load a RunConfig document and resolve its derived fields."""
    try:
        return resolve(RunConfig.model_validate(_load_mapping(path)))
    except ValueError as exc:
        typer.echo(f"[studio] {path} is not a valid configuration:\n{exc}", err=True)
        raise typer.Exit(code=2) from exc


def _run_set(path: Path) -> RunSet:
    try:
        return RunSet.model_validate(_load_mapping(path))
    except ValueError as exc:
        typer.echo(f"[studio] {path} is not a valid sweep:\n{exc}", err=True)
        raise typer.Exit(code=2) from exc


@app.command()
def run(
    config: Path = typer.Argument(..., help="RunConfig as YAML or JSON"),
    out: Path = typer.Option(Path("var/studio/runs"), "--out", help="where jobs and artefacts go"),
    label: str = typer.Option(
        "", "--label", help="human-readable name; identity is still the hash"
    ),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="block until the run finishes"),
    database: str | None = typer.Option(None, "--database", help="override STUDIO_DATABASE_URL"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="resolve and print the identity; touch nothing"
    ),
) -> None:
    """Run one configuration, end to end."""
    resolved = _resolved_config(config)
    typer.echo(f"[studio] config hash {resolved.config.config_hash()}")
    typer.echo(f"[studio] initial SO2  {resolved.config.injection.so2_initial_pptv:.6e} pptv")
    if dry_run:
        typer.echo("[studio] --dry-run: nothing written, nothing submitted")
        return

    from studio.runner import LocalSubprocessRunner

    factory, store = prepare(database, out)
    runner = LocalSubprocessRunner(out / "jobs")
    try:
        with session_scope(factory) as session:
            run_set_row = create_run_set(session, label=label or config.stem)
        run_id, state = submit_and_record(
            factory, store, runner, config=resolved, run_set=run_set_row, label=label, wait=wait
        )
    finally:
        runner.shutdown()
    typer.echo(f"[studio] run {run_id} -> {state}")
    if state != "succeeded" and wait:
        raise typer.Exit(code=1)


@app.command()
def sweep(
    sweep_file: Path = typer.Argument(..., help="RunSet as YAML or JSON: base config plus axes"),
    out: Path = typer.Option(Path("var/studio/runs"), "--out", help="where jobs and artefacts go"),
    plan: bool = typer.Option(
        False, "--plan", help="print what would run and submit NOTHING (mirrors run_ensemble plan)"
    ),
    database: str | None = typer.Option(None, "--database", help="override STUDIO_DATABASE_URL"),
) -> None:
    """Expand a sweep into runs, and submit them unless asked to plan."""
    run_set = _run_set(sweep_file)
    typer.echo(f"[studio] {run_set.size()} run(s) from {len(run_set.axes)} axis/axes")
    expanded = run_set.expand()
    if plan:
        for index, item in enumerate(expanded):
            typer.echo(f"  [{index}] {item.label or '(unlabelled)'}  {item.config_hash[:12]}")
        typer.echo("[studio] --plan: nothing submitted")
        return

    from studio.runner import LocalSubprocessRunner

    factory, store = prepare(database, out)
    runner = LocalSubprocessRunner(out / "jobs")
    failures = 0
    try:
        with session_scope(factory) as session:
            run_set_row = create_run_set(
                session,
                label=sweep_file.stem,
                axes=run_set.model_dump(mode="json")["axes"],
            )
        for item in expanded:
            run_id, state = submit_and_record(
                factory,
                store,
                runner,
                config=resolve(item.config),
                run_set=run_set_row,
                label=item.label,
            )
            typer.echo(f"[studio] {item.label or run_id} -> {state}")
            failures += state != "succeeded"
    finally:
        runner.shutdown()
    if failures:
        typer.echo(f"[studio] {failures} run(s) did not succeed", err=True)
        raise typer.Exit(code=1)


@app.command()
def status(
    run_id: str = typer.Argument(..., help="run id, as printed by `run` or `sweep`"),
    database: str | None = typer.Option(None, "--database", help="override STUDIO_DATABASE_URL"),
) -> None:
    """Show a run's persisted state.

    Reads the database, not memory: job state outlives the process that submitted it, so this works
    from a different shell, after a restart, or while the run is still going.
    """
    from studio.store.models import RunRow

    factory = session_factory(create_db_engine(database_url(database)))
    with session_scope(factory) as session:
        run = session.get(RunRow, run_id)
        if run is None:
            typer.echo(f"[studio] no run {run_id!r} in this database", err=True)
            raise typer.Exit(code=2)
        typer.echo(f"run        {run.id}")
        typer.echo(f"label      {run.label or '(none)'}")
        typer.echo(f"config     {run.config_hash}")
        typer.echo(f"created    {run.created_at.isoformat()}")
        reproducible = {True: "yes", False: "NO — a checkout was dirty", None: "unknown"}
        typer.echo(f"reproducible {reproducible[run.reproducible]}")
        for job in run.jobs:
            typer.echo(f"job        {job.id}  {job.state}  exit={job.exit_code}")
            for transition in job.transitions:
                typer.echo(
                    f"    {transition.at.isoformat()}  {transition.state}  {transition.detail}"
                )
        for artifact in run.artifacts:
            typer.echo(
                f"artifact   {artifact.kind:10s} {artifact.size_bytes:>10,d} B  {artifact.path}"
            )


def main() -> int:
    app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
