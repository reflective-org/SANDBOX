# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Persistence: the schema, the immutability rules, and the model/migration agreement.

Every test upgrades a real SQLite database **through the migrations**, never via
``Base.metadata.create_all``. Creating tables straight from the models would test a schema no
migration ever produced, leaving the migrations to be exercised for the first time on someone's real
database -- which is the failure Alembic-from-the-first-migration exists to prevent.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import inspect

from studio.resolve import apply_change, resolve, set_override
from studio.schema import RunConfig
from studio.store import (
    LocalDirectoryStore,
    artifact_for,
    create_db_engine,
    create_run,
    create_run_set,
    current_revision,
    ensure_config,
    record_artifact,
    record_job,
    record_summary,
    record_transition,
    runs_for_config,
    session_factory,
    session_scope,
    sha256_of,
    upgrade_to_head,
)

EXPECTED_TABLES = {
    "run_set",
    "run",
    "run_config",
    "job",
    "job_transition",
    "result_artifact",
    "dataset_version",
    "run_dataset",
    "run_summary",
}


@pytest.fixture
def database(tmp_path: Path):
    """A migrated SQLite database and its session factory."""
    url = f"sqlite:///{tmp_path / 'studio.db'}"
    upgrade_to_head(url)
    engine = create_db_engine(url)
    yield engine, session_factory(engine), url
    engine.dispose()


@pytest.fixture
def store(tmp_path: Path) -> LocalDirectoryStore:
    return LocalDirectoryStore(tmp_path / "artifacts")


@pytest.mark.tier_a
def test_migrating_creates_every_table(database) -> None:
    engine, _, _ = database
    assert EXPECTED_TABLES <= set(inspect(engine).get_table_names())
    assert current_revision(engine) is not None


@pytest.mark.tier_a
def test_upgrading_twice_is_idempotent(database) -> None:
    """Startup must be able to call this unconditionally."""
    engine, _, url = database
    before = current_revision(engine)
    upgrade_to_head(url)
    assert current_revision(engine) == before


@pytest.mark.tier_a
def test_the_models_and_the_migration_agree(database) -> None:
    """The drift test: autogenerate against a migrated database must find nothing to do.

    Without it, a column added to ``models.py`` without a migration works on every developer machine
    (where the table was created from the models by some other path) and fails on the first real
    deployment. Here it fails immediately, in seconds.
    """
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    engine, _, _ = database
    from studio.store.models import Base

    with engine.connect() as connection:
        context = MigrationContext.configure(connection, opts={"compare_type": True})
        diff = compare_metadata(context, Base.metadata)
    assert diff == [], (
        f"models.py and the migrations have drifted: {diff}. Generate a migration "
        f"(`alembic -c studio/store/alembic.ini revision --autogenerate`) rather than editing the "
        f"models alone."
    )


@pytest.mark.tier_a
def test_the_same_config_is_one_row(database) -> None:
    """Identity is the hash (ADR-006): CLI and API submissions converge rather than duplicating."""
    _, factory, _ = database
    config = resolve(RunConfig())
    with session_scope(factory) as session:
        first = ensure_config(session, config)
        second = ensure_config(session, resolve(RunConfig()))
        assert first.config_hash == second.config_hash
        assert first is second


@pytest.mark.tier_a
def test_an_edited_config_is_a_different_row(database) -> None:
    """Configs are immutable: an edit produces a new config, never a mutation (ADR-004)."""
    _, factory, _ = database
    original = resolve(RunConfig())
    edited = apply_change(original, "site.temperature_k", 213.0)
    with session_scope(factory) as session:
        first = ensure_config(session, original)
        second = ensure_config(session, edited)
        assert first.config_hash != second.config_hash
        assert first.resolved_config["site"]["temperature_k"] == 210.0
        assert second.resolved_config["site"]["temperature_k"] == 213.0


@pytest.mark.tier_a
def test_the_repository_offers_no_way_to_update_a_config() -> None:
    """Immutability is structural, not a convention someone has to remember."""
    import studio.store.repository as repository

    forbidden = [
        name
        for name in dir(repository)
        if any(word in name.lower() for word in ("update", "delete", "overwrite"))
        and not name.startswith("_")
    ]
    assert forbidden == [], f"repository exposes mutation helpers: {forbidden}"


@pytest.mark.tier_a
def test_a_stale_config_is_refused(database) -> None:
    """Persisting one would give an inconsistent set of numbers a permanent identity."""
    from studio.resolve import InconsistentConfigError

    _, factory, _ = database
    stale = apply_change(
        set_override(resolve(RunConfig()), "injection.so2_initial_pptv", 5.0e9),
        "site.temperature_k",
        213.0,
    )
    with pytest.raises(InconsistentConfigError), session_scope(factory) as session:
        ensure_config(session, stale)


@pytest.mark.tier_a
def test_a_run_records_its_provenance_and_reproducibility(database) -> None:
    """``reproducible`` describes the moment the run started, so it is stored, not recomputed."""
    _, factory, _ = database
    provenance = {
        "sandbox": {"commit": "a" * 40, "dirty": True},
        "submodules": {"tuvx-jax": {"commit": "b" * 40, "dirty": False}},
    }
    with session_scope(factory) as session:
        run_set = create_run_set(session, label="sweep")
        run = create_run(
            session,
            run_set=run_set,
            config=resolve(RunConfig()),
            label="case",
            provenance=provenance,
        )
        assert run.reproducible is False
        assert run.provenance["sandbox"]["commit"] == "a" * 40

        clean = create_run(
            session,
            run_set=run_set,
            config=resolve(RunConfig()),
            provenance={"sandbox": {"dirty": False}, "submodules": {}},
        )
        assert clean.reproducible is True


@pytest.mark.tier_a
def test_job_transitions_are_appended_not_replaced(database) -> None:
    """The transition list is the record; the state column is a convenience."""
    _, factory, _ = database
    with session_scope(factory) as session:
        run_set = create_run_set(session)
        run = create_run(session, run_set=run_set, config=resolve(RunConfig()))
        job = record_job(session, run=run, state="queued", work_dir="/tmp/x")
        record_transition(session, job=job, state="running", detail="launched")
        record_transition(session, job=job, state="failed", detail="exit 1", exit_code=1)

        assert job.state == "failed"
        assert job.exit_code == 1
        assert [t.state for t in job.transitions] == ["queued", "running", "failed"]
        assert all(t.at.tzinfo is not None for t in job.transitions), "timestamps must be aware"


@pytest.mark.tier_a
def test_an_artifact_stores_a_pointer_and_a_checksum(database, store, tmp_path: Path) -> None:
    """The bytes stay on disk; the database gets a path, a size and a hash."""
    _, factory, _ = database
    source = tmp_path / "state.npz"
    source.write_bytes(b"not really an npz, but bytes are bytes")

    with session_scope(factory) as session:
        run_set = create_run_set(session)
        run = create_run(session, run_set=run_set, config=resolve(RunConfig()))
        row = record_artifact(session, run=run, kind="state", source=source, store=store)
        run_id = run.id

    assert row.sha256 == sha256_of(source)
    assert row.size_bytes == source.stat().st_size
    assert not Path(row.path).is_absolute(), "paths are relative to the store root"
    assert store.verify(row.path, row.sha256)
    with session_scope(factory) as session:
        assert artifact_for(session, run_id, "state").sha256 == row.sha256


@pytest.mark.tier_a
def test_a_missing_artifact_raises_rather_than_recording_an_absence(database, store) -> None:
    """A run that was supposed to produce an artefact and did not is a failure to surface."""
    _, factory, _ = database
    with pytest.raises(FileNotFoundError), session_scope(factory) as session:
        run_set = create_run_set(session)
        run = create_run(session, run_set=run_set, config=resolve(RunConfig()))
        record_artifact(
            session, run=run, kind="state", source=Path("/nonexistent.npz"), store=store
        )


@pytest.mark.tier_a
def test_corruption_is_detectable(database, store, tmp_path: Path) -> None:
    """Why the checksum is stored: silent corruption and a tidied directory look the same."""
    _, factory, _ = database
    source = tmp_path / "summary.json"
    source.write_text('{"schema_version": "0.1.0"}', encoding="utf-8")
    with session_scope(factory) as session:
        run_set = create_run_set(session)
        run = create_run(session, run_set=run_set, config=resolve(RunConfig()))
        row = record_artifact(session, run=run, kind="summary", source=source, store=store)
        stored = store.open_path(row.path)
        key, digest = row.path, row.sha256

    assert store.verify(key, digest)
    stored.write_text('{"schema_version": "tampered"}', encoding="utf-8")
    assert not store.verify(key, digest)


@pytest.mark.tier_a
def test_a_summary_promotes_headline_scalars_for_querying(database) -> None:
    """So "every run where peak number exceeded X" is a query, not 810 deserialisations."""
    _, factory, _ = database
    summary = {
        "schema_version": "0.1.0",
        "termination": "completed",
        "flags": ["open_system_dilution"],
        "series": {
            "SO2": {"values": [3.3e9, 1.7e6]},
            "H2SO4": {"values": [0.0, 15.05, 2.1]},
            "total_n": {"values": [0.0, 3.07e6, 1.2e6]},
            "SA": {"values": [2.0, 41.3]},
        },
    }
    with session_scope(factory) as session:
        run_set = create_run_set(session)
        run = create_run(session, run_set=run_set, config=resolve(RunConfig()))
        row = record_summary(session, run=run, summary=summary)

    assert row.final_so2_pptv == 1.7e6
    assert row.peak_h2so4_pptv == 15.05
    assert row.peak_number_cm3 == 3.07e6
    assert row.final_surface_area == 41.3
    assert row.termination == "completed"
    assert row.summary["series"]["SO2"]["values"][-1] == row.final_so2_pptv, "column and JSON agree"


@pytest.mark.tier_a
def test_runs_for_a_config_are_the_cache_lookup(database) -> None:
    """Identical hash is necessary but not sufficient to reuse a result, so this returns runs.

    The caller still has to compare the model version in each run's provenance -- which is why this
    is not ``cached_result_for()`` returning a verdict.
    """
    _, factory, _ = database
    config = resolve(RunConfig())
    with session_scope(factory) as session:
        run_set = create_run_set(session)
        create_run(session, run_set=run_set, config=config, label="first")
        create_run(session, run_set=run_set, config=config, label="second")
        found = runs_for_config(session, config.config.config_hash())
        assert {run.label for run in found} == {"first", "second"}
        assert len({run.id for run in found}) == 2, "same config, two distinct runs"


@pytest.mark.tier_a
def test_foreign_keys_are_enforced_on_sqlite(database) -> None:
    """Without the pragma, SQLite ignores them and Postgres does not -- silent divergence."""
    from sqlalchemy.exc import IntegrityError

    from studio.store.models import JobRow

    _, factory, _ = database
    with pytest.raises(IntegrityError), session_scope(factory) as session:
        session.add(
            JobRow(
                id="j",
                run_id="does-not-exist",
                state="queued",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )
