# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Provenance records (ADR-006).

These build **real git repositories** in a temp directory rather than mocking ``subprocess``.
The module is a thin shell around git's behaviour, so a mocked git would test the mock: whether
``status --porcelain`` reports an untracked file, whether a missing ``.git`` fails as expected,
whether ``rev-parse`` in a fresh repo with no commits errors — those are the questions, and only
git answers them.

Cost: ~1 s for a handful of ``git init`` calls. Cheap enough for Tier A, and it means the dirty-tree
and not-a-checkout paths are genuinely exercised rather than asserted about a stub.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from studio.modelio.provenance import (
    MODEL_SUBMODULES,
    PROVENANCE_SCHEMA_VERSION,
    NotAGitCheckoutError,
    ProvenanceRecord,
    describe_checkout,
    record_for,
    repository_root,
)
from studio.resolve import ResolvedConfig, apply_change, resolve, set_override
from studio.schema import RunConfig


def _git(path: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "user.name=Test",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=str(path),
        check=True,
        capture_output=True,
    )


def _make_repo(path: Path, filename: str = "file.txt") -> Path:
    """A real git repo with one commit."""
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "--quiet")
    (path / filename).write_text("content\n", encoding="utf-8")
    _git(path, "add", filename)
    _git(path, "commit", "--quiet", "-m", "initial")
    return path


@pytest.fixture
def fake_sandbox(tmp_path: Path) -> Path:
    """A SANDBOX-shaped tree: a root repo with the three model submodules REGISTERED as such.

    Registered rather than merely nested, because the two differ to git: a nested repo the parent
    does not know about shows up in ``git status --porcelain`` as an untracked entry, so the parent
    reads as dirty. That is correct behaviour, and it is what the first version of this fixture
    tripped over -- the code was right and the fixture was lying about the shape of a real
    checkout.

    ``protocol.file.allow=always`` is required because git refuses local-path submodules by default
    (CVE-2022-39253). Safe here: the "remote" is a directory this test just created.
    """
    root = _make_repo(tmp_path / "SANDBOX")
    for name in MODEL_SUBMODULES:
        origin = _make_repo(tmp_path / "origins" / name)
        _git(
            root,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "--quiet",
            str(origin),
            name,
        )
    _git(root, "commit", "--quiet", "-m", "add submodules")
    return root


@pytest.fixture
def resolved() -> ResolvedConfig:
    return resolve(RunConfig())


@pytest.mark.tier_a
def test_the_record_carries_every_field_adr_006_requires(
    fake_sandbox: Path, resolved: ResolvedConfig
) -> None:
    """config hash, app version, SANDBOX SHA, three submodule SHAs, datasets, resolved config."""
    record = record_for(resolved, repo_root=fake_sandbox)

    assert record.schema_version == PROVENANCE_SCHEMA_VERSION
    assert record.config_hash == resolved.config.config_hash()
    assert record.studio_version  # studio.__version__, whatever it currently is
    assert len(record.sandbox.commit) == 40
    assert set(record.submodules) == set(MODEL_SUBMODULES)
    assert all(len(sub.commit) == 40 for sub in record.submodules.values())
    assert record.datasets == {}, "present and empty, so its absence is never ambiguous"
    assert record.recorded_at.tzinfo is not None, "a naive timestamp compares wrongly across zones"


@pytest.mark.tier_a
def test_the_recorded_config_is_the_resolved_one(
    fake_sandbox: Path, resolved: ResolvedConfig
) -> None:
    """ "What the model actually received", not what the user typed.

    The derived fields are the test: a record of the user's inputs would have ``None`` here, and
    would not let anyone reconstruct the run.
    """
    record = record_for(resolved, repo_root=fake_sandbox)
    injection = record.resolved_config["injection"]
    assert injection["plume_volume_cm3"] == 1.5e12
    assert injection["so2_initial_pptv"] == pytest.approx(3.309115922996412e9, rel=1e-15)
    assert record.resolved_config["schema_version"] == resolved.config.schema_version


@pytest.mark.tier_a
def test_an_override_is_recorded_with_the_value_in_force(fake_sandbox: Path) -> None:
    """A user-supplied derived value must be visible as an override, not silently indistinguishable
    from a computed one."""
    from studio.resolve import keep_override

    overridden = keep_override(
        set_override(resolve(RunConfig()), "injection.so2_initial_pptv", 5.0e9),
        "injection.so2_initial_pptv",
    )
    record = record_for(overridden, repo_root=fake_sandbox)
    assert record.overrides == {"injection.so2_initial_pptv": 5.0e9}
    assert record.resolved_config["injection"]["so2_initial_pptv"] == 5.0e9


@pytest.mark.tier_a
def test_a_clean_tree_is_reproducible(fake_sandbox: Path, resolved: ResolvedConfig) -> None:
    record = record_for(resolved, repo_root=fake_sandbox)
    assert record.is_reproducible
    assert record.dirty_checkouts == ()


@pytest.mark.tier_a
def test_a_dirty_sandbox_is_recorded_and_flags_the_run(
    fake_sandbox: Path, resolved: ResolvedConfig
) -> None:
    """The SHA no longer describes the code that ran -- exactly when someone wants to know."""
    (fake_sandbox / "file.txt").write_text("modified\n", encoding="utf-8")
    record = record_for(resolved, repo_root=fake_sandbox)

    assert record.sandbox.dirty
    assert record.sandbox.dirty_files, "the record must say WHAT was uncommitted"
    assert not record.is_reproducible
    assert record.dirty_checkouts == ("SANDBOX",)


@pytest.mark.tier_a
def test_an_untracked_file_counts_as_dirty(fake_sandbox: Path, resolved: ResolvedConfig) -> None:
    """``status --porcelain`` rather than ``diff --quiet``, on purpose.

    An untracked module that a run imported is exactly the kind of thing that makes a SHA a lie, and
    ``git diff`` would not see it.
    """
    (fake_sandbox / "scratch_module.py").write_text("x = 1\n", encoding="utf-8")
    assert record_for(resolved, repo_root=fake_sandbox).sandbox.dirty


@pytest.mark.tier_a
def test_a_dirty_submodule_flags_the_run_and_the_parent(
    fake_sandbox: Path, resolved: ResolvedConfig
) -> None:
    """The model lives in the submodules; a dirty one means the model that ran is not any commit.

    **Both** checkouts are flagged, and that is git being helpful rather than the record being
    imprecise: a registered submodule with a dirty working tree also shows up in the PARENT's
    ``status --porcelain`` as modified, because the parent's recorded submodule pointer no longer
    describes what is on disk. So an edited submodule cannot hide behind a clean-looking SANDBOX.
    """
    (fake_sandbox / "stratchem-jax" / "file.txt").write_text("edited\n", encoding="utf-8")
    record = record_for(resolved, repo_root=fake_sandbox)
    assert record.submodules["stratchem-jax"].dirty
    assert not record.is_reproducible
    assert record.dirty_checkouts == ("SANDBOX", "stratchem-jax")
    assert not record.submodules["tuvx-jax"].dirty, "only the edited submodule is dirty"


@pytest.mark.tier_a
def test_not_a_git_checkout_raises(tmp_path: Path, resolved: ResolvedConfig) -> None:
    """An empty SHA looks like an answer, so this refuses to produce one (ADR-005)."""
    plain = tmp_path / "not-a-repo"
    plain.mkdir()
    for name in MODEL_SUBMODULES:
        (plain / name).mkdir()
    with pytest.raises(NotAGitCheckoutError):
        record_for(resolved, repo_root=plain)


@pytest.mark.tier_a
def test_a_missing_submodule_raises(tmp_path: Path, resolved: ResolvedConfig) -> None:
    """The SANDBOX SHA alone does not pin the model (ADR-001), so a missing submodule is fatal."""
    root = _make_repo(tmp_path / "partial")
    _make_repo(root / "tuvx-jax")  # the other two are absent
    with pytest.raises(NotAGitCheckoutError, match="submodule"):
        record_for(resolved, repo_root=root)


@pytest.mark.tier_a
def test_a_repo_with_no_commits_raises(tmp_path: Path, resolved: ResolvedConfig) -> None:
    """``rev-parse HEAD`` has nothing to report, which is a failure rather than an empty string."""
    root = tmp_path / "empty"
    root.mkdir()
    _git(root, "init", "--quiet")
    with pytest.raises(NotAGitCheckoutError):
        describe_checkout(root)


@pytest.mark.tier_a
def test_a_stale_config_is_refused(fake_sandbox: Path) -> None:
    """Recording provenance for an inconsistent config would give it a respectable pedigree."""
    from studio.resolve import InconsistentConfigError

    stale = apply_change(
        set_override(resolve(RunConfig()), "injection.so2_initial_pptv", 5.0e9),
        "site.temperature_k",
        213.0,
    )
    with pytest.raises(InconsistentConfigError):
        record_for(stale, repo_root=fake_sandbox)


@pytest.mark.tier_a
def test_the_record_is_immutable(fake_sandbox: Path, resolved: ResolvedConfig) -> None:
    """Written once, before the run, never mutated (ADR-006)."""
    record = record_for(resolved, repo_root=fake_sandbox)
    with pytest.raises(ValueError, match="frozen"):
        record.config_hash = "tampered"  # type: ignore[misc]


@pytest.mark.tier_a
def test_the_record_round_trips_through_json(
    fake_sandbox: Path, resolved: ResolvedConfig, tmp_path: Path
) -> None:
    original = record_for(resolved, repo_root=fake_sandbox)
    restored = ProvenanceRecord.read(original.write(tmp_path / "provenance.json"))
    assert restored == original
    assert restored.is_reproducible == original.is_reproducible


@pytest.mark.tier_a
def test_the_repository_root_is_derived_from_the_package_not_the_cwd() -> None:
    """An API process started anywhere must still record the checkout the code came from."""
    root = repository_root()
    assert (root / "studio").is_dir()
    assert (root / "coupled").is_dir()


@pytest.mark.tier_a
def test_this_checkout_can_be_pinned(resolved: ResolvedConfig) -> None:
    """The real repository, not a synthetic one: submodules present, SHAs readable.

    Skips where the submodules are absent, which is the same condition every other model-touching
    test skips on -- including in CI.
    """
    root = repository_root()
    if not all((root / name / ".git").exists() for name in MODEL_SUBMODULES):
        pytest.skip("model submodules not checked out (`git submodule update --init`)")
    record = record_for(resolved, repo_root=root)
    assert len(record.sandbox.commit) == 40
    assert set(record.submodules) == set(MODEL_SUBMODULES)
