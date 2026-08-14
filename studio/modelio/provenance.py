# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""What produced a result: config identity, app version, model version (ADR-006).

The requirement is that **any figure traces to an exact configuration, model version and input
dataset checksum**. What existed before this module was a hand-composed case ID -- good design for a
fixed factorial, and genuinely useful, but it captures the *axes* rather than the resolved
configuration: everything held fixed across the 810-run ensemble (``ion_pair_rate=30``,
``day_of_year=172``, the whole background composition) is invisible in it, so two ensembles
differing only in a "fixed" value collide.

This is **the one place Studio shells out to git**, and it is deliberately strict about it:

* "Not a git checkout" **raises**. An empty SHA in a provenance record is worse than no record at
  all, because it looks like an answer (ADR-005).
* **A dirty working tree is recorded and flags the run.** Uncommitted changes mean the SHA does not
  describe the code that ran, and that is exactly the case where someone later wants to know.
* Submodule SHAs are read individually. The SANDBOX SHA alone does not pin the model: the three
  submodules are what contain it (ADR-001), and a submodule pointer that has moved without a commit
  here is invisible in the parent SHA.

The record is written **before execution begins** -- a run that dies in minute three still has its
provenance -- and is never mutated. An edited config is a new config and a new run (ADR-004).

Note what this module does NOT do: it does not import ``coupled``. Model *identity* is a question
about the checkout, not about the model's API, so recording it costs nothing.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

import studio
from studio.resolve import ResolvedConfig

#: The submodules that, together with the SANDBOX SHA, pin the model exactly (ADR-001).
MODEL_SUBMODULES: Final[tuple[str, ...]] = ("tuvx-jax", "stratchem-jax", "tomas-jax")

#: Bumped if the record's shape changes, so a stored record is never read under new semantics.
PROVENANCE_SCHEMA_VERSION: Final = "0.1.0"


class NotAGitCheckoutError(RuntimeError):
    """The repository root is not a git checkout, so the model cannot be pinned.

    Raised rather than recording an empty SHA: a provenance record that cannot say what ran is not a
    provenance record, and one that says ``""`` looks like it can.
    """


class GitCheckout(BaseModel):
    """The state of one checkout: its commit, and whether it was clean."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    commit: str
    #: True when ``git status --porcelain`` was non-empty. The commit then does NOT describe the
    #: code that ran, which is precisely when someone will want to know.
    dirty: bool
    #: The porcelain output, truncated. Enough to see WHAT was uncommitted without storing a diff.
    dirty_files: tuple[str, ...] = ()


class ProvenanceRecord(BaseModel):
    """Everything needed to say what produced a result. Written before the run, never mutated."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = PROVENANCE_SCHEMA_VERSION
    recorded_at: datetime
    #: Identity of the resolved config (ADR-006). Doubles as the cache key.
    config_hash: str
    #: The orchestration layer's version. The model has none -- it is pinned by SHA alone.
    studio_version: str
    #: The SANDBOX checkout: the coupling layer plus the paper pipeline.
    sandbox: GitCheckout
    #: The three submodules that contain the model itself.
    submodules: dict[str, GitCheckout]
    #: Input datasets consulted, by identifier -> checksum. Empty in Phase 0: the schema carries no
    #: dataset inputs yet. Present and empty rather than omitted, so its absence is never ambiguous.
    datasets: dict[str, str] = Field(default_factory=dict)
    #: The RESOLVED, post-derivation parameter set -- what the model actually received, not what the
    #: user typed. This is the field that makes the record self-contained.
    resolved_config: dict[str, Any]
    #: Any derived field the user overrode, with the value that was in force.
    overrides: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_reproducible(self) -> bool:
        """True when every checkout was clean, so the SHAs fully describe the code that ran."""
        return not self.sandbox.dirty and not any(sub.dirty for sub in self.submodules.values())

    @property
    def dirty_checkouts(self) -> tuple[str, ...]:
        """Names of checkouts with uncommitted changes. Empty when the run is reproducible."""
        names = ["SANDBOX"] if self.sandbox.dirty else []
        names.extend(name for name, sub in sorted(self.submodules.items()) if sub.dirty)
        return tuple(names)

    def write(self, path: Path) -> Path:
        """Write as JSON. Called before the run starts."""
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: Path) -> ProvenanceRecord:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def _git(repo_root: Path, *args: str) -> str:
    """Run git in ``repo_root`` and return stdout, or raise.

    Raises:
        NotAGitCheckoutError: If git fails for any reason -- not installed, not a checkout, a broken
            submodule. All of them mean the same thing here: the code that ran cannot be identified.
    """
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:  # git itself is missing
        raise NotAGitCheckoutError(
            f"git is not available, so the model cannot be pinned for a run in {repo_root}. "
            f"Provenance is required (ADR-006); it is not optional metadata."
        ) from exc
    if proc.returncode != 0:
        raise NotAGitCheckoutError(
            f"`git {' '.join(args)}` failed in {repo_root}: {proc.stderr.strip() or 'no output'}. "
            f"A provenance record with no commit is not a provenance record, so this raises rather "
            f"than recording an empty SHA."
        )
    return proc.stdout.strip()


def describe_checkout(path: Path, *, dirty_file_limit: int = 20) -> GitCheckout:
    """The commit and cleanliness of the checkout at ``path``.

    ``git status --porcelain`` rather than ``diff --quiet`` because it also reports untracked
    files, and an untracked module that a run imported is exactly the kind of thing that makes a
    SHA a lie.
    """
    commit = _git(path, "rev-parse", "HEAD")
    status = _git(path, "status", "--porcelain")
    lines = tuple(line.strip() for line in status.splitlines() if line.strip())
    return GitCheckout(
        path=str(path),
        commit=commit,
        dirty=bool(lines),
        dirty_files=lines[:dirty_file_limit],
    )


def repository_root() -> Path:
    """The SANDBOX root, derived from this file's location rather than the working directory.

    Deliberately not ``Path.cwd()``: a run submitted by an API process started anywhere at all must
    still record the checkout that the code came from.
    """
    return Path(studio.__file__).resolve().parent.parent


def record_for(
    config: ResolvedConfig,
    *,
    repo_root: Path | None = None,
    datasets: dict[str, str] | None = None,
) -> ProvenanceRecord:
    """Build the record for a config, at submit time.

    Raises:
        InconsistentConfigError: If the config has stale overrides. Recording provenance for a
            config whose numbers do not follow from each other would give an inconsistent run a
            respectable-looking pedigree.
        NotAGitCheckoutError: If any checkout cannot be identified.
    """
    config.require_consistent()
    root = repo_root or repository_root()
    submodules = {}
    for name in MODEL_SUBMODULES:
        path = root / name
        if not (path / ".git").exists():
            raise NotAGitCheckoutError(
                f"submodule {name} is not checked out at {path} (`git submodule update --init`). "
                f"The SANDBOX SHA alone does not pin the model: the three submodules are what "
                f"contain it (ADR-001)."
            )
        submodules[name] = describe_checkout(path)
    return ProvenanceRecord(
        recorded_at=datetime.now(UTC),
        config_hash=config.config.config_hash(),
        studio_version=studio.__version__,
        sandbox=describe_checkout(root),
        submodules=submodules,
        datasets=dict(datasets or {}),
        resolved_config=config.config.model_dump(mode="json"),
        overrides={path: record.value for path, record in sorted(config.overrides.items())},
    )


__all__ = [
    "MODEL_SUBMODULES",
    "PROVENANCE_SCHEMA_VERSION",
    "GitCheckout",
    "NotAGitCheckoutError",
    "ProvenanceRecord",
    "describe_checkout",
    "record_for",
    "repository_root",
]
