# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the Studio suite.

Run it explicitly -- root ``pytest`` sets ``testpaths = ["coupled/tests"]`` and that contract is
left alone (CLAUDE.md documents three separate suites; this is a fourth):

    pytest studio/tests -m tier_a      # fast; what CI runs
    pytest studio/tests -m tier_b      # full-case golden reproduction; minutes, nightly/manual

Tier markers are declared in the root ``pyproject.toml`` so ``--strict-markers`` accepts them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

#: SANDBOX root -- the directory `coupled/` and `studio/` live in. Tests that shell out to a fresh
#: interpreter need this as cwd, since neither package is pip-installed (ADR-001).
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The SANDBOX repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def paper_ensemble_runs(repo_root: Path) -> Path:
    """Directory holding the archived 810-run ensemble, or skip.

    ``coupled/paper_ensemble/runs*/`` is gitignored and regenerable, so it is absent on a fresh
    clone. Golden Tier B needs it and skips without it -- deliberately a skip and not a failure,
    because its absence says nothing about the code under test.
    """
    runs = repo_root / "coupled" / "paper_ensemble" / "runs"
    if not runs.is_dir():
        pytest.skip(
            f"archived ensemble not present at {runs} (gitignored and regenerable) -- "
            f"Tier B golden reproduction needs it; see "
            f"docs/studio/adr/ADR-009-golden-file-strategy.md"
        )
    return runs
