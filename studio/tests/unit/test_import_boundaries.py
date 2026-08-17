# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The package boundaries from ADR-001, enforced rather than documented.

``studio.schema``, ``studio.science`` and ``studio.resolve`` must be usable from a bare Python
session: no ``coupled``, no JAX, no database, no web framework. This is not tidiness.
``coupled.tomas_bridge`` and ``coupled.driver`` import JAX (and set ``jax_enable_x64``) at module
scope, and an API that validates a form on every keystroke cannot pay a JAX import.

Two complementary checks:

* a RUNTIME one, importing each clean package in a fresh interpreter and inspecting ``sys.modules``;
* a STATIC one, scanning the source tree, which catches an import added inside a function body where
  the runtime check would not reach it. ``CoupledScenario.__post_init__`` used to hold exactly that
  pattern -- a function-body ``from coupled.tomas_bridge import BACKGROUND_MODES`` that made merely
  *constructing* a scenario cost ~1 s (fixed in task 0.8 by ``coupled/backgrounds.py``) -- so it is
  not a hypothetical.

The runtime check must run in a fresh interpreter. Inspecting ``sys.modules`` in-process would be
meaningless: by then pytest and its plugins have imported plenty, and another test module importing
``studio.modelio`` -- which *is* allowed to import ``coupled`` -- would poison the result.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
STUDIO_ROOT = REPO_ROOT / "studio"

#: Packages that must stay importable without the model. See ``studio/__init__.py``.
#: ``studio.resolve`` is here because the API resolves a config on every keystroke.
CLEAN_PACKAGES = ["studio.schema", "studio.science", "studio.resolve"]

#: The single package permitted to import ``coupled`` (ADR-001). Adding to this set requires
#: amending ADR-001 in the same change.
MODEL_SEAM_PACKAGES = {"studio.modelio"}

#: Import roots that must not appear in a clean package. ``coupled`` is the model seam; ``jax`` and
#: ``jaxlib`` are what importing it costs; the rest would make the package unusable outside a
#: configured server.
FORBIDDEN_ROOTS = ["coupled", "jax", "jaxlib", "sqlalchemy", "fastapi", "alembic"]

_PROBE = textwrap.dedent("""
    import importlib, json, sys
    importlib.import_module({package!r})
    roots = {{name.split(".")[0] for name in sys.modules}}
    print(json.dumps(sorted(roots & set({forbidden!r}))))
    """)


def _forbidden_imports_after(package: str) -> list[str]:
    """Import ``package`` in a fresh interpreter; return which forbidden roots it pulled in."""
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(package=package, forbidden=FORBIDDEN_ROOTS)],
        capture_output=True,
        text=True,
        # Run from the repository root, where `coupled` IS importable. The test is only meaningful
        # if the forbidden import is available and simply not made.
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, f"importing {package} failed:\n{proc.stderr}"
    result: list[str] = json.loads(proc.stdout.strip())
    return result


def _modules_importing_coupled() -> set[str]:
    """Dotted module names under ``studio/`` whose source imports ``coupled``, at any nesting depth.

    Walks the AST rather than grepping, so a string mentioning ``coupled`` in a docstring -- of
    which this package has many -- is not a false positive.
    """
    offenders: set[str] = set()
    for path in sorted(STUDIO_ROOT.rglob("*.py")):
        if "tests" in path.relative_to(STUDIO_ROOT).parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                # level > 0 is a relative import, which can never reach `coupled`
                roots = {node.module.split(".")[0]} if node.module and not node.level else set()
            else:
                continue
            if "coupled" in roots:
                rel = path.relative_to(REPO_ROOT).with_suffix("")
                parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
                offenders.add(".".join(parts))
    return offenders


@pytest.mark.tier_a
@pytest.mark.parametrize("package", CLEAN_PACKAGES)
def test_clean_packages_do_not_import_the_model(package: str) -> None:
    """ADR-001: these packages must not reach into ``coupled`` or drag in JAX."""
    leaked = _forbidden_imports_after(package)
    assert leaked == [], (
        f"{package} imported {leaked}, breaking the boundary in ADR-001. "
        f"{CLEAN_PACKAGES} must be usable from a bare Python session; "
        f"only {sorted(MODEL_SEAM_PACKAGES)} may import `coupled`."
    )


@pytest.mark.tier_a
def test_only_the_model_seam_imports_coupled() -> None:
    """Statically: no module outside ``studio.modelio`` may import ``coupled``.

    Catches imports hidden inside function bodies, which the runtime check above cannot see.
    """
    offenders = _modules_importing_coupled()
    stray = {
        module
        for module in offenders
        # a submodule of the seam (studio.modelio.scenario) is the seam; a sibling package is not
        if not any(module == seam or module.startswith(f"{seam}.") for seam in MODEL_SEAM_PACKAGES)
    }
    assert stray == set(), (
        f"{sorted(stray)} import `coupled`, but ADR-001 names {sorted(MODEL_SEAM_PACKAGES)} as the "
        f"only model seam. Move the call behind studio.modelio, or amend ADR-001 in this change."
    )
