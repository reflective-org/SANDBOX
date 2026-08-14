# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""``air_number_density`` is a mirror of the model's. This is what makes that acceptable.

``studio.science`` may not import the model (ADR-001), so this one relation is duplicated. A
duplicate is only safe if divergence is detectable, so the test below imports the model's own
implementation and asserts EXACT agreement across the parameter range the runs actually use.

It skips -- rather than fails -- when the ``stratchem-jax`` submodule is not checked out, because
its absence says nothing about the code under test. CI does not check out the private submodules
(see ``.github/workflows/studio-ci.yml``), so in CI this is a skip and locally it is a real check.
That asymmetry is deliberate but worth knowing: a divergence would be caught on a developer machine,
not by CI.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from studio.science import air_number_density

#: The T-p corners the ensemble and the flagship runs use: 55 hPa / 210 K and 213 K (30N 20 km),
#: 120 hPa / 210 K (60N 15 km), 55 hPa / 215 K (the D1 clean run).
RUN_CONDITIONS = [(55.0, 210.0), (55.0, 213.0), (120.0, 210.0), (55.0, 215.0)]


@pytest.mark.tier_a
def test_known_value_at_the_golden_site() -> None:
    """55 hPa, 210 K -> 1.8956916099773243e18 molec cm^-3.

    Tolerance 1e-12 relative: the value is one multiply-divide chain on exact constants, so the only
    admissible difference is float64 rounding. Quoted to full precision on purpose -- a rounded
    literal here would have passed while hiding a wrong constant.
    """
    assert air_number_density(55.0, 210.0) == pytest.approx(1.8956916099773243e18, rel=1e-12)


@pytest.mark.tier_a
def test_scales_as_the_ideal_gas_law() -> None:
    """Linear in pressure, inverse in temperature -- the property, not just a value."""
    assert air_number_density(110.0, 210.0) == pytest.approx(
        2.0 * air_number_density(55.0, 210.0), rel=1e-15
    )
    assert air_number_density(55.0, 420.0) == pytest.approx(
        0.5 * air_number_density(55.0, 210.0), rel=1e-15
    )


@pytest.mark.tier_a
@pytest.mark.parametrize(("pressure", "temperature"), [(0.0, 210.0), (-1.0, 210.0)])
def test_non_physical_pressure_raises(pressure: float, temperature: float) -> None:
    with pytest.raises(ValueError, match="pressure must be > 0"):
        air_number_density(pressure, temperature)


@pytest.mark.tier_a
@pytest.mark.parametrize(("pressure", "temperature"), [(55.0, 0.0), (55.0, -3.0)])
def test_non_physical_temperature_raises(pressure: float, temperature: float) -> None:
    """Zero kelvin would be a division by zero; ``inf`` is not a useful answer (ADR-005)."""
    with pytest.raises(ValueError, match="temperature must be > 0"):
        air_number_density(pressure, temperature)


@pytest.mark.tier_a
def test_agrees_exactly_with_the_models_own_implementation(repo_root: Path) -> None:
    """The mirror check. Tolerance: exact -- same formula and constants, so any difference is real.

    Runs in a subprocess with ``stratchem-jax`` on ``sys.path``: importing the model's flat
    ``config`` module in-process would leave a ``config`` in ``sys.modules`` for every test that
    follows, which is exactly the kind of cross-test contamination the import-boundary test warns
    about.
    """
    stratchem = repo_root / "stratchem-jax"
    if not (stratchem / "config.py").is_file():
        pytest.skip(
            f"stratchem-jax submodule not checked out at {stratchem} "
            f"(`git submodule update --init`); the mirror check needs the model's own version"
        )
    probe = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(stratchem)!r})
        from config import air_number_density as model_impl
        print("\\n".join(repr(model_impl(p, t)) for p, t in {RUN_CONDITIONS!r}))
        """)
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=str(repo_root)
    )
    assert proc.returncode == 0, f"could not evaluate the model's implementation:\n{proc.stderr}"

    model_values = [float(line) for line in proc.stdout.split()]
    studio_values = [air_number_density(p, t) for p, t in RUN_CONDITIONS]
    assert model_values == studio_values, (
        "studio.science.air_number_density has diverged from stratchem-jax/config.py:55. "
        "It is a deliberate mirror (studio.science may not import the model); if the model's "
        "relation changed, change this one in the same commit and say so."
    )
