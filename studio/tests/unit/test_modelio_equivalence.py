# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The equivalence test: the schema is a faithful superset of what the ensemble actually ran.

``to_scenario`` applied to the golden ``RunConfig`` must produce a ``CoupledScenario``
**field-for-field identical** to ``run_ensemble.build_scenario()`` for case
``30N_20km__sabr220__D2med__a1p0__nuc1__cg1``. Tolerance: **exact**, on every field, including the
floats -- this is not a physics comparison, it is a claim that two code paths build the same object,
and "close" would mean one of them is doing arithmetic the other is not.

Proving it here is much cheaper and sharper than discovering it from a diverging result: a mismatch
names the field, now, instead of showing up as a 3 % difference in particle number after a 4-minute
run and a day of bisection.

These tests need the model, so they need the submodules checked out. CI does not check them out
(see ``.github/workflows/studio-ci.yml``), so this skips there and runs locally -- the same
asymmetry as the ``air_number_density`` mirror check, and worth the same caution.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from studio.resolve import resolve, set_override
from studio.schema import BackgroundAerosol, DilutionRegime, PhotolysisMode, RunConfig

#: The axis values for the golden case, exactly as ``run_ensemble.py:61-76`` spells them.
GOLDEN_AXES: dict[str, Any] = {
    "lat_alt": ("30N_20km", 30.0, 210.0, 55.0, 6.9104),
    "background": ("sabr220", "sabr_220", 20.0),
    "dilution": ("D2med", "D2"),
    "sticking": ("a1p0", 1.0),
    "nucleation": ("nuc1", 1.0),
    "coag": ("cg1", 1.0),
}


@pytest.fixture(scope="module")
def build_scenario(repo_root: Path) -> Any:
    """``run_ensemble.build_scenario``, or skip if the model is not checked out."""
    if not (repo_root / "stratchem-jax" / "config.py").is_file():
        pytest.skip(
            "model submodules not checked out (`git submodule update --init`); the equivalence "
            "test compares against run_ensemble.build_scenario, which needs them"
        )
    from coupled.paper_ensemble.run_ensemble import build_scenario as builder

    return builder


@pytest.fixture(scope="module")
def studio_scenario() -> Any:
    from studio.modelio.scenario import to_scenario

    return to_scenario(resolve(RunConfig()))


@pytest.mark.tier_a
def test_the_golden_case_is_the_schemas_default(build_scenario: Any) -> None:
    """``RunConfig()`` with no arguments IS the golden case; nothing has to be set up to get it."""
    reference = build_scenario(GOLDEN_AXES)
    # temperature_k is DERIVED since 0.4.0 (dataset selector), so the default config resolves it;
    # under the default USER dataset it must equal the entered value, which is the golden case's.
    config = resolve(RunConfig()).config
    assert config.site.temperature_k == reference.T
    assert config.site.pressure_mbar == reference.P
    assert config.background.aerosol is BackgroundAerosol.SABR_220
    assert config.dilution.regime is DilutionRegime.D2
    assert config.chemistry.photolysis is PhotolysisMode.TUVX


@pytest.mark.tier_a
def test_to_scenario_is_field_for_field_identical(
    build_scenario: Any, studio_scenario: Any
) -> None:
    """The whole point of the task. Every field, exact equality, no exceptions."""
    reference = dataclasses.asdict(build_scenario(GOLDEN_AXES))
    produced = dataclasses.asdict(studio_scenario)

    assert set(produced) == set(reference), "the two scenarios have different field sets"
    differing = {
        key: (reference[key], produced[key]) for key in reference if produced[key] != reference[key]
    }
    assert differing == {}, (
        "to_scenario diverges from run_ensemble.build_scenario for the golden case "
        f"(reference, produced): {differing}. The schema is meant to be a faithful superset; "
        f"either the mapping is wrong or the schema default is."
    )


@pytest.mark.tier_a
def test_the_initial_so2_matches_to_the_last_bit(build_scenario: Any, studio_scenario: Any) -> None:
    """Called out separately because it is the one value Studio DERIVES rather than passes through.

    Everything else is a copy; this one goes mass -> number density -> mixing ratio through
    ``studio.science`` while the ensemble does the same arithmetic inline. Exact equality is the
    evidence that consolidating that derivation changed nothing.
    """
    reference = build_scenario(GOLDEN_AXES)
    assert studio_scenario.concentrations["SO2"] == reference.concentrations["SO2"]
    assert studio_scenario.concentrations == reference.concentrations


@pytest.mark.tier_a
def test_the_constant_regime_maps_to_the_models_empty_string(build_scenario: Any) -> None:
    """The one enum value that is not the model's own string (``enums.py``).

    Asserted against a real ``CoupledScenario`` rather than against the mapping table, because the
    model validates ``dilution_regime`` in ``__post_init__`` -- if ``""`` ever stopped being the
    spelling, this fails here rather than in a run.
    """
    from studio.modelio.scenario import to_scenario

    config = RunConfig.model_validate(
        {**RunConfig().model_dump(), "dilution": {"regime": "constant"}}
    )
    assert to_scenario(resolve(config)).dilution_regime == ""


@pytest.mark.tier_a
@pytest.mark.parametrize(
    ("regime", "expected"),
    [("D1", "D1"), ("D2", "D2"), ("D3", "D3"), ("D5", "D5"), ("burst", "burst")],
)
def test_every_other_regime_passes_through_unchanged(
    build_scenario: Any, regime: str, expected: str
) -> None:
    """The claim that the enum values ARE the model's strings, checked rather than trusted."""
    from studio.modelio.scenario import to_scenario

    config = RunConfig.model_validate({**RunConfig().model_dump(), "dilution": {"regime": regime}})
    assert to_scenario(resolve(config)).dilution_regime == expected


@pytest.mark.tier_a
def test_an_unresolved_config_is_refused(build_scenario: Any) -> None:
    """A config that never went through the resolver has ``so2_initial_pptv = None``.

    Handing that to the model would start the plume with no SO2 -- a run that completes and means
    nothing. It raises instead (ADR-005).
    """
    from studio.modelio.scenario import to_scenario

    with pytest.raises(ValueError, match="so2_initial_pptv is unresolved"):
        to_scenario(RunConfig())


@pytest.mark.tier_a
def test_a_stale_config_is_refused(build_scenario: Any) -> None:
    """Converting a stale config would hand the model numbers that do not follow from each other."""
    from studio.modelio.scenario import to_scenario
    from studio.resolve import InconsistentConfigError, apply_change

    stale = apply_change(
        set_override(resolve(RunConfig()), "injection.so2_initial_pptv", 5.0e9),
        "site.temperature_k",
        213.0,
    )
    with pytest.raises(InconsistentConfigError):
        to_scenario(stale)


@pytest.mark.tier_a
def test_a_resolved_override_reaches_the_model(build_scenario: Any) -> None:
    """The other half: an override the user has settled must actually arrive."""
    from studio.modelio.scenario import to_scenario
    from studio.resolve import keep_override

    overridden = set_override(resolve(RunConfig()), "injection.so2_initial_pptv", 5.0e9)
    assert (
        to_scenario(keep_override(overridden, "injection.so2_initial_pptv")).concentrations["SO2"]
        == 5.0e9
    )
