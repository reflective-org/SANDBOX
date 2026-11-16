# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The emission system: one degree of freedom (schema 0.3.0).

The paper ensemble specifies the release geometrically -- 1 t over 15 km -- because that is what the
model consumes. A deployment is specified operationally: a platform at some speed, emitting at some
rate, for some time. These are not alternative models of the release; they are the same release
described from different ends, related by

    t = M / R        and        L = v * t

which leaves exactly ONE free choice among {rate, duration, length} once the mass and the speed are
fixed. ``injection.emission_input`` names which one is given, and the other two are derived.

What these tests protect:

* **All three descriptions agree.** Whichever you enter, the released plume is the same one. If it
  were not, the choice would be a physics decision disguised as a preference.
* **The default run is unchanged**, so no archived comparison shifts.
* **Nothing is ever unknown.** Every one of rate, duration and length always has a value, because
  the rate is what an operator recognises and the length is what sets the volume -- a "not
  applicable" in either would leave the stage half-legible.
"""

from __future__ import annotations

import pytest

from studio.resolve import apply_change, resolve
from studio.schema import RunConfig
from studio.schema.enums import EmissionInput

#: The default release, stated once: 1 t over 15 km at 250 m/s is 60 s of emission at 16.667 kg/s.
DEFAULT_MASS_KG = 1000.0
DEFAULT_SPEED_M_S = 250.0
DEFAULT_LENGTH_M = 15000.0
DEFAULT_DURATION_S = DEFAULT_LENGTH_M / DEFAULT_SPEED_M_S
DEFAULT_RATE_KG_S = DEFAULT_MASS_KG / DEFAULT_DURATION_S


def _with(selection: str, **given: object) -> RunConfig:
    return RunConfig.model_validate({"injection": {"emission_input": selection, **given}})


@pytest.mark.tier_a
def test_the_default_is_the_paper_ensembles_geometry() -> None:
    """Existing configs keep their meaning: the length is still what is entered."""
    config = RunConfig()
    assert config.injection.emission_input is EmissionInput.TRACK_LENGTH
    injection = resolve(config).config.injection
    assert injection.plume_length_m == DEFAULT_LENGTH_M
    assert injection.plume_volume_cm3 == 1.5e12


@pytest.mark.tier_a
@pytest.mark.parametrize(
    "selection",
    [EmissionInput.TRACK_LENGTH, EmissionInput.EMISSION_RATE, EmissionInput.EMISSION_DURATION],
)
def test_every_selection_describes_the_same_default_release(selection: EmissionInput) -> None:
    """Switching which quantity is entered must not move the plume.

    The three defaults are written as expressions of each other in the schema for exactly this
    reason. Agreement is to ~1e-16 rather than bit-exact: entering the RATE means the duration is
    ``1000 / (1000/60)``, which is ``59.99999999999999`` in IEEE 754, not 60. The initial mixing
    ratio -- the number the model actually consumes -- is identical to the last bit regardless.
    """
    injection = resolve(_with(selection.value)).config.injection
    reference = resolve(RunConfig()).config.injection

    assert injection.emission_duration_s == pytest.approx(DEFAULT_DURATION_S, rel=1e-12)
    assert injection.emission_rate_kg_s == pytest.approx(DEFAULT_RATE_KG_S, rel=1e-12)
    assert injection.plume_length_m == pytest.approx(DEFAULT_LENGTH_M, rel=1e-12)
    assert (
        injection.so2_initial_pptv == reference.so2_initial_pptv
    ), "the quantity the model consumes must be identical, not merely close"


@pytest.mark.tier_a
def test_nothing_is_ever_not_applicable() -> None:
    """All three of rate, duration and length carry a value under every selection."""
    for selection in EmissionInput:
        injection = resolve(_with(selection.value)).config.injection
        assert injection.emission_duration_s is not None, selection
        assert injection.emission_rate_kg_s is not None, selection
        assert injection.plume_length_m is not None, selection


@pytest.mark.tier_a
def test_giving_the_rate_derives_the_duration_then_the_length() -> None:
    """t = M/R, then L = v.t -- the chain as specified."""
    injection = resolve(
        _with("emission_rate", given_emission_rate_kg_s=5.0, platform_speed_m_s=200.0)
    ).config.injection
    assert injection.emission_duration_s == pytest.approx(1000.0 / 5.0)  # 200 s
    assert injection.plume_length_m == pytest.approx(200.0 * 200.0)  # 40 km
    assert injection.emission_rate_kg_s == 5.0, "the entered value is passed through unchanged"
    assert injection.plume_volume_cm3 == pytest.approx(40000.0 * 10.0 * 10.0 * 1e6)


@pytest.mark.tier_a
def test_giving_the_duration_derives_the_rate_then_the_length() -> None:
    """R = M/t, L = v.t."""
    injection = resolve(
        _with("emission_duration", given_emission_duration_s=100.0)
    ).config.injection
    assert injection.emission_duration_s == 100.0
    assert injection.emission_rate_kg_s == pytest.approx(1000.0 / 100.0)  # 10 kg/s
    assert injection.plume_length_m == pytest.approx(250.0 * 100.0)  # 25 km


@pytest.mark.tier_a
def test_giving_the_length_derives_the_duration_then_the_rate() -> None:
    """t = L/v, R = M/t -- the ensemble's direction, and the default."""
    injection = resolve(_with("track_length", given_track_length_m=30000.0)).config.injection
    assert injection.plume_length_m == 30000.0
    assert injection.emission_duration_s == pytest.approx(30000.0 / 250.0)  # 120 s
    assert injection.emission_rate_kg_s == pytest.approx(1000.0 / 120.0)  # 8.33 kg/s


@pytest.mark.tier_a
def test_the_concentration_follows_the_length_whichever_way_it_was_reached() -> None:
    """Twice the track, half the concentration -- by rate, by duration or by length alike."""
    reference = resolve(RunConfig()).config.injection
    doubled = {
        "by length": _with("track_length", given_track_length_m=2 * DEFAULT_LENGTH_M),
        "by duration": _with("emission_duration", given_emission_duration_s=2 * DEFAULT_DURATION_S),
        "by rate": _with("emission_rate", given_emission_rate_kg_s=DEFAULT_RATE_KG_S / 2),
        "by speed": _with("track_length", platform_speed_m_s=2 * DEFAULT_SPEED_M_S),
    }
    for how, config in doubled.items():
        injection = resolve(config).config.injection
        if how == "by speed":
            # Speed with the length FIXED changes the duration, not the plume: the same mass over
            # the same track, flown faster. The concentration must not move.
            assert injection.plume_length_m == pytest.approx(DEFAULT_LENGTH_M), how
            assert injection.so2_initial_pptv == pytest.approx(reference.so2_initial_pptv), how
            continue
        assert injection.plume_length_m == pytest.approx(2 * DEFAULT_LENGTH_M, rel=1e-12), how
        assert injection.so2_initial_pptv == pytest.approx(
            reference.so2_initial_pptv / 2, rel=1e-12
        ), how


@pytest.mark.tier_a
def test_a_faster_platform_at_a_fixed_rate_lays_a_longer_thinner_plume() -> None:
    """With the RATE fixed, speed does change the plume: same mass, more track."""
    slow = resolve(_with("emission_rate", platform_speed_m_s=100.0)).config.injection
    fast = resolve(_with("emission_rate", platform_speed_m_s=400.0)).config.injection
    assert fast.emission_duration_s == pytest.approx(
        slow.emission_duration_s
    ), "duration is mass over rate and does not involve speed"
    assert fast.plume_length_m == pytest.approx(4.0 * slow.plume_length_m)
    assert fast.so2_initial_pptv == pytest.approx(slow.so2_initial_pptv / 4.0, rel=1e-12)


@pytest.mark.tier_a
def test_the_unselected_values_are_kept_but_inert() -> None:
    """Switching selection must not lose what was typed into the others."""
    injection = resolve(
        _with("emission_rate", given_track_length_m=99000.0, given_emission_rate_kg_s=5.0)
    ).config.injection
    assert injection.given_track_length_m == 99000.0, "kept, not erased"
    # 1000 kg at 5 kg/s is 200 s, and 200 s at the default 250 m/s is 50 km -- not the 99 km typed.
    assert injection.plume_length_m == pytest.approx(50000.0), "but not used under this selection"


@pytest.mark.tier_a
def test_changing_the_selection_recomputes_the_whole_chain() -> None:
    """The selector is an input like any other, so switching it propagates through the DAG."""
    start = resolve(_with("emission_rate", given_emission_rate_kg_s=5.0, platform_speed_m_s=200.0))
    assert start.config.injection.plume_length_m == pytest.approx(40000.0)

    switched = apply_change(start, "injection.emission_input", EmissionInput.TRACK_LENGTH)
    assert switched.config.injection.plume_length_m == DEFAULT_LENGTH_M
    assert switched.config.injection.emission_duration_s == pytest.approx(
        75.0
    ), "15 km at the 200 m/s still in force"
    assert switched.is_consistent


@pytest.mark.tier_a
@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("given_emission_rate_kg_s", 0.0),
        ("given_emission_duration_s", 0.0),
        ("given_track_length_m", 0.0),
        ("platform_speed_m_s", 0.0),
    ],
)
def test_degenerate_inputs_are_refused(field_name: str, value: float) -> None:
    """Fail loud (ADR-005). A zero anywhere in this chain is an infinity somewhere downstream."""
    with pytest.raises(ValueError, match=r"greater than 0|must be positive|must be > 0"):
        resolve(_with("track_length", **{field_name: value}))


@pytest.mark.tier_a
def test_the_speed_default_is_marked_as_a_choice_not_a_measurement() -> None:
    """250 m/s is a decision, and the schema has to say so rather than imply an airframe.

    Nothing in the model or the paper ensemble carries a platform speed, so this default has no
    upstream source to cite. Asserted because a caveat nobody checks quietly disappears.
    """
    from studio.schema import run_config_json_schema

    meta = run_config_json_schema()["$defs"]["Injection"]["properties"]["platform_speed_m_s"][
        "x-studio"
    ]
    assert meta["provenance"] == "convention", "not paper_ensemble: the ensemble has no speed"
    assert "not from the paper ensemble" in meta["source"].lower()
    assert "airframe" in meta["caveat"].lower()


@pytest.mark.tier_a
def test_the_emission_system_is_on_one_stage() -> None:
    """The whole system is one decision, so it must not be split across stages to make it."""
    from studio.schema.layout import STAGES

    stage_of = {
        f: stage.id for stage in STAGES for section in stage.sections for f in section.fields
    }
    system = [
        "injection.so2_mass_kg",
        "injection.platform_speed_m_s",
        "injection.emission_input",
        "injection.given_track_length_m",
        "injection.given_emission_rate_kg_s",
        "injection.given_emission_duration_s",
        "injection.emission_duration_s",
        "injection.emission_rate_kg_s",
        "injection.plume_length_m",
    ]
    stages = {stage_of.get(path) for path in system}
    assert stages == {"plume_volume"}, f"the emission system is spread across {stages}"
