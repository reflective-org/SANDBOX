# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The emission basis: two ways to say the same release (schema 0.3.0).

The paper ensemble specifies the release geometrically -- 1 t into 10 m x 10 m x 15 km -- because
that is what the model consumes. A deployment is specified operationally: a platform flying at some
speed, emitting at some rate. Both describe the same plume, and the schema now accepts either.

What these tests protect:

* **The default run is unchanged.** Adding a parameterisation must not move the reference case, or
  every archived comparison silently shifts.
* **The two bases agree on the default config**, to floating point. If switching basis moved the
  plume, the choice would be a physics decision disguised as a preference.
* **Nothing is derived that does not follow.** Under MASS_AND_LENGTH there is no emission rate in
  play, so the duration is null rather than a number describing a release that is not happening.
"""

from __future__ import annotations

import pytest

from studio.resolve import apply_change, resolve
from studio.schema import RunConfig
from studio.schema.enums import EmissionBasis


def _rate_and_speed(**overrides: object) -> RunConfig:
    injection = {"emission_basis": "rate_and_speed", **overrides}
    return RunConfig.model_validate({"injection": injection})


@pytest.mark.tier_a
def test_the_default_basis_is_the_paper_ensembles() -> None:
    """Existing configs keep their meaning: the default is still mass and an entered length."""
    config = RunConfig()
    assert config.injection.emission_basis is EmissionBasis.MASS_AND_LENGTH
    resolved = resolve(config).config
    assert resolved.injection.plume_length_m == 15000.0
    assert resolved.injection.plume_volume_cm3 == 1.5e12
    assert (
        resolved.injection.emission_duration_s is None
    ), "no rate is in play under this basis, so a duration would describe a different release"


@pytest.mark.tier_a
def test_the_two_bases_agree_on_the_default_config() -> None:
    """Switching basis with untouched defaults must not move the plume.

    Not bit-exact, and the reason is worth stating: the default rate is ``1000 / (15000 / 250)``,
    and ``1000 / (1000 / 60)`` is ``59.99999999999999`` in IEEE 754, not 60. So the derived length
    lands within one part in 1e16 of the entered one. The initial mixing ratio -- the number the
    model actually consumes -- comes out identical to the last bit anyway.
    """
    entered = resolve(RunConfig()).config.injection
    derived = resolve(_rate_and_speed()).config.injection

    assert derived.emission_duration_s == pytest.approx(60.0, rel=1e-12)
    assert derived.plume_length_m == pytest.approx(entered.plume_length_m, rel=1e-12)
    assert derived.plume_volume_cm3 == pytest.approx(entered.plume_volume_cm3, rel=1e-12)
    assert (
        derived.so2_initial_pptv == entered.so2_initial_pptv
    ), "the quantity the model consumes must be identical, not merely close"


@pytest.mark.tier_a
def test_the_chain_is_mass_over_rate_then_speed_times_duration() -> None:
    """t = M / R, then L = v.t -- and the mixing ratio follows the length it produces."""
    resolved = resolve(_rate_and_speed(emission_rate_kg_s=5.0, platform_speed_m_s=200.0)).config
    injection = resolved.injection

    assert injection.emission_duration_s == pytest.approx(1000.0 / 5.0)  # 200 s
    assert injection.plume_length_m == pytest.approx(200.0 * 200.0)  # 40 km
    assert injection.plume_volume_cm3 == pytest.approx(40000.0 * 10.0 * 10.0 * 1e6)

    # 40 km instead of 15 km is 2.667x the volume, so 2.667x less concentrated.
    reference = resolve(RunConfig()).config.injection
    ratio = reference.so2_initial_pptv / injection.so2_initial_pptv
    assert ratio == pytest.approx(40000.0 / 15000.0, rel=1e-12)


@pytest.mark.tier_a
def test_a_faster_platform_lays_a_longer_and_thinner_plume() -> None:
    """Same mass, same rate, more speed: more track, lower concentration. The operational knob."""
    slow = resolve(_rate_and_speed(platform_speed_m_s=100.0)).config.injection
    fast = resolve(_rate_and_speed(platform_speed_m_s=400.0)).config.injection

    assert fast.emission_duration_s == pytest.approx(
        slow.emission_duration_s
    ), "duration is mass over rate and does not depend on speed"
    assert fast.plume_length_m == pytest.approx(4.0 * slow.plume_length_m)
    assert fast.so2_initial_pptv == pytest.approx(slow.so2_initial_pptv / 4.0, rel=1e-12)


@pytest.mark.tier_a
def test_the_entered_track_length_is_ignored_under_rate_and_speed() -> None:
    """The basis decides which input counts; the other is inert rather than quietly blended in."""
    resolved = resolve(_rate_and_speed(track_length_m=99000.0)).config.injection
    assert resolved.track_length_m == 99000.0, "the entered value is kept, not erased"
    assert resolved.plume_length_m == pytest.approx(
        15000.0, rel=1e-12
    ), "but the length the model uses comes from speed x duration under this basis"


@pytest.mark.tier_a
def test_changing_the_basis_recomputes_the_whole_chain() -> None:
    """The basis is an input like any other, so switching it propagates through the DAG."""
    start = resolve(_rate_and_speed(emission_rate_kg_s=5.0, platform_speed_m_s=200.0))
    assert start.config.injection.plume_length_m == pytest.approx(40000.0)

    switched = apply_change(start, "injection.emission_basis", EmissionBasis.MASS_AND_LENGTH)
    assert switched.config.injection.plume_length_m == 15000.0, "back to the entered track length"
    assert switched.config.injection.emission_duration_s is None
    assert switched.is_consistent


@pytest.mark.tier_a
def test_a_zero_rate_is_refused_rather_than_producing_an_infinite_plume() -> None:
    """Fail loud (ADR-005). A zero rate is an infinite emission, which is not a run anyone means."""
    with pytest.raises(ValueError, match=r"emission rate must be positive|greater than 0"):
        resolve(_rate_and_speed(emission_rate_kg_s=0.0))


@pytest.mark.tier_a
def test_the_speed_default_is_marked_as_a_choice_not_a_measurement() -> None:
    """250 m/s is a decision, and the schema has to say so rather than imply an airframe.

    Nothing in the model or the paper ensemble carries a platform speed at all, so this default has
    no upstream source to cite. The caveat is the honest form of that, and it is asserted here
    because a caveat nobody checks quietly disappears in a later edit.
    """
    from studio.schema import run_config_json_schema

    node = run_config_json_schema()["$defs"]["Injection"]["properties"]["platform_speed_m_s"]
    meta = node["x-studio"]
    assert meta["provenance"] == "convention", "not paper_ensemble: the ensemble has no speed"
    assert "not from the paper ensemble" in meta["source"].lower()
    assert "airframe" in meta["caveat"].lower() or "aircraft" in meta["caveat"].lower()


@pytest.mark.tier_a
def test_the_new_fields_are_on_the_wizard() -> None:
    """A schema field with no stage is invisible; the layout test enforces it, this names them."""
    from studio.schema.layout import laid_out_fields

    placed = set(laid_out_fields())
    for path in (
        "injection.emission_basis",
        "injection.emission_rate_kg_s",
        "injection.platform_speed_m_s",
        "injection.track_length_m",
        "injection.emission_duration_s",
    ):
        assert path in placed, f"{path} has no wizard stage"
