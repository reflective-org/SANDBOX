# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The GCR module refuses to compute, and these tests hold it to that.

An unimplemented derivation is only safe if it STAYS unimplemented until someone supplies a
citation. The obvious future failure is a well-meaning change that makes
``ion_pair_production_rate`` return "something reasonable" so a form stops erroring -- these tests
are what would fail in that PR, with the reason attached.
"""

from __future__ import annotations

import pytest

from studio.science import (
    MODEL_DEFAULT_ION_PAIR_RATE,
    PAPER_ENSEMBLE_ION_PAIR_RATE,
    ion_pair_production_rate,
)


@pytest.mark.tier_a
def test_the_parameterisation_refuses_to_guess() -> None:
    """No citation, no number (ADR-005). The error must say why and where it is tracked."""
    with pytest.raises(NotImplementedError) as excinfo:
        ion_pair_production_rate(altitude_km=20.0, latitude_deg=30.0, solar_cycle_phase=0.5)
    message = str(excinfo.value)
    assert "SCIENCE-6" in message
    assert "#63" in message
    assert "PAPER_ENSEMBLE_ION_PAIR_RATE" in message, (
        "the error must point at the constant that DOES have provenance, or the next person "
        "will invent a value rather than find it"
    )


@pytest.mark.tier_a
def test_it_refuses_for_every_input_including_the_ensembles_own_conditions() -> None:
    """~20 km / 30N is exactly where the uncited 30.0 came from, and it is not special-cased.

    Returning the known value at the known point and raising elsewhere would be the most tempting
    version of this mistake: it would look like a working function with gaps.
    """
    for altitude, latitude, phase in [(20.0, 30.0, 0.0), (15.0, 60.0, 1.0), (20.0, 30.0, 0.5)]:
        with pytest.raises(NotImplementedError):
            ion_pair_production_rate(
                altitude_km=altitude, latitude_deg=latitude, solar_cycle_phase=phase
            )


@pytest.mark.tier_a
def test_the_two_documented_constants_are_what_the_code_uses() -> None:
    """The ensemble's value and the model's default, both recorded, neither silently preferred."""
    assert PAPER_ENSEMBLE_ION_PAIR_RATE == 30.0
    assert MODEL_DEFAULT_ION_PAIR_RATE == 0.0


@pytest.mark.tier_a
def test_the_schema_default_matches_the_ensemble_constant() -> None:
    """One value, two places: the schema default and this constant must not drift apart.

    ``microphysics.ion_pair_rate`` defaults to the ensemble's 30.0 (ASSUMPTION-5) rather than the
    model's 0.0, and that choice is only defensible while both sides agree on what the ensemble
    used.
    """
    from studio.schema import RunConfig

    assert RunConfig().microphysics.ion_pair_rate == PAPER_ENSEMBLE_ION_PAIR_RATE
