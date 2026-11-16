# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The curated background picker and the custom bimodal distribution (schema 0.5.0).

Requested in review: offer SABRE-220/310/330 and the geoengineered stratosphere, hide the rest, and
allow a custom distribution from two (N, Dp, sigma) modes.

The essential distinction these tests pin: hidden is NOT removed. Tier B's archived cases use
cesm_g6, and a schema that refused it would disconnect the archive from its own configurations. So
the enum keeps every member, the picker offers four plus CUSTOM, and ``hidden_choices`` in the
field metadata is what the form generator filters on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from studio.resolve import resolve
from studio.schema import RunConfig, run_config_json_schema
from studio.schema.enums import BackgroundAerosol


def _custom(**background: object) -> RunConfig:
    return RunConfig.model_validate({"background": {"aerosol": "custom", **background}})


@pytest.mark.tier_a
def test_the_picker_offers_four_references_and_custom() -> None:
    """What a NEW config can choose: the three SABRE sets, aer_geo, and custom."""
    node = run_config_json_schema()["$defs"]["Background"]["properties"]["aerosol"]
    hidden = set(node["x-studio"]["hidden_choices"])
    offered = [
        v
        for v in (
            "redcircles",
            "sabr_330",
            "sabr_310",
            "sabr_220",
            "cesm_g6",
            "cesm_g6_amb",
            "aer_geo",
            "custom",
        )
        if v not in hidden
    ]
    assert offered == ["sabr_330", "sabr_310", "sabr_220", "aer_geo", "custom"]


@pytest.mark.tier_a
def test_hidden_is_not_removed() -> None:
    """Tier B's archive uses cesm_g6; the schema must keep accepting what it accepted."""
    config = RunConfig.model_validate({"background": {"aerosol": "cesm_g6"}})
    assert config.background.aerosol is BackgroundAerosol.CESM_G6
    from studio.tests.golden.paper_cases import BACKGROUNDS  # token -> (aerosol, so2)

    archived = {aerosol for aerosol, _ in BACKGROUNDS.values()}
    hidden = set(
        run_config_json_schema()["$defs"]["Background"]["properties"]["aerosol"]["x-studio"][
            "hidden_choices"
        ]
    )
    # Every archived background is either offered or hidden-but-valid; none may be MISSING.
    assert {b.value for b in archived} <= {e.value for e in BackgroundAerosol}
    assert hidden < {e.value for e in BackgroundAerosol}, "hidden names must be real members"


@pytest.mark.tier_a
def test_custom_modes_reach_the_scenario_as_entered(repo_root: Path) -> None:
    """(N1, Dp1, s1), (N2, Dp2, s2) -> the bridge's own custom-mode path, basis attached."""
    if not (repo_root / "stratchem-jax" / "config.py").is_file():
        pytest.skip("model submodules not checked out")
    from studio.modelio.scenario import to_scenario

    config = _custom(
        custom_n1_cm3=30.0,
        custom_dg1_um=0.1,
        custom_sigma1=1.5,
        custom_n2_cm3=2.0,
        custom_dg2_um=0.8,
        custom_sigma2=1.4,
        custom_basis="ambient",
    )
    scenario = to_scenario(resolve(config).config)
    assert scenario.background_dist == ((30.0, 0.1, 1.5), (2.0, 0.8, 1.4))
    # And the N2 = 0 default maps to ONE mode, because the model refuses zero-N entries.
    assert to_scenario(resolve(_custom()).config).background_dist == ((49.0, 0.12, 1.6),)
    with pytest.raises(ValueError, match="both modes have N = 0"):
        to_scenario(resolve(_custom(custom_n1_cm3=0.0)).config)
    assert scenario.background_modes_basis == "ambient"


@pytest.mark.tier_a
def test_named_backgrounds_carry_no_basis(repo_root: Path) -> None:
    """The basis field is only for custom modes; a named set's basis is the dataset's own."""
    if not (repo_root / "stratchem-jax" / "config.py").is_file():
        pytest.skip("model submodules not checked out")
    from studio.modelio.scenario import to_scenario

    scenario = to_scenario(resolve(RunConfig()).config)
    # The seam translates the campaign spelling to the model's internal key.
    assert scenario.background_dist == "sabr_220"
    assert scenario.background_modes_basis == ""


@pytest.mark.tier_a
def test_the_default_custom_distribution_is_sabre_220(repo_root: Path) -> None:
    """CUSTOM with untouched defaults seeds the SAME distribution as sabr_220, bin for bin.

    The defaults are SABRE-220's mode with N2 = 0, so switching to CUSTOM starts from a citable
    distribution rather than an invented one -- and this is the test that the zero mode really
    contributes nothing.
    """
    if not (repo_root / "stratchem-jax" / "config.py").is_file():
        pytest.skip("model submodules not checked out")
    from studio.modelio.preview import size_distribution

    named = size_distribution(RunConfig())
    custom = size_distribution(_custom())
    assert custom["dn_dlogdp"] == named["dn_dlogdp"], "same modes must seed the same bins"
    assert custom["total_cm3"] == named["total_cm3"]


@pytest.mark.tier_a
def test_a_second_mode_adds_particles_where_it_says(repo_root: Path) -> None:
    """N2 > 0 raises the total by ~N2 (x the STP factor) and moves mass to the coarse mode."""
    if not (repo_root / "stratchem-jax" / "config.py").is_file():
        pytest.skip("model submodules not checked out")
    import numpy as np

    from studio.modelio.preview import size_distribution

    base = size_distribution(_custom())
    bimodal = size_distribution(_custom(custom_n2_cm3=5.0))
    stp_factor = (5500.0 / 101325.0) * (273.15 / 210.0)
    assert bimodal["total_cm3"] - base["total_cm3"] == pytest.approx(5.0 * stp_factor, rel=0.02)
    # The added particles sit near Dp2 = 0.9 um, not under mode 1.
    dp = np.asarray(bimodal["dp_um"])
    added = np.asarray(bimodal["dn_dlogdp"]) - np.asarray(base["dn_dlogdp"])
    assert 0.5 < dp[int(np.argmax(added))] < 1.6


@pytest.mark.tier_a
def test_degenerate_modes_are_refused() -> None:
    """sigma = 1 is monodisperse (log-normal width zero divides by it); negatives are not counts."""
    with pytest.raises(ValueError):
        _custom(custom_sigma1=1.0)
    with pytest.raises(ValueError):
        _custom(custom_n1_cm3=-1.0)
    with pytest.raises(ValueError):
        _custom(custom_dg1_um=0.0)
