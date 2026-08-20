# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The preview panels.

Two things are worth testing here and one of them is not the arithmetic.

**That each curve is the model's own.** A preview exists to tell the user what the run will do, so a
panel that plots a re-derived formula is a lie with a plausible shape. ``test_the_dilution_curve_is
_the_models`` compares against ``coupled.dilution.volume_ratio`` element for element rather than
against a stored expectation, so the two cannot drift apart.

**That the units survive the seam.** ``stp_to_ambient_factor`` takes pressure in **Pa** while the
schema's canonical unit is mbar (ADR-003). Passing 55 where 5500 was meant scales the whole
distribution by exactly 100 and the plot still looks like a perfectly good size distribution -- I
made that mistake while building this. ``test_the_seeded_number_matches_the_declared_mode`` is the
assertion that catches it, and it is written against the *declared* mode concentration rather than
against whatever the code currently produces.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from studio.resolve import resolve
from studio.schema import RunConfig


@pytest.fixture(autouse=True)
def _needs_model(repo_root: Path) -> None:
    """Four of the five panels need the model; CI has no submodules (see studio-ci.yml)."""
    if not (repo_root / "stratchem-jax" / "solar.py").is_file():
        pytest.skip("model submodules not checked out (`git submodule update --init`)")


@pytest.mark.tier_a
def test_the_dilution_curve_is_the_models() -> None:
    """Element for element the same as ``coupled.dilution.volume_ratio``, not a lookalike."""
    import numpy as np

    from coupled import dilution
    from studio.modelio.preview import dilution_curve

    config = RunConfig()
    panel = dilution_curve(config)
    # Rebuilt exactly as the panel does. Round-tripping through the reported days reintroduces a
    # float error the comparison below is deliberately too strict to tolerate.
    seconds = np.linspace(0.0, config.schedule.duration_days * 86400.0, len(panel["days"]))
    for name, series in panel["regimes"].items():
        expected = np.asarray(dilution.volume_ratio(seconds, name), dtype=float)
        assert np.array_equal(
            np.asarray(series["volume_ratio"]), expected
        ), f"the {name} curve does not match the model's own volume_ratio"
    assert set(panel["regimes"]) == set(dilution.DILUTION_REGIMES), "every regime must be drawn"


@pytest.mark.tier_a
def test_the_constant_regime_says_it_uses_no_curve() -> None:
    """CONSTANT ignores the curves entirely, and the panel must not highlight one that is unused."""
    from studio.modelio.preview import dilution_curve

    config = RunConfig.model_validate({"dilution": {"regime": "constant"}})
    panel = dilution_curve(config)
    assert panel["selected"] == "constant"
    assert panel["uses_curve"] is False
    assert panel["constant_rate_per_s"] == config.dilution.rate_per_s


@pytest.mark.tier_a
def test_the_seeded_number_matches_the_declared_mode() -> None:
    """The unit trap, asserted from the declared mode rather than from the current output.

    ``sabr_220`` declares 49 cm^-3 at STP. At 210 K and 55 mbar the STP-to-ambient factor is
    (5500/101325) x (273.15/210) ~ 0.0706, so the seeded total must be ~3.46 cm^-3. Passing mbar
    where Pa was wanted gives 0.0346 -- a plot that looks entirely reasonable and is 100x wrong.
    """
    from coupled import backgrounds
    from studio.modelio.preview import size_distribution

    panel = size_distribution(RunConfig())
    modes = backgrounds.normalize_modes(backgrounds.BACKGROUND_MODES["sabr_220"])
    declared_stp = sum(mode[0] for mode in modes)
    factor = (5500.0 / 101325.0) * (273.15 / 210.0)
    expected = declared_stp * factor
    # 2% covers the bins clipping the far tails of the lognormal; 100x does not hide in that.
    assert panel["total_cm3"] == pytest.approx(
        expected, rel=0.02
    ), f"seeded {panel['total_cm3']} cm^-3, expected ~{expected} from {declared_stp} cm^-3 at STP"


@pytest.mark.tier_a
def test_the_distribution_peaks_at_the_declared_mode_diameter() -> None:
    """A distribution peaking somewhere other than its mode means the grid mapping is wrong."""
    from coupled import backgrounds
    from studio.modelio.preview import size_distribution

    panel = size_distribution(RunConfig())
    ((_, mode_diameter_um, _),) = backgrounds.normalize_modes(
        backgrounds.BACKGROUND_MODES["sabr_220"]
    )
    # Within one bin: the peak is the bin containing the mode, not the mode itself.
    assert panel["peak_dp_um"] == pytest.approx(mode_diameter_um, rel=0.10)
    assert panel["basis"] == "dry", "CAVEATS: dp_mid_um is dry; SA and radius are wet"


@pytest.mark.tier_a
@pytest.mark.parametrize("n_bins", [40, 80, 160])
def test_the_bin_grid_keeps_the_range_fixed(n_bins: int) -> None:
    """Choosing n_bins changes the resolution and nothing else -- the stage's whole claim."""
    from studio.modelio.preview import bin_grid

    panel = bin_grid(RunConfig.model_validate({"microphysics": {"n_bins": n_bins}}))
    grids = panel["grids"]
    assert panel["selected"] == str(n_bins)
    assert len(grids[str(n_bins)]["dp_um"]) == n_bins
    # The EDGES are what is pinned. Midpoints must not be compared across resolutions: a 40-bin
    # first bin is wider than an 80-bin one, so their geometric means differ by construction --
    # asserting on midpoints was this test's own bug, and it found a real one in the panel, which
    # was quoting a midpoint as `d_min_um` while the hint claimed the edge (1.7 nm vs 1.9 nm).
    for other in ("40", "80", "160"):
        assert grids[other]["edges_um"][0] == pytest.approx(grids["40"]["edges_um"][0], rel=1e-9)
        assert grids[other]["edges_um"][-1] == pytest.approx(grids["40"]["edges_um"][-1], rel=1e-9)
        assert len(grids[other]["edges_um"]) == len(grids[other]["dp_um"]) + 1
    assert grids[str(n_bins)]["mass_ratio"] == pytest.approx(2.0 ** (40.0 / n_bins))


@pytest.mark.tier_a
def test_sza_puts_the_sun_where_the_geometry_says() -> None:
    """Day 172 is the June solstice: at 30 N the sun reaches |30 - 23.44| degrees from vertical."""
    from studio.modelio.preview import sza_diurnal

    panel = sza_diurnal(RunConfig())
    assert panel["min_sza_deg"] == pytest.approx(abs(30.0 - 23.44), abs=0.5)
    # Summer at 30 N is roughly 13.9 h of daylight; sampling is 10-minute, hence the tolerance.
    assert panel["daylight_hours"] == pytest.approx(13.9, abs=0.3)
    assert panel["sun_up"] is True
    assert len(panel["hours"]) == len(panel["sza_deg"])


@pytest.mark.tier_a
def test_polar_night_is_reported_as_dark_rather_than_as_a_flat_line() -> None:
    """At 80 S in June the sun never rises, and the panel must say so."""
    from studio.modelio.preview import sza_diurnal

    polar = {"site": {"latitude_deg": -80.0}, "schedule": {"month": 6, "day_of_month": 21}}
    panel = sza_diurnal(RunConfig.model_validate(polar))
    assert panel["sun_up"] is False
    assert panel["daylight_hours"] == 0.0
    assert min(panel["sza_deg"]) > 90.0


@pytest.mark.tier_a
def test_concentration_is_inversely_proportional_to_volume() -> None:
    """The same mass in ten times the volume is a tenth the mixing ratio -- exactly."""
    from studio.modelio.preview import concentration_sensitivity

    panel = concentration_sensitivity(RunConfig())
    volumes, pptv = panel["volume_cm3"], panel["pptv"]
    products = [v * p for v, p in zip(volumes, pptv, strict=True)]
    assert max(products) == pytest.approx(min(products), rel=1e-12)
    # The marked point must be the config's own resolved value, not a point off the sampled curve.
    resolved = resolve(RunConfig()).config
    assert panel["current_pptv"] == pytest.approx(resolved.injection.so2_initial_pptv)
    assert panel["current_volume_cm3"] == pytest.approx(resolved.injection.plume_volume_cm3)
    # The sweep must bracket the current config, or the marker falls outside the axes.
    assert min(volumes) < panel["current_volume_cm3"] < max(volumes)


@pytest.mark.tier_a
def test_the_curves_follow_the_config() -> None:
    """A panel that ignored an edit would show a confident picture of the wrong configuration."""
    from studio.modelio.preview import dilution_curve, sza_diurnal

    short = dilution_curve(RunConfig.model_validate({"schedule": {"duration_days": 2}}))
    assert math.isclose(max(short["days"]), 2.0)

    winter = sza_diurnal(RunConfig.model_validate({"schedule": {"month": 12, "day_of_month": 21}}))
    summer = sza_diurnal(RunConfig.model_validate({"schedule": {"month": 6, "day_of_month": 21}}))
    assert winter["daylight_hours"] < summer["daylight_hours"], "30 N has shorter days in December"


@pytest.mark.tier_a
def test_the_cheap_panels_need_no_jax() -> None:
    """Stage 1 and stage 3 must not drag JAX into the process.

    Stage 1 is the landing stage; making it import JAX would put a 1.2 s stall on first paint for a
    few lines of trigonometry. Checked in a subprocess because any earlier test in this session may
    already have imported JAX for its own reasons.
    """
    import subprocess
    import sys

    script = (
        "import sys;"
        "from studio.schema import RunConfig;"
        "from studio.modelio.preview import sza_diurnal, concentration_sensitivity;"
        "sza_diurnal(RunConfig()); concentration_sensitivity(RunConfig());"
        "print('jax' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False", "the SZA/concentration panels must not import JAX"


@pytest.mark.tier_a
def test_the_dilution_panel_reports_the_equation_and_each_regimes_k() -> None:
    """The constant is INTROSPECTED from the model's segment tuples, never re-typed.

    Compared against the same tuples here, so if coupled/dilution.py changes a coefficient this
    fails on the spot rather than the panel quoting a k the model no longer uses.
    """
    from coupled import dilution
    from studio.modelio.preview import dilution_curve

    panel = dilution_curve(RunConfig())
    assert "t^0.8" in panel["equation"]["early"]
    assert "exp(k" in panel["equation"]["late"]
    for name, (_, segments) in dilution.DILUTION_REGIMES.items():
        expected = float(segments[1][1][2]) if len(segments) == 2 else None
        assert panel["regimes"][name]["k"] == expected, name


@pytest.mark.tier_a
def test_the_explored_k_uses_the_models_own_machinery() -> None:
    """At a named regime's k the custom curve must equal that regime's curve BIT FOR BIT.

    This is the assertion that the explorer is the model's `_two_piece` + `_eval_segment` and not a
    lookalike formula: a re-derivation would agree to a tolerance, not to the last bit.
    """
    from studio.modelio.preview import dilution_curve

    panel = dilution_curve(RunConfig(), {"explore_k": 8.89e-9})
    assert panel["custom"]["k"] == 8.89e-9
    assert panel["custom"]["volume_ratio"] == panel["regimes"]["D2"]["volume_ratio"]


@pytest.mark.tier_a
def test_an_unphysical_exploration_k_is_refused() -> None:
    """exp(k t^1.5) at k=1 over ten days is an overflow, not a curve (ADR-005)."""
    from studio.modelio.preview import dilution_curve

    with pytest.raises(ValueError, match="explore_k must be within"):
        dilution_curve(RunConfig(), {"explore_k": 1.0})
    assert dilution_curve(RunConfig(), {})["custom"] is None, "no params, no custom curve"
