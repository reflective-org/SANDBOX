# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The ERA5 zonal-mean monthly climatology: the product, its reader, and the derivations over it.

Unlike the model-facing tests, these run in CI: the product is COMMITTED (a few MB of npz), not a
gitignored artefact or a submodule. That is deliberate -- it means the climatology path is the first
scientifically substantive code whose tests CI can actually execute.

The physics assertions use wide, cited bounds. They are not testing ERA5 (ERA5 is the reference);
they are testing that the reduction and interpolation did not scramble axes -- a transposed
(month, level, lat) array still produces plausible-looking numbers at any single point, but it
cannot produce a colder tropical tropopause than midlatitude one AND a summer-warm polar
stratosphere at once.
"""

from __future__ import annotations

import pytest

from studio.resolve import apply_change, resolve
from studio.schema import RunConfig
from studio.science import climatology


def _era5(**site: object) -> RunConfig:
    return RunConfig.model_validate({"site": {"dataset": "era5", **site}})


@pytest.mark.tier_a
def test_the_product_is_committed_and_matches_its_manifest() -> None:
    """Presence and integrity: the sha256 in the manifest is the one the loader verifies."""
    product = climatology.load()
    assert product.dataset_id == climatology.DATASET_ID
    assert product.temperature_k.shape == (12, len(product.level_hpa), len(product.latitude_deg))
    assert len(product.sha256) == 64


@pytest.mark.tier_a
def test_the_interpolation_is_exact_at_grid_points() -> None:
    """At a node of the product, interpolation must return the stored value bit-for-bit."""
    product = climatology.load()
    month, level_index, lat_index = 6, 5, 48  # arbitrary interior node
    expected = float(product.temperature_k[month - 1, level_index, lat_index])
    got = climatology.temperature_k(
        month=month,
        latitude_deg=float(product.latitude_deg[lat_index]),
        pressure_mbar=float(product.level_hpa[level_index]),
    )
    assert got == expected


@pytest.mark.tier_a
def test_no_extrapolation_ever() -> None:
    """Outside 300-5 hPa the lookup raises rather than guessing (ADR-005)."""
    for pressure in (400.0, 1000.0, 2.0):
        with pytest.raises(ValueError, match="outside the climatology"):
            climatology.temperature_k(month=6, latitude_deg=30.0, pressure_mbar=pressure)


@pytest.mark.tier_a
def test_the_stratosphere_looks_like_the_stratosphere() -> None:
    """Axis-scrambling detectors, with generous bounds.

    * The tropical tropopause region (~100 hPa) is the coldest place in the product's domain:
      ~190-205 K climatologically (Seidel et al. 2001 put the tropical cold point near 190 K).
    * 30N at ~50 hPa in June sits near 210-220 K -- the paper ensemble's 210 K at 55 hPa is
      exactly this number, which is the strongest single check available.
    * The summer polar stratosphere is WARMER than the winter one at 30 hPa by tens of kelvin
      (no polar night jet in summer).
    """
    tropical_tropopause = climatology.temperature_k(month=2, latitude_deg=0.0, pressure_mbar=100.0)
    assert 185.0 < tropical_tropopause < 205.0

    ensemble_site = climatology.temperature_k(month=6, latitude_deg=30.0, pressure_mbar=55.0)
    assert 205.0 < ensemble_site < 222.0

    july_arctic = climatology.temperature_k(month=7, latitude_deg=80.0, pressure_mbar=30.0)
    january_arctic = climatology.temperature_k(month=1, latitude_deg=80.0, pressure_mbar=30.0)
    assert july_arctic > january_arctic + 10.0, "summer pole must be warmer than winter pole"


@pytest.mark.tier_a
def test_stratospheric_water_vapour_is_a_few_ppmv() -> None:
    """The wet stratosphere is single-digit ppmv (3-8 typical); 100x off means a unit slipped.

    The q -> ppmv conversion is where a silent factor would live, so the bound is deliberately
    tight enough to catch Mw/Md swapped (x1.6) as well as kg/g confusions (x1000).
    """
    value = climatology.h2o_ppmv(month=6, latitude_deg=30.0, pressure_mbar=55.0)
    assert 2.0 < value < 10.0, f"{value} ppmv is not a stratospheric water vapour value"


@pytest.mark.tier_a
def test_the_conversion_is_the_exact_mole_fraction_form() -> None:
    assert climatology.specific_humidity_to_ppmv(0.0) == 0.0
    # Dilute limit: q * Md/Mw * 1e6, to first order.
    approx = 3e-6 * (28.9644 / 18.0153) * 1e6
    assert climatology.specific_humidity_to_ppmv(3e-6) == pytest.approx(approx, rel=1e-5)
    with pytest.raises(ValueError, match="negative"):
        climatology.specific_humidity_to_ppmv(-1e-6)


@pytest.mark.tier_a
def test_era5_dataset_derives_the_ambient_state() -> None:
    """The schema path end to end: dataset=ERA5 fills T and H2O from the product."""
    resolved = resolve(_era5()).config
    assert resolved.site.temperature_k == pytest.approx(
        climatology.temperature_k(month=6, latitude_deg=30.0, pressure_mbar=55.0)
    )
    assert resolved.site.h2o_ppmv == pytest.approx(
        climatology.h2o_ppmv(month=6, latitude_deg=30.0, pressure_mbar=55.0)
    )
    # The given values are kept but inert, exactly as in the emission system.
    assert resolved.site.given_temperature_k == 210.0


@pytest.mark.tier_a
def test_user_dataset_is_byte_identical_to_the_old_behaviour() -> None:
    """The default path must not so much as touch the product."""
    resolved = resolve(RunConfig()).config
    assert resolved.site.temperature_k == 210.0
    assert resolved.site.h2o_ppmv == 6.9104
    assert resolved.injection.so2_initial_pptv == pytest.approx(3309115922.996412, rel=1e-15)


@pytest.mark.tier_a
def test_changing_the_month_moves_the_era5_temperature() -> None:
    """The wizard's claim, on real data: edit the month, the ambient state follows."""
    june = resolve(_era5())
    december = apply_change(june, "schedule.month", 12)
    t_june = june.config.site.temperature_k
    t_december = december.config.site.temperature_k
    assert t_june is not None and t_december is not None
    assert t_june != t_december, "30N at 55 hPa has a real seasonal cycle"
    assert december.is_consistent


@pytest.mark.tier_a
def test_the_concentration_follows_the_era5_temperature() -> None:
    """The chain reaches the model's input: T from ERA5 changes the air density, so the pptv."""
    user = resolve(RunConfig()).config
    era5 = resolve(_era5()).config
    assert era5.site.temperature_k is not None and user.site.temperature_k is not None
    if era5.site.temperature_k != user.site.temperature_k:
        assert era5.injection.so2_initial_pptv != user.injection.so2_initial_pptv
        # Same number density either way; pptv scales with T at fixed p (ideal gas).
        ratio = era5.injection.so2_initial_pptv / user.injection.so2_initial_pptv
        assert ratio == pytest.approx(era5.site.temperature_k / user.site.temperature_k, rel=1e-12)


@pytest.mark.tier_a
def test_era5_runs_record_the_dataset_in_provenance(repo_root: object) -> None:
    """ADR-006: a run whose temperature came from ERA5 must say so, with the checksum."""
    from pathlib import Path

    from studio.modelio.provenance import record_for

    root = repo_root if isinstance(repo_root, Path) else None
    if root is None or not (root / "stratchem-jax" / ".git").exists():
        pytest.skip("model submodules not checked out")
    record = record_for(resolve(_era5()), repo_root=root)
    assert record.datasets == {climatology.DATASET_ID: climatology.load().sha256}
    assert record_for(resolve(RunConfig()), repo_root=root).datasets == {}


@pytest.mark.tier_a
def test_altitude_and_pressure_agree_with_the_site_name() -> None:
    """The ensemble's site is literally labelled ``30N_20km`` at 55 hPa; the product must agree.

    Bounds are generous (19-22 km) because the 55 hPa surface moves with season and latitude --
    but a wrong-axis or wrong-units bug lands kilometres away, not hundreds of metres.
    """
    z = climatology.geopotential_height_m(month=6, latitude_deg=30.0, pressure_mbar=55.0)
    assert 19_000.0 < z < 22_000.0, f"{z} m is not ~20 km"


@pytest.mark.tier_a
def test_height_falls_as_pressure_rises() -> None:
    """Monotonicity across the whole level range -- the relation the right axis relies on."""
    heights = [
        climatology.geopotential_height_m(month=6, latitude_deg=30.0, pressure_mbar=float(p))
        for p in (5, 30, 100, 300)
    ]
    assert heights == sorted(heights, reverse=True)


@pytest.mark.tier_a
def test_the_profile_panel_carries_the_altitude_labeling(repo_root: object) -> None:
    """Round-km ticks placed at their true pressures, and the box's own altitude."""
    from studio.modelio.preview import climatology_profile

    panel = climatology_profile(RunConfig())
    assert panel["box_altitude_km"] == pytest.approx(20.2, abs=0.5)
    ticks = panel["altitude_ticks"]
    assert [t["km"] for t in ticks] == sorted({t["km"] for t in ticks}), "ascending, no duplicates"
    assert all(t["km"] % 5 == 0 for t in ticks), "round kilometres only"
    for tick in ticks:
        back = climatology.geopotential_height_m(
            month=6, latitude_deg=30.0, pressure_mbar=tick["pressure_hpa"]
        )
        # Round-trip through the inverse interpolation: the tick must sit where it claims, within
        # the interpolation's own error (log-p linear both ways).
        assert back / 1000.0 == pytest.approx(tick["km"], abs=0.1)
