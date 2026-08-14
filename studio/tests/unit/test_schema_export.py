# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""JSON Schema export and round-tripping.

The export is the contract with the web client (ADR-002): the form is generated from it, so a field
that loses its metadata on the way out is a field the UI cannot explain. Round-tripping is the
contract with everything else -- the CLI reads YAML, the API reads JSON, the runner writes the
resolved config next to the results, and all three must reconstruct the same object with the same
hash.
"""

from __future__ import annotations

import json

import pytest

from studio.schema import (
    EXTENSION_KEY,
    SCHEMA_ID,
    SCHEMA_VERSION,
    RunConfig,
    config_hash,
    field_catalogue,
    run_config_json_schema,
)


@pytest.mark.tier_a
def test_export_is_json_serialisable_and_identifies_itself() -> None:
    """A client must be able to tell which schema version it is holding."""
    schema = run_config_json_schema()
    json.dumps(schema)  # raises if anything in the export is not JSON
    assert schema["$id"] == SCHEMA_ID
    assert SCHEMA_VERSION in schema["$id"]
    assert schema["x-studio-schema-version"] == SCHEMA_VERSION
    assert schema["$schema"].startswith("https://json-schema.org/")


@pytest.mark.tier_a
def test_metadata_survives_the_export() -> None:
    """Unit, provenance and source must reach the client, or the form cannot explain a field."""
    schema = run_config_json_schema()
    site = schema["$defs"]["Site"]["properties"]
    temperature = site["temperature_k"][EXTENSION_KEY]
    assert temperature["unit"] == "K"
    assert temperature["provenance"] == "paper_ensemble"
    assert "TABLE_microphysics_parameters.md" in temperature["source"]
    assert temperature["range"] == {"gt": 0.0}
    assert site["temperature_k"]["description"]


@pytest.mark.tier_a
def test_every_leaf_field_appears_in_the_catalogue_with_a_default() -> None:
    """``RunConfig()`` must be constructible with no arguments -- it is the form's opening state.

    The only fields without a default are the derived ones, which are unresolved by design.
    """
    catalogue = field_catalogue()
    assert len(catalogue) >= 40
    missing_default = [path for path, meta in catalogue.items() if "default" not in meta]
    assert missing_default == [], f"fields with no default: {missing_default}"
    derived_unset = [
        path
        for path, meta in catalogue.items()
        if meta["provenance"] == "derived" and meta["default"] is not None
    ]
    assert derived_unset == []


@pytest.mark.tier_a
def test_catalogue_paths_are_the_same_paths_runset_axes_use() -> None:
    """One vocabulary. A path read from the schema must be usable as an axis path unchanged."""
    from studio.schema import resolve_path

    for path in field_catalogue():
        resolve_path(RunConfig, path)


@pytest.mark.tier_a
def test_json_round_trip_preserves_identity() -> None:
    """Serialise, parse, revalidate: same object, same hash."""
    original = RunConfig()
    restored = RunConfig.model_validate_json(original.model_dump_json())
    assert restored == original
    assert config_hash(restored) == config_hash(original)


@pytest.mark.tier_a
def test_round_trip_survives_a_non_default_config() -> None:
    """Defaults round-tripping proves little; a config with every group altered proves more."""
    original = RunConfig.model_validate(
        {
            "site": {
                "latitude_deg": -60.0,
                "longitude_deg": 175.0,
                "temperature_k": 213.0,
                "pressure_mbar": 120.0,
                "h2o_ppmv": 3.1673,
            },
            "schedule": {"day_of_year": 355, "start_utc_hour": 13.5, "duration_days": 60},
            "injection": {"so2_mass_kg": 2500.0, "plume_length_m": 30000.0},
            "background": {"aerosol": "cesm_g6", "so2_pptv": 100.0, "gas_pptv": {"O3": 1.2e6}},
            "dilution": {"regime": "constant", "rate_per_s": 2.0e-6, "zero_species": ("SO2",)},
            "microphysics": {"n_bins": 160, "condensation_alpha": 0.5, "ion_pair_rate": 0.0},
            "chemistry": {"photolysis": "sza", "so2_ho2_rate": 1e-16},
            "numerics": {"output_dt_s": 1200.0, "couple_dt_s": 300.0},
            "switches": {"aerosol_to_j": True},  # heating_to_t is refused (schema 0.2.0)
            "termination": {"max_wall_time_s": 60.0, "max_sim_time_days": 5.0},
        }
    )
    restored = RunConfig.model_validate_json(original.model_dump_json())
    assert restored == original
    assert config_hash(restored) == config_hash(original)
    assert config_hash(restored) != config_hash(RunConfig())


@pytest.mark.tier_a
def test_unknown_fields_are_rejected() -> None:
    """A typo'd key is a config that does not describe the run; accepting it silently is worse."""
    payload = RunConfig().model_dump()
    payload["site"]["temprature_k"] = 210.0
    with pytest.raises(ValueError, match="temprature_k"):
        RunConfig.model_validate(payload)


@pytest.mark.tier_a
def test_configs_are_immutable() -> None:
    """Identity is a hash of the content, so content that can change under it is a bug (ADR-004)."""
    config = RunConfig()
    with pytest.raises(ValueError, match="frozen"):
        config.site.temperature_k = 250.0  # type: ignore[misc]


@pytest.mark.tier_a
def test_background_evolves_accepts_only_false() -> None:
    """SCIENCE-5: the model's background is static, so True must fail rather than be ignored."""
    payload = RunConfig().model_dump()
    payload["dilution"]["background_evolves"] = True
    with pytest.raises(ValueError, match="background_evolves"):
        RunConfig.model_validate(payload)


@pytest.mark.tier_a
def test_the_temperature_feedback_cannot_be_enabled() -> None:
    """Decision (Ali, 2026-08-13): no temperature feedback while longwave radiation is missing.

    The model's heating term is shortwave-only, so enabling it does not make the thermodynamics
    more complete -- it makes them one-sided, and the resulting ~+1.2 K / 10 d drift is an artefact
    of the absent cooling. Refused outright rather than defaulted off, so it cannot be turned on by
    a form, a YAML file or a sweep axis without the schema changing first.
    """
    payload = RunConfig().model_dump()
    payload["switches"]["heating_to_t"] = True
    with pytest.raises(ValueError, match="heating_to_t"):
        RunConfig.model_validate(payload)
    assert RunConfig().switches.heating_to_t is False


@pytest.mark.tier_a
def test_a_sweep_cannot_enable_the_temperature_feedback_either() -> None:
    """The axis path is the one that would slip past a UI-level guard."""
    from studio.schema import Axis, RunSet

    runset = RunSet(axes=(Axis.over("heating", "switches.heating_to_t", {"on": True}),))
    with pytest.raises(ValueError, match="heating_to_t"):
        runset.expand()


@pytest.mark.tier_a
def test_bin_count_is_restricted_to_the_grids_the_model_has() -> None:
    """40/80/160 are the only TOMAS grids; 100 must fail here, not inside tomas_bridge."""
    payload = RunConfig().model_dump()
    payload["microphysics"]["n_bins"] = 100
    with pytest.raises(ValueError, match="n_bins"):
        RunConfig.model_validate(payload)
