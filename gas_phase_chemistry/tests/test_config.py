"""Sanity checks for the species bookkeeping and config (A1 scaffold)."""

from config import IDX, N_SPECIES, SPECIES, ModelConfig, air_number_density


def test_species_count_is_34():
    assert N_SPECIES == 34
    assert len(SPECIES) == 34


def test_species_order_matches_matlab():
    # Spot-check a few positions against the MATLAB column order (0-based here).
    assert SPECIES[0] == "Cl"
    assert SPECIES[10] == "O3"
    assert SPECIES[-2] == "SO2"
    assert SPECIES[-1] == "H2O2"


def test_idx_lookup_is_consistent():
    for i, name in enumerate(SPECIES):
        assert IDX[name] == i


def test_air_number_density_matches_matlab_formula():
    # MATLAB default scenario: T=210, P=68 -> M = 9.65e18 * P/T * (760/1013.25)
    M = air_number_density(P=68.0, T=210.0)
    assert abs(M - 9.65e18 * 68.0 / 210.0 * (760.0 / 1013.25)) < 1.0


def test_config_computes_M_when_not_given():
    cfg = ModelConfig(T=210.0, P=68.0)
    assert cfg.M is not None
    assert abs(cfg.M - air_number_density(68.0, 210.0)) < 1.0


def test_config_respects_explicit_M():
    cfg = ModelConfig(M=1.234e18)
    assert cfg.M == 1.234e18
