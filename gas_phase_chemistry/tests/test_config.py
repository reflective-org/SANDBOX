"""Sanity checks for the species bookkeeping and config (A1 scaffold)."""

from config import IDX, N_SPECIES, SPECIES, ModelConfig, air_number_density


def test_species_count_is_36():
    # 34 original (MATLAB) + SO3, H2SO4 appended for the sulfur-oxidation chain.
    assert N_SPECIES == 36
    assert len(SPECIES) == 36


def test_species_order_matches_matlab():
    # Spot-check a few positions against the MATLAB column order (0-based here).
    assert SPECIES[0] == "Cl"
    assert SPECIES[10] == "O3"
    # the original 34 end at SO2 (index 32) and H2O2 (index 33), unchanged
    assert SPECIES[32] == "SO2"
    assert SPECIES[33] == "H2O2"
    # appended sulfur-chain species
    assert SPECIES[34] == "SO3"
    assert SPECIES[35] == "H2SO4"


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
