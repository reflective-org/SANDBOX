"""Reaction reference numbering: R-numbers + MATLAB k-labels."""

from reactions import BY_KLABEL, BY_RNUMBER, MECHANISM, REACTIONS


def test_every_reaction_has_a_klabel():
    assert all(r.klabel for r in REACTIONS)


def test_counts():
    # 78 reactions total (k1..k72 with a/b branch suffixes); 75 active (3 disabled).
    assert len(REACTIONS) == 78
    assert len(MECHANISM.active) == 75


def test_rnumber_lookup_matches_position():
    for i, rxn in enumerate(REACTIONS, 1):
        assert BY_RNUMBER[f"R{i}"] is rxn


def test_klabel_lookup_is_unique_and_consistent():
    assert len(BY_KLABEL) == len(REACTIONS)            # labels are unique
    assert BY_KLABEL["k22"].equation == "OH + HNO3 -> H2O + NO3"
    assert BY_KLABEL["k20a"].equation == "O3 -> O2 + O"
    assert BY_KLABEL["k68"].equation == "SO2 + OH -> HO2"


def test_describe_includes_reference_numbers():
    text = MECHANISM.describe()
    assert "R1" in text and "(k1)" in text
    assert "R26" in text and "(k22)" in text
