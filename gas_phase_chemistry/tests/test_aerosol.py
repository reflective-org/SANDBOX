"""Validate aerosol.py against the Octave oracle (tests/fixtures/aerosol.csv).

The fixture was produced by running the original MATLAB h2so4wpATfxn.m through Octave
(see tests/octave/dump_oracles.m). The input rows span all three water-activity branches.
"""

import csv
import os

import pytest

from aerosol import h2so4wp_at

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "aerosol.csv")


def load_cases():
    with open(FIX) as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("row", load_cases())
def test_matches_octave(row):
    T, P, H2O = float(row["T"]), float(row["P"]), float(row["H2O"])
    wp, ml, aw = h2so4wp_at(T, P, H2O)

    # Octave prints 17 significant digits; the math is identical, so we expect agreement
    # to near double precision. Use a tight relative tolerance.
    assert wp == pytest.approx(float(row["h2so4wpAT"]), rel=1e-12, abs=1e-12)
    assert ml == pytest.approx(float(row["h2so4mlAT"]), rel=1e-12, abs=1e-12)
    assert aw == pytest.approx(float(row["a_WAT"]), rel=1e-12, abs=1e-12)


def test_branches_are_covered():
    # Make sure the fixture actually exercises all three water-activity regimes,
    # otherwise this test would silently miss a branch.
    aws = [float(r["a_WAT"]) for r in load_cases()]
    assert any(aw <= 0.05 for aw in aws)
    assert any(0.05 < aw < 0.85 for aw in aws)
    assert any(aw >= 0.85 for aw in aws)
