"""Validate gammas.py against the Octave oracle (tests/fixtures/gammas.csv).

The fixture was produced by running the original MATLAB hetgammasJPL00.m through Octave
(see tests/octave/dump_oracles.m), reusing the aerosol composition computed there.
"""

import csv
import os

import pytest

from gammas import hetgammas_jpl00

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "gammas.csv")


def load_cases():
    with open(FIX) as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("row", load_cases())
def test_matches_octave(row):
    # Feed the SAME aerosol composition (h2so4wp, a_W) the oracle used, so this isolates
    # the gamma calculation from any aerosol-port differences.
    Yhocl, Yclnh2o, Yclnhcl = hetgammas_jpl00(
        T=float(row["T"]),
        P=float(row["P"]),
        h2so4wp=float(row["h2so4wp"]),
        a_W=float(row["a_W"]),
        HCl=float(row["HCl"]),
        ClONO2=float(row["ClONO2"]),
        radius=float(row["radius"]),
        sts=float(row["sts"]),
    )

    assert Yhocl == pytest.approx(float(row["Yhocl"]), rel=1e-10, abs=1e-18)
    assert Yclnh2o == pytest.approx(float(row["Yclnh2o"]), rel=1e-10, abs=1e-18)
    assert Yclnhcl == pytest.approx(float(row["Yclnhcl"]), rel=1e-10, abs=1e-18)
