"""Unit tests for the declarative mechanism framework (M1).

Uses small toy mechanisms with hand-checkable stoichiometry, independent of the full
chemistry (which is validated separately against Octave).
"""

import math

import numpy as np
import pytest

from config import IDX, N_SPECIES
from mechanism import Env, Mechanism, parse_equation, react, photo, het, troe, khet


# ---- parsing -------------------------------------------------------------------------
def test_parse_basic():
    reac, prod = parse_equation("Cl + O3 -> ClO + O2")
    assert reac == {"Cl": 1, "O3": 1}
    assert prod == {"ClO": 1, "O2": 1}


def test_parse_strips_M_and_hv():
    reac, prod = parse_equation("ClO + ClO + M -> ClOOCl + M")
    assert reac == {"ClO": 2}          # ClO + ClO collapses to coefficient 2; M dropped
    assert prod == {"ClOOCl": 1}

    reac, prod = parse_equation("O3 -> O2 + O1D")  # (hv implied; here no hv token)
    assert reac == {"O3": 1}
    assert prod == {"O2": 1, "O1D": 1}


def test_parse_coefficients():
    reac, prod = parse_equation("ClOOCl -> 2 Cl + O2")
    assert reac == {"ClOOCl": 1}
    assert prod == {"Cl": 2, "O2": 1}


def test_parse_unknown_species_raises():
    with pytest.raises(ValueError):
        parse_equation("Cl + Xx -> ClO")


# ---- stoichiometry + dC/dt -----------------------------------------------------------
def test_single_reaction_dcdt():
    # Cl + O3 -> ClO + O2 with a constant rate coefficient k.
    k = 2.0e-11
    mech = Mechanism([react("Cl + O3 -> ClO + O2", lambda e: k)])

    conc = np.zeros(N_SPECIES)
    conc[IDX["Cl"]] = 1e5
    conc[IDX["O3"]] = 1e12
    env = Env(T=210.0, M=2.3e18)

    rate = k * 1e5 * 1e12
    d = mech.dCdt(env, conc)
    assert d[IDX["Cl"]] == pytest.approx(-rate)
    assert d[IDX["O3"]] == pytest.approx(-rate)
    assert d[IDX["ClO"]] == pytest.approx(+rate)
    assert d[IDX["O2"]] == pytest.approx(+rate)


def test_stoichiometric_coefficient_two():
    # ClO + ClO -> ClOOCl consumes two ClO per event; rate ~ [ClO]^2.
    k = 1e-13
    mech = Mechanism([react("ClO + ClO + M -> ClOOCl + M", lambda e: k)])
    conc = np.zeros(N_SPECIES)
    conc[IDX["ClO"]] = 3.0
    env = Env(T=210.0, M=2.3e18)

    rate = k * 3.0 ** 2
    d = mech.dCdt(env, conc)
    assert d[IDX["ClO"]] == pytest.approx(-2 * rate)
    assert d[IDX["ClOOCl"]] == pytest.approx(+rate)


def test_photolysis_day_night_scaling():
    mech = Mechanism([photo("O3 -> O2 + O1D", j45=4.7e-5)])
    conc = np.zeros(N_SPECIES)
    conc[IDX["O3"]] = 1e12

    day = mech.dCdt(Env(T=210, M=2.3e18, j_scale=1.0), conc)
    night = mech.dCdt(Env(T=210, M=2.3e18, j_scale=0.0), conc)
    assert day[IDX["O1D"]] == pytest.approx(4.7e-5 * 1e12)
    assert night[IDX["O1D"]] == 0.0


def test_heterogeneous_rate_first_order_in_driver():
    # Het uptake is first-order in the gas taken up (ClONO2), even though HCl is also
    # consumed stoichiometrically. So the rate must NOT depend on [HCl].
    mech = Mechanism([het("ClONO2 + HCl -> Cl2 + HNO3aq", gamma="Yclnhcl",
                          gasmass=97, driver="ClONO2")])
    conc = np.zeros(N_SPECIES)
    conc[IDX["ClONO2"]] = 1e8
    conc[IDX["HCl"]] = 5e9   # nonzero, but must not change the rate
    env = Env(T=210, M=2.3e18, SA=2.0, gammas={"Yclnhcl": 1e-3})

    expected_rate = khet(1e-3, 97, 210, 2.0) * 1e8   # k * [ClONO2] only
    d = mech.dCdt(env, conc)
    assert d[IDX["Cl2"]] == pytest.approx(expected_rate)
    assert d[IDX["ClONO2"]] == pytest.approx(-expected_rate)
    assert d[IDX["HCl"]] == pytest.approx(-expected_rate)   # consumed via stoichiometry
    assert d[IDX["HNO3aq"]] == pytest.approx(+expected_rate)


def test_inactive_reaction_excluded():
    mech = Mechanism([
        react("Cl + O3 -> ClO + O2", lambda e: 1e-11),
        react("Cl + C2H6 -> HCl", lambda e: 1e-11, active=False),
    ])
    assert len(mech.active) == 1
    assert mech.S.shape == (N_SPECIES, 1)


def test_troe_matches_manual():
    e = Env(T=220.0, M=2.0e18)
    got = troe(e, 1.6e-32, -4.5, 3.0e-12, -2.0)
    k0 = 1.6e-32 * (220 / 300) ** -4.5
    kinf = 3.0e-12 * (220 / 300) ** -2.0
    ratio = k0 * e.M / kinf
    expected = (k0 * e.M / (1 + ratio)) * 0.6 ** (1 / (1 + math.log10(ratio) ** 2))
    assert got == pytest.approx(expected)
