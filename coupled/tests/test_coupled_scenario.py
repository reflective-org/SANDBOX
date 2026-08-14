# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""CoupledScenario: the single coupled-model input, its switches, and validation."""

import os
import subprocess
import sys
import textwrap

import pytest

from coupled_scenario import CoupledScenario, Switches

_DEFAULT_YAML = os.path.join(os.path.dirname(__file__), "..", "scenarios", "coupled_default.yaml")
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_defaults():
    sc = CoupledScenario()
    assert sc.photolysis == "tuvx"
    assert sc.dt_couple == 600.0            # outer radiation/coupling step (two-level driver)
    assert sc.dt_rad == sc.dt_couple        # alias property
    # adaptive inner (micro) step defaults
    assert (sc.micro_eps, sc.micro_floor_s, sc.micro_cap_s) == (0.1, 1.0e-4, 20.0)
    # ALL processes default ON (full coupled physics is the default; the nucleation "runaway" that
    # once motivated defaults-off was a coarse-step artifact, fixed by the two-level driver)
    assert all([sc.switches.sulfur, sc.switches.nucleation, sc.switches.condensation,
                sc.switches.coagulation, sc.switches.aerosol_to_j, sc.switches.heating_to_t,
                sc.switches.dilution])


def test_load_default_yaml():
    sc = CoupledScenario.load(_DEFAULT_YAML)
    assert sc.photolysis == "tuvx"
    assert sc.switches.sulfur is True
    assert sc.concentrations["SO2"] == 953895
    assert sc.concentrations["NO"] == 450  # 'NO' quoted in YAML (unquoted NO is boolean false)


def test_save_load_round_trip(tmp_path):
    sc = CoupledScenario(photolysis="sza", dt_couple=60.0,
                         concentrations={"SO2": 1.0e6, "OH": 0.5})
    p = tmp_path / "s.yaml"
    sc.save(str(p))
    back = CoupledScenario.load(str(p))
    assert back.to_dict() == sc.to_dict()


def test_bad_photolysis_raises():
    with pytest.raises(ValueError):
        CoupledScenario(photolysis="referance")  # typo must NOT silently pass


def test_bad_dt_couple_raises():
    with pytest.raises(ValueError):
        CoupledScenario(dt_couple=0.0)


def test_all_switches_implemented_and_allowed():
    # Phases 3-6 wired every process switch; enabling all of them must NOT raise.
    CoupledScenario(switches=Switches(sulfur=True, nucleation=True, condensation=True,
                                      coagulation=True, aerosol_to_j=True, heating_to_t=True,
                                      dilution=True))


def test_unknown_key_rejected():
    with pytest.raises(ValueError):
        CoupledScenario.from_dict({"not_a_field": 1})


def test_output_dir_is_gone_and_says_why():
    # Removed, not honoured: run_coupled returns arrays and writes nothing, so the caller owns the
    # output path. An archived config carrying the key gets an explanation, not "unknown key".
    assert "output_dir" not in CoupledScenario.__dataclass_fields__
    with pytest.raises(ValueError, match="the CALLER chooses where to save"):
        CoupledScenario.from_dict({"output_dir": "coupled_output"})


def test_unknown_switch_key_raises_friendly_valueerror():
    # a typo'd switch name gives a clear ValueError, not a cryptic TypeError
    with pytest.raises(ValueError):
        CoupledScenario(switches={"sulfer": True})


def test_days_must_be_at_least_one():
    with pytest.raises(ValueError):
        CoupledScenario(days=0)     # a zero/negative run length is a config error, not a no-op
    with pytest.raises(ValueError):
        CoupledScenario(days=-1)
    CoupledScenario(days=1)          # one day is the minimum valid run


def test_dt_couple_must_not_exceed_DT():
    with pytest.raises(ValueError):
        CoupledScenario(DT=600.0, dt_couple=900.0)


def test_DT_must_be_multiple_of_dt_couple():
    with pytest.raises(ValueError):
        CoupledScenario(DT=600.0, dt_couple=250.0)   # 600 / 250 is not an integer
    CoupledScenario(DT=600.0, dt_couple=200.0)        # 600 / 200 = 3 -> ok


def test_bad_background_dist_raises():
    with pytest.raises(ValueError):
        CoupledScenario(background_dist="sabr_999")   # not a known name


# --- user-supplied lognormal background modes ------------------------------------------------
_USER_MODES = [(50.0, 0.10, 1.6), (2.0, 0.5, 1.4)]


def test_user_modes_are_accepted_and_canonicalized():
    sc = CoupledScenario(background_dist=_USER_MODES, background_modes_basis="stp")
    # lists (as YAML/JSON hand them back) canonicalize to a tuple of float 3-tuples
    assert sc.background_dist == ((50.0, 0.10, 1.6), (2.0, 0.5, 1.4))
    assert all(isinstance(v, float) for mode in sc.background_dist for v in mode)


def test_user_modes_round_trip_through_yaml(tmp_path):
    sc = CoupledScenario(background_dist=_USER_MODES, background_modes_basis="ambient")
    p = tmp_path / "s.yaml"
    sc.save(str(p))
    assert CoupledScenario.load(str(p)).to_dict() == sc.to_dict()


def test_user_modes_require_an_explicit_basis():
    # The STP->ambient factor is ~0.09 at 68 mbar/210 K, so the basis is an order-of-magnitude
    # decision and must never be defaulted.
    with pytest.raises(ValueError, match="background_modes_basis"):
        CoupledScenario(background_dist=_USER_MODES)
    with pytest.raises(ValueError, match="background_modes_basis"):
        CoupledScenario(background_dist=_USER_MODES, background_modes_basis="STP")   # not a member


def test_named_background_rejects_a_basis():
    # the named sets carry their own basis (backgrounds.AMBIENT_BACKGROUNDS); accepting a
    # contradicting one here would let a config claim a conversion that never happens
    with pytest.raises(ValueError, match="background_modes_basis"):
        CoupledScenario(background_dist="sabr_220", background_modes_basis="ambient")


@pytest.mark.parametrize("modes, why", [
    ([], "empty"),
    ([(0.0, 0.1, 1.6)], "N == 0"),
    ([(-1.0, 0.1, 1.6)], "N < 0"),
    ([(50.0, 0.0, 1.6)], "Dg == 0"),
    ([(50.0, -0.1, 1.6)], "Dg < 0"),
    ([(50.0, 0.1, 1.0)], "sigma_g == 1 -> log10(sigma_g) = 0, a divide by zero"),
    ([(50.0, 0.1, 0.8)], "sigma_g < 1"),
    ([(50.0, 0.1)], "not a 3-tuple"),
    ([(50.0, 0.1, 1.6, 2.0)], "too long"),
    ([50.0], "not a tuple at all"),
    ([(50.0, 0.1, "wide")], "non-numeric"),
    (42, "not a sequence"),
])
def test_degenerate_user_modes_raise(modes, why):
    with pytest.raises(ValueError):
        CoupledScenario(background_dist=modes, background_modes_basis="stp"), why


# ---------------------------------------------------------------------------------------------
# Constructing a scenario must not import JAX.
#
# ``__post_init__`` validates ``background_dist``; it used to do that via
# ``from coupled.tomas_bridge import BACKGROUND_MODES``, which imports jax and sets
# jax_enable_x64 -- ~1 s on the first CoupledScenario() in a process. The tables now live in the
# JAX-free ``coupled.backgrounds``.
#
# This MUST run in a fresh interpreter. An in-process ``"jax" not in sys.modules`` assertion is
# vacuous here: the rest of this suite imports the driver, so jax is already resident by the time
# any single test runs, and the assertion would pass (or fail) for reasons unrelated to this module.
# ---------------------------------------------------------------------------------------------
_NO_JAX_PROBE = textwrap.dedent("""
    import sys
    from coupled.coupled_scenario import CoupledScenario
    CoupledScenario()                                   # default: the tabulated background
    CoupledScenario(background_dist="sabr_220")         # a named lognormal mode set
    try:
        CoupledScenario(background_dist="not_a_background")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown background_dist must still be rejected")
    leaked = sorted({n.split(".")[0] for n in sys.modules} & {"jax", "jaxlib"})
    assert not leaked, f"constructing a CoupledScenario imported {leaked}"
    """)


def test_constructing_a_scenario_does_not_import_jax():
    proc = subprocess.run([sys.executable, "-c", _NO_JAX_PROBE],
                          capture_output=True, text=True, cwd=_REPO_ROOT)
    # cwd = repo root, where `coupled` (and therefore jax) IS importable: the check is only
    # meaningful if the expensive import is available and simply never made.
    assert proc.returncode == 0, f"probe failed:\n{proc.stdout}\n{proc.stderr}"
