"""Tests for the Scenario config (M3)."""

import csv
import os

import numpy as np
import pytest

from config import IDX, SPECIES
from driver import run, run_scenario
from scenario import Scenario

HERE = os.path.dirname(__file__)
FIXDIR = os.path.join(HERE, "fixtures")
DEFAULT_YAML = os.path.join(HERE, "..", "scenarios", "default.yaml")


def test_load_default_yaml():
    s = Scenario.load(DEFAULT_YAML)
    assert s.T == 210 and s.P == 68 and s.opt == 1
    assert s.photolysis == "reference"


def test_roundtrip_yaml_and_json(tmp_path):
    s = Scenario(T=205, P=83, days=3, initial_overrides={"HNO3": 2360})
    for ext in (".yaml", ".json"):
        p = tmp_path / f"scn{ext}"
        s.save(str(p))
        loaded = Scenario.load(str(p))
        assert loaded.to_dict() == s.to_dict()


def test_unknown_key_rejected():
    with pytest.raises(ValueError):
        Scenario.from_dict({"T": 210, "bogus": 1})


def test_unknown_species_override_rejected():
    with pytest.raises(ValueError):
        Scenario(initial_overrides={"Xx": 1.0})


def test_bad_photolysis_mode_rejected():
    with pytest.raises(ValueError):
        Scenario(photolysis="sunshine")


def test_initial_overrides_applied():
    base = Scenario(P=68)
    over = Scenario(P=68, initial_overrides={"BrO": 7.0})
    M = base.to_config().M
    assert over.initial_state()[IDX["BrO"]] == pytest.approx(7.0 * 1e-12 * M)
    # Other species untouched.
    assert over.initial_state()[IDX["HCl"]] == pytest.approx(base.initial_state()[IDX["HCl"]])


def test_run_scenario_matches_run_cfg():
    # A Scenario with default params must give the same trajectory as run(cfg) directly.
    s = Scenario(T=210, P=68, SA=2, WTR=5, Yn2o5=0.1, opt=1, days=1, DT=1800)
    t_s, x_s = run_scenario(s)
    cfg = s.to_config()
    t_c, x_c = run(cfg, td=s.td, tn=s.tn, days=s.days, DT=s.DT)
    assert np.allclose(t_s, t_c)
    assert np.allclose(x_s, x_c)


def test_run_scenario_conserves_chlorine():
    # On the JPL-19-5 branch the trajectory no longer matches the legacy JPL-11 Octave
    # fixture, so we check that a scenario run is well-formed and conserves inorganic
    # chlorine (a rate-independent invariant) instead of comparing to the fixture.
    s = Scenario(T=210.0, P=68.0, SA=2.0, WTR=5.0, Yn2o5=0.1, opt=1, days=2, DT=600.0)
    _t, x = run_scenario(s)
    assert np.all(np.isfinite(x))
    cly = (x[:, IDX["Cl"]] + x[:, IDX["ClO"]] + 2 * x[:, IDX["ClOOCl"]] + x[:, IDX["ClONO2"]]
           + x[:, IDX["HCl"]] + 2 * x[:, IDX["Cl2"]] + x[:, IDX["HOCl"]] + x[:, IDX["OClO"]]
           + x[:, IDX["BrCl"]])
    assert np.max(np.abs(cly - cly[0])) / cly[0] < 1e-2
