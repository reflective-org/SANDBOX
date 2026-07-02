# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""CoupledScenario: the single coupled-model input, its switches, and validation."""

import os

import pytest

from coupled_scenario import CoupledScenario, Switches

_DEFAULT_YAML = os.path.join(os.path.dirname(__file__), "..", "scenarios", "coupled_default.yaml")


def test_defaults():
    sc = CoupledScenario()
    assert sc.photolysis == "tuvx"
    assert sc.dt_couple == 120.0
    assert sc.switches.sulfur is True
    # every not-yet-implemented switch defaults off
    assert not any([sc.switches.nucleation, sc.switches.condensation, sc.switches.coagulation,
                    sc.switches.aerosol_to_j, sc.switches.heating_to_t, sc.switches.dilution])


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


def test_enabling_unimplemented_switch_raises():
    # A config can't silently claim a capability the model doesn't have yet.
    with pytest.raises(NotImplementedError):
        CoupledScenario(switches=Switches(dilution=True))
    with pytest.raises(NotImplementedError):
        CoupledScenario(switches={"nucleation": True})


def test_unknown_key_rejected():
    with pytest.raises(ValueError):
        CoupledScenario.from_dict({"not_a_field": 1})


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
