"""Tests for SZA-dependent photolysis (M5).

Reference mode must be unchanged (covered by the other tests); here we check the new
SZA-driven behaviour and the j_scale logic.
"""

import math

import numpy as np
import pytest

from config import IDX, ModelConfig
from driver import run_scenario
from rhs import _j_scale
from scenario import Scenario
from solar import cos_solar_zenith, photolysis_scale


def test_photolysis_scale_reference_points():
    # 45 deg -> 1.0; horizon/below -> 0.0; overhead -> ~1.414.
    assert photolysis_scale(math.cos(math.radians(45))) == pytest.approx(1.0)
    assert photolysis_scale(0.0) == 0.0
    assert photolysis_scale(-0.3) == 0.0
    assert photolysis_scale(1.0) == pytest.approx(1.0 / math.cos(math.radians(45)))


def test_reference_jscale_day_night():
    day = ModelConfig(photolysis="reference", SZA=1)
    night = ModelConfig(photolysis="reference", SZA=0)
    assert _j_scale(0.0, day) == 1.0
    assert _j_scale(0.0, night) == 0.0


def test_sza_jscale_zero_at_night_peak_at_noon():
    # Equator, equinox, longitude 0: midnight (UTC 0) -> 0; noon (UTC 12) -> sun overhead.
    cfg = ModelConfig(photolysis="sza", latitude=0.0, longitude=0.0,
                      day_of_year=80, start_utc_hour=0.0)
    j_midnight = _j_scale(0.0, cfg)                 # t=0 -> UTC 0
    j_noon = _j_scale(12 * 3600.0, cfg)             # t=12h -> UTC 12
    assert j_midnight == 0.0
    # Overhead sun -> scale ~ 1/cos45 ~ 1.41.
    assert j_noon == pytest.approx(1.0 / math.cos(math.radians(45)), abs=0.02)


def test_sza_jscale_matches_solar():
    cfg = ModelConfig(photolysis="sza", latitude=30.0, longitude=10.0,
                      day_of_year=172, start_utc_hour=3.0)
    t = 5 * 3600.0
    total_hours = 3.0 + 5.0
    cosz = cos_solar_zenith(30.0, 10.0, 172 + total_hours / 24.0, total_hours % 24.0)
    assert _j_scale(t, cfg) == pytest.approx(photolysis_scale(cosz))


def test_sza_run_completes_and_has_diurnal_cycle():
    # A short SZA-mode run should integrate fine and show a day/night cycle in a
    # photolysis-driven species (e.g. NO2 builds at night, OH collapses at night).
    s = Scenario(photolysis="sza", latitude=20.0, longitude=0.0, day_of_year=80,
                 start_utc_hour=0.0, days=2, DT=900.0)
    t, x = run_scenario(s)
    assert t[-1] == pytest.approx(2 * 24 * 3600.0)
    oh = x[:, IDX["OH"]]
    # OH should vary strongly between day and night (min near zero at night).
    assert oh.max() > 0
    assert oh.min() / oh.max() < 0.2
