"""Tests for solar geometry (M4), checked against well-known solar-position facts."""

import math

import pytest

from solar import (
    cos_solar_zenith,
    equation_of_time,
    is_daytime,
    solar_declination,
    solar_zenith_angle,
)


def test_declination_at_solstices():
    # Declination reaches ~ +/-23.44 deg at the solstices (Jun ~172, Dec ~355).
    jun = math.degrees(solar_declination(172))
    dec = math.degrees(solar_declination(355))
    assert jun == pytest.approx(23.44, abs=0.4)
    assert dec == pytest.approx(-23.44, abs=0.4)


def test_declination_near_zero_at_equinoxes():
    # Equinoxes near day 80 (Mar 21) and 266 (Sep 23): declination ~ 0.
    assert abs(math.degrees(solar_declination(80))) < 1.5
    assert abs(math.degrees(solar_declination(266))) < 1.5


def test_equation_of_time_reasonable_range():
    # EoT stays within about +/-17 minutes over the year.
    vals = [equation_of_time(d) for d in range(1, 366, 5)]
    assert max(vals) < 17.0
    assert min(vals) > -17.0


def test_overhead_sun_equator_equinox_noon():
    # Equator, equinox, local solar noon at longitude 0 (UTC noon) -> sun nearly overhead.
    sza = solar_zenith_angle(latitude=0.0, longitude=0.0, day_of_year=80, utc_hour=12.0)
    assert sza < 2.0


def test_night_is_below_horizon():
    # Longitude 0 at local midnight (UTC 00:00) -> sun well below the horizon everywhere mid-lat.
    assert not is_daytime(latitude=45.0, longitude=0.0, day_of_year=80, utc_hour=0.0)
    assert cos_solar_zenith(45.0, 0.0, 80, 0.0) < 0.0


def test_higher_latitude_larger_zenith_at_noon():
    # At equinox solar noon, SZA at noon ~ |latitude|, so it grows with latitude.
    sza0 = solar_zenith_angle(0.0, 0.0, 80, 12.0)
    sza30 = solar_zenith_angle(30.0, 0.0, 80, 12.0)
    sza60 = solar_zenith_angle(60.0, 0.0, 80, 12.0)
    assert sza0 < sza30 < sza60
    assert sza30 == pytest.approx(30.0, abs=2.0)
    assert sza60 == pytest.approx(60.0, abs=2.0)


def test_longitude_shifts_solar_noon():
    # At longitude +15 deg East, solar noon occurs 1 hour earlier in UTC (11:00 UTC).
    sza = solar_zenith_angle(latitude=0.0, longitude=15.0, day_of_year=80, utc_hour=11.0)
    assert sza < 2.0
