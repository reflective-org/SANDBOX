"""Solar geometry: solar zenith angle (SZA) from latitude, longitude, date, and time.

This provides the astronomical input the photolysis code needs once it becomes SZA-dependent
(the model currently uses a fixed 45-deg daytime / off-at-night switch). Nothing here changes
the chemistry; it just answers "where is the sun?".

Formulas follow the standard NOAA solar-position algorithm with Spencer's (1971) Fourier
series for the solar declination and the equation of time. Accuracy is ~0.1-0.5 deg, which is
ample for scaling photolysis rates.

Conventions:
    latitude   : degrees, North positive
    longitude  : degrees, East positive
    day_of_year: 1 = Jan 1 ... 365/366
    utc_hour   : hours (UTC), e.g. 13.5 = 13:30 UTC
"""

from __future__ import annotations

import math

_DEG = math.pi / 180.0


def _fractional_year(day_of_year: float, utc_hour: float) -> float:
    """Spencer's fractional year angle gamma (radians)."""
    return 2.0 * math.pi / 365.0 * (day_of_year - 1.0 + (utc_hour - 12.0) / 24.0)


def solar_declination(day_of_year: float, utc_hour: float = 12.0) -> float:
    """Solar declination (radians), Spencer (1971) Fourier series."""
    g = _fractional_year(day_of_year, utc_hour)
    return (
        0.006918
        - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
        - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
        - 0.002697 * math.cos(3 * g) + 0.001480 * math.sin(3 * g)
    )


def equation_of_time(day_of_year: float, utc_hour: float = 12.0) -> float:
    """Equation of time (minutes), Spencer (1971): apparent minus mean solar time."""
    g = _fractional_year(day_of_year, utc_hour)
    return 229.18 * (
        0.000075
        + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
        - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g)
    )


def cos_solar_zenith(latitude: float, longitude: float,
                     day_of_year: float, utc_hour: float) -> float:
    """Cosine of the solar zenith angle.

    cos(SZA) = sin(lat)sin(dec) + cos(lat)cos(dec)cos(H), where H is the hour angle.
    Positive => sun above the horizon; negative => below (night).
    """
    dec = solar_declination(day_of_year, utc_hour)
    eot = equation_of_time(day_of_year, utc_hour)

    # True solar time (minutes): UTC clock time, corrected by longitude (4 min/deg East)
    # and the equation of time.
    true_solar_time = utc_hour * 60.0 + 4.0 * longitude + eot
    hour_angle = (true_solar_time / 4.0 - 180.0) * _DEG  # radians; 0 at solar noon

    latr = latitude * _DEG
    return (math.sin(latr) * math.sin(dec)
            + math.cos(latr) * math.cos(dec) * math.cos(hour_angle))


def solar_zenith_angle(latitude: float, longitude: float,
                       day_of_year: float, utc_hour: float) -> float:
    """Solar zenith angle in degrees (0 = sun overhead, 90 = horizon, >90 = below horizon)."""
    cosz = max(-1.0, min(1.0, cos_solar_zenith(latitude, longitude, day_of_year, utc_hour)))
    return math.degrees(math.acos(cosz))


def is_daytime(latitude: float, longitude: float,
               day_of_year: float, utc_hour: float) -> bool:
    """True when the sun is above the horizon (SZA < 90 deg)."""
    return cos_solar_zenith(latitude, longitude, day_of_year, utc_hour) > 0.0


# Reference solar zenith angle for the tabulated J-values (they are quoted at 45 deg).
_COS45 = math.cos(45.0 * _DEG)


def photolysis_scale(cos_sza: float) -> float:
    """Normalized-cosine photolysis scaling factor from cos(SZA).

    factor = max(cos SZA, 0) / cos(45 deg)
      * 1.0 when the sun is at 45 deg (so it reproduces the tabulated J-values),
      * 0.0 when the sun is at or below the horizon (night),
      * up to ~1.41 with the sun overhead.

    A single shape is applied to all photolysis reactions (a first-cut approximation; a
    per-reaction J(SZA) table can replace this later without changing the call sites).
    """
    return max(cos_sza, 0.0) / _COS45
