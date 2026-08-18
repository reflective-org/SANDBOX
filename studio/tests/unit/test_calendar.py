# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The date: a month and a day on a fixed non-leap calendar (SCIENCE-1).

SCIENCE-1 settled the climatology as a **monthly average over years**, which decides the shape of
the date. There is no year to give -- accepting one would imply the run used that year's weather --
so a date is a month and a day, and ``day_of_year`` is derived rather than entered. Entering both
would allow a config whose month and day disagree: the climatology read at one date, the sun at
another.

The calendar is non-leap by declaration. That is a convention, so it is pinned here rather than
trusted: on it, 21 June is day 172, which is exactly what the paper ensemble uses.
"""

from __future__ import annotations

import pytest

from studio.resolve import resolve
from studio.schema import RunConfig
from studio.science.calendar import (
    CUMULATIVE_DAYS,
    DAYS_IN_MONTH,
    DAYS_IN_YEAR,
    day_of_year,
    days_in_month,
    describe,
    month_and_day,
)


@pytest.mark.tier_a
def test_the_calendar_is_a_non_leap_year() -> None:
    """365 days, February 28. The convention itself, asserted."""
    assert sum(DAYS_IN_MONTH) == DAYS_IN_YEAR == 365
    assert days_in_month(2) == 28
    assert CUMULATIVE_DAYS[0] == 0, "nothing precedes January"


@pytest.mark.tier_a
def test_21_june_is_day_172_as_the_ensemble_has_it() -> None:
    """The one value that ties this convention to the archived runs."""
    assert day_of_year(month=6, day_of_month=21) == 172


@pytest.mark.tier_a
def test_the_year_is_covered_end_to_end() -> None:
    assert day_of_year(month=1, day_of_month=1) == 1
    assert day_of_year(month=12, day_of_month=31) == DAYS_IN_YEAR


@pytest.mark.tier_a
def test_every_day_round_trips() -> None:
    """The mapping and its inverse agree for all 365 days, not just the ones anyone tried."""
    for day in range(1, DAYS_IN_YEAR + 1):
        month, day_of_month = month_and_day(day)
        assert day_of_year(month=month, day_of_month=day_of_month) == day


@pytest.mark.tier_a
@pytest.mark.parametrize(
    ("month", "day"),
    [(2, 29), (2, 31), (4, 31), (6, 31), (9, 31), (11, 31)],
)
def test_a_date_that_does_not_exist_is_refused(month: int, day: int) -> None:
    """31 February is a typo, not a date.

    Clamping it to the 28th would run a day the user did not choose, and 29 February is refused for
    the same reason the calendar is non-leap: there is no year in which to have one.
    """
    with pytest.raises(ValueError, match="does not exist"):
        day_of_year(month=month, day_of_month=day)


@pytest.mark.tier_a
@pytest.mark.parametrize("month", [0, 13, -1, 99])
def test_a_month_out_of_range_is_refused_not_wrapped(month: int) -> None:
    """Indexing a tuple with 13 would raise something unhelpful; 0 would silently mean December."""
    with pytest.raises(ValueError, match="month must be 1-12"):
        days_in_month(month)


@pytest.mark.tier_a
def test_describe_names_the_month_and_gives_no_year() -> None:
    assert describe(month=6, day_of_month=21) == "21 June"
    assert "2026" not in describe(month=6, day_of_month=21), "a monthly climatology has no year"


@pytest.mark.tier_a
def test_the_schema_derives_the_day_of_year_from_the_date() -> None:
    """The default config still resolves to the ensemble's day 172, now via the date."""
    config = RunConfig()
    assert (config.schedule.month, config.schedule.day_of_month) == (6, 21)
    assert config.schedule.day_of_year is None, "derived, so unset until resolved"
    assert resolve(config).config.schedule.day_of_year == 172


@pytest.mark.tier_a
def test_the_schema_refuses_a_date_that_does_not_exist() -> None:
    """Validated in the schema as well as the derivation, so the error names the fields typed."""
    with pytest.raises(ValueError, match="non-leap calendar"):
        RunConfig.model_validate({"schedule": {"month": 2, "day_of_month": 30}})


@pytest.mark.tier_a
def test_moving_the_date_moves_the_day_of_year() -> None:
    """The wizard's claim: change the month, and what depends on it follows."""
    december = resolve(RunConfig.model_validate({"schedule": {"month": 12, "day_of_month": 21}}))
    assert december.config.schedule.day_of_year == 355
    assert december.is_consistent


@pytest.mark.tier_a
def test_the_model_seam_refuses_an_unresolved_date(repo_root: object) -> None:
    """A raw config must not reach the model: it would pick a solar declination of its own."""
    from studio.modelio.scenario import to_scenario

    with pytest.raises(ValueError, match=r"schedule\.day_of_year is unresolved|so2_initial_pptv"):
        to_scenario(RunConfig())
