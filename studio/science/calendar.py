# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Calendar arithmetic, on a fixed non-leap year.

Not physics, but it belongs here for the same reason the physics does: it is a **convention** that
several things depend on, and a second copy of it would eventually disagree with the first.

**Why non-leap.** SCIENCE-1 settled the climatology as a monthly average over years, so a run has no
year -- a date is a month and a day, and February has 28 days by declaration. The alternative,
accepting a year, would imply the meteorology came from that year's weather, which it does not.

The consequence is stated rather than hidden: on this calendar 21 June is day 172, which is what the
paper ensemble uses (``TABLE_microphysics_parameters.md``: "day 172, ~21 June"). In a leap year the
same date is day 173, and the solar declination differs by about 0.01 degrees -- far below the
uncertainty in anything the declination feeds. Choosing the fixed calendar buys a config that means
one thing forever; the cost is that arithmetic.
"""

from __future__ import annotations

from typing import Final

#: Days in each month of a non-leap year, January first.
DAYS_IN_MONTH: Final[tuple[int, ...]] = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

#: Day of year before each month starts, non-leap: ``CUMULATIVE_DAYS[5] == 151``, so 1 June is
#: day 152 and 21 June is day 172 -- the ensemble's value.
CUMULATIVE_DAYS: Final[tuple[int, ...]] = tuple(sum(DAYS_IN_MONTH[:index]) for index in range(12))

#: Month names, for labels and error messages. Index 0 is January.
MONTH_NAMES: Final[tuple[str, ...]] = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

#: Days in a non-leap year. Named so the 365 in a range check is not a bare literal.
DAYS_IN_YEAR: Final[int] = 365


def days_in_month(month: int) -> int:
    """Length of ``month`` (1-12) on the non-leap calendar.

    Raises:
        ValueError: If the month is out of range, rather than indexing past the end of the tuple.
    """
    if not 1 <= month <= 12:
        raise ValueError(f"month must be 1-12, got {month}")
    return DAYS_IN_MONTH[month - 1]


def day_of_year(*, month: int, day_of_month: int) -> int:
    """Day of year for a (month, day) pair on the non-leap calendar.

    Keyword-only: ``day_of_year(6, 21)`` and ``day_of_year(21, 6)`` are both plausible-looking calls
    and only one is right, which is exactly the mistake worth making impossible.

    Raises:
        ValueError: If the day does not exist in that month. 31 February is a typo, not a date,
            and clamping it to the 28th would run a day the user did not choose.
    """
    length = days_in_month(month)
    if not 1 <= day_of_month <= length:
        raise ValueError(
            f"{MONTH_NAMES[month - 1]} has {length} days on the non-leap calendar, "
            f"so day {day_of_month} does not exist"
        )
    return CUMULATIVE_DAYS[month - 1] + day_of_month


def month_and_day(day: int) -> tuple[int, int]:
    """The inverse: (month, day of month) for a day of year.

    For labels, and for the round-trip test that pins the mapping.
    """
    if not 1 <= day <= DAYS_IN_YEAR:
        raise ValueError(
            f"day of year must be 1-{DAYS_IN_YEAR} on the non-leap calendar, got {day}"
        )
    for index in range(11, -1, -1):
        if day > CUMULATIVE_DAYS[index]:
            return index + 1, day - CUMULATIVE_DAYS[index]
    raise AssertionError(  # pragma: no cover
        "unreachable: day >= 1 always exceeds CUMULATIVE_DAYS[0] == 0"
    )


def describe(*, month: int, day_of_month: int) -> str:
    """A human date, e.g. ``21 June``. No year, because the climatology has none."""
    days_in_month(month)  # validates the month before it is used as an index
    return f"{day_of_month} {MONTH_NAMES[month - 1]}"


__all__ = [
    "CUMULATIVE_DAYS",
    "DAYS_IN_MONTH",
    "DAYS_IN_YEAR",
    "MONTH_NAMES",
    "day_of_year",
    "days_in_month",
    "describe",
    "month_and_day",
]
