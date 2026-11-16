# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The measured tolerances, as code, and the comparison that applies them.

**Every number here was measured before it was asserted** and is traceable to
``REFERENCE_TOLERANCES.md`` in this directory, which records the SHAs, the environment and the
per-quantity deviations it came from (task 0.7, issue #70). None of them is a value somebody chose
because a test failed.

The rule that follows: **if an assertion here starts failing, re-run the measurement and update the
record — do not widen the number.** A tolerance widened to make a test pass is a test that no longer
tests anything, and this file exists to make that a visible edit rather than a quiet one.

Two subtleties the measurement turned up, both encoded below rather than left to whoever writes the
next assertion:

* **Near-zero series must be floored.** Unguarded relative error reaches 4.24e+04 on night-time
  ``O1D``/``O`` at O(1e-35) molec cm^-3 oscillating about zero -- the archived value is literally
  negative on a species that peaks around 3 molec cm^-3. Comparing only samples above
  ``1e-6 x the series' own peak`` is what makes the comparison mean anything.
* **``dp_mid_um`` is not exact.** The archive writes ``10**(0.5*(log10 a + log10 b))``; Studio
  writes ``sqrt(a*b)`` (task 0.5, deliberately). Algebraically identical, ~8e-16 apart in float64.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

#: Final and peak values of the quantities a result is actually read for. Measured <= 2.9e-14 across
#: both reference cases; 1e-12 is ~35x headroom and still ~6 orders below any meaningful physics
#: change, so a real regression cannot hide under it.
RTOL_HEADLINE: Final = 1e-12

#: Any stored series, compared sample by sample. Measured worst 3.4e-12 (H2SO4, day 1.34); 1e-10
#: covers the worst trace species (Cl2, 1.1e-11) with ~10x margin.
RTOL_SERIES: Final = 1e-10

#: Per-bin size distribution. Per-bin conditioning is worse than integrated number, so it gets its
#: own looser number; measured worst bin 2.0e-12.
RTOL_SIZE_DISTRIBUTION: Final = 1e-10

#: Photolysis J. Measured 1.09e-13 and identical in both reference cases -- TUV-x is the most
#: reproducible part of the pipeline.
RTOL_PHOTOLYSIS: Final = 1e-11

#: Dry bin mid-points. NOT exact: see the module docstring. A few ULP, measured 8.1e-16.
RTOL_BIN_EDGES: Final = 1e-15

#: Arrays measured bit-identical, because they are analytic rather than integrated. Any difference
#: at all is a real bug, so these are compared exactly and deliberately have no tolerance.
EXACT_ARRAYS: Final = ("t", "V_ratio", "T")

#: A sample is compared only if it exceeds this fraction of its own series' peak. Below it, relative
#: error is meaningless (see the module docstring); an absolute-floor assertion would be the way to
#: cover those, and is not attempted here rather than being faked.
NEAR_ZERO_FRACTION_OF_PEAK: Final = 1e-6

#: Species excluded outright: they spend most of the run at O(1e-35) and oscillate about zero, so
#: even the floor above leaves too few comparable samples to mean anything.
EXCLUDED_SPECIES: Final = ("O1D", "O")

FloatArray = npt.NDArray[np.float64]


def worst_relative_deviation(
    fresh: npt.ArrayLike, reference: npt.ArrayLike, *, floor_by_peak: bool = True
) -> float:
    """Worst relative deviation between two series, ignoring samples below the near-zero floor.

    Returns 0.0 when nothing is comparable, rather than NaN: a series entirely below its own floor
    carries no information either way, and propagating NaN into an assertion would fail for the
    wrong reason.
    """
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(fresh, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: reference {a.shape} vs fresh {b.shape}")
    magnitude = np.abs(a)
    if floor_by_peak:
        peak = float(np.nanmax(magnitude)) if magnitude.size else 0.0
        comparable = magnitude > peak * NEAR_ZERO_FRACTION_OF_PEAK
    else:
        comparable = magnitude > 0.0
    comparable &= np.isfinite(a) & np.isfinite(b)
    if not np.any(comparable):
        return 0.0
    return float(np.max(np.abs(b[comparable] - a[comparable]) / magnitude[comparable]))


def assert_series_matches(
    name: str, fresh: npt.ArrayLike, reference: npt.ArrayLike, rtol: float
) -> None:
    """Compare a series, or raise with the measured number and where to look.

    The message names the tolerance's provenance on purpose: the first instinct on a failure here is
    to widen the number, and the right response is to re-measure.
    """
    worst = worst_relative_deviation(fresh, reference)
    if worst > rtol:
        raise AssertionError(
            f"{name}: worst relative deviation {worst:.3e} exceeds {rtol:.0e}.\n"
            f"This tolerance was MEASURED (see REFERENCE_TOLERANCES.md in this directory), not "
            f"chosen. Re-run the measurement and update that record with the new SHAs -- do not "
            f"widen this number to make the test pass."
        )


def endpoint_deviations(fresh: npt.ArrayLike, reference: npt.ArrayLike) -> dict[str, float]:
    """Relative deviation of the FINAL and PEAK values of a series.

    Separate from :func:`worst_relative_deviation` because the record gives these two a tighter
    tolerance than the series they come from -- ``1e-12`` against ``1e-10``. Conflating them is an
    easy mistake with a misleading symptom: it looks like a reproduction failure when it is a test
    reading the wrong row.
    """
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(fresh, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: reference {a.shape} vs fresh {b.shape}")
    out: dict[str, float] = {}
    for label, reference_value, fresh_value in (
        ("final", float(a[-1]), float(b[-1])),
        ("peak", float(np.nanmax(a)), float(np.nanmax(b))),
    ):
        if reference_value == 0.0:
            continue  # a zero endpoint has no relative deviation; the series check still covers it
        out[label] = abs(fresh_value - reference_value) / abs(reference_value)
    return out


def assert_headline_matches(name: str, fresh: npt.ArrayLike, reference: npt.ArrayLike) -> None:
    """Final and peak at ``RTOL_HEADLINE``. What a result is actually read for."""
    for label, deviation in endpoint_deviations(fresh, reference).items():
        if deviation > RTOL_HEADLINE:
            raise AssertionError(
                f"{name} ({label}): relative deviation {deviation:.3e} exceeds "
                f"{RTOL_HEADLINE:.0e}. This is the ENDPOINT tolerance; the series that produced it "
                f"has its own, looser one ({RTOL_SERIES:.0e}). Both were measured -- see "
                f"REFERENCE_TOLERANCES.md in this directory."
            )


def assert_exact(name: str, fresh: npt.ArrayLike, reference: npt.ArrayLike) -> None:
    """Bit-for-bit, for the analytic arrays. Any difference is a real bug."""
    a = np.asarray(reference)
    b = np.asarray(fresh)
    if not np.array_equal(a, b):
        differing = int(np.sum(a != b)) if a.shape == b.shape else -1
        raise AssertionError(
            f"{name} is expected to be bit-identical (analytic, not integrated) but "
            f"{differing} element(s) differ. Worst relative deviation "
            f"{worst_relative_deviation(b, a, floor_by_peak=False):.3e}."
        )


__all__ = [
    "EXACT_ARRAYS",
    "EXCLUDED_SPECIES",
    "NEAR_ZERO_FRACTION_OF_PEAK",
    "RTOL_BIN_EDGES",
    "RTOL_HEADLINE",
    "RTOL_PHOTOLYSIS",
    "RTOL_SERIES",
    "RTOL_SIZE_DISTRIBUTION",
    "assert_exact",
    "assert_headline_matches",
    "assert_series_matches",
    "endpoint_deviations",
    "worst_relative_deviation",
]
