# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Size-distribution reductions, and the claim that the repository's "two conventions" are one.

The headline test here is ``test_the_repositorys_two_spellings_are_the_same_quantity``: it measures
the difference between the two forms in the code rather than asserting they are equivalent on paper,
because "these are the same" is the kind of statement that is easy to believe and expensive to be
wrong about.
"""

from __future__ import annotations

import numpy as np
import pytest

from studio.science import bin_midpoints_um, dlog10_dp, dn_dlogdp

#: A TOMAS-like 80-bin grid: dry Dp from 1.7 nm to 17.5 um, geometric (the real grid's mass ratio is
#: 2**(40/n_bins), which is geometric in diameter too). Built here rather than imported from the
#: model because Tier A must not pay a JAX import; the property under test is grid-shape-independent
#: and is also checked on a deliberately irregular grid below.
EDGES_UM = np.geomspace(1.7e-3, 17.5, 81)


@pytest.mark.tier_a
def test_the_repositorys_two_spellings_are_the_same_quantity() -> None:
    """``10**(0.5*(log a + log b))`` vs ``sqrt(a*b)``, and ``log b - log a`` vs ``log(b/a)``.

    Tolerance 1e-14 relative, against a measured ~7e-16 (mid-point) and ~5e-15 (dlog10Dp) on this
    grid: a few ULP of float64, i.e. rounding, not a modelling difference. The plan's note that the
    repository carries "two different mid-point expressions" is a misreading of two spellings of one
    expression, and this is the evidence for saying so.
    """
    log_edges = np.log10(EDGES_UM)
    run_ensemble_mid = 10 ** (0.5 * (log_edges[:-1] + log_edges[1:]))  # run_ensemble.py:148
    run_ensemble_dlog = log_edges[1:] - log_edges[:-1]  # run_ensemble.py:148

    np.testing.assert_allclose(bin_midpoints_um(EDGES_UM), run_ensemble_mid, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(dlog10_dp(EDGES_UM), run_ensemble_dlog, rtol=1e-14, atol=0.0)


@pytest.mark.tier_a
def test_equivalence_holds_on_an_irregular_grid_too() -> None:
    """Not an artefact of a perfectly geometric grid: same check where bin widths vary wildly."""
    edges = np.array([1e-3, 2e-3, 5e-3, 1e-2, 3e-1, 1.0, 17.5])
    log_edges = np.log10(edges)
    np.testing.assert_allclose(
        bin_midpoints_um(edges), 10 ** (0.5 * (log_edges[:-1] + log_edges[1:])), rtol=1e-14
    )
    np.testing.assert_allclose(dlog10_dp(edges), log_edges[1:] - log_edges[:-1], rtol=1e-14)


@pytest.mark.tier_a
def test_midpoint_is_geometric_not_arithmetic() -> None:
    """On a log axis the arithmetic mean is not the centre, and the difference is not small.

    For a bin spanning a decade the two differ by ~28 %, which would be visible as a shifted mode
    diameter in every size-distribution figure.
    """
    edges = np.array([0.1, 1.0])
    assert bin_midpoints_um(edges)[0] == pytest.approx(np.sqrt(0.1), rel=1e-15)
    arithmetic = 0.55
    assert abs(bin_midpoints_um(edges)[0] - arithmetic) / arithmetic > 0.25


@pytest.mark.tier_a
def test_dn_dlogdp_normalises_by_bin_width() -> None:
    """The defining property: equal counts in unequal bins are NOT an equal dN/dlogDp."""
    edges = np.array([1.0, 10.0, 1000.0])  # widths 1 and 2 in log10
    result = dn_dlogdp(np.array([100.0, 100.0]), edges)
    np.testing.assert_allclose(result, [100.0, 50.0], rtol=1e-15)


@pytest.mark.tier_a
def test_dn_dlogdp_handles_a_time_series() -> None:
    """``state.npz`` stores ``n_cm3`` as (n_times, n_bins); the size axis is last."""
    counts = np.tile(np.linspace(1.0, 80.0, 80), (5, 1))
    result = dn_dlogdp(counts, EDGES_UM)
    assert result.shape == (5, 80)
    np.testing.assert_allclose(result[0], counts[0] / dlog10_dp(EDGES_UM), rtol=1e-15)


@pytest.mark.tier_a
def test_integrating_dn_dlogdp_recovers_the_total_number() -> None:
    """sum(dN/dlogDp * dlogDp) == sum(N) -- the conservation check behind the normalisation."""
    counts = np.linspace(0.5, 40.0, 80)
    recovered = float(np.sum(dn_dlogdp(counts, EDGES_UM) * dlog10_dp(EDGES_UM)))
    assert recovered == pytest.approx(float(counts.sum()), rel=1e-14)


@pytest.mark.tier_a
def test_off_by_one_in_the_edge_count_raises() -> None:
    """The mistake worth catching: n edges for n bins broadcasts into a plausible wrong answer."""
    with pytest.raises(ValueError, match="edges must have exactly one more element"):
        dn_dlogdp(np.ones(80), EDGES_UM[:-1])
    with pytest.raises(ValueError, match="edges must have exactly one more element"):
        dn_dlogdp(np.ones(79), EDGES_UM)


@pytest.mark.tier_a
@pytest.mark.parametrize(
    ("edges", "message"),
    [
        (np.array([1.0]), "at least 2 bin edges"),
        (np.array([[1.0, 2.0], [3.0, 4.0]]), "must be 1-D"),
        (np.array([0.0, 1.0]), "must be > 0 um"),
        (np.array([-1.0, 1.0]), "must be > 0 um"),
        (np.array([1.0, 0.5, 2.0]), "strictly increasing"),
        (np.array([1.0, 1.0]), "strictly increasing"),
        (np.array([1.0, np.nan]), "must all be finite"),
    ],
)
def test_malformed_edges_raise(edges: np.ndarray, message: str) -> None:
    """A logarithmic grid has preconditions; violating them silently yields NaN, not an error."""
    with pytest.raises(ValueError, match=message):
        bin_midpoints_um(edges)


@pytest.mark.tier_a
def test_negative_counts_raise() -> None:
    """A negative number concentration is a corrupted input, not a small one."""
    with pytest.raises(ValueError, match="must be >= 0"):
        dn_dlogdp(np.array([1.0, -1.0]), np.array([1.0, 2.0, 3.0]))
