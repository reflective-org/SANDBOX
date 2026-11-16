# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Size-distribution reductions: bin mid-points, dlog10Dp, and dN/dlogDp.

Four copies exist in the repository, written two ways:

* ``run_ensemble.py:147-149`` and ``analyses/.../make_background_overlays.py:41-44``
  -- ``dp_mid = 10**(0.5*(log10(edges[:-1]) + log10(edges[1:])))``,
  ``dlogdp = log10(edges)[1:] - log10(edges)[:-1]``
* ``coupled/run_dilution_d1_clean.py:587-589``
  -- ``dp_mid = sqrt(edges[:-1]*edges[1:])``, ``dlogdp = log10(edges[1:]/edges[:-1])``

**These are the same quantity.** ``10**(0.5*(log a + log b)) == sqrt(a*b)`` and
``log b - log a == log(b/a)`` identically; the plan's note that the repository uses "two different
mid-point expressions" is, on inspection, two spellings of one expression. Measured on an 80-bin
TOMAS-like grid they differ by <= 7e-16 (mid-point) and <= 5e-15 (dlog10Dp) relative -- a few ULP of
float64 rounding, not a modelling difference. That is worth stating plainly, because "there are two
conventions in the code" would otherwise become a thing people believe and work around.

This module implements the geometric-mean form (``sqrt(a*b)``): fewer operations, and no
intermediate logarithm to round. The equivalence is asserted by test, so the choice cannot
silently start mattering.

The real trap in this area is not the formula. It is that **``dp_mid_um`` and ``dNdlogDp`` in
``state.npz`` are DRY diameters** while ``SA`` and ``radius_cm`` in the same file are WET -- see
``docs/studio/CAVEATS.md``. This module computes numbers; it does not know which basis its inputs
are on, so its callers must, and ``RunSummary`` (task 0.4) has to declare it per array.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _validated_edges(edges_um: npt.ArrayLike) -> FloatArray:
    """Bin edges as a float array, or raise. Strictly increasing and positive."""
    edges = np.asarray(edges_um, dtype=np.float64)
    if edges.ndim != 1:
        raise ValueError(f"bin edges must be 1-D, got shape {edges.shape}")
    if edges.size < 2:
        raise ValueError(f"need at least 2 bin edges to define a bin, got {edges.size}")
    if not np.all(np.isfinite(edges)):
        raise ValueError("bin edges must all be finite")
    if np.any(edges <= 0.0):
        raise ValueError("bin edges must be > 0 um (the grid is logarithmic in diameter)")
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError("bin edges must be strictly increasing")
    return edges


def bin_midpoints_um(edges_um: npt.ArrayLike) -> FloatArray:
    """Geometric mid-point diameter of each bin [um].

    Geometric, not arithmetic: the TOMAS grid is logarithmic in mass (ratio ``2**(40/n_bins)``), so
    the arithmetic mean of two edges is not the centre of the bin on the axis these are plotted on.
    """
    edges = _validated_edges(edges_um)
    return np.sqrt(edges[:-1] * edges[1:])


def dlog10_dp(edges_um: npt.ArrayLike) -> FloatArray:
    """Width of each bin in log10(diameter): the ``dlogDp`` a size distribution is normalised by."""
    edges = _validated_edges(edges_um)
    return np.log10(edges[1:] / edges[:-1])


def dn_dlogdp(number_per_cm3: npt.ArrayLike, edges_um: npt.ArrayLike) -> FloatArray:
    """Normalise per-bin number concentration [cm^-3] to dN/dlogDp [cm^-3].

    Accepts a single spectrum ``(n_bins,)`` or a time series ``(n_times, n_bins)``; the last axis is
    the size axis, matching ``state.npz``'s ``n_cm3``.

    Normalising is what makes bins comparable across grids: raw per-bin counts on a 40-bin grid and
    an 80-bin grid are not the same curve, and only the normalised form is.

    Raises:
        ValueError: If the bin count does not match the edges. This is the mistake worth catching --
            passing ``n_bins + 1`` edges is right and passing ``n_bins`` is a silent off-by-one that
            numpy would broadcast into a plausible-looking wrong answer.
    """
    counts = np.asarray(number_per_cm3, dtype=np.float64)
    widths = dlog10_dp(edges_um)
    if counts.ndim not in (1, 2):
        raise ValueError(f"number concentration must be 1-D or 2-D, got shape {counts.shape}")
    if counts.shape[-1] != widths.size:
        raise ValueError(
            f"got {counts.shape[-1]} bins but {widths.size + 1} edges define {widths.size} bins; "
            f"edges must have exactly one more element than bins"
        )
    if np.any(counts < 0.0):
        raise ValueError("number concentration must be >= 0 cm^-3")
    return counts / widths


__all__ = ["FloatArray", "bin_midpoints_um", "dlog10_dp", "dn_dlogdp"]
