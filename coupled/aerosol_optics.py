# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Spectral aerosol optics from a TomasState, for the aerosol->photolysis feedback (Phase 4).

Builds per-(layer, wavelength) optical depth / single-scattering albedo / asymmetry g on the TUV-x
wavelength grid, so the photolysis solver's actinic flux (and hence J) responds to the aerosol.

Method (AD-4.1): call tomas-jax's Bohren-Huffman Mie (`bhmie_qsca_jax`) per wavelength x per bin on the
FIXED geometric-mean bin radii (TOMAS's own radiative-forcing radius convention) with a FIXED complex
refractive index -> a precomputed Mie table Qext/Qsca/gsca of shape (n_wl, n_bins) that is reused every
step (it depends only on the fixed radii + index). Per step, the evolving number Nk weights the table:
  b_ext(lambda) = sum_k (Nk/boxvol) * pi r_k^2 * Qext(lambda,k)      [1/cm]
  b_sca(lambda) = sum_k (Nk/boxvol) * pi r_k^2 * Qsca(lambda,k)
  SSA(lambda)   = b_sca / b_ext ;  g(lambda) = sum_k g*b_sca_k / b_sca   (scattering-weighted)
Vertical placement (AD-4.2, OPEN): the box extinction is spread UNIFORMLY over a configurable altitude
band; each layer in the band gets OD = b_ext(lambda) * dz. Refractive index is wavelength-independent
(no n(lambda) data; documented approximation, DEFERRED).
"""

from __future__ import annotations

import numpy as np

from . import tomas_bridge as tb   # puts tomas_jax on sys.path

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from tomas_jax.physics.bhmie import bhmie_qsca_jax  # noqa: E402
from tomas_jax.physics.radiative_forcing import precompute_mie_properties, REFINDEX_SULFATE  # noqa: E402

from tuvx_photolysis.radiators import RadiatorOpticalProps  # noqa: E402

#: fixed complex refractive index for sulfate (no wavelength dependence -- AD-4.1 / DEFERRED).
REFRACTIVE_INDEX = REFINDEX_SULFATE   # 1.4 + 1e-8j


def bin_radii_m(xk=None) -> np.ndarray:
    """Geometric-mean dry bin radii [m] (the TOMAS radiative-forcing convention), shape (n_bins,)."""
    if xk is None:
        xk = tb.tcfg.xk_boundaries()
    return np.asarray(precompute_mie_properties(xk).radii, dtype=float)


class MieTable:
    """Precomputed per-(wavelength, bin) Mie efficiencies on fixed radii. Reused every step."""

    def __init__(self, wavelength_nm, radii_m=None, refractive_index=REFRACTIVE_INDEX):
        self.wavelength_nm = np.asarray(wavelength_nm, dtype=float)
        self.radii_m = bin_radii_m() if radii_m is None else np.asarray(radii_m, dtype=float)
        self.refractive_index = complex(refractive_index)
        n_wl, n_bin = self.wavelength_nm.size, self.radii_m.size
        # size parameter x = 2*pi*r/lambda for every (wavelength, bin) pair
        lam_m = self.wavelength_nm[:, None] * 1.0e-9
        x = (2.0 * np.pi * self.radii_m[None, :]) / lam_m          # (n_wl, n_bin)
        qext, qsca, gsca = jax.vmap(bhmie_qsca_jax, in_axes=(0, None))(
            jnp.asarray(x.reshape(-1)), self.refractive_index)
        self.Qext = np.asarray(qext, dtype=float).reshape(n_wl, n_bin)
        self.Qsca = np.asarray(qsca, dtype=float).reshape(n_wl, n_bin)
        self.gsca = np.asarray(gsca, dtype=float).reshape(n_wl, n_bin)


def bulk_optics(state, mie: MieTable):
    """Column extinction/scattering coefficients + SSA + g per wavelength from a TomasState.

    Returns ``(b_ext, b_sca, ssa, g)`` each shape ``(n_wl,)``; ``b_ext``/``b_sca`` in [1/cm].
    """
    n_cm3 = np.asarray(state.Nk, dtype=float) / float(state.boxvol)    # #/cm^3, (n_bin,)
    r_cm = mie.radii_m * 100.0                                         # m -> cm
    geo = np.pi * r_cm ** 2                                            # geometric cross section [cm^2]
    # per wavelength: sum over bins of n * geo * Q
    b_ext = (mie.Qext * (n_cm3 * geo)[None, :]).sum(axis=1)            # (n_wl,)
    b_sca = (mie.Qsca * (n_cm3 * geo)[None, :]).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ssa = np.where(b_ext > 0.0, b_sca / b_ext, 0.0)
        g_num = (mie.gsca * mie.Qsca * (n_cm3 * geo)[None, :]).sum(axis=1)
        g = np.where(b_sca > 0.0, g_num / b_sca, 0.0)
    return b_ext, b_sca, ssa, g


def aerosol_optical_props(state, wavelength_nm, height_edges_km, band_km, mie: MieTable | None = None):
    """Per-(layer, wavelength) aerosol ``RadiatorOpticalProps`` for the TUV-x column.

    The box extinction coefficient is spread uniformly over the altitude band ``band_km=(z_lo, z_hi)``
    (AD-4.2): each layer whose center is in the band gets ``OD = b_ext(lambda) * dz``; layers outside
    get 0. ``height_edges_km`` is the level grid (n_levels,), bottom-up; there are n_levels-1 layers.
    """
    if mie is None:
        mie = MieTable(wavelength_nm)
    b_ext, _b_sca, ssa, g = bulk_optics(state, mie)                    # each (n_wl,)
    edges = np.asarray(height_edges_km, dtype=float)
    dz_cm = np.diff(edges) * 1.0e5                                     # km -> cm, (n_layers,)
    z_center = 0.5 * (edges[:-1] + edges[1:])
    z_lo, z_hi = float(band_km[0]), float(band_km[1])
    in_band = (z_center >= z_lo) & (z_center <= z_hi)                  # (n_layers,)
    # OD(layer, wl) = b_ext(wl) * dz(layer) for in-band layers, else 0
    od = (in_band * dz_cm)[:, None] * b_ext[None, :]                  # (n_layers, n_wl)
    ssa_2d = np.broadcast_to(ssa[None, :], od.shape)
    g_2d = np.broadcast_to(g[None, :], od.shape)
    return RadiatorOpticalProps(od, ssa_2d, g_2d, is_air=False)
