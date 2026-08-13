# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Climatology access and derived environment quantities (Phase 1).

ERA5 via the Copernicus CDS API (BLOCKING-5, answered: credentials available). Raw gridded fields
are NEVER committed and never loaded into the relational store; the committed artefact is a reduced,
derived product read via ``xarray`` (ADR-004). Preprocessing lives in ``data/pipelines/`` as a
committed, re-runnable pipeline with recorded input identifiers and output checksums.

This package exposes DERIVED quantities -- tropopause height per definition, and T / p / H2O at a
target level -- not raw grids.

Nothing here exists yet, and none of it exists in the wider repository either: grep for "tropopause"
hits only literature notes in ``tomas-jax``. Phase 1 is genuinely new code.

Open before the preprocessing pipeline is written, not after:

* SCIENCE-1 -- zonal mean vs longitude-resolved; monthly climatology vs daily vs a specific
  reanalysis timestep. These give materially different tropopause heights and temperatures, and they
  determine the reduced product's dimensions. Note longitude and date are required anyway for the
  solar zenith angle, so a longitude-resolved option should exist even if the default is zonal-mean.

Required behaviour once built:

* Three tropopause definitions -- WMO lapse-rate (default), cold-point, dynamical (2 PVU). They
  differ by 1-3 km, so the stage-1 figure shows all available definitions rather than implying the
  spread away.
* Interpolation method (in pressure level and latitude) is documented and unit-tested against
  hand-computed cases -- not whichever xarray default happens to be in force.
* A missing value RAISES with a message naming the requested point and the product's coverage
  (ADR-005). It never returns a nearest neighbour silently.
* ERA5 stratospheric water vapour is biased dry; it is available but is not the recommended source
  for ``h2o_mixing_ratio_ppmv``. Carried as a user-facing caveat.
"""

from __future__ import annotations

__all__: list[str] = []
