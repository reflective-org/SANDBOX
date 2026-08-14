# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Background aerosol size distributions -- the *data*, with NO model imports.

Split out of ``tomas_bridge`` so that validating a scenario does not cost a JAX import.
``CoupledScenario.__post_init__`` has to check ``background_dist``, and importing ``tomas_bridge``
to do it pulled in ``jax`` (which also sets ``jax_enable_x64``) -- ~1 s on the first
``CoupledScenario(...)`` in a process. A form-validating API cannot pay that per keystroke.

**This module must stay free of numpy/jax/scipy and of any ``coupled`` sibling that imports them.**
It is pure data + pure-Python validation; ``tomas_bridge`` re-exports the tables (so existing
``tomas_bridge.BACKGROUND_MODES`` references keep working) and does the actual seeding.
"""

from __future__ import annotations

#: The tabulated (non-lognormal) background: Marianna's digitized "red circles" distribution,
#: loaded by ``background_aerosol_distribution.get_initial_state`` in ``tomas_bridge``.
TABULATED_BACKGROUND = "redcircles"

# --- background aerosol size distributions as (multi-)lognormal modes, DIAMETER basis. Each entry is
# a list of (N [cm^-3, STP], Dg [um], sigma_g). DIGITIZED (approximate) from the SABR / CESM plots the
# user provided; see coupled/analyses/paper_ensemble/DECISIONS.md for the source figures and the
# overlay-verification. "redcircles" (Marianna, tabulated loader) stays the default and is NOT here. ---
# N chosen so each mode's PEAK dN/dlogDp = N/(sqrt(2pi)*log10(sigma_g)) matches the value read off the
# source plot (the most reliable digitized feature): SABR 330->~1000, 220->~95; CESM Aitken->~50,
# Accumulation->~12, Coarse->~0.5 cm^-3 STP.
BACKGROUND_MODES = {
    "sabr_330": [(810.0, 0.045, 2.1)],                                    # young air (high N2O), peak ~1000
    "sabr_310": [(205.0, 0.060, 1.8)],                                    # mid air (310-320 ppbv), peak ~320
    "sabr_220": [(49.0, 0.12, 1.6)],                                      # aged air (low N2O), peak ~95
    "cesm_g6":  [(22.0, 0.040, 1.5), (5.3, 0.20, 1.5), (0.18, 0.90, 1.4)],  # CESM G6 SAI (r->D x2)
    # AER 2D geoengineered stratosphere (Pierce et al. fig. 2, gray curve: 5 Mt-S/yr, 95 nm case).
    # Dg = 0.30 um (mode radius 0.15 um) and sigma_g = 1.7 fitted to the curve; N = 120 cm^-3 per
    # user spec (paper caption quotes 50 cm^-3). Values are AMBIENT -> no STP conversion on seeding.
    "aer_geo":  [(120.0, 0.30, 1.7)],
    # CESM G6 with the source plot read as AMBIENT (user-confirmed): same modes as cesm_g6 but
    # seeded without the STP->ambient factor. cesm_g6 is kept unchanged so the original 810-run
    # ensemble stays reproducible.
    "cesm_g6_amb": [(22.0, 0.040, 1.5), (5.3, 0.20, 1.5), (0.18, 0.90, 1.4)],
}
# mode sets specified at AMBIENT conditions (seeding skips the STP->ambient factor)
AMBIENT_BACKGROUNDS = {"aer_geo", "cesm_g6_amb"}
