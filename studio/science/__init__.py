# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Scientific derivations: plume volume, initial concentration, size-distribution reductions, GCR.

Consolidation as much as new code. Each derivation below existed several times over in this
repository, and this is the one cited, tested implementation:

* **Plume volume and injected mass -> initial concentration** (``plume.py``). Six copies today, and
  they disagree: the ensemble uses a 15 km track and the D1 flagship a 30 km one -- a factor of two
  in V0 for the same injected mass. Which is right depends on what t = 0 means (SCIENCE-2, #54).
* **dN/dlogDp and bin mid-points** (``size_distribution.py``). Four copies, written two ways which
  turn out to be algebraically identical -- measured agreement to a few ULP.
* **Air number density** (``air.py``). A deliberate MIRROR of the model's, because this package may
  not import the model; a test asserts they agree exactly.
* **GCR ion-pair rate** (``gcr.py``). Raises ``NotImplementedError``: the value in use is uncited
  and the honest thing is to say so (SCIENCE-6, #63).

**Reused, not rewritten.** These are already implemented and tested in the model, so Studio reaches
them through ``studio/modelio`` (the only package allowed to import ``coupled``) rather than keeping
a second copy here:

* ``coupled.dilution.volume_ratio`` / ``kdil_from_regime`` -- plume expansion V(t)/V0 and k_dil
* ``coupled.aerosol_props`` -- surface area, effective wet radius, H2SO4 weight percent
* ``coupled.units`` -- number density <-> mass per grid cell

Every physical constant lives in ``constants.py`` with its source, including the two that are
deliberately the model's rounded values rather than the best-known ones -- Studio inherits the
model's constants so that Phase 0 can reproduce its runs, and says so at the point of use.

This package must not import ``coupled``, the API, or the database (see ``studio/__init__.py``).
"""

from __future__ import annotations

from studio.science.air import air_number_density
from studio.science.constants import (
    AIR_NUMBER_DENSITY_COEFF,
    AVOGADRO,
    AVOGADRO_GAS_MODEL,
    AVOGADRO_SEAM_RELATIVE_DIFFERENCE,
    CM3_PER_M3,
    G_PER_KG,
    H2SO4_MOLAR_MASS_G_PER_MOL,
    MBAR_TO_TORR,
    PPMV_PER_MOLE_FRACTION,
    PPTV_PER_MOLE_FRACTION,
    SO2_MOLAR_MASS_G_PER_MOL,
)
from studio.science.gcr import (
    MODEL_DEFAULT_ION_PAIR_RATE,
    PAPER_ENSEMBLE_ION_PAIR_RATE,
    ion_pair_production_rate,
)
from studio.science.plume import (
    initial_mixing_ratio_pptv,
    injected_number_density,
    number_density_to_pptv,
    plume_volume_cm3,
    pptv_to_number_density,
)
from studio.science.size_distribution import bin_midpoints_um, dlog10_dp, dn_dlogdp

__all__ = [
    "AIR_NUMBER_DENSITY_COEFF",
    "AVOGADRO",
    "AVOGADRO_GAS_MODEL",
    "AVOGADRO_SEAM_RELATIVE_DIFFERENCE",
    "CM3_PER_M3",
    "G_PER_KG",
    "H2SO4_MOLAR_MASS_G_PER_MOL",
    "MBAR_TO_TORR",
    "MODEL_DEFAULT_ION_PAIR_RATE",
    "PAPER_ENSEMBLE_ION_PAIR_RATE",
    "PPMV_PER_MOLE_FRACTION",
    "PPTV_PER_MOLE_FRACTION",
    "SO2_MOLAR_MASS_G_PER_MOL",
    "air_number_density",
    "bin_midpoints_um",
    "dlog10_dp",
    "dn_dlogdp",
    "initial_mixing_ratio_pptv",
    "injected_number_density",
    "ion_pair_production_rate",
    "number_density_to_pptv",
    "plume_volume_cm3",
    "pptv_to_number_density",
]
