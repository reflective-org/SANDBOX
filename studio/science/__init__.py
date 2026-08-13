# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Scientific derivations: plume volume, initial concentration, size-distribution reductions, GCR.

Purpose is consolidation as much as new code. Several of these derivations already exist in the
repository three to five times over, with drifting constants -- one implementation each, cited and
tested (task 0.5):

* V0 and injected mass -> initial concentration. Five copies today, with two different V0 values
  (``run_ensemble.py:41-46`` uses a 15 km track, ``run_dilution_d1_clean.py:61`` a 30 km one).
* dN/dlogDp and bin diameter edges. Four copies, two mid-point expressions.

Reused rather than rewritten -- these are already correct and tested:

* ``coupled.dilution.volume_ratio`` / ``kdil_from_regime`` (plume expansion V(t)/V0 and k_dil)
* ``config.air_number_density`` (stratchem-jax)
* ``coupled.aerosol_props`` (surface area, effective wet radius, H2SO4 weight percent)
* ``coupled.units`` (mass <-> number density; note the DELIBERATE ~0.036% Avogadro mismatch at the
  gas/TOMAS seam documented at ``coupled/units.py:18-22`` -- inherited, not silently "corrected")

Genuinely new: a galactic-cosmic-ray ion-pair parameterisation. Today ``ion_pair_rate`` is a bare
30.0 with no derivation. The default should not be zero and should depend on altitude, latitude and
solar-cycle phase -- but absent an agreed citation it raises ``NotImplementedError`` rather than
returning a plausible number (ADR-005).

No magic numbers: physical constants live in one module with sources, and any numeric literal in
scientific code needs a named constant and a citation.

This package must not import ``coupled``, the API, or the database (see ``studio/__init__.py``).
"""

from __future__ import annotations

__all__: list[str] = []
