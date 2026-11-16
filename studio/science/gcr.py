# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Galactic-cosmic-ray ion-pair production rate.

**This module deliberately does not compute anything.** It is the clearest case in the project of
the rule in ``studio/CLAUDE.md``: a number that is needed, that has no agreed source, and that would
be trivially easy to fabricate convincingly.

What exists today is a bare ``30.0`` cm^-3 s^-1 in the paper ensemble (``run_ensemble.py:102``),
described in ``TABLE_microphysics_parameters.md`` as "galactic cosmic rays at ~20 km" with no
citation; and the model's own default of ``0.0``, which switches off ion-induced nucleation
altogether (``coupled/coupled_scenario.py:117``). So the two available values are an uncited
constant and a value that disables a physical process.

The rate genuinely varies -- roughly a factor of two over the solar cycle, and strongly with
latitude (geomagnetic cutoff) and altitude. A function of those three arguments is the right shape.
Inventing its coefficients is not, because the result would arrive with the same confidence as a
computed one and feed the most sensitive part of the system: the ion-induced channels of Dunne et
al. (2016) nucleation.

So :func:`ion_pair_production_rate` raises, and :data:`PAPER_ENSEMBLE_ION_PAIR_RATE` is available
for anyone who wants the ensemble's constant *as a constant*, with its provenance attached.

Tracked as SCIENCE-6, issue #63.
"""

from __future__ import annotations

from typing import Final

#: The paper ensemble's fixed value [ion pairs cm^-3 s^-1] (``run_ensemble.py:102``;
#: ``TABLE_microphysics_parameters.md``, "30 ion pairs cm^-3 s^-1, galactic cosmic rays at ~20 km").
#: Uncited. Use it to reproduce the ensemble, not as a general-purpose value -- it is a single
#: number for a quantity that varies with altitude, latitude and solar-cycle phase.
PAPER_ENSEMBLE_ION_PAIR_RATE: Final = 30.0

#: The MODEL's default, which disables the ion-induced nucleation channels entirely
#: (``coupled/coupled_scenario.py:117``). Recorded so that "the default is 0" is discoverable here
#: rather than surprising someone whose nucleation quietly lost a channel.
MODEL_DEFAULT_ION_PAIR_RATE: Final = 0.0

_SCIENCE_6 = (
    "SCIENCE-6 (issue #63): the GCR ion-pair production rate has no agreed parameterisation. "
    "The paper ensemble uses a fixed, uncited 30.0 cm^-3 s^-1 at ~20 km and the model defaults to "
    "0.0, which disables ion-induced nucleation entirely."
)


def ion_pair_production_rate(
    altitude_km: float, latitude_deg: float, solar_cycle_phase: float
) -> float:
    """Ion-pair production rate [cm^-3 s^-1]. **Not implemented, by decision.**

    Args:
        altitude_km: Box altitude.
        latitude_deg: Geographic latitude; the geomagnetic cutoff rigidity, and therefore the
            ionisation rate, is a strong function of it.
        solar_cycle_phase: Phase in [0, 1], 0 at solar minimum (maximum GCR flux).

    Raises:
        NotImplementedError: Always. Absent a citable parameterisation, returning a plausible number
            would be worse than failing: it would look computed. Use
            :data:`PAPER_ENSEMBLE_ION_PAIR_RATE` explicitly if you want the ensemble's constant.
    """
    raise NotImplementedError(
        f"ion_pair_production_rate(altitude_km={altitude_km}, latitude_deg={latitude_deg}, "
        f"solar_cycle_phase={solar_cycle_phase}) is not implemented. {_SCIENCE_6} Set "
        f"microphysics.ion_pair_rate explicitly -- studio.science.gcr."
        f"PAPER_ENSEMBLE_ION_PAIR_RATE is that constant with its provenance attached."
    )


__all__ = [
    "MODEL_DEFAULT_ION_PAIR_RATE",
    "PAPER_ENSEMBLE_ION_PAIR_RATE",
    "ion_pair_production_rate",
]
