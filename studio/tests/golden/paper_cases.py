# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Case ID -> ``RunConfig``, for the Tier-B archive comparison.

The 810-run ensemble names each case by its axis levels joined with ``__`` --
``30N_20km__sabr220__D2med__a1p0__nuc1__cg1`` -- and that name IS the parameter set
(``run_ensemble.py:84``). This module reverses it, so a Tier-B test can say "reproduce this archived
directory" and get the config that produced it.

Lives under ``studio/tests/`` rather than in ``studio/``: task 0.2 deliberately kept the paper
ensemble's axes out of the package, on the grounds that they would earn a home there when something
needed them. Tier B needs them *as test data*, which is not the same as needing a preset library, so
they stay here until production asks.

Token tables are copied from ``run_ensemble.py:61-76`` and checked against it by
``test_tier_b_archive.py`` -- a token that stops matching is a case that would silently be run with
the wrong parameters and compared against the right archive.
"""

from __future__ import annotations

from typing import Final

from studio.resolve import ResolvedConfig, resolve
from studio.schema import BackgroundAerosol, DilutionRegime, RunConfig

#: site token -> (latitude, T [K], p [mbar], H2O [ppmv]). ``run_ensemble.py:61-65``.
SITES: Final[dict[str, tuple[float, float, float, float]]] = {
    "30N_20km": (30.0, 210.0, 55.0, 6.9104),
    "60N_15km": (60.0, 210.0, 120.0, 3.1673),
    "30N_20km_213K": (30.0, 213.0, 55.0, 10.1834),
}

#: background token -> (aerosol distribution, background SO2 [pptv]). ``run_ensemble.py:67-71``.
BACKGROUNDS: Final[dict[str, tuple[BackgroundAerosol, float]]] = {
    "sabr330": (BackgroundAerosol.SABR_330, 20.0),
    "sabr220": (BackgroundAerosol.SABR_220, 20.0),
    "cesm": (BackgroundAerosol.CESM_G6, 100.0),
}

#: dilution token -> regime. ``run_ensemble.py:72-73``.
DILUTIONS: Final[dict[str, DilutionRegime]] = {
    "D1low": DilutionRegime.D1,
    "D2med": DilutionRegime.D2,
    "D3high": DilutionRegime.D3,
    "burst": DilutionRegime.BURST,
    "D5vhigh": DilutionRegime.D5,
}

#: condensation alpha, nucleation scale, coagulation scale. ``run_ensemble.py:74-76``.
STICKING: Final[dict[str, float]] = {"a0p5": 0.5, "a1p0": 1.0}
NUCLEATION: Final[dict[str, float]] = {"nuc0p01": 0.01, "nuc1": 1.0, "nuc100": 100.0}
COAGULATION: Final[dict[str, float]] = {"cg0p5": 0.5, "cg1": 1.0, "cg2": 2.0}

#: The ensemble's fixed values, which are the schema's defaults: 80 bins, day 172, 00:00, 10 days,
#: ion pair rate 30, SO2+HO2 1e-18, aerosol->J and heating off. Asserted, not assumed, by
#: ``test_tier_b_archive.py``.
ENSEMBLE_BINS: Final = 80
ENSEMBLE_DAYS: Final = 10


def config_for_case(case_id: str) -> ResolvedConfig:
    """The resolved config that produced the archived directory ``case_id``.

    Raises:
        ValueError: On an unknown token or the wrong number of them. A mistyped case would otherwise
            be run with default parameters and compared against a real archive, which fails in a way
            that looks like a physics regression.
    """
    tokens = case_id.split("__")
    if len(tokens) != 6:
        raise ValueError(
            f"case id {case_id!r} has {len(tokens)} tokens, expected 6: "
            f"site__background__dilution__sticking__nucleation__coag"
        )
    site, background, dilution, sticking, nucleation, coagulation = tokens
    for token, table, what in (
        (site, SITES, "site"),
        (background, BACKGROUNDS, "background"),
        (dilution, DILUTIONS, "dilution"),
        (sticking, STICKING, "sticking"),
        (nucleation, NUCLEATION, "nucleation"),
        (coagulation, COAGULATION, "coagulation"),
    ):
        if token not in table:
            raise ValueError(f"unknown {what} token {token!r}; known: {sorted(table)}")

    latitude, temperature, pressure, h2o = SITES[site]
    aerosol, background_so2 = BACKGROUNDS[background]
    payload = RunConfig().model_dump()
    payload["site"].update(
        latitude_deg=latitude,
        temperature_k=temperature,
        pressure_mbar=pressure,
        h2o_ppmv=h2o,
    )
    payload["background"].update(aerosol=aerosol.value, so2_pptv=background_so2)
    payload["dilution"]["regime"] = DILUTIONS[dilution].value
    payload["microphysics"].update(
        condensation_alpha=STICKING[sticking],
        nucleation_rate_scale=NUCLEATION[nucleation],
        coag_kernel_scale=COAGULATION[coagulation],
        n_bins=ENSEMBLE_BINS,
    )
    payload["schedule"]["duration_days"] = ENSEMBLE_DAYS
    return resolve(RunConfig.model_validate(payload))


#: The curated Tier-B set: D1/D2/D3/burst against the clean and loaded backgrounds (ADR-009 asks for
#: 4-6). All are ``cg1`` deliberately -- ``REFERENCE_TOLERANCES.md`` records that cg0p5/cg2 may
#: straddle the tomas-jax commit that wired ``coag_kernel_scale`` through, so adopting one as a
#: golden case needs its own measurement first.
TIER_B_CASES: Final[tuple[str, ...]] = (
    "30N_20km__sabr220__D1low__a1p0__nuc1__cg1",
    "30N_20km__sabr220__D2med__a1p0__nuc1__cg1",
    "30N_20km__sabr220__D3high__a1p0__nuc1__cg1",
    "30N_20km__sabr220__burst__a1p0__nuc1__cg1",
    "30N_20km__sabr330__D2med__a1p0__nuc1__cg1",
    "30N_20km__sabr330__burst__a1p0__nuc1__cg1",
)

__all__ = [
    "BACKGROUNDS",
    "COAGULATION",
    "DILUTIONS",
    "ENSEMBLE_BINS",
    "ENSEMBLE_DAYS",
    "NUCLEATION",
    "SITES",
    "STICKING",
    "TIER_B_CASES",
    "config_for_case",
]
