# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Tier B: reproduce the archived 810-run ensemble, at the measured tolerance.

**Nightly or manual, never in CI.** Six 10-day cases at ~4.6 min each is ~28 minutes, and the
archive it compares against is gitignored, so a fresh clone does not have it. Run it explicitly:

    pytest studio/tests/golden -m tier_b

What it asserts is **reproduction of results produced in July 2026 by a different toolchain**,
which is a different claim from Tier A's "Studio has not drifted since its own fixture".
Bit-for-bit is already known to be false -- ~31 % of gas state elements differ -- so every tolerance
here comes from ``REFERENCE_TOLERANCES.md``, measured before it was asserted.

**A failure here is not automatically a regression.** A JAX or diffrax bump moves these numbers; the
correct response is to re-run ``measure_deviation.py``, update the record with the new SHAs, and
decide whether the new deviation is acceptable -- not to widen a constant until the test passes. The
assertion messages say so at the point of failure, where the temptation is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from studio.tests.golden.paper_cases import (
    ENSEMBLE_BINS,
    ENSEMBLE_DAYS,
    TIER_B_CASES,
    config_for_case,
)
from studio.tests.golden.tolerances import (
    EXACT_ARRAYS,
    RTOL_HEADLINE,
    RTOL_SERIES,
    RTOL_SIZE_DISTRIBUTION,
    assert_exact,
    endpoint_deviations,
    worst_relative_deviation,
)

#: Gas species compared at the headline tolerance, by name.
_GAS_KEYS = ("SO2", "SO3", "H2SO4", "OH")

#: Aerosol series compared at the series tolerance.
_SERIES_KEYS = ("SA", "radius_cm", "h2so4wp", "particulate_S", "total_n")


@pytest.fixture(scope="module")
def archive_root(repo_root: Path) -> Path:
    """The archived ensemble, or skip. Read-only: this is irreplaceable reference data."""
    runs = repo_root / "coupled" / "paper_ensemble" / "runs"
    if not runs.is_dir():
        pytest.skip(
            f"archived ensemble not present at {runs} (gitignored and regenerable); Tier B "
            f"reproduces it and cannot run without it"
        )
    if not (repo_root / "stratchem-jax" / "config.py").is_file():
        pytest.skip("model submodules not checked out (`git submodule update --init`)")
    return runs


@pytest.fixture(scope="module")
def reproduced(
    archive_root: Path, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, dict[str, Any]]:
    """Re-run every curated case once, and pair each with its archived counterpart.

    Module-scoped because each case costs ~4.6 min: running them once and sharing the arrays across
    assertions is the difference between 28 minutes and several hours.
    """
    from studio.modelio.execute import run_and_write

    out_root = tmp_path_factory.mktemp("tier_b_golden")
    paired: dict[str, dict[str, Any]] = {}
    for case_id in TIER_B_CASES:
        archived_path = archive_root / case_id / "state.npz"
        if not archived_path.is_file():
            continue  # a case absent from this machine's archive is a skip, not a failure
        fresh_path = run_and_write(config_for_case(case_id), out_root / case_id)["state"]
        with (
            np.load(fresh_path, allow_pickle=True) as fresh,
            np.load(archived_path, allow_pickle=True) as archived,
        ):
            paired[case_id] = {
                "fresh": {key: fresh[key] for key in fresh.files},
                "archived": {key: archived[key] for key in archived.files},
            }
    if not paired:
        pytest.skip(f"none of {list(TIER_B_CASES)} is present in {archive_root}")
    return paired


@pytest.mark.tier_b
def test_the_curated_set_covers_the_dilution_regimes_and_both_backgrounds() -> None:
    """ADR-009 asks for 4-6 cases across D1/D2/D3/burst x sabr220/sabr330. Assert the set, cheaply.

    Pure — it needs neither the model nor the archive, so a mis-curated set is caught in
    milliseconds rather than 28 minutes in.
    """
    assert 4 <= len(TIER_B_CASES) <= 6
    regimes = {config_for_case(case).config.dilution.regime.value for case in TIER_B_CASES}
    backgrounds = {config_for_case(case).config.background.aerosol.value for case in TIER_B_CASES}
    assert {"D1", "D2", "D3", "burst"} <= regimes
    assert {"sabre_220", "sabre_330"} <= backgrounds
    for case in TIER_B_CASES:
        config = config_for_case(case).config
        assert config.microphysics.n_bins == ENSEMBLE_BINS
        assert config.schedule.duration_days == ENSEMBLE_DAYS
        assert config.microphysics.coag_kernel_scale == 1.0, (
            "the curated set stays on cg1 until the coag_kernel_scale coverage gap recorded in "
            "REFERENCE_TOLERANCES.md is measured"
        )


@pytest.mark.tier_b
def test_every_case_reproduces(reproduced: dict[str, dict[str, Any]]) -> None:
    """One test over all cases, reporting every deviation before failing.

    Deliberately not parametrised per case: after 28 minutes of compute, "SO2 failed in case 3" is
    much less useful than the whole table. A per-case failure would also hide whether the deviation
    is systematic or specific to one regime, which is the first thing to want to know.
    """
    failures: list[str] = []
    #: every deviation, not only the ones that exceed: 28 minutes of compute should produce a
    #: measurement, not just a verdict. Printed below so a passing run still reports numbers.
    observed: dict[str, tuple[float, float]] = {}
    for case_id, pair in sorted(reproduced.items()):
        fresh, archived = pair["fresh"], pair["archived"]
        species = [str(name) for name in archived["species"]]

        for key in EXACT_ARRAYS:
            try:
                assert_exact(f"{case_id}/{key}", fresh[key], archived[key])
            except AssertionError as exc:
                failures.append(str(exc))

        # Two tolerances, from two different rows of the record: the endpoints a result is read
        # for (1e-12) and the series they come from (1e-10). Applying the endpoint number to a
        # whole series is the mistake this harness made on its first real run.
        for name in _GAS_KEYS:
            index = species.index(name)  # BY NAME
            series_fresh, series_archived = fresh["x"][:, index], archived["x"][:, index]
            for label, deviation in endpoint_deviations(series_fresh, series_archived).items():
                observed[f"{case_id}/{name} ({label})"] = (deviation, RTOL_HEADLINE)
                if deviation > RTOL_HEADLINE:
                    failures.append(
                        f"{case_id}/{name} {label}: {deviation:.3e} > {RTOL_HEADLINE:.0e}"
                    )
            worst = worst_relative_deviation(series_fresh, series_archived)
            observed[f"{case_id}/{name} (series)"] = (worst, RTOL_SERIES)
            if worst > RTOL_SERIES:
                failures.append(f"{case_id}/{name} series: {worst:.3e} > {RTOL_SERIES:.0e}")

        for key in _SERIES_KEYS:
            for label, deviation in endpoint_deviations(fresh[key], archived[key]).items():
                observed[f"{case_id}/{key} ({label})"] = (deviation, RTOL_HEADLINE)
                if deviation > RTOL_HEADLINE:
                    failures.append(
                        f"{case_id}/{key} {label}: {deviation:.3e} > {RTOL_HEADLINE:.0e}"
                    )
            worst = worst_relative_deviation(fresh[key], archived[key])
            observed[f"{case_id}/{key} (series)"] = (worst, RTOL_SERIES)
            if worst > RTOL_SERIES:
                failures.append(f"{case_id}/{key} series: {worst:.3e} > {RTOL_SERIES:.0e}")

        worst = worst_relative_deviation(fresh["dNdlogDp"], archived["dNdlogDp"])
        observed[f"{case_id}/dNdlogDp (per bin)"] = (worst, RTOL_SIZE_DISTRIBUTION)
        if worst > RTOL_SIZE_DISTRIBUTION:
            failures.append(f"{case_id}/dNdlogDp: {worst:.3e} > {RTOL_SIZE_DISTRIBUTION:.0e}")

    print(f"\n{'quantity':52s} {'worst rel':>11s}  tolerance")
    for label, (deviation, tolerance) in sorted(observed.items()):
        print(f"{label:52s} {deviation:11.3e}  {tolerance:.0e}")

    assert not failures, (
        "Tier-B reproduction deviates beyond the MEASURED tolerances:\n  "
        + "\n  ".join(failures)
        + "\n\nThese tolerances were measured, not chosen (see REFERENCE_TOLERANCES.md in this "
        "directory). A toolchain bump moves them: re-run measure_deviation.py, update that record "
        "with the new SHAs, and decide whether the new deviation is acceptable. Do not widen the "
        "constants to make this pass."
    )


@pytest.mark.tier_b
def test_the_runs_are_physically_recognisable(reproduced: dict[str, dict[str, Any]]) -> None:
    """The floor under the tolerances: a tolerance test cannot tell that a run did nothing."""
    for case_id, pair in sorted(reproduced.items()):
        fresh = pair["fresh"]
        species = [str(name) for name in fresh["species"]]
        so2 = fresh["x"][:, species.index("SO2")]
        assert so2[-1] < so2[0], f"{case_id}: SO2 must be consumed"
        assert fresh["x"][:, species.index("H2SO4")].max() > 0.0, f"{case_id}: H2SO4 must form"
        assert fresh["total_n"].max() > 0.0, f"{case_id}: particles must form"
        assert fresh["V_ratio"][-1] > 1.0, f"{case_id}: the plume must expand"
