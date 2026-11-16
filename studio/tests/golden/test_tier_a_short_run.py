# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Tier A: one short real run against a committed fixture.

This is the tier that catches **drift in Studio's own pipeline** -- a changed derivation, a
reordered operator split, a dependency bump that moves the solver. It compares against a fixture
this project generated (``make_fixture.py``), not against the 2026-07 archive; reproducing the
archive is Tier B's job and costs ~4.6 min per case.

**Honest limitation, stated because it undercuts the plan's intent:** the plan calls Tier A "CI,
seconds", but CI does not check out the private submodules, so the model cannot run there. In CI
this module **skips**, and Tier A there is the pure schema/units/DAG/hash/expansion tests; locally
it runs. Fixing that means giving CI a deploy key for the submodules, which is its own change --
tracked rather than quietly ignored.

Cost when it does run: ~19 s for the run plus the comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from studio.tests.golden.make_fixture import (
    FIXTURE_PATH,
    STRIDE,
    TIER_A_BINS,
    TIER_A_DAYS,
    reduce_state,
    run_tier_a_case,
    tier_a_config,
)
from studio.tests.golden.tolerances import (
    EXACT_ARRAYS,
    RTOL_BIN_EDGES,
    RTOL_PHOTOLYSIS,
    RTOL_SERIES,
    RTOL_SIZE_DISTRIBUTION,
    assert_exact,
    assert_headline_matches,
    assert_series_matches,
)

#: Series compared sample-by-sample at the series tolerance.
_SERIES_KEYS = ("SA", "radius_cm", "h2so4wp", "particulate_S", "total_n")

#: Gas species compared at the headline tolerance. The sulfur chain plus its oxidant: what a result
#: is actually read for. Near-zero species are excluded by the floor in ``tolerances.py``.
_GAS_KEYS = ("SO2", "SO3", "H2SO4", "OH")


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    """The committed fixture, or skip if it has not been generated."""
    if not FIXTURE_PATH.is_file():
        pytest.skip(
            f"no Tier-A fixture at {FIXTURE_PATH}; generate it with "
            f"`python -m studio.tests.golden.make_fixture`"
        )
    with np.load(FIXTURE_PATH, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


@pytest.fixture(scope="module")
def fresh(repo_root: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Re-run the Tier-A case now and reduce it the same way. ~19 s."""
    if not (repo_root / "stratchem-jax" / "config.py").is_file():
        pytest.skip(
            "model submodules not checked out (`git submodule update --init`); the Tier-A golden "
            "run needs them. CI does not check them out, so this skips there -- see the module "
            "docstring."
        )
    out_dir = tmp_path_factory.mktemp("tier_a_golden")
    return reduce_state(run_tier_a_case(out_dir))


@pytest.mark.tier_a
def test_the_case_is_what_the_fixture_was_built_from(reference: dict[str, Any]) -> None:
    """Guard against comparing a re-run of one case against a fixture built from another.

    Cheap, and it fails clearly. Without it, changing ``TIER_A_DAYS`` would produce a shape mismatch
    deep inside a series comparison instead of saying "the fixture is stale".
    """
    config = tier_a_config().config
    assert config.schedule.duration_days == TIER_A_DAYS
    assert config.microphysics.n_bins == TIER_A_BINS
    assert len(reference["dp_mid_um"]) == TIER_A_BINS

    # the reduction rule, restated: every STRIDE-th sample plus the last one
    n_full = int(reference["n_times_full"])
    expected = sorted(set(range(0, n_full, STRIDE)) | {n_full - 1})
    assert list(reference["kept_indices"]) == expected


@pytest.mark.tier_a
def test_the_time_axis_is_bit_identical(fresh: dict[str, Any], reference: dict[str, Any]) -> None:
    """``t`` is built from the terminator schedule, not integrated. Any drift is a real bug."""
    assert_exact("t", fresh["t"], reference["t"])


@pytest.mark.tier_a
@pytest.mark.parametrize("key", EXACT_ARRAYS)
def test_analytic_arrays_are_bit_identical(
    key: str, fresh: dict[str, Any], reference: dict[str, Any]
) -> None:
    """``t``, ``V_ratio`` and ``T``: analytic, so exact equality is the right assertion."""
    assert_exact(key, fresh[key], reference[key])


@pytest.mark.tier_a
@pytest.mark.parametrize("species", _GAS_KEYS)
def test_gas_species_reproduce(
    species: str, fresh: dict[str, Any], reference: dict[str, Any]
) -> None:
    names = [str(name) for name in reference["species"]]
    index = names.index(species)  # BY NAME, never by position
    # Endpoints at the headline tolerance, the series at its own looser one -- two different rows of
    # REFERENCE_TOLERANCES.md. Conflating them is what broke Tier B's first real run.
    assert_headline_matches(species, fresh["x"][:, index], reference["x"][:, index])
    assert_series_matches(
        f"{species} (series)", fresh["x"][:, index], reference["x"][:, index], RTOL_SERIES
    )


@pytest.mark.tier_a
@pytest.mark.parametrize("key", _SERIES_KEYS)
def test_aerosol_series_reproduce(
    key: str, fresh: dict[str, Any], reference: dict[str, Any]
) -> None:
    assert_series_matches(key, fresh[key], reference[key], RTOL_SERIES)


@pytest.mark.tier_a
def test_the_size_distribution_reproduces(fresh: dict[str, Any], reference: dict[str, Any]) -> None:
    """Per bin, over the whole stored series -- not just the final spectrum."""
    assert_series_matches(
        "dNdlogDp", fresh["dNdlogDp"], reference["dNdlogDp"], RTOL_SIZE_DISTRIBUTION
    )
    assert_series_matches("n_cm3", fresh["n_cm3"], reference["n_cm3"], RTOL_SIZE_DISTRIBUTION)


@pytest.mark.tier_a
def test_photolysis_reproduces(fresh: dict[str, Any], reference: dict[str, Any]) -> None:
    """J is the most reproducible part of the pipeline (measured 1.09e-13), so drift here is signal.

    Compared per reaction, not summed: a compensating pair of errors across two reactions would
    survive a total and is exactly the kind of thing this tier exists to catch.
    """
    assert_exact("J_tmid", fresh["J_tmid"], reference["J_tmid"])
    equations = [str(name) for name in reference["J_equations"]]
    for index, equation in enumerate(equations):
        assert_series_matches(
            f"J[{equation}]", fresh["J"][:, index], reference["J"][:, index], RTOL_PHOTOLYSIS
        )


@pytest.mark.tier_a
def test_the_dry_bin_edges_reproduce(fresh: dict[str, Any], reference: dict[str, Any]) -> None:
    """Not exact: Studio and the archive spell the geometric mean differently (see tolerances.py).

    Within Studio's own pipeline they should agree bit-for-bit, so this passing at 1e-15 rather than
    exactly would itself be information -- the tolerance is the one measured against the archive.
    """
    assert_series_matches("dp_mid_um", fresh["dp_mid_um"], reference["dp_mid_um"], RTOL_BIN_EDGES)


@pytest.mark.tier_a
def test_the_run_is_still_physically_recognisable(fresh: dict[str, Any]) -> None:
    """A sanity floor under the tolerances: a comparison can only be meaningful if the run happened.

    All four assertions would hold for any correct run of this case, and none of them would hold for
    a run that silently did nothing -- which is the failure a tolerance-based test cannot see.
    """
    names = [str(name) for name in fresh["species"]]
    so2 = fresh["x"][:, names.index("SO2")]
    h2so4 = fresh["x"][:, names.index("H2SO4")]
    assert so2[-1] < so2[0], "SO2 must be consumed"
    assert h2so4.max() > 0.0, "H2SO4 must be produced"
    assert fresh["total_n"].max() > 0.0, "particles must form"
    assert fresh["V_ratio"][-1] > 1.0, "the plume must expand"
