# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Build the Tier-A golden fixture: a short run, reduced on a UNIFORM stride.

    python -m studio.tests.golden.make_fixture            # writes tier_a_short_run.npz
    python -m studio.tests.golden.make_fixture --check    # reports drift, writes nothing

**Uniform stride, not an adaptive or thinned grid.** A coarsening grid aliases the morning
particle-number spike by up to 8x (ADR-009, and ``CAVEATS.md``), so a fixture built on one would
encode the aliasing and then assert it forever. Every ``STRIDE``-th stored sample, and the last one
so the endpoint is always present.

The fixture is committed because it is small (tens of kB) and because Tier A must not depend on the
3.6 MB archive, which is gitignored and absent on a fresh clone. Regenerating it is a deliberate act
with a recorded reason -- ``--check`` exists so drift can be measured without overwriting the
reference by accident.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from studio.resolve import resolve
from studio.schema import RunConfig
from studio.tests.golden.tolerances import (
    RTOL_HEADLINE,
    RTOL_SERIES,
    worst_relative_deviation,
)

#: Where the fixture lives. Next to the tests that read it.
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "tier_a_short_run.npz"

#: 1 day, 40 bins: the cheapest run that still exercises the whole pipeline -- gas chemistry, TUV-x
#: photolysis, all three microphysics processes and dilution. ~21 s on an M-series CPU.
TIER_A_DAYS = 1
TIER_A_BINS = 40

#: Keep every 4th sample. 1 day at 600 s nominal is ~147 stored samples, so this is 38 -- enough
#: for a series comparison to be meaningful, small enough to commit.
STRIDE = 4

#: Arrays reduced along time. Everything a Tier-A assertion compares.
_TIME_SERIES_KEYS = (
    "t",
    "x",
    "SA",
    "radius_cm",
    "h2so4wp",
    "particulate_S",
    "T",
    "V_ratio",
    "total_n",
    "n_cm3",
    "dNdlogDp",
)

#: Arrays that are not time series and are stored whole.
_WHOLE_KEYS = ("species", "M", "dp_mid_um", "J_equations")

#: Photolysis is on the INTERVAL axis (n_times - 1), not the sample axis, so it strides separately.
#: Included because ``REFERENCE_TOLERANCES.md`` measured it as the most reproducible part of the
#: pipeline (1.09e-13), which makes any drift there a strong signal rather than noise.
_INTERVAL_SERIES_KEYS = ("J_tmid", "J")


def tier_a_config() -> Any:
    """The Tier-A case: the golden case's defaults, shortened and coarsened."""
    payload = RunConfig().model_dump()
    payload["schedule"]["duration_days"] = TIER_A_DAYS
    payload["microphysics"]["n_bins"] = TIER_A_BINS
    return resolve(RunConfig.model_validate(payload))


def run_tier_a_case(out_dir: Path) -> Path:
    """Run the case and return the path to its ``state.npz``."""
    from studio.modelio.execute import run_and_write

    return run_and_write(tier_a_config(), out_dir)["state"]


def reduce_state(npz_path: Path) -> dict[str, np.ndarray]:
    """Uniform-stride reduction of a ``state.npz``, keeping the final sample."""
    with np.load(npz_path, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    n_times = len(data["t"])
    keep = sorted(set(range(0, n_times, STRIDE)) | {n_times - 1})
    reduced: dict[str, np.ndarray] = {"kept_indices": np.asarray(keep), "n_times_full": n_times}
    for key in _TIME_SERIES_KEYS:
        reduced[key] = np.asarray(data[key])[keep]
    n_intervals = len(data["J_tmid"])
    keep_intervals = sorted(set(range(0, n_intervals, STRIDE)) | {n_intervals - 1})
    reduced["kept_intervals"] = np.asarray(keep_intervals)
    for key in _INTERVAL_SERIES_KEYS:
        reduced[key] = np.asarray(data[key])[keep_intervals]
    for key in _WHOLE_KEYS:
        reduced[key] = np.asarray(data[key])
    return reduced


def write_fixture(out_dir: Path) -> Path:
    """Run the case, reduce it, write the fixture."""
    state = run_tier_a_case(out_dir)
    reduced = reduce_state(state)
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIXTURE_PATH, **reduced)
    size_kb = FIXTURE_PATH.stat().st_size / 1024
    print(
        f"wrote {FIXTURE_PATH} ({size_kb:.0f} kB): {len(reduced['kept_indices'])} of "
        f"{reduced['n_times_full']} samples, stride {STRIDE}"
    )
    return FIXTURE_PATH


def check_against_fixture(out_dir: Path) -> int:
    """Re-run and report drift against the committed fixture. Writes nothing."""
    if not FIXTURE_PATH.is_file():
        print(f"no fixture at {FIXTURE_PATH}; run without --check first", file=sys.stderr)
        return 2
    fresh = reduce_state(run_tier_a_case(out_dir))
    with np.load(FIXTURE_PATH, allow_pickle=True) as archive:
        reference = {key: archive[key] for key in archive.files}
    species = [str(name) for name in reference["species"]]
    print(f"{'quantity':22s} {'worst rel':>12s}  tolerance")
    for name, rtol in (("SO2", RTOL_HEADLINE), ("H2SO4", RTOL_HEADLINE)):
        index = species.index(name)
        worst = worst_relative_deviation(fresh["x"][:, index], reference["x"][:, index])
        print(f"{name:22s} {worst:12.3e}  {rtol:.0e}")
    for key in ("total_n", "SA", "particulate_S", "dNdlogDp"):
        worst = worst_relative_deviation(fresh[key], reference[key])
        print(f"{key:22s} {worst:12.3e}  {RTOL_SERIES:.0e}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="re-run and report drift; do not overwrite the fixture"
    )
    parser.add_argument(
        "--work-dir", type=Path, default=None, help="where to run (default: a tempdir)"
    )
    args = parser.parse_args()

    import tempfile

    with tempfile.TemporaryDirectory(prefix="studio-golden-") as tmp:
        out_dir = args.work_dir or Path(tmp)
        return check_against_fixture(out_dir) if args.check else (write_fixture(out_dir) and 0)


if __name__ == "__main__":
    raise SystemExit(main())
