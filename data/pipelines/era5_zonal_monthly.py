# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""ERA5 -> the committed zonal-mean monthly climatology (task 1.1, SCIENCE-1, BLOCKING-5).

Produces ``studio/science/data/era5_zonal_monthly_v1.npz`` and its manifest. The product is
(month x level x latitude) for temperature, specific humidity and geopotential height -- a few MB,
committed to the repository with a checksum, so no run ever fetches anything (ADR-006: the raw ERA5
fields are NOT redistributed; the Copernicus licence permits derived products with attribution).

Conventions, each a decision recorded in ASSUMPTIONS.md (ASSUMPTION-9):

* **Years 1991-2020** -- the WMO standard climate normal, so "the June climatology" cites a
  standard rather than a habit.
* **Zonal mean, monthly mean** -- SCIENCE-1's answer (issue #53), decided 2026-08-18.
* **2.5-degree grid** at retrieval. The zonal mean over 144 longitudes is insensitive to this, and
  it keeps the download ~10^2 MB instead of gigabytes. Latitude resolution of the product is 2.5
  degrees; the lookup interpolates.
* **15 pressure levels, 300-5 hPa** -- the lower-to-middle stratosphere the box lives in. The
  lookup refuses to extrapolate outside this range.

Usage (this script does NOT run in Studio's locked venv -- it needs ``cdsapi``; make a scratch env):

    python -m venv /tmp/cds && /tmp/cds/bin/pip install cdsapi netCDF4 numpy
    /tmp/cds/bin/python data/pipelines/era5_zonal_monthly.py fetch  /path/to/raw/
    /tmp/cds/bin/python data/pipelines/era5_zonal_monthly.py reduce /path/to/raw/

``fetch`` needs ``~/.cdsapirc``. ``reduce`` writes the product and prints the sha256 that goes into
the manifest; commit both files together.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

#: Pressure levels [hPa], descending altitude. The box's native coordinate is pressure (ADR-003).
LEVELS_HPA = [5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300]

#: WMO standard climate normal.
YEARS = list(range(1991, 2021))

VARIABLES = {"temperature": "t", "specific_humidity": "q", "geopotential": "z"}

#: Identifier recorded in every run's provenance when the ERA5 dataset is selected.
DATASET_ID = "era5_zonal_monthly_v1"

#: Standard gravity [m s^-2], to turn geopotential into geopotential height (WMO value).
STANDARD_GRAVITY = 9.80665

PRODUCT = Path(__file__).resolve().parents[2] / "studio" / "science" / "data"


def fetch(raw_dir: Path) -> None:
    """One CDS request per variable: each queues independently, and one failure loses one file."""
    import cdsapi

    client = cdsapi.Client(quiet=True)
    for variable in VARIABLES:
        target = raw_dir / f"era5_{variable}.nc"
        print(f"requesting {variable} -> {target}")
        client.retrieve(
            "reanalysis-era5-pressure-levels-monthly-means",
            {
                "product_type": ["monthly_averaged_reanalysis"],
                "variable": [variable],
                "pressure_level": [str(level) for level in LEVELS_HPA],
                "year": [str(year) for year in YEARS],
                "month": [f"{month:02d}" for month in range(1, 13)],
                "time": ["00:00"],
                "data_format": "netcdf",
                "grid": [2.5, 2.5],
            },
            str(target),
        )


def reduce(raw_dir: Path) -> None:
    """Raw monthly fields -> (month x level x lat) climatology, zonal mean then multi-year mean."""
    import netCDF4
    import numpy as np

    arrays: dict[str, "np.ndarray"] = {}
    latitude = level = None
    for variable, short in VARIABLES.items():
        with netCDF4.Dataset(raw_dir / f"era5_{variable}.nc") as ds:
            data = np.asarray(ds[short][:], dtype=np.float64)  # (time, level, lat, lon)
            times = netCDF4.num2date(ds["valid_time"][:], ds["valid_time"].units)
            months = np.asarray([t.month for t in times])
            years = np.asarray([t.year for t in times])
            latitude = np.asarray(ds["latitude"][:], dtype=np.float64)
            level = np.asarray(ds["pressure_level"][:], dtype=np.float64)
        expected = len(YEARS) * 12
        if data.shape[0] != expected:
            raise SystemExit(f"{variable}: {data.shape[0]} months, expected {expected} -- refetch")
        if sorted(set(years)) != YEARS:
            raise SystemExit(f"{variable}: years {min(years)}-{max(years)} != {YEARS[0]}-{YEARS[-1]}")
        zonal = data.mean(axis=3)  # over longitude
        monthly = np.stack([zonal[months == m].mean(axis=0) for m in range(1, 13)])
        arrays[short] = monthly.astype(np.float32)  # (12, level, lat)

    assert latitude is not None and level is not None
    # BOTH axes ascending, so the lookup's searchsorted works without a flip at read time. CDS
    # delivers latitude 90..-90 and pressure levels 300..5; trusting either order as-is is how the
    # first cut of this pipeline put the 300 hPa row under every lookup -- caught by the
    # stratosphere-shaped physics tests (T(55 hPa) came back 240 K, a tropospheric number).
    lat_order = np.argsort(latitude)
    latitude = latitude[lat_order]
    level_order = np.argsort(level)
    level = level[level_order]
    for short in arrays:
        arrays[short] = arrays[short][:, level_order][:, :, lat_order]

    PRODUCT.mkdir(parents=True, exist_ok=True)
    out = PRODUCT / f"{DATASET_ID}.npz"
    np.savez_compressed(
        out,
        latitude_deg=latitude.astype(np.float32),
        level_hpa=level.astype(np.float32),
        month=np.arange(1, 13, dtype=np.int16),
        temperature_k=arrays["t"],
        specific_humidity_kg_kg=arrays["q"],
        geopotential_height_m=(arrays["z"] / STANDARD_GRAVITY).astype(np.float32),
    )
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    manifest = {
        "dataset_id": DATASET_ID,
        "source": "ERA5 monthly averaged reanalysis on pressure levels (Copernicus CDS)",
        "doi": "10.24381/cds.6860a573",
        "licence": (
            "Contains modified Copernicus Climate Change Service information (1991-2020). "
            "Derived product; raw ERA5 fields are not redistributed."
        ),
        "convention": "zonal mean, then 1991-2020 mean, per calendar month (SCIENCE-1, issue #53)",
        "years": [YEARS[0], YEARS[-1]],
        "grid_deg": 2.5,
        "levels_hpa": LEVELS_HPA,
        "created": datetime.date.today().isoformat(),
        "sha256": digest,
    }
    (PRODUCT / f"{DATASET_ID}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out} ({size_mb:.1f} MB), sha256={digest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fetch", "reduce"])
    parser.add_argument("raw_dir", type=Path)
    args = parser.parse_args()
    (fetch if args.command == "fetch" else reduce)(args.raw_dir)
