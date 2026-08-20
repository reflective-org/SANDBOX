# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Reading the committed ERA5 zonal-mean monthly climatology (SCIENCE-1).

The product is built by ``data/pipelines/era5_zonal_monthly.py`` and committed at
``studio/science/data/era5_zonal_monthly_v1.npz`` with a manifest carrying its sha256. This module
is the ONLY reader: it verifies the checksum once per process, interpolates, and refuses to
extrapolate. No fetching happens here or anywhere at run time (ADR-006).

Interpolation choices, stated because they are conventions:

* **Linear in latitude** between the product's 2.5-degree rows. Zonal-mean fields are smooth at
  this scale; the residual against the 0.25-degree native grid is far below interannual spread.
* **Linear in log-pressure** between levels, because temperature is closer to linear in log-p than
  in p through the stratosphere (pressure falls exponentially with height).
* **No extrapolation.** Outside 300-5 hPa the derivation raises rather than guessing (ADR-005): a
  plausible temperature for 400 hPa from a stratospheric product is exactly the fabricated number
  this project forbids.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np
import numpy.typing as npt

#: Molar masses [g mol^-1]: dry air (US Standard Atmosphere 1976), water (CODATA).
MOLAR_MASS_DRY_AIR_G_PER_MOL: Final = 28.9644
MOLAR_MASS_WATER_G_PER_MOL: Final = 18.0153

#: The committed product this module reads.
DATASET_ID: Final = "era5_zonal_monthly_v1"

_DATA_DIR = Path(__file__).resolve().parent / "data"


class Climatology(NamedTuple):
    """The product in memory. Arrays are (month, level, latitude); axes ascend except level."""

    latitude_deg: npt.NDArray[np.floating]
    level_hpa: npt.NDArray[np.floating]
    temperature_k: npt.NDArray[np.floating]
    specific_humidity_kg_kg: npt.NDArray[np.floating]
    geopotential_height_m: npt.NDArray[np.floating]
    dataset_id: str
    sha256: str


class ClimatologyUnavailableError(FileNotFoundError):
    """The committed product is missing -- regenerate it, never substitute for it."""


@lru_cache(maxsize=1)
def load() -> Climatology:
    """Load and checksum-verify the product, once per process.

    Raises:
        ClimatologyUnavailableError: If the product or its manifest is absent. The message carries
            the regeneration commands, because the reader who hits this is holding a fresh clone.
        ValueError: If the file does not match the manifest's sha256 -- an edited or corrupted
            product must not silently feed runs (ADR-006).
    """
    npz_path = _DATA_DIR / f"{DATASET_ID}.npz"
    manifest_path = _DATA_DIR / f"{DATASET_ID}.json"
    if not npz_path.is_file() or not manifest_path.is_file():
        raise ClimatologyUnavailableError(
            f"the ERA5 climatology product is not present at {npz_path}. It is committed to the "
            f"repository, so this means an incomplete checkout or a build that stripped data "
            f"files. To regenerate: data/pipelines/era5_zonal_monthly.py (fetch, then reduce; "
            f"needs ~/.cdsapirc). Never substitute a fabricated profile."
        )
    manifest = json.loads(manifest_path.read_text())
    digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    if digest != manifest["sha256"]:
        raise ValueError(
            f"{npz_path.name} does not match its manifest: sha256 {digest} != "
            f"{manifest['sha256']}. The product or the manifest was modified; regenerate both "
            f"together with data/pipelines/era5_zonal_monthly.py."
        )
    with np.load(npz_path) as data:
        # Refuse, not fix: the interpolation's searchsorted requires ascending axes, and the first
        # cut of the pipeline shipped descending levels. A product violating this is regenerated,
        # never silently reordered here -- the checksum pins WHAT was read, so the reader must not
        # change what it means.
        for axis in ("latitude_deg", "level_hpa"):
            values = data[axis]
            if not np.all(np.diff(values) > 0):
                raise ValueError(
                    f"{npz_path.name}: {axis} is not strictly ascending; the product predates the "
                    f"axis-order fix -- regenerate it with data/pipelines/era5_zonal_monthly.py"
                )
        return Climatology(
            latitude_deg=data["latitude_deg"].astype(np.float64),
            level_hpa=data["level_hpa"].astype(np.float64),
            temperature_k=data["temperature_k"].astype(np.float64),
            specific_humidity_kg_kg=data["specific_humidity_kg_kg"].astype(np.float64),
            geopotential_height_m=data["geopotential_height_m"].astype(np.float64),
            dataset_id=str(manifest["dataset_id"]),
            sha256=digest,
        )


def _interpolate(
    field: npt.NDArray[np.floating],
    *,
    month: int,
    latitude_deg: float,
    pressure_mbar: float,
    climatology: Climatology,
) -> float:
    """Bilinear in (latitude, log-pressure) at an exact month. Refuses to extrapolate."""
    if not 1 <= month <= 12:
        raise ValueError(f"month must be 1-12, got {month}")
    lats = climatology.latitude_deg
    if not lats[0] <= latitude_deg <= lats[-1]:
        raise ValueError(f"latitude {latitude_deg} outside the product's {lats[0]}..{lats[-1]}")
    levels = climatology.level_hpa  # strictly ascending; enforced at load
    p_lo, p_hi = float(levels.min()), float(levels.max())
    if not p_lo <= pressure_mbar <= p_hi:
        raise ValueError(
            f"pressure {pressure_mbar} mbar is outside the climatology's {p_lo:g}-{p_hi:g} hPa. "
            f"The product covers the lower-to-middle stratosphere; extrapolating a stratospheric "
            f"profile would fabricate a value (ADR-005)."
        )
    plane = field[month - 1]  # (level, lat)

    # latitude bracket
    j = int(np.searchsorted(lats, latitude_deg))
    j0, j1 = max(j - 1, 0), min(j, len(lats) - 1)
    w_lat = 0.0 if j0 == j1 else (latitude_deg - lats[j0]) / (lats[j1] - lats[j0])

    # log-pressure bracket (ascending, enforced at load)
    logp = np.log(levels)
    target = float(np.log(pressure_mbar))
    i = int(np.searchsorted(logp, target))
    i0, i1 = max(i - 1, 0), min(i, len(levels) - 1)
    w_p = 0.0 if i0 == i1 else (target - logp[i0]) / (logp[i1] - logp[i0])

    v00, v01 = plane[i0, j0], plane[i0, j1]
    v10, v11 = plane[i1, j0], plane[i1, j1]
    return float(
        (1 - w_p) * ((1 - w_lat) * v00 + w_lat * v01) + w_p * ((1 - w_lat) * v10 + w_lat * v11)
    )


def temperature_k(*, month: int, latitude_deg: float, pressure_mbar: float) -> float:
    """Climatological temperature at (month, latitude, pressure)."""
    c = load()
    return _interpolate(
        c.temperature_k,
        month=month,
        latitude_deg=latitude_deg,
        pressure_mbar=pressure_mbar,
        climatology=c,
    )


def specific_humidity_to_ppmv(q_kg_kg: float) -> float:
    """Specific humidity (kg water / kg moist air) -> volume mixing ratio [ppmv].

    Exact mole-fraction form, not the dilute approximation: x = (q/Mw) / (q/Mw + (1-q)/Md).
    At stratospheric q ~ 3e-6 the two differ negligibly, but the exact form costs nothing and
    removes an approximation nobody needs to remember.
    """
    if q_kg_kg < 0.0:
        raise ValueError(f"specific humidity cannot be negative, got {q_kg_kg}")
    moles_water = q_kg_kg / MOLAR_MASS_WATER_G_PER_MOL
    moles_dry = (1.0 - q_kg_kg) / MOLAR_MASS_DRY_AIR_G_PER_MOL
    return moles_water / (moles_water + moles_dry) * 1e6


def h2o_ppmv(*, month: int, latitude_deg: float, pressure_mbar: float) -> float:
    """Climatological water vapour at (month, latitude, pressure), in the model's ppmv.

    Carries the same caveat as the schema field: reanalysis stratospheric water vapour is biased
    dry (BLOCKING-5); MLS as a dedicated H2O source is issue #94.
    """
    c = load()
    q = _interpolate(
        c.specific_humidity_kg_kg,
        month=month,
        latitude_deg=latitude_deg,
        pressure_mbar=pressure_mbar,
        climatology=c,
    )
    return specific_humidity_to_ppmv(q)


def geopotential_height_m(*, month: int, latitude_deg: float, pressure_mbar: float) -> float:
    """Climatological geopotential height at (month, latitude, pressure).

    The product's own z field (ERA5 geopotential / g0), so pressure-to-altitude is the atmosphere's
    actual relation at this latitude and month rather than a standard-atmosphere approximation --
    the 55 hPa surface is ~500 m higher in the tropics than at the pole, and this carries that.
    """
    c = load()
    return _interpolate(
        c.geopotential_height_m,
        month=month,
        latitude_deg=latitude_deg,
        pressure_mbar=pressure_mbar,
        climatology=c,
    )


def profile(*, month: int, latitude_deg: float) -> dict[str, list[float]]:
    """T and H2O against pressure at (month, latitude) -- the stage-1 preview panel's data."""
    c = load()
    out_t, out_q, out_z = [], [], []
    for level in c.level_hpa:
        pressure = float(level)
        out_t.append(temperature_k(month=month, latitude_deg=latitude_deg, pressure_mbar=pressure))
        out_q.append(h2o_ppmv(month=month, latitude_deg=latitude_deg, pressure_mbar=pressure))
        out_z.append(
            geopotential_height_m(month=month, latitude_deg=latitude_deg, pressure_mbar=pressure)
        )
    return {
        "level_hpa": [float(level) for level in c.level_hpa],
        "temperature_k": out_t,
        "h2o_ppmv": out_q,
        "geopotential_height_m": out_z,
    }


__all__ = [
    "DATASET_ID",
    "MOLAR_MASS_DRY_AIR_G_PER_MOL",
    "MOLAR_MASS_WATER_G_PER_MOL",
    "Climatology",
    "ClimatologyUnavailableError",
    "geopotential_height_m",
    "h2o_ppmv",
    "load",
    "profile",
    "specific_humidity_to_ppmv",
    "temperature_k",
]
