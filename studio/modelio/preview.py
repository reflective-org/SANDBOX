# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Cheap previews of what a configuration implies, for the wizard's per-stage panels.

Spec section 8 asks for *"interactive preview panels (dilution curve, size distribution builder,
SZA/OH diurnal)"* that *"must respond in well under a second"* and *"never invoke the full model"*.

**Every curve here is the model's own implementation, not a lookalike.** That is the whole design
constraint: a preview that plots a re-derived formula would eventually disagree with what the run
does, and the disagreement would be invisible -- the picture would still look like a dilution curve.
So the dilution curve is ``coupled.dilution.volume_ratio``, the size distribution is the state
``coupled.tomas_bridge.initial_tomas_state`` actually seeds, and the bin grid is TOMAS's own. This
module converts and shapes for display; it computes no physics of its own.

Measured on this machine (after the one-off import below):

    dilution curve, 300 points     0.1 ms
    size distribution, 80 bins     1.2 ms   (46 ms on the very first call)
    SZA, 145 points                0.2 ms
    bin grid                       0.4 ms

**The cost that matters is the import**, not the arithmetic: ``import coupled`` pulls JAX and takes
~1.2 s. It happens once per process, lazily, inside the functions below -- so a wizard that never
opens a panel never pays it, and ``/api/config/resolve`` stays free of it (the import-boundary tests
assert exactly that).

A unit trap this module exists to keep on one side of the seam: ``stp_to_ambient_factor`` takes
pressure in **Pa**, while the schema's canonical unit is mbar (ADR-003). Passing 55 where 5500 was
wanted silently scales the whole distribution by 100, and the plot still looks entirely plausible.
Going through ``to_scenario`` is what avoids that, which is why the size-distribution preview builds
a real scenario rather than calling the seeding helper directly.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

import numpy as np

from studio.resolve import resolve
from studio.schema import RunConfig
from studio.science.size_distribution import bin_midpoints_um, dn_dlogdp

#: Points in a preview curve. Enough that a log-x dilution curve is smooth over ten days, small
#: enough that the JSON is a few kB.
_CURVE_POINTS = 300

#: Points in the diurnal SZA curve: one every ten minutes, so sunrise lands within ten minutes.
_DIURNAL_POINTS = 145


def _repo_root() -> str:
    """The SANDBOX checkout root (this file is ``<root>/studio/modelio/preview.py``)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _solar() -> Any:
    """The gas model's ``solar`` module, imported without dragging JAX in.

    ``coupled.model_bridge`` does this same ``sys.path`` insert and is the canonical version, but
    importing it costs ~1.2 s and loads JAX for what is a few lines of NumPy trigonometry. The
    stage-1 panel is on the landing stage, so that is worth avoiding.
    ``test_solar_comes_from_the_model`` asserts this resolves to the submodule's own file.
    """
    gas = os.path.join(_repo_root(), "stratchem-jax")
    if not os.path.isdir(gas):  # legacy vendored layout, as model_bridge also handles
        gas = os.path.join(_repo_root(), "gas_phase_chemistry")
    if gas not in sys.path:
        sys.path.insert(0, gas)
    import solar

    return solar


def sza_diurnal(config: RunConfig) -> dict[str, Any]:
    """Solar zenith angle over the 24 hours of the configured day (stage 1).

    Photolysis follows the sun, so this is the panel that says what "day 172 at 30 degrees" actually
    means for the chemistry: how long the sun is up, how high it gets, and where the configured
    release hour sits in that.
    """
    solar = _solar()
    # Resolved first: day_of_year is derived from the month and day (SCIENCE-1), so a raw config has
    # it as None and the panel would draw a curve for a date nobody chose.
    resolved = resolve(config).config
    latitude = resolved.site.latitude_deg
    longitude = resolved.site.longitude_deg
    day = resolved.schedule.day_of_year
    hours = [24.0 * i / (_DIURNAL_POINTS - 1) for i in range(_DIURNAL_POINTS)]
    sza = [float(solar.solar_zenith_angle(latitude, longitude, day, h)) for h in hours]
    daylight = [h for h, z in zip(hours, sza, strict=True) if z < 90.0]
    return {
        "hours": hours,
        "sza_deg": sza,
        "release_hour": resolved.schedule.start_utc_hour,
        "release_sza_deg": float(
            solar.solar_zenith_angle(latitude, longitude, day, resolved.schedule.start_utc_hour)
        ),
        "min_sza_deg": min(sza),
        # Counted from the sampled points rather than solved for: at ten-minute resolution this is
        # exact to ten minutes, and saying so beats implying a precision the sampling does not have.
        "daylight_hours": round(len(daylight) * 24.0 / _DIURNAL_POINTS, 2),
        "sun_up": bool(daylight),
    }


def dilution_curve(config: RunConfig) -> dict[str, Any]:
    """V(t)/V0 for the configured regime, with every other regime for comparison (stage 4).

    The others are drawn because the choice between them is the point of the stage: D2 against D3
    is a factor of three in dilution rate, far easier to judge as two curves than as two names.
    """
    from coupled import dilution

    days = config.schedule.duration_days
    seconds = np.linspace(0.0, days * 86400.0, _CURVE_POINTS)
    regimes = {}
    for name in dilution.DILUTION_REGIMES:
        ratio = np.asarray(dilution.volume_ratio(seconds, name), dtype=float)
        regimes[name] = {
            "label": str(dilution.DILUTION_REGIMES[name][0]),
            "volume_ratio": [float(v) for v in ratio],
            "final": float(ratio[-1]),
        }
    selected = (
        config.dilution.regime.value
        if hasattr(config.dilution.regime, "value")
        else str(config.dilution.regime)
    )
    return {
        "hours": [float(s) / 3600.0 for s in seconds],
        "days": [float(s) / 86400.0 for s in seconds],
        "regimes": regimes,
        "selected": selected,
        # CONSTANT ignores the curves entirely and uses a fixed rate, so the panel must say so
        # rather than highlighting a curve that will not be used.
        "uses_curve": selected != "constant",
        "constant_rate_per_s": config.dilution.rate_per_s,
    }


def size_distribution(config: RunConfig) -> dict[str, Any]:
    """The background distribution TOMAS will actually be seeded with (stage 5).

    Built through ``to_scenario`` and ``initial_tomas_state`` -- the real path -- so the numbers are
    the run's, including the STP-to-ambient conversion that a hand-assembled call gets wrong.
    """
    from coupled import tomas_bridge
    from studio.modelio.scenario import to_scenario

    scenario = to_scenario(resolve(config).config)
    state = tomas_bridge.initial_tomas_state(scenario)
    number_per_cell = np.asarray(state.Nk, dtype=float)
    number_per_cm3 = number_per_cell / tomas_bridge.BOXVOL_CM3
    edges = np.asarray(
        tomas_bridge._bad._xk_to_dp_um(np.asarray(tomas_bridge._grid_for(len(number_per_cm3)))),
        dtype=float,
    )
    density = dn_dlogdp(number_per_cm3, edges)
    midpoints = bin_midpoints_um(edges)
    peak = int(np.argmax(density)) if density.size else 0
    return {
        # DRY diameters (CAVEATS: dp_mid_um is dry, SA/radius_cm are wet).
        "dp_um": [float(d) for d in midpoints],
        "dn_dlogdp": [float(v) for v in density],
        "edges_um": [float(e) for e in edges],
        "n_bins": int(number_per_cm3.size),
        "total_cm3": float(number_per_cm3.sum()),
        "peak_dp_um": float(midpoints[peak]) if density.size else 0.0,
        "peak_dn_dlogdp": float(density[peak]) if density.size else 0.0,
        "basis": "dry",
    }


def bin_grid(config: RunConfig) -> dict[str, Any]:
    """The TOMAS mass grid at the configured resolution (stage 7).

    The diameter range is fixed and the mass ratio is 2**(40/n_bins), so choosing n_bins moves the
    resolution and nothing else -- which is much clearer as bin widths across the size range than as
    the number 80.
    """
    from coupled import tomas_bridge

    grids = {}
    for n_bins in (40, 80, 160):
        edges = np.asarray(
            tomas_bridge._bad._xk_to_dp_um(np.asarray(tomas_bridge._grid_for(n_bins))), dtype=float
        )
        midpoints = bin_midpoints_um(edges)
        # Width in decades: what a bin actually resolves on a log-diameter axis.
        widths = np.diff(np.log10(edges))
        grids[str(n_bins)] = {
            "dp_um": [float(d) for d in midpoints],
            # Edges, not just midpoints: the edges are what is pinned across resolutions (the range
            # is fixed and only the subdivision moves), so the midpoints necessarily differ between
            # grids and are the wrong thing to quote as the range.
            "edges_um": [float(e) for e in edges],
            "width_decades": [float(w) for w in widths],
            "mass_ratio": 2.0 ** (40.0 / n_bins),
            "diameter_ratio": float(2.0 ** (40.0 / n_bins)) ** (1.0 / 3.0),
        }
    return {
        "grids": grids,
        "selected": str(config.microphysics.n_bins),
        "d_min_um": grids["40"]["edges_um"][0],
        "d_max_um": grids["40"]["edges_um"][-1],
    }


def concentration_sensitivity(config: RunConfig) -> dict[str, Any]:
    """Initial SO2 mixing ratio against plume volume, with this config marked (stage 3).

    The panel is the SCIENCE-2 caveat made visible. t = 0 is the moment the volume is defined
    (answered 2026-08-18), so V0 is a modelling choice -- and on a log-log axis its consequence is a
    straight line the reader can walk along: this is the sensitivity of the initial concentration to
    a choice the model does not make for you.

    Pure ``studio.science``: no model import, so this panel costs nothing and needs no JAX.
    """
    from studio.science.air import air_number_density
    from studio.science.constants import SO2_MOLAR_MASS_G_PER_MOL
    from studio.science.plume import initial_mixing_ratio_pptv

    resolved = resolve(config).config
    volume = resolved.injection.plume_volume_cm3
    if volume is None or volume <= 0.0:
        raise ValueError("plume_volume_cm3 is unresolved; resolve the config before previewing")
    air = air_number_density(resolved.site.pressure_mbar, resolved.site.temperature_k)
    decades = 3.0
    volumes = np.logspace(
        math.log10(volume) - decades, math.log10(volume) + decades, _CURVE_POINTS // 3
    )
    pptv = [
        initial_mixing_ratio_pptv(
            mass_kg=resolved.injection.so2_mass_kg,
            molar_mass_g_per_mol=SO2_MOLAR_MASS_G_PER_MOL,
            volume_cm3=float(v),
            pressure_mbar=resolved.site.pressure_mbar,
            temperature_k=resolved.site.temperature_k,
        )
        for v in volumes
    ]
    return {
        "volume_cm3": [float(v) for v in volumes],
        "pptv": [float(p) for p in pptv],
        "current_volume_cm3": float(volume),
        "current_pptv": float(resolved.injection.so2_initial_pptv or 0.0),
        "background_pptv": resolved.background.so2_pptv,
        "air_number_density_cm3": air,
    }


#: Panel name -> builder. The API exposes exactly these, so a typo in a panel name is a 404 naming
#: the ones that exist rather than an empty chart.
PANELS = {
    "sza": sza_diurnal,
    "dilution": dilution_curve,
    "size-distribution": size_distribution,
    "bins": bin_grid,
    "concentration": concentration_sensitivity,
}

__all__ = [
    "PANELS",
    "bin_grid",
    "concentration_sensitivity",
    "dilution_curve",
    "size_distribution",
    "sza_diurnal",
]
