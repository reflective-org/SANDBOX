"""Visual checkpoint for solar geometry (M4): diurnal solar zenith angle.

Plots SZA over two days for a few latitudes, with night (SZA > 90 deg) shaded.

Usage:
    python3 tests/plot_solar.py
"""

import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src-python"))
from solar import solar_zenith_angle  # noqa: E402

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def main():
    hours = np.linspace(0, 48, 48 * 12 + 1)        # two days, UTC
    day_of_year = 80                                # ~equinox
    longitude = 0.0

    fig, ax = plt.subplots(figsize=(11, 6))
    for lat, c in [(0.0, "C0"), (30.0, "C1"), (60.0, "C2")]:
        sza = [solar_zenith_angle(lat, longitude, day_of_year, h % 24) for h in hours]
        ax.plot(hours, sza, c, lw=1.8, label=f"lat = {lat:g}°")

    ax.axhline(90, color="0.4", ls="--", lw=0.9)
    ax.text(0.5, 91, "horizon (SZA = 90°)", fontsize=8, color="0.4")
    # Shade where the sun is below the horizon for the equator case (reference day/night).
    sza_eq = np.array([solar_zenith_angle(0.0, longitude, day_of_year, h % 24) for h in hours])
    ax.fill_between(hours, 0, 180, where=sza_eq > 90, color="0.9", zorder=0)

    ax.set_ylim(0, 180)
    ax.set_xlim(0, 48)
    ax.set_xlabel("Time (hours UTC)")
    ax.set_ylabel("Solar zenith angle (deg)")
    ax.set_title(f"Diurnal solar zenith angle  (day-of-year {day_of_year}, longitude 0°)")
    ax.legend()
    ax.grid(alpha=0.3)

    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "solar_sza.png")
    fig.savefig(out, dpi=110)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
