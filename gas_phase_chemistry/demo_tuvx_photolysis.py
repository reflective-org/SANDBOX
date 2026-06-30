"""Smoke test / demo: diurnal TUV-x photolysis rate constants for the frank-model box.

Computes J(t) over a day at the box conditions using the TUV-x (JAX) port via
``tuvx_photolysis_adapter`` and plots a few representative reactions. Run with the port installed::

    pip install -e ..
    python demo_tuvx_photolysis.py

Writes ``tuvx_diurnal_J.png`` next to this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tuvx_photolysis_adapter import j_values_for, REACTION_MAP, FALLBACK_REACTIONS


@dataclass
class _Scenario:
    P: float = 68.0
    latitude: float = 0.0
    longitude: float = 0.0
    day_of_year: float = 80.0       # ~21 March (equinox)
    start_utc_hour: float = 0.0


def main():
    cfg = _Scenario()
    hours = np.linspace(0.0, 24.0, 49)
    reactions = ["NO2 -> NO + O", "NO3 -> NO2 + O", "Cl2 -> 2 Cl",
                 "ClONO2 -> Cl + NO3", "BrO -> Br + O", "OClO -> O + ClO"]

    series = {r: [] for r in reactions}
    for h in hours:
        jv = j_values_for(cfg, t_seconds=h * 3600.0)
        for r in reactions:
            series[r].append(jv.get(r, 0.0))

    fig, ax = plt.subplots(figsize=(8, 5))
    for r in reactions:
        ax.plot(hours, series[r], label=r)
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1e-1)
    ax.set_xlabel("UTC hour")
    ax.set_ylabel("J (s$^{-1}$)")
    ax.set_title(
        f"TUV-x diurnal photolysis rate constants\n"
        f"box: {cfg.P:.0f} mbar, lat {cfg.latitude:.0f}, day {cfg.day_of_year:.0f}"
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    out = Path(__file__).resolve().parent / "tuvx_diurnal_J.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"covered reactions: {len(REACTION_MAP)}; fallback (reference-scaled): {sorted(FALLBACK_REACTIONS)}")
    noon = j_values_for(cfg, t_seconds=12 * 3600.0)
    print("J at local noon (s^-1):")
    for r in reactions:
        print(f"  {r:22s} {noon[r]:.3e}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
