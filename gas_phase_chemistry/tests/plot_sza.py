"""Visual checkpoint for SZA-dependent photolysis (M5).

Top: photolysis scaling vs time -- the new SZA mode (smooth cosine, sun-driven) vs the
reference mode (square wave from the prescribed day/night schedule).
Bottom: OH for both modes.

Usage:
    python3 tests/plot_sza.py
"""

import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src-python"))
from config import IDX  # noqa: E402
from driver import run_scenario  # noqa: E402
from rhs import _j_scale  # noqa: E402
from scenario import Scenario  # noqa: E402

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def reference_scale(t_s, td, tn):
    """Reference-mode photolysis scale (1 day / 0 night) from the td/tn schedule."""
    th = t_s / 3600.0
    if th <= td:
        return 1.0
    th -= td
    return 0.0 if (th % (tn + td)) < tn else 1.0


def main():
    common = dict(T=210, P=68, SA=2, WTR=5, opt=1, days=2, DT=600.0)
    ref = Scenario(photolysis="reference", td=14, tn=10, **common)
    sza = Scenario(photolysis="sza", latitude=20.0, longitude=0.0, day_of_year=80,
                   start_utc_hour=0.0, **common)

    t_ref, x_ref = run_scenario(ref)
    t_sza, x_sza = run_scenario(sza)

    cfg_sza = sza.to_config()
    js = [_j_scale(t, cfg_sza) for t in t_sza]
    jr = [reference_scale(t, ref.td, ref.tn) for t in t_ref]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(t_sza / 3600.0, js, "C1", lw=1.8, label="SZA mode (sun-driven)")
    ax1.plot(t_ref / 3600.0, jr, "C0", lw=1.4, label="reference (day/night square wave)")
    ax1.axhline(1.0, color="0.6", ls=":", lw=0.8)
    ax1.text(1, 1.02, "= tabulated J (45°)", fontsize=8, color="0.4")
    ax1.set_ylabel("photolysis scale (J / J$_{45°}$)")
    ax1.set_title("M5: SZA-dependent photolysis vs reference mode  "
                  "(lat 20°, equinox)")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    ax2.plot(t_sza / 3600.0, x_sza[:, IDX["OH"]] / cfg_sza.M * 1e12, "C1", lw=1.8,
             label="OH — SZA mode")
    ax2.plot(t_ref / 3600.0, x_ref[:, IDX["OH"]] / ref.to_config().M * 1e12, "C0", lw=1.4,
             label="OH — reference")
    ax2.set_xlabel("Time (hr)")
    ax2.set_ylabel("OH (pptv)")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "photolysis_sza.png")
    fig.savefig(out, dpi=110)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
