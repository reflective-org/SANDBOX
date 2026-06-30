"""Impact of the JPL 19-5 rate update: new rates vs the legacy JPL-11 reference.

Overlays the Phase A trajectory with the updated JPL 19-5 rates (this branch) on the legacy
JPL-11 Octave reference (tests/fixtures/trajectory.csv), at the same scenario/settings, so the
effect of the rate update on each species is visible.

Usage:
    python3 tests/plot_jpl195_impact.py
"""

import csv
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src-python"))
from config import IDX, SPECIES, ModelConfig  # noqa: E402
from driver import run  # noqa: E402

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def main():
    rows = list(csv.DictReader(open(os.path.join(FIX, "trajectory.csv"))))
    meta = {r[0]: r[1] for r in csv.reader(open(os.path.join(FIX, "trajectory_meta.csv")))
            if r and r[0] != "key"}
    M = float(meta["M"])
    t_ref = np.array([float(r["time"]) for r in rows]) / 3600.0

    cfg = ModelConfig(T=float(meta["T"]), P=float(meta["P"]), M=M, SA=float(meta["SA"]),
                      WTR=float(meta["WTR"]), Yn2o5=float(meta["Yn2o5"]),
                      opt=int(float(meta["opt"])))
    t, x = run(cfg, td=float(meta["td"]), tn=float(meta["tn"]),
               days=int(meta["days"]), DT=float(meta["DT"]))
    t_hr = t / 3600.0

    species = [("HCl", "r"), ("ClONO2", "g"), ("NO2", "m"), ("ClO", "b"),
               ("OH", "c"), ("HO2", "orange")]
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, c in species:
        ref = np.array([float(r[name]) for r in rows]) / M * 1e12
        new = x[:, IDX[name]] / M * 1e12
        ax.plot(t_ref, ref, color=c, lw=4, alpha=0.30)                 # legacy JPL-11 (pale)
        ax.plot(t_hr, new, color=c, lw=1.5, label=name)                # JPL 19-5 (solid)
    ax.set_title("Rate-constant update impact: JPL 19-5 (thin) vs legacy JPL-11 (thick pale)")
    ax.set_xlabel("Time (hr)"); ax.set_ylabel("Mixing ratio (pptv)")
    ax.legend(fontsize=9, ncol=3); ax.grid(alpha=0.3)

    # Largest end-of-run change, for a quick quantitative headline.
    worst, wn = 0.0, None
    for name in SPECIES:
        ref = np.array([float(r[name]) for r in rows])
        scale = max(np.max(np.abs(ref)), 1e-30)
        rel = abs(x[-1, IDX[name]] - ref[-1]) / scale
        if rel > worst:
            worst, wn = rel, name
    ax.text(0.02, 0.02, f"largest end-of-run change: {worst*100:.1f}% ({wn})",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "jpl195_impact.png")
    fig.savefig(out, dpi=110)
    print(f"Wrote {out}  (largest end-of-run change {worst*100:.1f}% in {wn})")


if __name__ == "__main__":
    main()
