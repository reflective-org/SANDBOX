"""Visual validation: overlay the Python port on the Octave reference trajectory.

Confirms that driver.run (SciPy BDF) reproduces the Octave/MATLAB reference (ode15s).
Reads tests/fixtures/trajectory.csv for the reference and re-runs the Python driver with
the same settings.

Usage:
    python3 tests/plot_validation.py
"""

import csv
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src-python"))
from config import IDX, ModelConfig  # noqa: E402
from driver import run  # noqa: E402

FIX = os.path.join(os.path.dirname(__file__), "fixtures")          # input CSV fixtures
FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")  # output figures


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

    def ref(name):
        return np.array([float(r[name]) for r in rows]) / M * 1e12

    def py(name):
        return x[:, IDX[name]] / M * 1e12

    species = [("HCl", "r"), ("ClONO2", "g"), ("ClO", "b"), ("NO2", "m"),
               ("OH", "c"), ("HO2", "orange")]

    fig, ax = plt.subplots(1, 1, figsize=(11, 6))
    for name, c in species:
        ax.plot(t_ref, ref(name), color=c, lw=4, alpha=0.30)            # reference: thick pale
        ax.plot(t_hr, py(name), color=c, lw=1.4, label=name)            # python: thin solid
    ax.set_title("Python port (thin) over Octave reference (thick pale) -- they overlap")
    ax.set_xlabel("Time (hr)")
    ax.set_ylabel("Mixing ratio (pptv)")
    ax.legend(fontsize=9, ncol=3)
    ax.grid(alpha=0.3)

    # Annotate the worst-case agreement so the match is quantified, not just visual.
    worst_name, worst = None, 0.0
    from config import SPECIES
    for name in SPECIES:
        xo = np.array([float(r[name]) for r in rows])
        xp = x[:, IDX[name]]
        scale = max(np.max(np.abs(xo)), 1e-30)
        rel = np.max(np.abs(xp - xo)) / scale
        if rel > worst:
            worst, worst_name = rel, name
    ax.text(0.02, 0.02, f"max relative difference: {worst:.2e} ({worst_name})",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "trajectory_validation.png")
    fig.savefig(out, dpi=110)
    print(f"Wrote {out}  (worst relative difference {worst:.2e} in {worst_name})")


if __name__ == "__main__":
    main()
