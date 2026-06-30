"""Quick visual checkpoint for the Octave reference trajectory (fixture from A2).

This is NOT the model's plotting code (that is the future plotting.py port). It only
reads tests/fixtures/trajectory.csv and draws the main species so we can eyeball the
ground truth the Python port will be validated against.

Usage:
    python3 tests/plot_reference.py
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIX = os.path.join(HERE, "fixtures")                       # input CSV fixtures
FIGDIR = os.path.join(HERE, "..", "figures")               # output figures


def load_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    cols = {k: [float(r[k]) for r in rows] for k in rows[0]}
    return cols


def main():
    traj = load_csv(os.path.join(FIX, "trajectory.csv"))
    meta = {r[0]: r[1] for r in csv.reader(open(os.path.join(FIX, "trajectory_meta.csv")))
            if r and r[0] != "key"}
    M = float(meta["M"])

    t_hr = [s / 3600.0 for s in traj["time"]]

    def ppt(name):
        # molec/cm^3 -> pptv, as the MATLAB figures plot
        return [v / M * 1e12 for v in traj[name]]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        f"Octave reference trajectory  (T={meta['T']} K, P={meta['P']} mbar, "
        f"H2O={meta['WTR']} ppm, SA={meta['SA']} um^2/cm^3, opt={meta['opt']})",
        fontsize=12,
    )

    # (1) Chlorine partitioning -- mirrors MATLAB Figure 1
    ax = axes[0, 0]
    for name, c in [("HCl", "r"), ("ClONO2", "g"), ("ClO", "b"),
                    ("Cl2", "y"), ("ClOOCl", "c"), ("NO2", "m")]:
        ax.plot(t_hr, ppt(name), c, label=name, lw=1.8)
    ax.set_title("Chlorine / NO2 partitioning")
    ax.set_xlabel("Time (hr)"); ax.set_ylabel("Mixing ratio (pptv)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (2) HOx -- mirrors MATLAB Figure 3
    ax = axes[0, 1]
    ax.plot(t_hr, ppt("OH"), "b", label="OH", lw=1.8)
    ax.plot(t_hr, ppt("HO2"), "r", label="HO2", lw=1.8)
    ax.set_title("HOx")
    ax.set_xlabel("Time (hr)"); ax.set_ylabel("Mixing ratio (pptv)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (3) Bromine -- mirrors MATLAB Figure 5
    ax = axes[1, 0]
    for name, c in [("BrO", "b"), ("BrCl", "r"), ("BrONO2", "g"),
                    ("Br", "m"), ("HOBr", "c")]:
        ax.plot(t_hr, ppt(name), c, label=name, lw=1.8)
    ax.set_title("Bromine")
    ax.set_xlabel("Time (hr)"); ax.set_ylabel("Mixing ratio (pptv)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (4) Ozone as fraction of its starting value -- mirrors MATLAB Figure 2
    ax = axes[1, 1]
    o3 = traj["O3"]
    ax.plot(t_hr, [v / o3[0] for v in o3], "b", lw=1.8)
    ax.set_title("Ozone (fraction of start)")
    ax.set_xlabel("Time (hr)"); ax.set_ylabel("O3 / O3(0)")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "trajectory_reference.png")
    fig.savefig(out, dpi=110)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
