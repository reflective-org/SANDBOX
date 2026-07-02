"""Compare O3 photolysis driven by the reference 4.7e-5*cos(SZA) vs the TUV-x O(1D) J.

Runs the model_input.yaml scenario twice, identical except for the O3-photolysis J source:
  * baseline: strip the TUV-x O(1D) key so reactions.py falls back to 4.7e-5 * j_scale;
  * tuvx:     use the absolute TUV-x O(1D)-channel J.
Writes overlay plots (baseline vs tuvx) for the most affected species and prints a summary.
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import IDX
from driver import integrate_sza
from scenario import Scenario
import tuvx_photolysis_adapter as ad

_p = argparse.ArgumentParser()
_p.add_argument("--config", default="scenarios/model_input.yaml")
_p.add_argument("--out", default="o3_tuvx_comparison")
_args = _p.parse_args()
CFG = _args.config
OUT = _args.out
SPECIES = ["O3", "OH", "HO2", "O1D", "O", "NO2", "NO", "HNO3", "ClO", "N2O5", "SO2", "H2O2"]

_orig_j = ad.j_values_for

def _strip_o3(cfg, t):
    d = dict(_orig_j(cfg, t))
    d.pop("O3 -> O2 + O1D", None)   # force reactions.py to the 4.7e-5 * j_scale fallback
    return d

def run(label):
    sc = Scenario.load(CFG)
    cfg = sc.to_config()
    x0 = sc.initial_state()
    t, states = integrate_sza(cfg, x0, days=sc.days, DT=sc.DT)
    return t, states, cfg

def ppt(states, cfg, name):
    return states[:, IDX[name]] / cfg.M * 1e12

# --- baseline (reference 4.7e-5 * cos SZA) ---
ad.j_values_for = _strip_o3
tb, sb, cfg = run("baseline 4.7e-5*cosSZA")
# --- tuvx O(1D) J ---
ad.j_values_for = _orig_j
tt, st, _ = run("tuvx O(1D) J")

os.makedirs(OUT, exist_ok=True)
days_b, days_t = tb / 86400.0, tt / 86400.0

print(f"\n{'species':8s} {'baseline':>14s} {'tuvx':>14s} {'Δ%':>9s}   (final mixing ratio, pptv)")
for name in SPECIES:
    yb, yt = ppt(sb, cfg, name), ppt(st, cfg, name)
    fb, ft = yb[-1], yt[-1]
    dpct = (ft / fb - 1.0) * 100.0 if fb != 0 else float("nan")
    print(f"{name:8s} {fb:14.4g} {ft:14.4g} {dpct:+8.3f}%")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(days_b, np.where(yb > 0, yb, np.nan), label="baseline 4.7e-5·cos(SZA)", lw=1.3)
    ax.plot(days_t, np.where(yt > 0, yt, np.nan), label="TUV-x O(1D) J", lw=1.3, ls="--")
    if np.nanmax(yt) > 0:
        ax.set_yscale("log")
    ax.set_xlabel("day"); ax.set_ylabel(f"{name} (pptv)")
    ax.set_title(f"{name}: O3-photolysis J source")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=110)
    plt.close(fig)

print(f"\nWrote overlay plots to {OUT}/")
