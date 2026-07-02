# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Phase 2.5 end-to-end validation of the coupled driver (real TUV-x port).

Runs the operator-split JAX driver AND the NumPy mirror on the same tuvx scenario, then reports:
  * H2SO4 production + sulfur-atom conservation (physical sanity), and
  * JAX-vs-NumPy cross-backend agreement (matched J-handling -> isolates solver difference).
Writes plots to coupled/validation/. Run:  python -m coupled.validate_coupled
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coupled import CoupledScenario
from coupled.driver import run_coupled
from coupled.reference_numpy import run_coupled_numpy
from config import IDX, SPECIES

_OUT = os.path.join(os.path.dirname(__file__), "validation")
_COMP = {"O2": 2.1e11, "O3": 1.18e6, "CH4": 1.6e6, "H2O": 5.0e6, "SO2": 953895.0, "C2H6": 200.0,
         "ClO": 10.0, "ClONO2": 127.0, "HCl": 777.0, "N2O5": 20.0, "NO": 450.0, "NO2": 450.0,
         "OH": 0.5, "HO2": 3.0}


def main():
    sc = CoupledScenario(T=210.0, P=68.0, SA=2.0, WTR=5.0, Yn2o5=0.1, opt=1,
                         latitude=0.0, longitude=0.0, day_of_year=80, start_utc_hour=0.0,
                         days=2, DT=3600.0, dt_couple=3600.0, photolysis="tuvx",
                         concentrations=_COMP)
    t, x = run_coupled(sc)               # JAX, real TUV-x port
    tn, xn = run_coupled_numpy(sc)       # NumPy mirror, same intervals + frozen midpoint J
    days = t / 86400.0
    M = x[0, IDX["O2"]] / 0.21           # air number density (O2 = 0.21 M)

    def ppt(arr, name):
        return arr[:, IDX[name]] / M * 1e12

    S = x[:, IDX["SO2"]] + x[:, IDX["SO3"]] + x[:, IDX["H2SO4"]]
    drift = (S[-1] - S[0]) / S[0]
    # worst cross-backend species disagreement (relative to each species' peak). Restrict to species
    # with a non-negligible peak: near-zero species (e.g. Br-family when no Br is present) give a
    # meaningless huge relative diff on ~0/~0, which the parity TEST handles with an atol floor.
    peaks = {n: np.max(np.abs(xn[:, IDX[n]])) for n in SPECIES}
    ref = max(peaks.values())
    worst = max((np.max(np.abs(x[:, IDX[n]] - xn[:, IDX[n]])) / peaks[n], n)
                for n in SPECIES if peaks[n] > 1e-6 * ref)

    os.makedirs(_OUT, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in ("SO2", "SO3", "H2SO4"):
        y = ppt(x, name)
        ax.plot(days, np.where(y > 0, y, np.nan), lw=1.3, label=name)
    ax.set_yscale("log"); ax.set_xlabel("day"); ax.set_ylabel("pptv")
    ax.set_title("Coupled driver (tuvx): gas-phase sulfur oxidation")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "coupled_sulfur.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(days, S / S[0], color="C3", lw=1.5)
    ax.set_xlabel("day"); ax.set_ylabel("total S / initial")
    ax.set_title(f"Sulfur conservation (drift {drift:+.2e})")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "coupled_conservation.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in ("H2SO4", "ClO", "OH"):
        ax.plot(days, ppt(x, name), lw=1.4, label=f"{name} (JAX)")
        ax.plot(days, ppt(xn, name), lw=1.0, ls="--", label=f"{name} (NumPy)")
    ax.set_yscale("log"); ax.set_xlabel("day"); ax.set_ylabel("pptv")
    ax.set_title(f"JAX vs NumPy (matched J); worst rel diff = {worst[0]:.2e} ({worst[1]})")
    ax.legend(fontsize=7); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "coupled_jax_vs_numpy.png"), dpi=110); plt.close(fig)

    print(f"coupled tuvx run: {sc.days} days, dt_couple={sc.dt_couple}s, {len(t)} outer steps")
    print(f"SO2   {ppt(x,'SO2')[0]:.1f} -> {ppt(x,'SO2')[-1]:.1f} pptv")
    print(f"H2SO4 0 -> {ppt(x,'H2SO4')[-1]:.3e} pptv (produced)")
    print(f"sulfur drift: {drift:+.3e}")
    print(f"JAX vs NumPy worst relative species diff: {worst[0]:.3e} ({worst[1]})")
    print(f"wrote plots to {_OUT}/")


if __name__ == "__main__":
    main()
