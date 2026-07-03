# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Phase 4 gate: aerosol -> photolysis feedback. Sweep aerosol loading and show the box-altitude
photolysis rates J respond to the aerosol optical depth. Writes coupled/validation/phase4_*.png.

Run:  python -m coupled.validate_phase4
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coupled import CoupledScenario
from coupled import tomas_bridge as tb
from coupled import aerosol_optics as ao
from coupled.model_bridge import to_model_config
from tuvx_photolysis_adapter import _compute_j_values, calculator_grids

_OUT = os.path.join(os.path.dirname(__file__), "validation")


def _col_od_550(props, wl_nm):
    k = int(np.argmin(np.abs(wl_nm - 550.0)))
    return float(props.optical_depth[:, k].sum())


def main():
    sc = CoupledScenario(T=210.0, P=68.0, WTR=5.0, latitude=0.0, day_of_year=80, start_utc_hour=12.0,
                         days=1, DT=3600.0, dt_couple=3600.0, photolysis="tuvx",
                         aerosol_band_km=(10.0, 30.0),
                         concentrations={"O2": 2.1e11, "O3": 1.18e6, "HCl": 777.0, "ClONO2": 127.0})
    cfg = to_model_config(sc)
    wl_nm, height_edges = calculator_grids(cfg)
    mie = ao.MieTable(wl_nm)
    st0 = tb.initial_tomas_state(sc)

    j0 = _compute_j_values(cfg, 0.0, aerosol_props=None)                 # no-aerosol baseline
    reactions = [k for k in j0 if j0[k] > 0.0]

    scales = [0.0, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5]                          # scale Nk to sweep OD
    ods, jrows = [], []
    for s in scales:
        st = st0._replace(Nk=st0.Nk * s)
        props = ao.aerosol_optical_props(st, wl_nm, height_edges, sc.aerosol_band_km, mie=mie)
        ods.append(_col_od_550(props, wl_nm))
        j = _compute_j_values(cfg, 0.0, aerosol_props=(None if s == 0.0 else props))
        jrows.append(j)
    ods = np.array(ods)

    os.makedirs(_OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in reactions[:8]:
        jvals = np.array([jr[name] for jr in jrows])
        ax.plot(ods, jvals / jvals[0], marker="o", lw=1.3, label=name.split("+hv")[0])
    ax.set_xlabel("aerosol column optical depth @ 550 nm")
    ax.set_ylabel("J / J(no aerosol)")
    ax.set_title("Phase 4: box-altitude photolysis rates vs aerosol optical depth")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "phase4_j_vs_od.png"), dpi=110); plt.close(fig)

    print(f"Phase 4 aerosol->photolysis sweep (box altitude, noon, band {sc.aerosol_band_km} km)")
    print(f"OD@550: {np.array2string(ods, precision=3)}")
    for name in reactions[:6]:
        jvals = np.array([jr[name] for jr in jrows])
        print(f"  {name:28s} J/J0: {np.array2string(jvals / jvals[0], precision=3)}")
    print(f"wrote plots to {_OUT}/")


if __name__ == "__main__":
    main()
