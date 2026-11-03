# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Initial-plume-volume (emission density) sweep: the SAME 1 tonne of SO2 released into a
V0 that is 2 / 5 / 10 / 20 / 50 / 100 x larger than the standard 10 m x 10 m x 15 km, i.e.
the initial SO2 concentration is divided by that factor. V0 enters ONLY through the initial
concentration (dilution is V(t)/V0, intensive), so this is an emission-density experiment.

Fixed: baseline 1 (30N / 20 km / 210 K), SABR-220 aged-air background, D2 med dilution,
alpha x1, nuc x1, coag x1, midnight release. The x1 reference is the standard ensemble run.

Output -> runs_boxsize/. Case ids: 30N_20km__sabr220__D2med__v<F>.

CLI (from SANDBOX/): python -m coupled.paper_ensemble.run_boxsize plan | one <i> | run <lo> <hi>
"""
import dataclasses
import os
import sys

from coupled.paper_ensemble import run_ensemble as _re

_HERE = os.path.dirname(os.path.abspath(__file__))
_re._OUT = os.path.join(_HERE, "runs_boxsize")

FACTORS = [2, 5, 10, 20, 50, 100]
_BG = ("sabr220", "sabr_220", 20.0)
DILUTIONS = [("D2med", "D2"), ("D1low", "D1")]
_AX = dict(background=_BG, sticking=("a1p0", 1.0),
           nucleation=("nuc1", 1.0), coag=("cg1", 1.0))


def _cases():
    return [(f"30N_20km__sabr220__{dl[0]}__v{f}",
             dict(lat_alt=_re.LAT_ALT[0], vol_factor=float(f), dilution=dl, **_AX))
            for dl in DILUTIONS for f in FACTORS]


_build_orig = _re.build_scenario


def _build(ax):
    sc = _build_orig(ax)
    f = float(ax.get("vol_factor", 1.0))
    conc = {**sc.concentrations, "SO2": sc.concentrations["SO2"] / f}
    return dataclasses.replace(sc, concentrations=conc)


_re.all_cases = _cases
_re.build_scenario = _build


def main(argv):
    _re.main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
