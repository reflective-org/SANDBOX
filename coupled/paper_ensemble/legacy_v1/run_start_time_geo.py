# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Start-time sensitivity in the GEOENGINEERED background: 16 runs = 2 sites (30N/20km,
60N/15km) x 8 start hours, D2 med only, aer_geo background (bg SO2 = 100 pptv),
alpha/nuc/coag x1. Companion to run_start_time.py (clean SABR-220).

Output -> runs_start_time_geo/. Case ids: <site>__aergeo__D2med__h<HH>.

CLI (from SANDBOX/):
  python -m coupled.paper_ensemble.run_start_time_geo plan | one <i> | run <lo> <hi>
"""
import dataclasses
import itertools
import os
import sys

from coupled.paper_ensemble import run_ensemble as _re

_HERE = os.path.dirname(os.path.abspath(__file__))
_re._OUT = os.path.join(_HERE, "runs_start_time_geo")

START_HOURS = [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0]
_BG = ("aergeo", "aer_geo", 100.0)
_D2 = ("D2med", "D2")
_X1 = dict(sticking=("a1p0", 1.0), nucleation=("nuc1", 1.0), coag=("cg1", 1.0))
_SITES = [la for la in _re.LAT_ALT if la[0] in ("30N_20km", "60N_15km")]


def _cases():
    cases = []
    for la, h in itertools.product(_SITES, START_HOURS):
        cid = f"{la[0]}__aergeo__D2med__h{int(h):02d}"
        cases.append((cid, dict(lat_alt=la, background=_BG, dilution=_D2,
                                start_hour=h, **_X1)))
    return cases


_build_orig = _re.build_scenario


def _build(ax):
    sc = _build_orig(ax)
    if "start_hour" in ax:
        sc = dataclasses.replace(sc, start_utc_hour=float(ax["start_hour"]))
    return sc


_re.all_cases = _cases
_re.build_scenario = _build


def main(argv):
    _re.main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
