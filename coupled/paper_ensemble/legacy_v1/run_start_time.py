# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Start-time-of-day sensitivity ensemble: 120 runs = 3 lat/alt x 5 dilution x 8 start hours
(00/03/06/09/12/15/18/21 UTC at longitude 0 = local solar time).

Fixed: SABR-220 background (bg SO2 = 20 pptv), condensation alpha x1, nucleation x1, coag x1.
Everything else identical to run_ensemble.py (delegated). Output -> runs_start_time/.
Case ids: <site>__sabr220__<dilution>__h<HH>.

CLI (from SANDBOX/):
  python -m coupled.paper_ensemble.run_start_time plan | one <i> | run <lo> <hi>
"""
import dataclasses
import itertools
import os
import sys

from coupled.paper_ensemble import run_ensemble as _re

_HERE = os.path.dirname(os.path.abspath(__file__))
_re._OUT = os.path.join(_HERE, "runs_start_time")

START_HOURS = [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0]
_BG = ("sabr220", "sabr_220", 20.0)
_X1 = dict(sticking=("a1p0", 1.0), nucleation=("nuc1", 1.0), coag=("cg1", 1.0))


def _cases():
    cases = []
    for la, dl, h in itertools.product(_re.LAT_ALT, _re.DILUTION, START_HOURS):
        cid = f"{la[0]}__sabr220__{dl[0]}__h{int(h):02d}"
        cases.append((cid, dict(lat_alt=la, background=_BG, dilution=dl,
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
