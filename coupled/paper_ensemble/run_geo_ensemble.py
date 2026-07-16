# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Geoengineered-background ensemble: the same 270 combinations as the main 810-run sweep
(3 lat/alt x 5 dilution x 2 sticking x 3 nucleation x 3 coag), for TWO backgrounds -> 540 runs:

  * aer_geo     -- AER 2D geoengineered stratosphere (Pierce fig. 2 gray curve): one lognormal,
                   N = 120 cm^-3, Dg = 0.30 um, sigma_g = 1.7, AMBIENT (no STP conversion).
  * cesm_g6_amb -- the CESM G6 modes re-read as AMBIENT (the original cesm_g6 runs wrongly
                   applied the STP->ambient factor, ~14.5x too dilute).

Background SO2 = 100 pptv for both. Everything else identical to run_ensemble.py (delegated).
Output -> coupled/paper_ensemble/runs_geo/.

CLI (from SANDBOX/), same modes as run_ensemble:
  python -m coupled.paper_ensemble.run_geo_ensemble plan | one <i> | run <lo> <hi>
"""
import os
import sys

from coupled.paper_ensemble import run_ensemble as _re

_HERE = os.path.dirname(os.path.abspath(__file__))

# override the background axis and the output dir; all machinery is reused from run_ensemble
_re.BACKGROUND = [("aergeo", "aer_geo", 100.0),
                  ("cesm", "cesm_g6_amb", 100.0)]
_re._OUT = os.path.join(_HERE, "runs_geo")


def main(argv):
    _re.main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
