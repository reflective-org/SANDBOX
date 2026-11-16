# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Site baselines shared by the v2 paper pipeline (run_v2.py, make_spinup_ic.py).

``LAT_ALT`` is the July-2026 ensemble's lat+alt baseline table, moved here verbatim from the v1
``run_ensemble.py`` (now ``legacy_v1/run_ensemble.py``, which keeps its own frozen copy) so the v2
pipeline does not depend on ``legacy_v1/``. Rows: (label, latitude [deg], T [K], P [hPa],
WTR [ppmv for RH = 3% at that T, P]).
"""

LAT_ALT = [
    ("30N_20km",      30.0, 210.0,  55.0,  6.9104),
    ("60N_15km",      60.0, 210.0, 120.0,  3.1673),
    ("30N_20km_213K", 30.0, 213.0,  55.0, 10.1834),
]
