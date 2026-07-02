# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Put the coupled/ package dir on sys.path so tests can import its modules by plain name
(mirrors gas_phase_chemistry/conftest.py)."""

import os
import sys

# coupled/ on path -> flat imports (`from units import ...`) for the lightweight config/units tests.
sys.path.insert(0, os.path.dirname(__file__))
# repo root on path -> package imports (`from coupled.model_bridge import ...`) for the bridge/driver,
# which reach into gas_phase_chemistry. (Kept separate from the flat style; unifying is a DEFERRED item.)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
