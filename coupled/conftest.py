# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Put the coupled/ package dir on sys.path so tests can import its modules by plain name
(mirrors gas_phase_chemistry/conftest.py)."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
