"""Pytest configuration for the gas-phase chemistry model.

The chemistry modules live flat in this directory and import each other by plain module name
(e.g. ``from config import ModelConfig``). Add this directory to ``sys.path`` so both the tests
and the modules resolve those imports.
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# The optional JAX/Diffrax "Phase-B" solver (jaxmodel/) needs extra deps installed via
# `pip install -e ".[jax-chemistry]"`. Skip its tests when those deps are absent so the default
# `pytest` run (NumPy/SciPy core + the TUV-x coupling) stays clean.
collect_ignore_glob = []
if importlib.util.find_spec("diffrax") is None or importlib.util.find_spec("lineax") is None:
    collect_ignore_glob.append("tests/test_jax_*.py")
