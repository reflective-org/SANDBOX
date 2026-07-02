# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Pytest setup for the coupled tests: enable JAX's PERSISTENT compilation cache.

Much of each coupled test's wall time is XLA compilation of the (jitted) gas solve + TOMAS step. The
persistent cache stores those kernels on disk so REPEATED test runs (and xdist workers) reuse them
instead of recompiling every time -- the single biggest lever for fast local iteration. Combine with
``pytest -n auto`` (pytest-xdist) to run the independent coupled-run tests across cores.

Override the location with ``JAX_COMPILATION_CACHE_DIR``; nothing here changes numerical results.
"""
import os

import jax

_CACHE = os.environ.get("JAX_COMPILATION_CACHE_DIR",
                        os.path.expanduser("~/.cache/jax_coupled"))
os.makedirs(_CACHE, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", _CACHE)
# cache every compiled kernel (defaults skip small/fast ones, which are most of ours)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
