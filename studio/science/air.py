# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Air number density.

This is the one function here that duplicates a model function rather than reusing it, and the
duplication is forced: ``studio.science`` may not import ``coupled`` or the model's flat modules
(ADR-001), because the packages must be usable from a bare Python session and importing the model
costs a JAX import.

So it is a MIRROR, not a fork. It is one line, it cites the line it mirrors, and
``studio/tests/unit/test_science_air.py`` asserts that the two agree exactly whenever the
``stratchem-jax`` submodule is checked out. If the model's relation ever changes, that test fails --
which is the property that makes a mirror acceptable and a quiet copy not.
"""

from __future__ import annotations

from studio.science.constants import AIR_NUMBER_DENSITY_COEFF, MBAR_TO_TORR


def air_number_density(pressure_mbar: float, temperature_k: float) -> float:
    """Air number density M [molec cm^-3] at ``pressure_mbar`` and ``temperature_k``.

    Mirrors ``stratchem-jax/config.py:55`` (``M = 9.65e18*P/T*conv``), itself a port of the MATLAB
    ``runconcs_het.m``. The coefficient folds in the mbar->torr conversion.

    Raises:
        ValueError: On a non-positive pressure or temperature. There is no sensible fallback: a
            zero temperature is a division by zero and a negative pressure is not a state (ADR-005).
    """
    if pressure_mbar <= 0.0:
        raise ValueError(f"pressure must be > 0 mbar, got {pressure_mbar}")
    if temperature_k <= 0.0:
        raise ValueError(f"temperature must be > 0 K, got {temperature_k}")
    return AIR_NUMBER_DENSITY_COEFF * pressure_mbar / temperature_k * MBAR_TO_TORR


__all__ = ["air_number_density"]
