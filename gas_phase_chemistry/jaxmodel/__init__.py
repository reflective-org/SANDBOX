"""JAX implementation of the gas-phase chemistry box model (Phase B).

This is a JAX-native re-implementation of the validated NumPy/SciPy model (Phase A). It
reuses the backend-agnostic structure -- the species ordering (``config.SPECIES``) and the
parsed reaction stoichiometry (``mechanism``/``reactions``) -- but evaluates the chemistry
with ``jax.numpy`` so the whole model can be JIT-compiled, differentiated, and vmapped.

Phase A stays the reference: the JAX results are validated against it (and it, in turn,
against the original MATLAB via Octave).

float64 is REQUIRED: concentrations span ~1e0 to ~1e18 molec/cm^3, so the default float32
would lose all accuracy. We enable 64-bit globally on import.
"""

import jax

jax.config.update("jax_enable_x64", True)
