# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Box temperature tendency from radiative heating (Phase 5).

dT/dt = [H_gas + H_aerosol_SW] / (n_air * cp_molec), where
  * H_gas = [O3] * Σ_channels heating_per_absorber(box)   -- O3 photochemical heating (heating_rates.F90
    port; O2 deferred, AD-5.1),
  * H_aerosol_SW = Σ_λ actinic(λ,box) * b_abs(λ) * E_photon(λ)   -- aerosol shortwave absorption from the
    Mie optics (b_abs = b_ext - b_sca). Longwave aerosol heating is NOT included (AD-5.4, OPEN) and is
    the dominant strat-sulfate term, so this UNDERESTIMATES aerosol heating.
Night (sza>=90) -> 0 (the TUV-x solve leaves a small twilight residual that is gated here, matching the
J path). cp_molec = 7/2 kB for diatomic air at constant pressure (AD-5.3).
"""

from __future__ import annotations

import numpy as np

from . import model_bridge  # noqa: F401  (gas_phase_chemistry on sys.path)
from .aerosol_optics import bulk_optics

from config import IDX                                   # noqa: E402  (gas model)
from tuvx_photolysis_adapter import compute_box_heating  # noqa: E402

KB = 1.380649e-23                       # Boltzmann constant [J/K]
CP_MOLEC = 3.5 * KB                     # air heat capacity per molecule, constant pressure [J/K]
_HC_J_NM = 6.626068e-34 * 2.99792458e8 * 1.0e9   # h*c [J*nm], matches the port


def box_dTdt(cfg, conc, t_seconds, aerosol_props=None, tstate=None, mie=None):
    """Box temperature tendency [K/s] at model time ``t_seconds``. Returns 0.0 at night.

    ``conc`` is the gas state [molec/cm^3] (for [O3]). If ``tstate`` + ``mie`` are given, the aerosol
    shortwave-absorption term is added using the SAME box actinic flux from the heating solve (which
    includes the aerosol iff ``aerosol_props`` was passed).
    """
    res = compute_box_heating(cfg, t_seconds, aerosol_props=aerosol_props)
    if res is None:
        return 0.0                      # night: no shortwave heating
    h_box, flux_box, wl_nm = res
    o3 = float(conc[IDX["O3"]])                                   # molec/cm^3
    h_gas = o3 * float(sum(h_box.values()))                       # J/cm^3/s
    h_aer = 0.0
    if tstate is not None and mie is not None:
        b_ext, b_sca, _ssa, _g = bulk_optics(tstate, mie)         # 1/cm per wavelength
        b_abs = np.maximum(b_ext - b_sca, 0.0)
        e_photon = _HC_J_NM / wl_nm                               # J
        h_aer = float(np.sum(flux_box * b_abs * e_photon))        # J/cm^3/s
    n_air = float(cfg.M)                                          # molec/cm^3
    return (h_gas + h_aer) / (n_air * CP_MOLEC)                   # K/s
