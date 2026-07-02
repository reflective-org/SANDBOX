"""Model configuration and species bookkeeping for the gas-phase chemistry box model.

This is the Python port of the shared state that the MATLAB code passed around through
`global` variables (see the `global T P M SA WTR SZA Yn2o5 count opt` line at the top of
``runconcs_het.m`` and ``concs_het.m``).

Instead of globals, we collect that state in a single ``ModelConfig`` object and pass it
explicitly into the right-hand-side function. This keeps the code easy to follow and makes
it safe to run many scenarios without hidden shared state.

Nothing in this module does any chemistry yet -- it only defines:
  * ``SPECIES``      : the names of the 34 modelled species, in the exact order MATLAB used
  * ``IDX``          : a name -> index lookup, so code can say ``x[IDX["O3"]]`` instead of ``x[10]``
  * ``ModelConfig``  : the physical scenario (temperature, pressure, aerosol, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------------------
# Species ordering
# ---------------------------------------------------------------------------------------
# The state vector `x` has 36 entries. Entries 1-34 MUST match the column order used in the
# MATLAB code so that a Python run can be compared directly against the MATLAB/Octave output.
# The reference order is the comment block at the top of ``concs_het.m``:
#
#   x = [Cl, ClO, ClOOCl, ClONO2, HCl, Cl2, NO, NO2, O, O2, O3, CH3, CH4, HNO3, H2O, HOCl,
#        N2O5, NO3, OH, HO2, O1D, HONO, HNO4, OClO, Br, BrO, BrONO2, BrCl, HBr, HOBr,
#        HNO3aq, C2H6, SO2, H2O2]
#
# (SO2 and H2O2 are the two species "FK" added at the end of the original 34.)
# SO3 and H2SO4 (35-36) are APPENDED for the gas-phase sulfur-oxidation chain (SO2->SO3->H2SO4);
# keeping them last leaves indices 1-34 unchanged, so reference-mode results are unaffected.
SPECIES = [
    "Cl", "ClO", "ClOOCl", "ClONO2", "HCl", "Cl2", "NO", "NO2",   # 1-8
    "O", "O2", "O3", "CH3", "CH4", "HNO3", "H2O", "HOCl",         # 9-16
    "N2O5", "NO3", "OH", "HO2", "O1D", "HONO", "HNO4", "OClO",    # 17-24
    "Br", "BrO", "BrONO2", "BrCl", "HBr", "HOBr", "HNO3aq",       # 25-31
    "C2H6", "SO2", "H2O2",                                        # 32-34
    "SO3", "H2SO4",                                               # 35-36 (sulfur chain; appended)
]

N_SPECIES = len(SPECIES)  # 36

# Name -> position in the state vector (0-based, unlike MATLAB's 1-based indexing).
IDX = {name: i for i, name in enumerate(SPECIES)}


# Number-density conversion constant used in ``runconcs_het.m``:
#   conv = 760 / 1013.25   (millibar -> torr, used inside the M calculation)
MBAR_TO_TORR = 760.0 / 1013.25


def air_number_density(P: float, T: float) -> float:
    """Air number density M in molecules/cm^3.

    Mirrors the MATLAB line ``M = 9.65e18*P/T*conv`` from ``runconcs_het.m`` (the ``conv``
    factor folds in the mbar->torr conversion). ``P`` is in mbar and ``T`` in kelvin.
    """
    return 9.65e18 * P / T * MBAR_TO_TORR


@dataclass
class ModelConfig:
    """Physical scenario shared by the driver and the right-hand-side function.

    These are the former MATLAB ``global`` variables:

      T      : temperature (K)
      P      : pressure (mbar)
      M      : air number density (molecules/cm^3); if left as None it is computed from P, T
      SA     : aerosol surface area density (um^2/cm^3), typically 1-5
      WTR    : water vapour (ppm) -- used by the O(1D) photolysis workaround
      Yn2o5  : uptake coefficient (gamma) for N2O5 + H2O, set manually (~0.05-0.2)
      SZA    : solar zenith-angle flag, 1 = daytime, 0 = nighttime (toggles photolysis)
      opt    : O3-photolysis mode. 1 = "Science paper" O(1D) workaround (default),
               0 = full O(1D) chemistry with reduced absolute tolerance
      count  : diagnostic counter of right-hand-side evaluations (parity with MATLAB)
    """

    T: float = 210.0
    P: float = 68.0
    SA: float = 2.0
    WTR: float = 5.0
    Yn2o5: float = 0.1
    SZA: int = 1
    opt: int = 1
    M: float | None = None
    count: int = field(default=0)
    # Photolysis handling. "reference" uses the fixed 45-deg J-values with the day/night
    # flag SZA (the validated MATLAB behaviour). "sza" scales J by the real solar zenith
    # angle computed from the location/date/time below.
    photolysis: str = "reference"
    latitude: float = 0.0       # deg North   (only used when photolysis == "sza")
    longitude: float = 0.0      # deg East
    day_of_year: float = 80.0
    start_utc_hour: float = 6.0  # UTC hour at model time t = 0

    #: photolysis modes understood by the model (validated so a typo can't silently mis-gate,
    #: e.g. quietly enabling the non-reference sulfur chain -- see reactions.build_env).
    PHOTOLYSIS_MODES = ("reference", "sza", "tuvx")

    def __post_init__(self) -> None:
        # Match MATLAB: M is derived from P and T unless explicitly provided.
        if self.M is None:
            self.M = air_number_density(self.P, self.T)
        if self.photolysis not in self.PHOTOLYSIS_MODES:
            raise ValueError(
                f"photolysis must be one of {self.PHOTOLYSIS_MODES}, got {self.photolysis!r}")
