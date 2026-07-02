# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""CoupledScenario -- the single input for the coupled SANDBOX model.

One editable object (loadable from one YAML/JSON) describing a coupled run: environment,
location/date, run schedule (including the operator-split coupling step ``dt_couple``), the
photolysis mode, the initial gas composition, and per-process **switches**. Later phases (TOMAS
microphysics, aerosol->photolysis radiation, radiative heating, dilution) read their switch here.

Phase 2 scaffolding: only the gas chemistry + photolysis are wired. Switches for not-yet-implemented
processes must stay OFF (enabling one raises, so a config can never silently claim a capability the
model doesn't have yet -- the same "no silent assumptions" guard as the photolysis-mode validation).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

#: Photolysis drivers understood by the model (validated -> no silent mis-gate of the sulfur chain).
PHOTOLYSIS_MODES = ("reference", "sza", "tuvx")

#: Process switches that are actually wired today. Phase 3 wired the TOMAS microphysics trio; Phase 4
#: aerosol_to_j; Phase 5 heating_to_t; Phase 6 dilution -- so all switches are now implemented.
_IMPLEMENTED_SWITCHES = frozenset(
    {"sulfur", "nucleation", "condensation", "coagulation", "aerosol_to_j", "heating_to_t", "dilution"})


@dataclass
class Switches:
    """Per-process on/off flags for the coupled system (see CoupledScenario)."""

    # NOTE: `sulfur` is NOT yet honored by any driver (Phase 2 has no coupled driver). Today the
    # sulfur chain is gated by photolysis mode (Env.sulfur_chain = photolysis != "reference").
    # Phase 2.2 makes THIS switch the single source of truth for the gate; until then it is
    # descriptive only. Keep photolysis mode and this flag consistent to avoid confusion.
    sulfur: bool = True            # gas-phase SO2->SO3->H2SO4 chain (Phase 1; active in non-reference)
    nucleation: bool = False       # TOMAS (Phase 3)
    condensation: bool = False     # TOMAS (Phase 3)
    coagulation: bool = False      # TOMAS (Phase 3)
    aerosol_to_j: bool = False     # aerosol optics -> TUV-x radiation (Phase 4)
    heating_to_t: bool = False     # radiative heating -> box temperature (Phase 5)
    dilution: bool = False         # dilution + background entrainment (Phase 6)

    def validate(self) -> None:
        on = {name for name, val in asdict(self).items() if val}
        not_yet = sorted(on - _IMPLEMENTED_SWITCHES)
        if not_yet:
            raise NotImplementedError(
                f"switch(es) {not_yet} are not implemented yet; keep them off until their phase "
                f"lands. Implemented now: {sorted(_IMPLEMENTED_SWITCHES)}.")


@dataclass
class CoupledScenario:
    # --- environment ---
    T: float = 210.0          # temperature (K)
    P: float = 68.0           # pressure (mbar)
    SA: float = 2.0           # aerosol surface area (um^2/cm^3) -- prescribed until TOMAS (Phase 3)
    WTR: float = 5.0          # water vapour (ppm)
    Yn2o5: float = 0.1        # N2O5 + H2O uptake coefficient
    opt: int = 1              # O3-photolysis mode (1 = Science-paper O(1D) workaround)

    # --- location & date ---
    latitude: float = 0.0
    longitude: float = 0.0
    day_of_year: int = 80
    start_utc_hour: float = 6.0

    # --- run schedule ---
    days: int = 10            # number of days to simulate
    DT: float = 600.0         # output time step (s)
    dt_couple: float = 120.0  # operator-split outer coupling step (s); drives the J recompute

    # --- photolysis ---
    photolysis: str = "tuvx"

    # --- dilution (Phase 6) ---
    # First-order relaxation rate [1/s] toward the initial (background) box state when
    # switches.dilution is on (AD-6.2; default ~ 1/(10 days)). The V(t) schedule is deferred.
    dilution_rate: float = 1.157e-6

    # --- sensitivity knobs (Phase 7; free multipliers, default 1.0; for Phase-8 sweeps) ---
    nucleation_rate_scale: float = 1.0   # -> TOMAS make_step nucleation fn_scale
    condensation_alpha: float = 1.0      # -> TomasState alpha (Fuchs accommodation coefficient), (0,1]
    coag_kernel_scale: float = 1.0       # NOT wired in tomas-jax yet (AD-7.2): must stay 1.0 (raises)

    # --- aerosol -> photolysis (Phase 4) ---
    # Altitude band (km) over which the box aerosol is spread in the TUV-x radiative-transfer column
    # when switches.aerosol_to_j is on (AD-4.2, FLAGGED OPEN -- the band sets the feedback magnitude).
    aerosol_band_km: tuple = (15.0, 25.0)

    # --- switches & output ---
    switches: Switches = field(default_factory=Switches)
    output_dir: str = "coupled_output"

    # --- initial gas composition (pptv); species omitted start at 0 ---
    concentrations: dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.switches, dict):           # allow a plain dict from YAML/JSON
            valid = set(Switches.__dataclass_fields__)
            unknown = set(self.switches) - valid
            if unknown:                                # friendly error (not a cryptic TypeError)
                raise ValueError(f"Unknown switch(es) {sorted(unknown)}; valid: {sorted(valid)}")
            self.switches = Switches(**self.switches)
        if self.photolysis not in PHOTOLYSIS_MODES:
            raise ValueError(f"photolysis must be one of {PHOTOLYSIS_MODES}, got {self.photolysis!r}")
        if self.dt_couple <= 0:
            raise ValueError(f"dt_couple must be > 0, got {self.dt_couple}")
        if self.days < 1:
            raise ValueError(f"days must be >= 1, got {self.days}")
        band = tuple(float(v) for v in self.aerosol_band_km)   # YAML/JSON give a list
        if len(band) != 2 or band[0] >= band[1] or band[0] < 0.0:
            raise ValueError(f"aerosol_band_km must be (lo, hi) km with 0 <= lo < hi, got {band}")
        self.aerosol_band_km = band
        if self.dilution_rate < 0.0:
            raise ValueError(f"dilution_rate must be >= 0, got {self.dilution_rate}")
        if self.nucleation_rate_scale < 0.0:
            raise ValueError(f"nucleation_rate_scale must be >= 0, got {self.nucleation_rate_scale}")
        if not (0.0 < self.condensation_alpha <= 1.0):
            raise ValueError(f"condensation_alpha must be in (0, 1], got {self.condensation_alpha}")
        if self.coag_kernel_scale != 1.0:   # not wired in tomas-jax -- fail loud, don't silently ignore
            raise NotImplementedError(
                "coag_kernel_scale is not wired yet (tomas-jax has no coagulation-kernel scale knob); "
                "it must stay 1.0 until that Phase-8/tomas-jax change lands (see AD-7.2, DEFERRED.md).")
        if self.dt_couple > self.DT:
            raise ValueError(f"dt_couple ({self.dt_couple}) must be <= output step DT ({self.DT})")
        # dt_couple drives sub-stepping within an output interval, so DT must be a whole multiple of it
        if abs(self.DT / self.dt_couple - round(self.DT / self.dt_couple)) > 1e-9:
            raise ValueError(f"DT ({self.DT}) must be an integer multiple of dt_couple ({self.dt_couple})")
        self.switches.validate()

    # ----------------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "CoupledScenario":
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown CoupledScenario keys: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: str) -> "CoupledScenario":
        ext = os.path.splitext(path)[1].lower()
        with open(path) as f:
            if ext in (".yaml", ".yml"):
                import yaml
                data = yaml.safe_load(f)
            elif ext == ".json":
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config extension {ext!r}; use .yaml or .json")
        return cls.from_dict(data or {})

    def save(self, path: str) -> None:
        ext = os.path.splitext(path)[1].lower()
        with open(path, "w") as f:
            if ext in (".yaml", ".yml"):
                import yaml
                yaml.safe_dump(self.to_dict(), f, sort_keys=False)
            elif ext == ".json":
                json.dump(self.to_dict(), f, indent=2)
            else:
                raise ValueError(f"Unsupported config extension {ext!r}; use .yaml or .json")
