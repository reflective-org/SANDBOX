"""Scenario: a single, editable description of a model run.

A ``Scenario`` collects everything needed to run the box model -- the environment
(temperature, pressure, aerosol, water), the location and date (for solar zenith angle),
the day/night schedule, and any initial-condition overrides -- in one object that can be
loaded from a YAML or JSON file. This lets a scientist set up a run by editing a config
file instead of touching code.

Example (YAML)::

    T: 210            # temperature (K)
    P: 68             # pressure (mbar) -- also selects the initial-condition preset
    SA: 2             # aerosol surface area (um^2/cm^3)
    WTR: 5            # water vapour (ppm)
    Yn2o5: 0.1        # N2O5 + H2O uptake coefficient
    opt: 1            # O3-photolysis mode (1 = Science-paper workaround)
    latitude: 20      # deg North (used by SZA-dependent photolysis)
    longitude: 0      # deg East
    day_of_year: 80
    td: 14            # daytime length (hours)
    tn: 10            # nighttime length (hours)
    days: 5
    DT: 600           # output time step (seconds)
    photolysis: reference   # 'reference' (fixed 45 deg day/night) or 'sza' (added in M5)
    initial_overrides:      # optional, pptv; merged onto the pressure-level preset
      HNO3: 3470
      BrO: 3
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from config import IDX, ModelConfig, air_number_density
from driver import initial_concentrations

# Photolysis modes understood by the model.
_REFERENCE = "reference"   # fixed 45-deg daytime J-values, on by day / off by night
_SZA = "sza"               # SZA-dependent scaling of the reference J-values
_TUVX = "tuvx"             # absolute per-reaction J from the TUV-x (JAX) port at the box altitude
_MODES = (_REFERENCE, _SZA, _TUVX)


@dataclass
class Scenario:
    # --- environment ---
    T: float = 210.0          # temperature (K)
    P: float = 68.0           # pressure (mbar); also selects the IC preset
    SA: float = 2.0           # aerosol surface area (um^2/cm^3)
    WTR: float = 5.0          # water vapour (ppm)
    Yn2o5: float = 0.1        # N2O5 + H2O uptake coefficient
    opt: int = 1              # O3-photolysis mode

    # --- location & date (for SZA-dependent photolysis) ---
    latitude: float = 0.0       # deg North
    longitude: float = 0.0      # deg East
    day_of_year: int = 80       # 1..365
    start_utc_hour: float = 6.0  # UTC hour at model time t = 0 (used in 'sza' mode)

    # --- run controls ---
    td: float = 14.0          # daytime length (hours)
    tn: float = 10.0          # nighttime length (hours)
    days: int = 5             # number of full day/night cycles after the first daytime
    DT: float = 600.0         # output time step (seconds)

    photolysis: str = _REFERENCE

    # --- initial conditions ---
    # Two ways to set the initial state, both in pptv (parts per trillion by volume):
    #   * initial_overrides: a few species merged onto the built-in pressure-level preset.
    #   * concentrations:    the FULL initial state, species -> pptv; species omitted start at 0.
    #                        If non-empty, this takes precedence over the preset + overrides,
    #                        making the file completely self-contained.
    initial_overrides: dict = field(default_factory=dict)
    concentrations: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.photolysis not in _MODES:
            raise ValueError(f"photolysis must be one of {_MODES}, got {self.photolysis!r}")
        for src in (self.initial_overrides, self.concentrations):
            for name in src:
                if name not in IDX:
                    raise ValueError(f"unknown species {name!r} in initial conditions")

    # ----------------------------------------------------------------------------------
    def to_config(self) -> ModelConfig:
        """The per-evaluation physical config the right-hand side needs."""
        return ModelConfig(T=self.T, P=self.P, SA=self.SA, WTR=self.WTR,
                           Yn2o5=self.Yn2o5, opt=self.opt,
                           photolysis=self.photolysis, latitude=self.latitude,
                           longitude=self.longitude, day_of_year=self.day_of_year,
                           start_utc_hour=self.start_utc_hour)

    def initial_state(self) -> np.ndarray:
        """Initial concentrations (molec/cm^3).

        If ``concentrations`` is given it defines the full state directly (pptv; omitted species = 0);
        otherwise the built-in pressure-level preset is used with ``initial_overrides`` merged in.
        """
        M = air_number_density(self.P, self.T)
        if self.concentrations:
            x0 = np.zeros(len(IDX))
            for name, ppt in self.concentrations.items():
                x0[IDX[name]] = float(ppt) * 1e-12 * M   # pptv -> molec/cm^3 (float() is robust
                #                                          to YAML unsigned-exponent strings)
            return x0
        x0 = initial_concentrations(self.P, M, self.WTR)
        for name, ppt in self.initial_overrides.items():
            x0[IDX[name]] = float(ppt) * 1e-12 * M   # pptv -> molec/cm^3
        return x0

    # ----------------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown scenario keys: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: str) -> "Scenario":
        """Load a Scenario from a .yaml/.yml or .json file."""
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
        """Save the Scenario to a .yaml/.yml or .json file."""
        ext = os.path.splitext(path)[1].lower()
        with open(path, "w") as f:
            if ext in (".yaml", ".yml"):
                import yaml
                yaml.safe_dump(self.to_dict(), f, sort_keys=False)
            elif ext == ".json":
                json.dump(self.to_dict(), f, indent=2)
            else:
                raise ValueError(f"Unsupported config extension {ext!r}; use .yaml or .json")
