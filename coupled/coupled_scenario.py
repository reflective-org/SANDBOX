# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""CoupledScenario -- the single input for the coupled SANDBOX model.

One editable object (loadable from one YAML/JSON) describing a coupled run: environment,
location/date, run schedule (including the operator-split coupling step ``dt_couple``), the
photolysis mode, the initial gas composition, and per-process **switches**. Later phases (TOMAS
microphysics, aerosol->photolysis radiation, radiative heating, dilution) read their switch here.

It describes the PHYSICS, not the bookkeeping: there is no output path, because ``run_coupled``
returns arrays and writes nothing -- the caller owns where results land.

Phase 2 scaffolding: only the gas chemistry + photolysis are wired. Switches for not-yet-implemented
processes must stay OFF (enabling one raises, so a config can never silently claim a capability the
model doesn't have yet -- the same "no silent assumptions" guard as the photolysis-mode validation).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

# Deliberately the JAX-free table module, NOT ``coupled.tomas_bridge``: this dataclass is constructed
# by form validation and must stay cheap to import and to build (see coupled/backgrounds.py).
# Absolute (not relative) because this module is also imported FLAT as ``coupled_scenario`` with
# coupled/ on sys.path -- see coupled/conftest.py -- where a relative import has no package to resolve.
from coupled.backgrounds import (BACKGROUND_MODES, MODE_BASES, TABULATED_BACKGROUND,
                                 normalize_modes)

#: Sheared-Gaussian dilution case names (paper runs v2). Mirrors ``coupled.dilution.GAUSSIAN_CASES``
#: without importing it (that pulls in tomas_jax; this module must stay cheap) -- a test keeps the
#: two in sync.
GAUSSIAN_DILUTION_CASES = ("C1", "C2", "C3", "C3a", "C4", "C4a", "C5", "C5a")

#: LEGACY Schumann dilution regimes (v1; removed from the model and the paper by PI decision L20).
#: Accepted only with ``legacy_dilution=True``; mirrors ``coupled.dilution_legacy_schumann.
#: DILUTION_REGIMES`` (a test keeps the two in sync).
LEGACY_SCHUMANN_REGIMES = ("D1", "D2", "D3", "D5", "burst")

#: Photolysis drivers understood by the model (validated -> no silent mis-gate of the sulfur chain).
PHOTOLYSIS_MODES = ("reference", "sza", "tuvx")

#: Process switches that are actually wired today. Phase 3 wired the TOMAS microphysics trio; Phase 4
#: aerosol_to_j; Phase 5 heating_to_t; Phase 6 dilution -- so all switches are now implemented.
_IMPLEMENTED_SWITCHES = frozenset(
    {"sulfur", "nucleation", "condensation", "coagulation", "aerosol_to_j", "heating_to_t", "dilution"})

#: Fields that used to exist. Loading an archived config that still carries one must say what
#: happened, not just "unknown key" -- the config was valid when it was written.
_REMOVED_FIELDS = {
    "output_dir": ("removed -- the driver never read it. ``run_coupled`` returns arrays and writes "
                   "nothing; the CALLER chooses where to save the .npz. Drop the key and pass the "
                   "path to whatever writes the output."),
}


@dataclass
class Switches:
    """Per-process on/off flags for the coupled system (see CoupledScenario)."""

    # NOTE: `sulfur` is NOT yet honored by any driver (Phase 2 has no coupled driver). Today the
    # sulfur chain is gated by photolysis mode (Env.sulfur_chain = photolysis != "reference").
    # Phase 2.2 makes THIS switch the single source of truth for the gate; until then it is
    # descriptive only. Keep photolysis mode and this flag consistent to avoid confusion.
    # All processes ON by default: the coupled system is the full physics. Individual processes are
    # turned OFF (or scaled via the sensitivity knobs) only for deliberate sub-case / sensitivity runs.
    # (The nucleation "runaway" that once motivated defaults-off was a coarse-outer-step operator-split
    # artifact; the two-level adaptive micro-stepping driver resolves it, so full physics is safe.)
    sulfur: bool = True            # gas-phase SO2->SO3->H2SO4 chain (Phase 1)
    nucleation: bool = True        # TOMAS (Phase 3)
    condensation: bool = True      # TOMAS (Phase 3)
    coagulation: bool = True       # TOMAS (Phase 3)
    aerosol_to_j: bool = True      # aerosol optics -> TUV-x radiation (Phase 4)
    # CAUTION: the heating term is SW-only -- no longwave cooling yet (AD-5.4) -- so leaving it on
    # gives a ONE-SIDED ~+1.2 K / 10 d warm drift, not a complete energy balance. Science scenarios
    # (e.g. the D1 run script) turn it off until LW cooling lands.
    heating_to_t: bool = True      # radiative heating -> box temperature (Phase 5; SW-only, AD-5.4)
    dilution: bool = True          # dilution + background entrainment (Phase 6)

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
    # Outer radiation/coupling step (s): the cadence at which the (expensive) TUV-x J and aerosol
    # optics are recomputed and frozen. Within each outer step the gas<->TOMAS<->dilution coupling is
    # resolved on an ADAPTIVE fine->coarse micro-step (see micro_* below); the outer step no longer
    # needs to be tiny. Exposed as ``dt_rad`` (alias property). 600 s default; 300 s also fine.
    dt_couple: float = 600.0

    # --- adaptive inner (micro) time step for the gas<->TOMAS handoff (two-level integration) ---
    # Each micro-step is chosen so the gas H2SO4 PRODUCED over the step stays below ``micro_eps`` of
    # the standing H2SO4 (production/standing <= micro_eps; the operator-split coupling error), bounded
    # to [``micro_floor_s``, ``micro_cap_s``]: halve when over eps, double when under eps/2. Particle
    # number N is NOT controlled by default (see coupled.driver.micro_consume for why). Reaching the
    # floor while still over eps ACCEPTS the (finite) step with a printed note; it raises only if
    # TOMAS goes non-finite at the floor.
    micro_eps: float = 0.1
    micro_floor_s: float = 1.0e-4
    micro_cap_s: float = 20.0
    # Opt-in nucleation-aware control (default OFF = bitwise the production-only controller above):
    # with ``micro_nuc_eps`` > 0 a micro-step is also refined until the fractional INCREASE of total
    # particle number per accepted step, dN / max(N, micro_nuc_nref_cm3), is <= micro_nuc_eps. The
    # reference floor (#/cm^3, ambient) keeps dN/N from refining to an infeasible step count while N
    # is tiny (clean box before the burst): 1e3 cm^-3 is ~2 orders above the stratospheric background
    # (~1-30 cm^-3), so the control engages only once nucleation adds a population that matters.
    micro_nuc_eps: float = 0.0
    micro_nuc_nref_cm3: float = 1.0e3
    # Production-fraction denominator: "envelope" (default, v1: the sink-free envelope, which lets
    # the step reach micro_cap_s regardless of the condensation sink) or "standing" (the gas H2SO4
    # actually standing, so micro_eps bounds the uptake fraction ~CS*dt). See micro_consume.
    micro_prod_ref: str = "envelope"
    # When a step's H2SO4 production reaches TOMAS: "end" (default, v1: all up front, first-order
    # split) or "mid" (envelope at the step midpoint: symmetric Strang split, second order).
    micro_handoff: str = "end"

    # --- photolysis ---
    photolysis: str = "tuvx"

    # --- dilution (Phase 6) ---
    # First-order relaxation rate [1/s] toward the background when switches.dilution is on and
    # dilution_regime is "" (constant rate; AD-6.2). Ignored when a regime is set.
    dilution_rate: float = 1.157e-6
    # Time-varying dilution regime "" (constant) | a sheared-Gaussian case C1..C5a, from the plume
    # volume expansion V(t)/V0 (coupled/dilution.GAUSSIAN_CASES; paper runs v2). The v1 Schumann
    # regimes D1 | D2 | D3 | D5 | burst are LEGACY (L20) and accepted only with legacy_dilution=True.
    dilution_regime: str = ""
    # Opt in to the legacy Schumann regimes (coupled/dilution_legacy_schumann.py; kept for the studio).
    # False (default): a Schumann dilution_regime is rejected.
    legacy_dilution: bool = False
    # Max ln(V(t1)/V(t0)) per outer interval: intervals where the plume expands faster are bisected
    # (and a Kz burst's start/end become edges) so the operator-split dilution step stays small.
    # 0 (default) = off, i.e. plain dt_couple intervals as in the v1 paper runs. v2 uses 0.1.
    dilution_max_dlnv: float = 0.0
    # Gas species zeroed in the dilution BACKGROUND (entrained air); all others keep their initial
    # value. Empty () => background is the full initial state (AD-6.3). Names must be valid species.
    dilution_zero_species: tuple = ()
    # Explicit background mixing ratios (pptv) for the dilution background, applied AFTER
    # dilution_zero_species (an override wins over a zero). Species not listed keep the
    # zero-species/initial-state background. E.g. {"SO2": 15.0} = clean stratosphere with 15 ppt SO2.
    dilution_background: dict = field(default_factory=dict)

    # Ion-pair production rate [pairs/cm^3/s] for the Dunne-2016 ion-induced nucleation channels
    # (TOMAS ``fion``). 0 (default) disables ion-induced nucleation (neutral channels unaffected);
    # galactic-cosmic-ray values at ~20 km are O(10) pairs/cm^3/s.
    ion_pair_rate: float = 0.0
    # TOMAS size resolution: 40 (standard, mass-doubling) or 80 (high-res, sqrt(2) mass ratio);
    # both span dry Dp 1.7 nm - 17.5 um. Everything downstream (initial state, Mie table, optics)
    # follows the state's own xk grid.
    tomas_nbins: int = 40
    # Background aerosol size distribution seeded into the initial TomasState. Either a NAME --
    # "redcircles" (Marianna, default) uses the tabulated loader, "sabr_330"/"sabr_220"/"cesm_g6"/...
    # seed a (multi-)lognormal from coupled.backgrounds.BACKGROUND_MODES (digitized from SABR/CESM
    # plots -- see paper_ensemble) -- or a USER-SUPPLIED list of lognormal modes
    # [(N [cm^-3], Dg [um], sigma_g), ...] on a DIAMETER basis, normalized to a tuple of tuples.
    background_dist: str | tuple = TABULATED_BACKGROUND
    # Number basis of a USER-SUPPLIED background_dist mode list: "stp" or "ambient". REQUIRED for a
    # mode list and REJECTED for a name (the named sets carry their own basis via
    # backgrounds.AMBIENT_BACKGROUNDS). Not defaulted: the STP->ambient factor is ~0.09 at
    # 68 mbar/210 K, so a wrong basis is an order-of-magnitude error in N (see backgrounds.py).
    background_modes_basis: str = ""
    # Rate constant [cm^3/molec/s] for SO2 + HO2 -> SO3 + OH (JPL 19-5 I34). JPL gives only an UPPER
    # LIMIT (~1e-18) and recommends NO products, so this is a deliberate sensitivity knob: 0.0
    # eliminates the channel; 1e-18/1e-17/1e-16 scan the plausible range. Only active in the sulfur
    # chain (non-reference photolysis). Default 1e-18 = the current hardcoded value (no behavior change).
    so2_ho2_rate: float = 1.0e-18

    # --- sensitivity knobs (Phase 7; free multipliers, default 1.0; for Phase-8 sweeps) ---
    nucleation_rate_scale: float = 1.0   # -> TOMAS make_step nucleation fn_scale
    condensation_alpha: float = 1.0      # -> TomasState alpha (Fuchs accommodation coefficient), (0,1]
    coag_kernel_scale: float = 1.0       # -> TOMAS make_step kernel multiplier (AD-7.2)
    # --- TOMAS process options (paper runs v2; defaults reproduce the v1 runs) ---
    water_every_process: bool = False    # re-equilibrate aerosol water after every process (box.f order)
    nuc_org_mass_frac: float = 0.1       # share of nucleated cluster mass put in the organic slot; 0 = all SO4
    # Conserve MOLES of sulfur at every gas->particle sulfate transfer (PI decision L13). TOMAS's particle
    # slot Mk[:,SRTSO4] holds SO4 mass (MW 96) while the gas Gc[SRTSO4] holds H2SO4 (MW 98): True adds
    # 96/98 kg SO4 per kg H2SO4 condensed/nucleated (TOMAS make_step molar_so4_transfer) AND makes the
    # coupled readers of that slot (particulate_S, h2so4wp) use MW 96. False = the kg-for-kg v1 behaviour
    # (slot read as H2SO4-equivalent mass, AD-3.8), bitwise.
    molar_so4_transfer: bool = False

    # --- aerosol -> photolysis (Phase 4) ---
    # Where the box aerosol sits in the TUV-x RT column when switches.aerosol_to_j is on. By DEFAULT
    # it is PRESSURE-ANCHORED: a plume of vertical extent ``aerosol_thickness_km`` centered on the box
    # altitude (derived from the input pressure P), so it tracks P and doesn't need an absolute km
    # window. ``aerosol_band_km`` (if set) overrides this with an ABSOLUTE (lo, hi) km band.
    aerosol_thickness_km: float = 1.0          # plume vertical extent (km), anchored on the box altitude
    aerosol_band_km: tuple | None = None       # optional ABSOLUTE (lo, hi) km override; None -> anchored

    # --- switches ---
    # (No output path here: ``run_coupled`` returns arrays and writes nothing -- see _REMOVED_FIELDS.)
    switches: Switches = field(default_factory=Switches)

    # --- initial gas composition (pptv); species omitted start at 0 ---
    concentrations: dict = field(default_factory=dict)

    @property
    def dt_rad(self) -> float:
        """Alias for ``dt_couple`` -- the outer radiation/coupling step (J recompute cadence)."""
        return self.dt_couple

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
        if not (0.0 < self.micro_eps <= 1.0):
            raise ValueError(f"micro_eps must be in (0, 1], got {self.micro_eps}")
        if self.micro_floor_s <= 0.0:
            raise ValueError(f"micro_floor_s must be > 0, got {self.micro_floor_s}")
        if self.micro_cap_s < self.micro_floor_s:
            raise ValueError(f"micro_cap_s ({self.micro_cap_s}) must be >= micro_floor_s "
                             f"({self.micro_floor_s})")
        if not self.micro_nuc_eps >= 0.0:            # also rejects NaN
            raise ValueError(f"micro_nuc_eps must be >= 0 (0 = off), got {self.micro_nuc_eps}")
        if self.micro_prod_ref not in ("envelope", "standing"):
            raise ValueError(f"micro_prod_ref must be 'envelope' or 'standing', got "
                             f"{self.micro_prod_ref!r}")
        if self.micro_handoff not in ("end", "mid"):
            raise ValueError(f"micro_handoff must be 'end' or 'mid', got {self.micro_handoff!r}")
        if not self.micro_nuc_nref_cm3 > 0.0:
            raise ValueError(f"micro_nuc_nref_cm3 must be > 0 (denominator floor of dN/N), got "
                             f"{self.micro_nuc_nref_cm3}")
        if self.days < 1:
            raise ValueError(f"days must be >= 1, got {self.days}")
        if self.aerosol_thickness_km <= 0.0:
            raise ValueError(f"aerosol_thickness_km must be > 0, got {self.aerosol_thickness_km}")
        if self.aerosol_band_km is not None:   # optional absolute override -> validate if given
            band = tuple(float(v) for v in self.aerosol_band_km)   # YAML/JSON give a list
            if len(band) != 2 or band[0] >= band[1] or band[0] < 0.0:
                raise ValueError(f"aerosol_band_km must be (lo, hi) km with 0 <= lo < hi, got {band}")
            self.aerosol_band_km = band
        if self.dilution_rate < 0.0:
            raise ValueError(f"dilution_rate must be >= 0, got {self.dilution_rate}")
        self.dilution_zero_species = tuple(self.dilution_zero_species)   # YAML list -> tuple
        if self.dilution_regime in LEGACY_SCHUMANN_REGIMES and not self.legacy_dilution:
            raise ValueError(
                f"dilution_regime {self.dilution_regime!r} is a legacy Schumann regime, removed from "
                f"the coupled model (PI decision L20); use a sheared-Gaussian case "
                f"{GAUSSIAN_DILUTION_CASES}, or set legacy_dilution=True to run it through "
                f"coupled/dilution_legacy_schumann.py")
        _VALID_REGIMES = ("", *GAUSSIAN_DILUTION_CASES,
                          *(LEGACY_SCHUMANN_REGIMES if self.legacy_dilution else ()))
        if self.dilution_regime not in _VALID_REGIMES:
            raise ValueError(f"dilution_regime must be one of {_VALID_REGIMES}, got {self.dilution_regime!r}")
        if self.dilution_max_dlnv < 0.0:
            raise ValueError(f"dilution_max_dlnv must be >= 0, got {self.dilution_max_dlnv}")
        if self.nucleation_rate_scale < 0.0:
            raise ValueError(f"nucleation_rate_scale must be >= 0, got {self.nucleation_rate_scale}")
        if not (0.0 < self.condensation_alpha <= 1.0):
            raise ValueError(f"condensation_alpha must be in (0, 1], got {self.condensation_alpha}")
        if not (0.0 <= self.nuc_org_mass_frac <= 1.0):
            raise ValueError(f"nuc_org_mass_frac must be in [0, 1], got {self.nuc_org_mass_frac}")
        if self.coag_kernel_scale < 0.0:   # now wired (AD-7.2): free multiplier on the coag kernel
            raise ValueError(f"coag_kernel_scale must be >= 0, got {self.coag_kernel_scale}")
        if isinstance(self.background_dist, str):
            if self.background_dist not in (TABULATED_BACKGROUND, *BACKGROUND_MODES):
                raise ValueError(f"background_dist must be {TABULATED_BACKGROUND!r}, one of "
                                 f"{sorted(BACKGROUND_MODES)}, or a list of (N, Dg, sigma_g) "
                                 f"lognormal modes, got {self.background_dist!r}")
            if self.background_modes_basis:
                raise ValueError(
                    f"background_modes_basis={self.background_modes_basis!r} applies only to a "
                    f"user-supplied background_dist mode list; the named distribution "
                    f"{self.background_dist!r} carries its own basis (backgrounds."
                    f"AMBIENT_BACKGROUNDS). Leave it empty.")
        else:                                   # user-supplied lognormal modes
            self.background_dist = normalize_modes(self.background_dist)
            if self.background_modes_basis not in MODE_BASES:
                raise ValueError(
                    f"a user-supplied background_dist mode list needs an explicit "
                    f"background_modes_basis, one of {list(MODE_BASES)}; got "
                    f"{self.background_modes_basis!r}. It is not defaulted because the STP->ambient "
                    f"factor is ~0.09 at 68 mbar/210 K -- guessing it is an order-of-magnitude "
                    f"error in the background number concentration.")
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
            removed = sorted(unknown & set(_REMOVED_FIELDS))
            if removed:   # name what happened rather than "unknown key" on a config that once worked
                raise ValueError("; ".join(f"{k}: {_REMOVED_FIELDS[k]}" for k in removed))
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
