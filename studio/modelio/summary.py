# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""``RunSummary`` -- the versioned, queryable reduction of a run.

Comparison views and figures read this; they never read the raw ``state.npz`` (ADR-004). The npz is
3.6 MB per case, carries 36 gas species and an (n_times, n_bins) size distribution, and knows
nothing about which of its arrays are wet and which are dry. A summary that fixes those three
things is what makes 810 runs comparable without opening 810 archives.

**Every array declares its basis.** In ``state.npz``, ``dp_mid_um`` and ``dNdlogDp`` are DRY
diameters while ``SA`` and ``radius_cm`` are WET -- the same file, no labelling, and the difference
is a factor of a few in radius at stratospheric humidity. Here it is a required field on every
series, so a plot axis cannot be labelled by guesswork.

Two more traps encoded rather than documented:

* **Species are indexed by NAME** from the npz's own ``species`` list. An existing analysis script
  hard-codes ``_SO2, _SO3, _H2SO4 = 32, 34, 35``; if the mechanism ever gains a species, that script
  silently plots the wrong one.
* **The time axis comes from the stored ``t``**, never from ``i * DT``. Outer steps snap to the
  terminator, so the mean step is ~592 s against a nominal 600 s -- about half a day of drift by day
  36.

This module does not import ``coupled``; it only reads arrays. It lives here because reading the
model's output format is the model seam's job, not because it needs the model.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

#: Bumped whenever the reduction changes shape or meaning. Stored in every summary so a comparison
#: view can refuse to plot two runs reduced under different rules rather than plotting them anyway.
SUMMARY_SCHEMA_VERSION = "0.2.0"

#: Seconds per day, for the time axis. Named rather than inline (studio/CLAUDE.md).
SECONDS_PER_DAY = 86400.0


class Basis(StrEnum):
    """Whether a quantity is on a dry or an ambient (wet) basis. Required on every series."""

    #: Particle without its water. ``dp_mid_um`` and ``dNdlogDp`` in ``state.npz``.
    DRY = "dry"
    #: Includes condensed water at ambient conditions. ``SA`` and ``radius_cm`` in ``state.npz``.
    WET = "wet"
    #: Not a particle-size quantity, so the distinction does not apply (gas mixing ratios, T, V/V0).
    NOT_APPLICABLE = "not_applicable"


class TerminationReason(StrEnum):
    """Why a run stopped. Never inferred from the data."""

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TERMINATED_ON_LIMIT = "terminated_on_limit"
    #: No record exists. The archived ensemble predates provenance capture (ADR-006), so a summary
    #: built from one of its npz files is UNKNOWN rather than assumed to have completed.
    UNKNOWN = "unknown"


class SummaryFlag(StrEnum):
    """Machine-readable caveats that must travel with the numbers."""

    #: Reduced from an archived npz with no provenance record: no config hash, no model version.
    NO_PROVENANCE_RECORD = "no_provenance_record"
    #: Dilution was active, so the box is an open system and sulfur is not expected to be conserved.
    OPEN_SYSTEM_DILUTION = "open_system_dilution"
    #: The run stopped on a limit rather than converging. Never presented as a converged result.
    STOPPED_ON_LIMIT = "stopped_on_limit"


class Series(BaseModel):
    """One scalar time series, with everything needed to plot it honestly."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    values: tuple[float, ...]
    unit: str
    basis: Basis
    description: str


class SizeDistribution(BaseModel):
    """The final size spectrum. Dry diameters, and it says so."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    diameter_um: tuple[float, ...]
    dn_dlogdp_cm3: tuple[float, ...]
    number_cm3: tuple[float, ...]
    basis: Basis = Basis.DRY
    #: Total number concentration, i.e. sum of the per-bin counts -- not of dN/dlogDp.
    total_number_cm3: float


class SizeDistributionHistory(BaseModel):
    """The number spectrum over time: dN/dlogDp on (time x bin), DRY diameters.

    Added in summary 0.2.0, prompted by review: the final spectrum alone hides nucleation, because
    by the end of a run the burst has grown and coagulated out of the small bins -- the reviewer
    read that as "no nucleation" when the npz showed dN/dlogDp peaking at 1.9e7 cm^-3 twelve hours
    in. This carries what a banana plot and a time slider need.

    Time is decimated by a UNIFORM stride (never a coarsening grid -- CAVEATS.md: non-uniform
    resampling aliases the morning number spikes by up to 8x), capped so a 60-day run stays a few
    hundred kB of JSON. ``stride`` says what was kept, so nobody mistakes the sampling for the
    model's own resolution.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    time_days: tuple[float, ...]
    diameter_um: tuple[float, ...]
    #: Row i is dN/dlogDp [cm^-3] across the bins at time_days[i].
    dn_dlogdp_cm3: tuple[tuple[float, ...], ...]
    basis: Basis = Basis.DRY
    #: Every ``stride``-th stored step was kept (1 = everything).
    stride: int


class ConservationCheck(BaseModel):
    """A budget check, or an explicit statement that it does not apply.

    The honest part is ``status``. In-box sulfur is **not** expected to be conserved when dilution
    is on: the box is an open system and entrainment removes plume sulfur while adding background
    sulfur. Reporting a large "residual" for a diluting run would be reporting the dilution, and
    reporting a small one would mean something had gone wrong. So the check is computed only when
    the run is closed, and otherwise says why not -- rather than producing a number that invites a
    reader to draw a conclusion from it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str
    #: ``(S_end - S_start) / S_start`` for a closed box; ``None`` when not applicable.
    relative_residual: float | None = None
    initial_value: float | None = None
    final_value: float | None = None
    unit: str = "molec cm^-3"
    reason: str = ""


class RunSummary(BaseModel):
    """The reduction of one run. Versioned, and self-describing about basis and provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SUMMARY_SCHEMA_VERSION
    #: Identity of the config that produced this run (ADR-006). ``None`` for archived runs, which
    #: have no provenance record -- flagged rather than left ambiguous.
    config_hash: str | None = None
    label: str | None = None
    termination: TerminationReason = TerminationReason.UNKNOWN
    flags: tuple[SummaryFlag, ...] = ()
    time_days: tuple[float, ...] = ()
    series: dict[str, Series] = Field(default_factory=dict)
    final_size_distribution: SizeDistribution | None = None
    size_distribution_history: SizeDistributionHistory | None = None
    sulfur_conservation: ConservationCheck | None = None

    def write(self, path: Path) -> Path:
        """Write as JSON next to the run's ``state.npz``."""
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: Path) -> RunSummary:
        """Read one back, validating it against this version of the schema."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


#: Gas species reduced to mixing-ratio series, by NAME. Extending this list is the supported way to
#: add a series; indexing by position is how the existing analysis scripts got it wrong.
_GAS_SERIES: tuple[tuple[str, str], ...] = (
    ("SO2", "sulfur dioxide, the injected species"),
    ("SO3", "sulfur trioxide, the intermediate"),
    ("H2SO4", "gas-phase sulfuric acid, the condensable"),
    ("OH", "hydroxyl radical, the oxidant that starts the chain"),
    ("HO2", "hydroperoxyl radical"),
    ("O3", "ozone"),
)

#: Aerosol/environment series carried straight through, with the basis each one is actually on.
_DIRECT_SERIES: tuple[tuple[str, str, Basis, str], ...] = (
    ("SA", "um^2 cm^-3", Basis.WET, "aerosol surface area density (wet)"),
    ("radius_cm", "cm", Basis.WET, "effective particle radius (wet)"),
    ("h2so4wp", "1", Basis.NOT_APPLICABLE, "H2SO4 weight fraction of the aerosol"),
    ("particulate_S", "molec cm^-3", Basis.NOT_APPLICABLE, "sulfur held in the particle phase"),
    ("T", "K", Basis.NOT_APPLICABLE, "box temperature"),
    ("total_n", "cm^-3", Basis.DRY, "total particle number concentration"),
    ("V_ratio", "1", Basis.NOT_APPLICABLE, "plume volume expansion V(t)/V0"),
)


def _particle_mass_series(data: dict[str, Any]) -> Series | None:
    """Dry particulate mass as H2SO4-equivalent [ug m^-3], from the stored sulfur count.

    ``particulate_S`` is molecules of S per cm^3 in the particle phase. Under ASSUMPTION-8 (all
    aerosol is pure sulfate) every particulate S atom sits in one H2SO4 unit, so mass follows from
    the molar mass and Avogadro alone -- a unit conversion, not new physics, which is why it is
    allowed to live here. DRY mass: the water the wet particle carries is deliberately excluded,
    and the name says so.

        ug/m^3 = molec/cm^3 * (M_H2SO4 / N_A) [g] * 1e6 [ug/g] * 1e6 [cm^3/m^3]
    """
    if "particulate_S" not in data:
        return None
    from studio.science.constants import AVOGADRO, H2SO4_MOLAR_MASS_G_PER_MOL

    count = np.asarray(data["particulate_S"], dtype=np.float64)
    mass = count * (H2SO4_MOLAR_MASS_G_PER_MOL / AVOGADRO) * 1e12
    return Series(
        values=tuple(float(v) for v in mass),
        unit="ug m^-3",
        basis=Basis.DRY,
        description=(
            "particulate mass as dry H2SO4-equivalent (ASSUMPTION-8: pure sulfate; excludes "
            "aerosol water)"
        ),
    )


#: Species whose number density IS its sulfur content -- each carries exactly one S atom. Used for
#: the closed-box budget check. ``particulate_S`` is added separately; it is already a sulfur count.
_SULFUR_GAS_SPECIES = ("SO2", "SO3", "H2SO4")


def summarise_state_npz(
    path: Path,
    *,
    label: str | None = None,
    config_hash: str | None = None,
    termination: TerminationReason = TerminationReason.UNKNOWN,
) -> RunSummary:
    """Reduce a ``state.npz`` to a :class:`RunSummary`.

    ``termination`` is an argument rather than something inferred from the arrays: the npz records
    what happened to the state, not why the loop stopped. A run that hit a wall-clock limit and one
    that finished look identical here, and guessing would be the difference between "converged" and
    "cut short" (ADR-005).
    """
    with np.load(path, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}

    missing = {"t", "x", "species", "M"} - set(data)
    if missing:
        raise ValueError(f"{path} is missing required arrays {sorted(missing)}")

    # From the STORED t, never i * DT: outer steps snap to the terminator (~592 s vs 600 s nominal).
    time_s = np.asarray(data["t"], dtype=np.float64)
    air_number_density = float(data["M"])
    species = [str(name) for name in data["species"]]
    state = np.asarray(data["x"], dtype=np.float64)

    series: dict[str, Series] = {}
    for name, description in _GAS_SERIES:
        if name not in species:
            continue  # a mechanism without this species is not an error, just fewer series
        column = state[:, species.index(name)]  # BY NAME. Never by position.
        series[name] = Series(
            values=tuple(column / air_number_density * 1.0e12),
            unit="pptv",
            basis=Basis.NOT_APPLICABLE,
            description=description,
        )
    for key, unit, basis, description in _DIRECT_SERIES:
        if key in data:
            series[key] = Series(
                values=tuple(np.asarray(data[key], dtype=np.float64)),
                unit=unit,
                basis=basis,
                description=description,
            )

    flags: list[SummaryFlag] = []
    if config_hash is None:
        flags.append(SummaryFlag.NO_PROVENANCE_RECORD)
    if termination is TerminationReason.TERMINATED_ON_LIMIT:
        flags.append(SummaryFlag.STOPPED_ON_LIMIT)

    diluting = "V_ratio" in data and float(np.max(np.asarray(data["V_ratio"]))) > 1.0
    if diluting:
        flags.append(SummaryFlag.OPEN_SYSTEM_DILUTION)

    mass_series = _particle_mass_series(data)
    if mass_series is not None:
        series["particle_mass_ug_m3"] = mass_series

    # Particulate sulfur as a mixing ratio, converted with the SAME air number density the gas
    # series use -- so "H2SO4 in gas and in particles" is one quantity on one axis (pptv), instead
    # of pptv and molec cm^-3 sharing a scale, which is the dual-axis lie in disguise. The raw
    # molec cm^-3 series stays alongside for anyone comparing against the npz.
    if "particulate_S" in data:
        particulate = np.asarray(data["particulate_S"], dtype=np.float64)
        series["particulate_S_pptv"] = Series(
            values=tuple(float(v) for v in particulate / air_number_density * 1e12),
            unit="pptv",
            basis=Basis.NOT_APPLICABLE,
            description="particle-phase sulfur as a mixing ratio (per S atom, ASSUMPTION-8)",
        )

    return RunSummary(
        config_hash=config_hash,
        label=label,
        termination=termination,
        flags=tuple(flags),
        time_days=tuple(time_s / SECONDS_PER_DAY),
        series=series,
        final_size_distribution=_final_size_distribution(data),
        size_distribution_history=_size_distribution_history(data),
        sulfur_conservation=_sulfur_budget(data, species, state, diluting=diluting),
    )


#: Cap on kept time samples in the history. 240 keeps a 60-day run's spectrum near 350 kB of JSON
#: while a 10-minute-step 1-day run (147 steps) passes through whole.
_HISTORY_MAX_SAMPLES = 240


def _size_distribution_history(data: dict[str, Any]) -> SizeDistributionHistory | None:
    """The (time x bin) spectrum, uniformly strided to at most ``_HISTORY_MAX_SAMPLES`` rows."""
    if "dNdlogDp" not in data or "dp_mid_um" not in data or "t" not in data:
        return None
    spectrum = np.asarray(data["dNdlogDp"], dtype=float)
    times = np.asarray(data["t"], dtype=float) / 86400.0
    stride = max(1, int(np.ceil(len(times) / _HISTORY_MAX_SAMPLES)))
    kept = slice(None, None, stride)
    return SizeDistributionHistory(
        time_days=tuple(float(v) for v in times[kept]),
        diameter_um=tuple(float(v) for v in np.asarray(data["dp_mid_um"], dtype=float)),
        dn_dlogdp_cm3=tuple(tuple(float(v) for v in row) for row in spectrum[kept]),
        stride=stride,
    )


def _final_size_distribution(data: dict[str, Any]) -> SizeDistribution | None:
    """The last spectrum. ``dp_mid_um`` and ``dNdlogDp`` are DRY (see the module docstring)."""
    if not {"dp_mid_um", "dNdlogDp", "n_cm3"} <= set(data):
        return None
    diameters = np.asarray(data["dp_mid_um"], dtype=np.float64)
    dn_dlogdp = np.asarray(data["dNdlogDp"], dtype=np.float64)[-1]
    counts = np.asarray(data["n_cm3"], dtype=np.float64)[-1]
    return SizeDistribution(
        diameter_um=tuple(diameters),
        dn_dlogdp_cm3=tuple(dn_dlogdp),
        number_cm3=tuple(counts),
        total_number_cm3=float(counts.sum()),
    )


def _sulfur_budget(
    data: dict[str, Any], species: list[str], state: np.ndarray, *, diluting: bool
) -> ConservationCheck:
    """In-box sulfur at the start and end, and a residual only when the box is closed."""
    present = [name for name in _SULFUR_GAS_SPECIES if name in species]
    if not present:
        return ConservationCheck(
            status="not_applicable",
            reason="the mechanism carries none of SO2, SO3, H2SO4",
        )
    total = np.zeros(state.shape[0], dtype=np.float64)
    for name in present:
        total = total + state[:, species.index(name)]
    if "particulate_S" in data:
        total = total + np.asarray(data["particulate_S"], dtype=np.float64)
    initial, final = float(total[0]), float(total[-1])

    if diluting:
        return ConservationCheck(
            status="not_applicable",
            initial_value=initial,
            final_value=final,
            reason=(
                "dilution was active (V(t)/V0 > 1), so the box is an open system: entrainment "
                "removes plume sulfur and adds background sulfur. A residual here would be a "
                "measure of the dilution, not of conservation. Start and end values are reported "
                "so the decay is visible."
            ),
        )
    if initial == 0.0:
        return ConservationCheck(
            status="not_applicable",
            initial_value=initial,
            final_value=final,
            reason="no sulfur at t = 0, so a relative residual is undefined",
        )
    return ConservationCheck(
        status="computed",
        relative_residual=(final - initial) / initial,
        initial_value=initial,
        final_value=final,
        reason="closed box (no dilution): total sulfur should be conserved",
    )


__all__ = [
    "SECONDS_PER_DAY",
    "SUMMARY_SCHEMA_VERSION",
    "Basis",
    "ConservationCheck",
    "RunSummary",
    "Series",
    "SizeDistribution",
    "SummaryFlag",
    "TerminationReason",
    "summarise_state_npz",
]
