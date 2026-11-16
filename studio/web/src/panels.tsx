// Copyright (C) 2026 University Corporation for Atmospheric Research
// SPDX-License-Identifier: Apache-2.0
/**
 * One preview panel per stage: what this configuration implies, before spending compute.
 *
 * Each panel plots numbers the **server** computed from the model's own implementation
 * (`studio/modelio/preview.py`). Nothing here derives physics -- the panels shape and label, and
 * that split is deliberate: a curve re-derived in the browser would eventually disagree with the
 * run, and it would still look exactly like a dilution curve while doing so.
 *
 * Two panels are the exception and say so: the parcel diagram (stage 2) draws the geometry the user
 * typed, and the species panel (stage 6) draws the mixing ratios they entered. Both are the config
 * itself, not a computation over it.
 */

import { useEffect, useState } from "react";
import { Chart, type Series } from "./Chart";
import { HelpTip } from "./HelpTip";
import { display, valueAt } from "./schema";
import { tickLabel } from "./scales";

interface PanelProps {
  config: Record<string, unknown>;
  /** Fetch a preview panel; supplied by App so panels do not each know about the API. */
  load: (
    panel: string,
    signal: AbortSignal,
    params?: Record<string, unknown>,
  ) => Promise<Record<string, unknown>>;
  /** Change a config field -- the same server round-trip every Field control uses. Panels use it
   *  for graph-adjacent controls like the dilution regime chips, so choosing on the graph IS
   *  choosing in the config. */
  onChange?: ((path: string, value: unknown) => void) | undefined;
}

/** Fetch-with-state, shared by every server-computed panel. */
function usePanel(
  name: string,
  config: Record<string, unknown>,
  load: PanelProps["load"],
  params?: Record<string, unknown>,
): { data: Record<string, unknown> | null; error: string; loading: boolean } {
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  // Keyed on the serialised config AND the panel parameters: the panel must follow every edit,
  // and an explorer input (the dilution k) is an edit to the picture even though it is not one to
  // the config.
  const key = JSON.stringify([config, params ?? null]);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    load(name, controller.signal, params)
      .then((payload) => {
        setData(payload);
        setError("");
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        // The first request in a server process imports JAX and takes about a second; anything that
        // fails after that is a real error and is shown rather than swallowed.
        setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [name, key, load]);
  return { data, error, loading };
}

function Frame({
  title,
  hint,
  help,
  loading,
  error,
  children,
}: {
  title: string;
  // `hint` is VISIBLE and for data (totals, extremes -- numbers the reader scans for). `help` is
  // the explanation, behind a CircleHelp hover per the design review: boxes hold controls and
  // numbers, not paragraphs. `| undefined` because exactOptionalPropertyTypes is on.
  hint?: string | undefined;
  help?: React.ReactNode;
  loading?: boolean | undefined;
  error?: string | undefined;
  children?: React.ReactNode;
}) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h3>
          {title}
          {help ? <HelpTip label={`about ${title}`}>{help}</HelpTip> : null}
        </h3>
        {loading ? <span className="panel-status">computing…</span> : null}
      </div>
      {hint ? <p className="panel-hint">{hint}</p> : null}
      {error ? <p className="chart-error">{error}</p> : null}
      {children}
    </section>
  );
}

const nums = (data: Record<string, unknown> | null, key: string): number[] =>
  Array.isArray(data?.[key]) ? (data[key] as number[]) : [];

/** Stage 1 — the sun over the configured day. Photolysis follows it, so the chemistry does too. */
export function SzaPanel({ config, load }: PanelProps) {
  const { data, error, loading } = usePanel("sza", config, load);
  const hours = nums(data, "hours");
  const sza = nums(data, "sza_deg");
  const daylight = typeof data?.daylight_hours === "number" ? data.daylight_hours : null;
  const releaseHour = typeof data?.release_hour === "number" ? data.release_hour : null;
  const releaseSza = typeof data?.release_sza_deg === "number" ? data.release_sza_deg : null;

  // Night is shaded rather than drawn: SZA past 90 degrees is the sun below the horizon, where the
  // exact angle stops meaning anything for photolysis.
  const nightBands: { from: number; to: number }[] = [];
  let start: number | null = null;
  hours.forEach((h, i) => {
    const dark = (sza[i] ?? 0) >= 90;
    if (dark && start === null) start = h;
    if ((!dark || i === hours.length - 1) && start !== null) {
      nightBands.push({ from: start, to: h });
      start = null;
    }
  });

  return (
    <Frame
      title="Solar zenith angle over the day"
      hint={
        daylight === null
          ? undefined
          : `${daylight} h of daylight · minimum SZA ${tickLabel(
              typeof data?.min_sza_deg === "number" ? data.min_sza_deg : 0,
            )}°`
      }
      help="Photolysis follows the sun: this is what the configured day and latitude mean for the chemistry. Shaded bands are night, where photolysis stops. The release marker shows the solar angle the plume first sees."
      loading={loading}
      {...(error ? { error } : {})}
    >
      <Chart
        series={[{ name: "sza", xs: hours, ys: sza, label: "SZA" }]}
        xLabel="hour (UTC)"
        yLabel="SZA (°)"
        xDomain={[0, 24]}
        yDomain={[0, 180]}
        bands={nightBands}
        markers={
          releaseHour !== null && releaseSza !== null
            ? [{ x: releaseHour, y: releaseSza, label: "release" }]
            : []
        }
        format={(v) => `${v.toFixed(1)}`}
        caption="Release time is marked; move it to change the photolysis the plume first sees."
      />
    </Frame>
  );
}


/** Stage 1b — the ERA5 profile this latitude and month imply, with the box marked on it. */
export function ClimatologyPanel({ config, load }: PanelProps) {
  const { data, error, loading } = usePanel("climatology", config, load);
  const levels = nums(data, "level_hpa");
  const temperature = nums(data, "temperature_k");
  const boxP = typeof data?.box_pressure_mbar === "number" ? data.box_pressure_mbar : null;
  const boxT = typeof data?.box_temperature_k === "number" ? data.box_temperature_k : null;
  const boxKm = typeof data?.box_altitude_km === "number" ? data.box_altitude_km : null;
  const heights = nums(data, "geopotential_height_m");
  const h2o = nums(data, "h2o_ppmv");
  const selected = typeof data?.selected_dataset === "string" ? data.selected_dataset : "user";
  const altitudeTicks = Array.isArray(data?.altitude_ticks)
    ? (data.altitude_ticks as { km: number; pressure_hpa: number }[]).map((tick) => ({
        y: tick.pressure_hpa,
        label: `${tick.km}`,
      }))
    : [];

  return (
    <Frame
      title="ERA5 temperature profile"
      hint={
        data
          ? selected === "era5"
            ? "the marked point IS the run's temperature"
            : "context — this run uses the typed values (dataset: user)"
          : undefined
      }
      help={
        data
          ? `ERA5 zonal-mean monthly climatology, 1991–2020 (${String(data.dataset_id)}). The profile is drawn at this latitude and month; switch "Ambient state from" to era5 and the run derives its temperature and water vapour from it.`
          : undefined
      }
      loading={loading}
      {...(error ? { error } : {})}
    >
      <Chart
        series={[{ name: "T", xs: temperature, ys: levels }]}
        xLabel="temperature (K)"
        yLabel="pressure (hPa)"
        yLog
        yReverse
        yDomain={[5, 300]}
        height={280}
        rightTicks={altitudeTicks}
        rightLabel="altitude (km)"
        hoverAxis="y"
        hoverReadout={(index) => {
          const rows = [
            { name: "p", value: `${(levels[index] ?? 0).toFixed(0)} hPa` },
            { name: "z", value: `${((heights[index] ?? 0) / 1000).toFixed(1)} km` },
            { name: "T", value: `${(temperature[index] ?? 0).toFixed(1)} K` },
          ];
          const water = h2o[index];
          if (water !== undefined) rows.push({ name: "H₂O", value: `${water.toFixed(2)} ppmv` });
          return rows;
        }}
        markers={
          boxP !== null && boxT !== null
            ? [{ x: boxT, y: boxP, label: boxKm !== null ? `the box · ${boxKm.toFixed(1)} km` : "the box" }]
            : []
        }
        format={(v) => v.toFixed(1)}
        caption="Up on the chart is up in the atmosphere. Altitude on the right is the same axis in kilometres — the product's own geopotential at this latitude and month, not a standard atmosphere."
      />
    </Frame>
  );
}

/** Stage 2 — the parcel, drawn to scale. The config itself, not a computation over it. */
export function ParcelPanel({ config }: PanelProps) {
  const length = Number(valueAt(config, "injection.plume_length_m") ?? 0);
  const width = Number(valueAt(config, "injection.plume_width_m") ?? 0);
  const heightM = Number(valueAt(config, "injection.plume_height_m") ?? 0);
  const volume = valueAt(config, "injection.plume_volume_cm3");
  const duration = valueAt(config, "injection.emission_duration_s");
  const rate = valueAt(config, "injection.emission_rate_kg_s");
  const speed = valueAt(config, "injection.platform_speed_m_s");
  const specifiedBy = String(valueAt(config, "injection.emission_input") ?? "");

  // The along-track length dwarfs the cross-section (15 km against 10 m), so a single scale would
  // render the parcel as a line. The two views are scaled independently and each says so.
  const crossMax = Math.max(width, heightM, 1);
  const boxW = (width / crossMax) * 90;
  const boxH = (heightM / crossMax) * 90;

  return (
    <Frame
      title="Parcel geometry"
      help="Drawn from the values entered, at two different scales — the along-track length is three orders of magnitude larger than the cross-section. t = 0 is the moment this volume is defined; how the parcel formed is out of scope (SCIENCE-2)."
    >
      <div className="parcel">
        <svg viewBox="0 0 320 130" role="img" aria-label="parcel cross-section">
          <rect x={110} y={20} width={boxW} height={boxH} className="parcel-box" />
          <text x={110 + boxW / 2} y={14} textAnchor="middle" className="parcel-dim">
            {width} m wide
          </text>
          <text x={104} y={20 + boxH / 2} textAnchor="end" dy="0.32em" className="parcel-dim">
            {heightM} m
          </text>
          <text x={160} y={125} textAnchor="middle" className="parcel-caption">
            cross-section
          </text>
        </svg>
        <div className="parcel-facts">
          <div>
            <span className="k">along-track</span>
            <span className="v">{(length / 1000).toFixed(2)} km</span>
          </div>
          <div>
            <span className="k">cross-section</span>
            <span className="v">
              {width} × {heightM} m
            </span>
          </div>
          <div>
            <span className="k">emission</span>
            <span className="v">
              {display(rate)} kg/s × {display(duration)} s
            </span>
          </div>
          <div>
            <span className="k">at</span>
            <span className="v">{display(speed)} m/s</span>
          </div>
          <div>
            <span className="k">V₀</span>
            <span className="v">{display(volume)} cm³</span>
          </div>
          <p className="parcel-note">
            Given <code>{specifiedBy.replace(/_/g, " ")}</code>; the rest of the emission follows
            from it.
          </p>

        </div>
      </div>
    </Frame>
  );
}

/** Stage 3 — initial mixing ratio against plume volume, with this config marked. */
export function ConcentrationPanel({ config, load }: PanelProps) {
  const { data, error, loading } = usePanel("concentration", config, load);
  const volumes = nums(data, "volume_cm3");
  const pptv = nums(data, "pptv");
  const currentV = typeof data?.current_volume_cm3 === "number" ? data.current_volume_cm3 : null;
  const currentP = typeof data?.current_pptv === "number" ? data.current_pptv : null;
  const background = typeof data?.background_pptv === "number" ? data.background_pptv : null;

  const series: Series[] = [{ name: "pptv", xs: volumes, ys: pptv, label: "SO₂" }];
  if (background !== null && background > 0 && volumes.length) {
    series.push({
      name: "background",
      xs: [volumes[0] ?? 1, volumes[volumes.length - 1] ?? 1],
      ys: [background, background],
      muted: true,
      dashed: true,
      label: "background",
    });
  }

  return (
    <Frame
      title="Initial SO₂ against plume volume"
      help="A straight line on log–log: the same mass in ten times the volume is a tenth the mixing ratio. The initial volume is a modelling choice (SCIENCE-2) — this panel is your result's sensitivity to it. Where the dashed background line crosses, the plume is indistinguishable from ambient."
      loading={loading}
      {...(error ? { error } : {})}
    >
      <Chart
        series={series}
        xLabel="V₀ (cm³)"
        yLabel="SO₂ (pptv)"
        xLog
        yLog
        markers={
          currentV !== null && currentP !== null
            ? [{ x: currentV, y: currentP, label: "this config" }]
            : []
        }
        caption="Where the dashed background line crosses, the plume is indistinguishable from ambient."
      />
    </Frame>
  );
}

/** Stage 4 — the dilution curves: the equation, its one constant, and the regimes as chips. */
export function DilutionPanel({ config, load, onChange }: PanelProps) {
  // The explorer's k: a draft string so typing does not refetch per keystroke; committed on Enter
  // or blur, like every other text control. Empty commits to "no custom curve".
  const [kDraft, setKDraft] = useState("");
  const [exploreK, setExploreK] = useState<number | null>(null);
  const { data, error, loading } = usePanel(
    "dilution",
    config,
    load,
    exploreK !== null ? { explore_k: exploreK } : undefined,
  );
  const days = nums(data, "days");
  const regimes = (data?.regimes ?? {}) as Record<
    string,
    { label: string; volume_ratio: number[]; final: number; k: number | null }
  >;
  const selected = typeof data?.selected === "string" ? data.selected : "";
  const usesCurve = data?.uses_curve !== false;
  const [kError, setKError] = useState("");
  const equation = (data?.equation ?? null) as {
    early: string;
    late: string;
    k_unit: string;
    k_min: number;
    k_max: number;
  } | null;
  const custom = (data?.custom ?? null) as { k: number; volume_ratio: number[] } | null;
  const selectedK = regimes[selected]?.k ?? null;

  const commitK = () => {
    if (kDraft.trim() === "") {
      setKError("");
      setExploreK(null);
      return;
    }
    const value = Number(kDraft);
    if (!Number.isFinite(value)) {
      setKError("not a number — e.g. 1.5e-8");
      setExploreK(null);
      return;
    }
    // Validate against the bounds the server declared, BEFORE any request: a bad k previously
    // came back as a panel-level 422 that replaced the panel body, taking this very input with
    // it -- an error state with no way out short of reloading. Reported from use.
    if (equation && (value < equation.k_min || value > equation.k_max)) {
      setKError(
        `k must be within ${equation.k_min.toExponential(0)} … ${equation.k_max.toExponential(0)}`,
      );
      setExploreK(null);
      return;
    }
    setKError("");
    setExploreK(value);
  };

  const series: Series[] = Object.entries(regimes).map(([name, regime]) => ({
    name,
    xs: days,
    ys: regime.volume_ratio,
    muted: name !== selected,
    label: name === selected ? `${name} — ${regime.label}` : name,
  }));
  if (custom) {
    series.push({
      name: "custom",
      xs: days,
      ys: custom.volume_ratio,
      dashed: true,
      color: "var(--gold)",
      label: `k = ${custom.k.toExponential(2)}`,
    });
  }

  return (
    <Frame
      title="Volume expansion V(t)/V₀"
      hint={
        usesCurve
          ? undefined
          : `CONSTANT regime: fixed ${display(data?.constant_rate_per_s)} s⁻¹ — these curves are not used.`
      }
      help="The regimes share one functional form and differ ONLY in the constant k. Click a chip to select the run's regime (it sets dilution.regime); type a custom k to see the family between and beyond them. A custom k is a picture, not a runnable configuration — the model runs named regimes or a constant rate."
      loading={loading}
      {...(error ? { error } : {})}
    >
      {equation ? (
        <div className="equation-card">
          <div className="equation">
            <code>{equation.early}</code>
            <code>{equation.late}</code>
          </div>
          <div className="equation-k">
            k ({equation.k_unit}) ={" "}
            <strong>{selectedK !== null ? selectedK.toExponential(3) : "—"}</strong>
          </div>
        </div>
      ) : null}

      <div className="regime-chips">
        {Object.entries(regimes).map(([name, regime]) => (
          <button
            key={name}
            type="button"
            className={`chip${name === selected ? " current" : ""}`}
            onClick={() => onChange?.("dilution.regime", name)}
            title={regime.label}
          >
            <span className="chip-name">{name}</span>
            <span className="chip-k">
              {regime.k !== null ? regime.k.toExponential(2) : "4-piece"}
            </span>
          </button>
        ))}
        <button
          type="button"
          className={`chip${selected === "constant" ? " current" : ""}`}
          onClick={() => onChange?.("dilution.regime", "constant")}
          title="fixed first-order rate; ignores the curves"
        >
          <span className="chip-name">constant</span>
          <span className="chip-k">rate</span>
        </button>
        <span className="explore">
          <label htmlFor="explore-k">try k</label>
          <input
            id="explore-k"
            type="text"
            inputMode="decimal"
            placeholder="e.g. 1.5e-8"
            value={kDraft}
            className={kError ? "invalid" : ""}
            onChange={(e) => setKDraft(e.target.value)}
            onBlur={commitK}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                commitK();
              }
            }}
          />
          {kError ? <span className="explore-error">{kError}</span> : null}
        </span>
      </div>

      <Chart
        series={series}
        xLabel="days"
        yLabel="V(t)/V₀"
        yLog
        height={260}
        format={(v) => tickLabel(v)}
        caption="Hover to read the expansion at a given time. The dashed gold curve, if present, is the explored k."
      />
    </Frame>
  );
}

/** Stage 5 — the background distribution TOMAS will actually be seeded with. */
export function SizeDistributionPanel({ config, load }: PanelProps) {
  const { data, error, loading } = usePanel("size-distribution", config, load);
  const dp = nums(data, "dp_um");
  const density = nums(data, "dn_dlogdp");
  const total = typeof data?.total_cm3 === "number" ? data.total_cm3 : null;
  const peakDp = typeof data?.peak_dp_um === "number" ? data.peak_dp_um : null;
  const peak = typeof data?.peak_dn_dlogdp === "number" ? data.peak_dn_dlogdp : null;

  return (
    <Frame
      title="Background size distribution"
      hint={
        total === null
          ? undefined
          : `${tickLabel(total)} cm⁻³ total · ${display(data?.n_bins)} bins`
      }
      help="The background distribution TOMAS is actually seeded with — the condensation sink the fresh plume competes against for H₂SO₄. Diameters are DRY; surface area and radius in results are wet."
      loading={loading}
      {...(error ? { error } : {})}
    >
      <Chart
        series={[{ name: "dNdlogDp", xs: dp, ys: density, label: "dN/dlogD" }]}
        xLabel="dry Dp (µm)"
        yLabel="dN/dlogDp (cm⁻³)"
        xLog
        markers={peakDp !== null && peak !== null ? [{ x: peakDp, y: peak, label: "mode" }] : []}
        caption="This is the condensation sink the fresh plume competes against for H₂SO₄."
      />
    </Frame>
  );
}

/** Stage 6 — the gas environment, as entered. */
export function SpeciesPanel({ config }: PanelProps) {
  const gases = (valueAt(config, "background.gas_pptv") ?? {}) as Record<string, number>;
  const so2 = Number(valueAt(config, "background.so2_pptv") ?? 0);
  const entries = Object.entries(gases)
    .concat(so2 > 0 ? [["SO2 (background)", so2]] : [])
    .filter(([, v]) => Number.isFinite(v) && v > 0)
    .sort((a, b) => b[1] - a[1]);

  if (!entries.length) {
    return (
      <Frame title="Background gas composition">
        <p className="empty">No background gases set — every species starts at the mechanism's own
        initial condition.</p>
      </Frame>
    );
  }

  const max = Math.max(...entries.map(([, v]) => v));
  const min = Math.min(...entries.map(([, v]) => v));
  const decades = Math.log10(max / min) || 1;

  return (
    <Frame
      title="Background gas composition"
      help="Log scale: these span orders of magnitude, and the oxidants at the bottom set how fast SO₂ becomes H₂SO₄. Species not listed start at the mechanism's own initial condition."
    >
      <div className="bars">
        {entries.map(([name, value]) => (
          <div className="bar-row" key={name}>
            <span className="bar-name">{name}</span>
            <div className="bar-track">
              <div
                className="bar-fill"
                style={{ width: `${Math.max(2, (1 - Math.log10(max / value) / decades) * 100)}%` }}
              />
            </div>
            <span className="bar-value">{tickLabel(value)}</span>
          </div>
        ))}
      </div>
    </Frame>
  );
}

/** Stage 7 — what choosing n_bins actually changes. */
export function BinsPanel({ config, load }: PanelProps) {
  const { data, error, loading } = usePanel("bins", config, load);
  const grids = (data?.grids ?? {}) as Record<
    string,
    { dp_um: number[]; width_decades: number[]; mass_ratio: number }
  >;
  const selected = typeof data?.selected === "string" ? data.selected : "";

  const series: Series[] = Object.entries(grids).map(([n, grid]) => ({
    name: n,
    xs: grid.dp_um,
    ys: grid.width_decades,
    muted: n !== selected,
    label: `${n} bins`,
  }));

  return (
    <Frame
      title="Size resolution"
      help="Bin width in decades of diameter across a FIXED range (1.7 nm – 17.5 µm). Doubling the bins halves the width; the range does not move. Narrower bins resolve a nucleation burst that a coarse grid smears across one bin."
      loading={loading}
      {...(error ? { error } : {})}
    >
      <Chart
        series={series}
        xLabel="dry Dp (µm)"
        yLabel="bin width (decades)"
        xLog
        height={200}
        format={(v) => v.toFixed(4)}
        caption="Narrower bins resolve a nucleation burst that a coarse grid smears across one bin."
      />
    </Frame>
  );
}

/** Stage id -> panel. Stages absent from here have no honest plot yet, and show nothing. */
function EnvironmentPanels(props: PanelProps) {
  return (
    <>
      <ClimatologyPanel {...props} />
      <SzaPanel {...props} />
    </>
  );
}

export const PANELS_BY_STAGE: Record<string, (props: PanelProps) => React.ReactElement> = {
  environment: EnvironmentPanels,
  plume_volume: ParcelPanel,
  initial_concentration: ConcentrationPanel,
  dilution: DilutionPanel,
  background_aerosol: SizeDistributionPanel,
  background_species: SpeciesPanel,
  physics: BinsPanel,
};
