// Copyright (C) 2026 University Corporation for Atmospheric Research
// SPDX-License-Identifier: Apache-2.0
/**
 * A run's results: the six time series a microphysicist scans first, the banana plot, and the
 * number / surface-area spectra with a time slider and a log-linear toggle.
 *
 * Everything is read from the RunSummary (ADR-004), whose 0.2.0 schema carries the time-resolved
 * spectrum precisely because the final spectrum alone HID nucleation: by the end of a run the
 * burst has grown and coagulated out of the small bins, and a reviewer reasonably read that as "no
 * nucleation" while the data peaked at 1.9e7 cm^-3 twelve hours in.
 *
 * The surface-area spectrum is the number spectrum times pi Dp^2 -- the definitional surface of a
 * sphere at the DRY diameter, computed here because it is geometry over served numbers, not
 * physics. It is NOT the model's wet SA series (that one is a separate quantity on a wet basis).
 */

import { useState } from "react";
import { Chart, type Series as ChartSeries } from "./Chart";
import { Heatmap } from "./Heatmap";
import { StackedArea } from "./StackedArea";
import { display } from "./schema";
import { tickLabel } from "./scales";
import type { RunBrief } from "./types";

interface SummarySeries {
  values: number[];
  unit: string;
  basis: string;
  description: string;
}

export interface RunSummaryPayload {
  time_days: number[];
  termination: string;
  flags: string[];
  /** Daylight per time step from the run's own photolysis; [] when the npz had no J. */
  daylight?: boolean[];
  series: Record<string, SummarySeries>;
  final_size_distribution?: {
    diameter_um: number[];
    dn_dlogdp_cm3: number[];
    total_number_cm3: number;
    basis: string;
  };
  size_distribution_history?: {
    time_days: number[];
    diameter_um: number[];
    dn_dlogdp_cm3: number[][];
    basis: string;
    stride: number;
  };
}

/** One small-multiple time series straight from the summary, by NAME. */
function TimeSeriesChart({
  summary,
  title,
  names,
  yLabel,
  yLog = true,
  night = [],
}: {
  summary: RunSummaryPayload;
  title: string;
  names: string[];
  yLabel: string;
  yLog?: boolean;
  night?: { from: number; to: number }[];
}) {
  const series: ChartSeries[] = [];
  for (const name of names) {
    const entry = summary.series[name];
    if (!entry) continue;
    series.push({
      name,
      xs: summary.time_days,
      ys: entry.values,
      label: name === "particle_mass_ug_m3" ? "mass" : name === "particulate_S_pptv" ? "S(p)" : name,
      muted: series.length > 0,
    });
  }
  if (!series.length) return null;
  return (
    <section>
      <h3>{title}</h3>
      <Chart series={series} xLabel="days" yLabel={yLabel} yLog={yLog} height={200} bands={night} />
    </section>
  );
}

export function Results({
  run,
  summary,
  onClose,
}: {
  run: RunBrief;
  summary: RunSummaryPayload | null;
  onClose: () => void;
}) {
  const [timeIndex, setTimeIndex] = useState<number | null>(null);
  const [logY, setLogY] = useState(true);

  // Night intervals from the run's own daylight flags -- contiguous dark spans as [from, to] in
  // days, drawn behind every time series so the OH/HO2 diurnal cycle reads against real day/night.
  const nightBands = (() => {
    const out: { from: number; to: number }[] = [];
    const dl = summary?.daylight;
    const days = summary?.time_days ?? [];
    if (!dl || dl.length !== days.length) return out;
    let start: number | null = null;
    for (let i = 0; i < dl.length; i++) {
      if (!dl[i] && start === null) start = days[i]!;
      if ((dl[i] || i === dl.length - 1) && start !== null) {
        out.push({ from: start, to: days[i]! });
        start = null;
      }
    }
    return out;
  })();

  // Sulfur budget: gas-phase S (SO2+SO3+H2SO4, one S atom each, in pptv) vs particle S (pptv),
  // as the gas FRACTION over time. Both are mixing ratios the summary already carries, so this is
  // a normalization, not new physics.
  const sulfurGasFraction = (() => {
    if (!summary) return null;
    const gasNames = ["SO2", "SO3", "H2SO4"];
    const n = summary.time_days.length;
    const gas = new Array(n).fill(0);
    for (const name of gasNames) {
      const s = summary.series[name];
      if (s) for (let i = 0; i < n; i++) gas[i] += s.values[i] ?? 0;
    }
    const particle = summary.series["particulate_S_pptv"];
    if (!particle) return null;
    const frac = gas.map((g, i) => {
      const total = g + (particle.values[i] ?? 0);
      return total > 0 ? g / total : 1;
    });
    return frac;
  })();

  const headline = run.headline;
  const stats: { label: string; value: string }[] = headline
    ? [
        { label: "final SO₂", value: `${tickLabel(headline.final_so2_pptv ?? 0)} pptv` },
        { label: "peak H₂SO₄", value: `${tickLabel(headline.peak_h2so4_pptv ?? 0)} pptv` },
        { label: "peak number", value: `${tickLabel(headline.peak_number_cm3 ?? 0)} cm⁻³` },
        {
          label: "final SA (wet)",
          value: `${tickLabel(headline.final_surface_area ?? 0)} µm² cm⁻³`,
        },
      ]
    : [];

  const history = summary?.size_distribution_history;
  const index =
    history === undefined
      ? 0
      : Math.min(timeIndex ?? history.time_days.length - 1, history.time_days.length - 1);
  const spectrum = history?.dn_dlogdp_cm3[index];
  const surface =
    history && spectrum
      ? spectrum.map((v, b) => {
          const dp = history.diameter_um[b] ?? 0;
          return v * Math.PI * dp * dp; // um^2 cm^-3 per dlogDp, DRY diameter
        })
      : null;

  return (
    <div className="results">
      <div className="results-head">
        <div>
          <h2>{run.label || run.run_id.slice(0, 8)}</h2>
          <p className="results-meta">
            <span className={`run-state state-${run.state}`}>{run.state}</span>
            {summary ? <span> · terminated: {summary.termination}</span> : null}
            {run.reproducible ? null : <span className="dirty"> · dirty checkout</span>}
            <span className="run-hash"> · {run.config_hash.slice(0, 12)}</span>
          </p>
        </div>
        <button type="button" className="ghost" onClick={onClose}>
          ← back to configure
        </button>
      </div>

      {!summary ? (
        <p className="empty">
          {run.state === "succeeded"
            ? "loading summary…"
            : `No summary yet — the run is ${run.state}. This view fills in when it finishes.`}
        </p>
      ) : (
        <>
          <div className="stat-tiles">
            {stats.map((stat) => (
              <div className="stat" key={stat.label}>
                <span className="stat-value">{stat.value}</span>
                <span className="stat-label">{stat.label}</span>
              </div>
            ))}
          </div>
          {summary.flags.length ? (
            <p className="results-flags">flags: {summary.flags.join(", ")}</p>
          ) : null}

          <div className="results-charts">
            <TimeSeriesChart
              summary={summary}
              title="Total particle number"
              names={["total_n"]}
              yLabel="N (cm⁻³)"
              night={nightBands}
            />
            <TimeSeriesChart
              summary={summary}
              title="Particle mass (dry H₂SO₄-eq)"
              names={["particle_mass_ug_m3"]}
              yLabel="mass (µg m⁻³)"
              night={nightBands}
            />
            <TimeSeriesChart
              summary={summary}
              title="SO₂ (gas)"
              names={["SO2"]}
              yLabel="SO₂ (pptv)"
              night={nightBands}
            />
            <TimeSeriesChart
              summary={summary}
              title="H₂SO₄: gas and particle (pptv)"
              names={["H2SO4", "particulate_S_pptv"]}
              yLabel="mixing ratio (pptv)"
              night={nightBands}
            />
            <TimeSeriesChart
              summary={summary}
              title="Oxidants: OH and HO₂"
              names={["OH", "HO2"]}
              yLabel="mixing ratio (pptv)"
              night={nightBands}
            />
            <TimeSeriesChart
              summary={summary}
              title="Aerosol surface area (wet)"
              names={["SA"]}
              yLabel="SA (µm² cm⁻³)"
              night={nightBands}
            />
          </div>

          {sulfurGasFraction ? (
            <section className="budget-panel">
              <h3>Sulfur budget — gas vs particle</h3>
              <StackedArea
                timeDays={summary.time_days}
                lower={sulfurGasFraction}
                lowerLabel="SO₂ gas"
                lowerColor="var(--gold)"
                upperLabel="in particles"
                upperColor="var(--steel)"
                night={nightBands}
              />
            </section>
          ) : null}

          {history ? (
            <div className="results-distributions">
              <section className="banana">
                <h3>Size distribution over time — the banana plot</h3>
                <Heatmap
                  timeDays={history.time_days}
                  diameterUm={history.diameter_um}
                  values={history.dn_dlogdp_cm3}
                />
                <p className="panel-hint">
                  Nucleation is the dark tongue in the smallest bins; growth bends it up and to the
                  right. Dry diameters
                  {history.stride > 1 ? ` · every ${history.stride}ᵗʰ stored step` : ""}.
                </p>
              </section>

              <section>
                <div className="dist-controls">
                  <label htmlFor="time-slider">
                    t = {(history.time_days[index] ?? 0).toFixed(2)} d (
                    {((history.time_days[index] ?? 0) * 24).toFixed(1)} h)
                  </label>
                  <input
                    id="time-slider"
                    type="range"
                    min={0}
                    max={history.time_days.length - 1}
                    value={index}
                    onChange={(e) => setTimeIndex(Number(e.target.value))}
                  />
                  <span className="dist-toggle" role="group" aria-label="y axis scale">
                    <button
                      type="button"
                      className={`chip${logY ? " current" : ""}`}
                      onClick={() => setLogY(true)}
                    >
                      <span className="chip-name">log y</span>
                    </button>
                    <button
                      type="button"
                      className={`chip${logY ? "" : " current"}`}
                      onClick={() => setLogY(false)}
                    >
                      <span className="chip-name">linear y</span>
                    </button>
                  </span>
                </div>
                <div className="results-charts">
                  <section>
                    <h3>Number size distribution</h3>
                    <Chart
                      series={[
                        {
                          name: "dnd",
                          xs: history.diameter_um,
                          ys: spectrum ?? [],
                          label: "dN/dlogD",
                        },
                      ]}
                      xLabel="dry Dp (µm)"
                      yLabel="dN/dlogDp (cm⁻³)"
                      xLog
                      yLog={logY}
                      height={230}
                    />
                  </section>
                  <section>
                    <h3>Surface-area size distribution</h3>
                    <Chart
                      series={[
                        {
                          name: "dsd",
                          xs: history.diameter_um,
                          ys: surface ?? [],
                          label: "dS/dlogD",
                        },
                      ]}
                      xLabel="dry Dp (µm)"
                      yLabel="dS/dlogDp (µm² cm⁻³)"
                      xLog
                      yLog={logY}
                      height={230}
                      caption="π·Dp²·dN/dlogDp at the DRY diameter — not the model's wet SA series above."
                    />
                  </section>
                </div>
              </section>
            </div>
          ) : summary.final_size_distribution ? (
            <section>
              <h3>Final size distribution</h3>
              <Chart
                series={[
                  {
                    name: "dnd",
                    xs: summary.final_size_distribution.diameter_um,
                    ys: summary.final_size_distribution.dn_dlogdp_cm3,
                    label: "dN/dlogD",
                  },
                ]}
                xLabel="dry Dp (µm)"
                yLabel="dN/dlogDp (cm⁻³)"
                xLog
                height={230}
                caption={`${display(summary.final_size_distribution.total_number_cm3)} cm⁻³ total, dry basis. This summary predates the time-resolved spectrum — re-run for the banana plot.`}
              />
            </section>
          ) : null}
        </>
      )}
    </div>
  );
}
