// Copyright (C) 2026 University Corporation for Atmospheric Research
// SPDX-License-Identifier: Apache-2.0
/**
 * A finished run, read from its RunSummary -- the first results view in the wizard.
 *
 * Everything here comes from the summary (ADR-004: comparison views and figures read the summary,
 * never the raw npz). Series are indexed by NAME from the summary's own dict -- the caveat the
 * summary module exists to enforce -- and each chart states its basis where it matters.
 *
 * Two charts rather than one: SO2/H2SO4 are mixing ratios [pptv] and particle number is [cm^-3] --
 * different quantities, so they get separate axes on separate charts rather than a dual-axis lie.
 */

import { Chart, type Series as ChartSeries } from "./Chart";
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
  series: Record<string, SummarySeries>;
  final_size_distribution?: {
    diameter_um: number[];
    dn_dlogdp_cm3: number[];
    total_number_cm3: number;
    basis: string;
  };
}

function gasSeries(summary: RunSummaryPayload): ChartSeries[] {
  const wanted: [string, boolean][] = [
    ["SO2", false],
    ["H2SO4", false],
    ["OH", true],
  ];
  const out: ChartSeries[] = [];
  for (const [name, muted] of wanted) {
    const series = summary.series[name];
    if (!series) continue; // a mechanism without the species is fewer series, not an error
    out.push({ name, xs: summary.time_days, ys: series.values, label: name, muted });
  }
  return out;
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
  // Headline scalars live on the RUN (database columns filled at finalise time); the summary JSON
  // carries the series and the spectrum. Read each from where it actually is.
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
  const total = summary?.series["total_n"];

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
            <section>
              <h3>Gas phase</h3>
              <Chart
                series={gasSeries(summary)}
                xLabel="days"
                yLabel="mixing ratio (pptv)"
                yLog
                height={230}
                caption="SO₂ falls as it oxidises and dilutes; H₂SO₄ is the condensable it becomes."
              />
            </section>
            {total ? (
              <section>
                <h3>Particles</h3>
                <Chart
                  series={[
                    {
                      name: "total_n",
                      xs: summary.time_days,
                      ys: total.values,
                      label: "total N",
                    },
                  ]}
                  xLabel="days"
                  yLabel="number (cm⁻³)"
                  yLog
                  height={230}
                  caption="Total particle number: nucleation bursts up, coagulation grinds down."
                />
              </section>
            ) : null}
            {summary.final_size_distribution ? (
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
                  caption={`${display(summary.final_size_distribution.total_number_cm3)} cm⁻³ total, dry basis.`}
                />
              </section>
            ) : null}
          </div>
        </>
      )}
    </div>
  );
}
