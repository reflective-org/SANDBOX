// Copyright (C) 2026 University Corporation for Atmospheric Research
// SPDX-License-Identifier: Apache-2.0
/**
 * The sulfur budget: what share of the plume's sulfur is still SO2 gas versus in particles, over
 * time, as a normalized 0-100% stack. Learned from the reference viz (plume_dynamics' "sulfur
 * budget"), the panel that makes gas-to-particle conversion legible at a glance where two separate
 * decaying curves do not.
 *
 * Two bands only, so two categorical hues in fixed order (gold = gas, steel = particle) with a
 * 2px surface gap between them and both directly labelled -- no legend box needed. Night bands
 * shade behind, from the run's own photolysis, because conversion pauses in the dark.
 */

import { linearScale } from "./scales";

const WIDTH = 680;
const PAD = { top: 12, right: 96, bottom: 34, left: 44 };

export interface Band {
  from: number;
  to: number;
}

export function StackedArea({
  timeDays,
  lower,
  lowerLabel,
  lowerColor,
  upperLabel,
  upperColor,
  night = [],
  height = 220,
}: {
  timeDays: number[];
  /** Fraction 0..1 in the LOWER band at each time; the upper band fills the rest. */
  lower: number[];
  lowerLabel: string;
  lowerColor: string;
  upperLabel: string;
  upperColor: string;
  night?: Band[];
  height?: number;
}) {
  if (timeDays.length < 2) return <p className="empty">not enough time steps to plot a budget</p>;
  const x = linearScale([timeDays[0]!, timeDays[timeDays.length - 1]!], [PAD.left, WIDTH - PAD.right]);
  const y = linearScale([0, 1], [height - PAD.bottom, PAD.top]);

  // Lower band: baseline up to `lower`. Upper band: `lower` up to 1, drawn from the top down so a
  // 2px gap between the two fills reads as a seam, not a smear (the spacer rule).
  const lowerPath =
    `M${x(timeDays[0]!).toFixed(1)},${y(0).toFixed(1)}` +
    timeDays.map((t, i) => `L${x(t).toFixed(1)},${y(lower[i] ?? 0).toFixed(1)}`).join("") +
    `L${x(timeDays[timeDays.length - 1]!).toFixed(1)},${y(0).toFixed(1)}Z`;
  const upperPath =
    `M${x(timeDays[0]!).toFixed(1)},${y(1).toFixed(1)}` +
    timeDays.map((t, i) => `L${x(t).toFixed(1)},${y((lower[i] ?? 0) + 0.004).toFixed(1)}`).join("") +
    `L${x(timeDays[timeDays.length - 1]!).toFixed(1)},${y(1).toFixed(1)}Z`;

  const lastLower = lower[lower.length - 1] ?? 0;

  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${WIDTH} ${height}`} role="img" aria-label={`${lowerLabel} versus ${upperLabel} over time`}>
        {night.map((band, i) => (
          <rect
            key={`n-${i}`}
            className="chart-band"
            x={x(band.from)}
            width={Math.max(0, x(band.to) - x(band.from))}
            y={PAD.top}
            height={height - PAD.bottom - PAD.top}
          />
        ))}
        <path d={upperPath} style={{ fill: upperColor }} />
        <path d={lowerPath} style={{ fill: lowerColor }} />
        {[0, 0.25, 0.5, 0.75, 1].map((frac) => (
          <text key={frac} className="chart-tick" x={PAD.left - 6} y={y(frac)} textAnchor="end" dy="0.32em">
            {frac * 100}%
          </text>
        ))}
        {x.ticks(6).map((tick) => (
          <text key={tick} className="chart-tick" x={x(tick)} y={height - PAD.bottom + 15} textAnchor="middle">
            {tick}
          </text>
        ))}
        {/* Direct labels at the right edge, each in its band's colour -- identity without a legend. */}
        <text className="chart-series-label" x={WIDTH - PAD.right + 6} y={y(lastLower / 2)} dy="0.32em" style={{ fill: lowerColor }}>
          {lowerLabel}
        </text>
        <text className="chart-series-label" x={WIDTH - PAD.right + 6} y={y(lastLower + (1 - lastLower) / 2)} dy="0.32em" style={{ fill: upperColor }}>
          {upperLabel}
        </text>
        <text className="chart-axis-label" x={PAD.left} y={height - 4}>
          days
        </text>
      </svg>
      <figcaption>
        By the end, {(lastLower * 100).toFixed(0)}% of plume sulfur is still SO₂ gas;{" "}
        {((1 - lastLower) * 100).toFixed(0)}% has converted to particles.
      </figcaption>
    </figure>
  );
}
