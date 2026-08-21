// Copyright (C) 2026 University Corporation for Atmospheric Research
// SPDX-License-Identifier: Apache-2.0
/**
 * The banana plot: dN/dlogDp over (time x diameter), colour = log10 of the value.
 *
 * Sequential single hue (paper -> navy), because the quantity is a magnitude; the colour scale is
 * logarithmic and the colourbar says so with decade labels. A nucleation burst appears exactly as
 * the textbook banana: a dark tongue starting in the smallest bins that bends up and to the right
 * as the mode grows -- which is the picture that answers "I do not see any nucleation".
 */

import { linearScale, logScale, tickLabel } from "./scales";

const WIDTH = 680;
const PAD = { top: 12, right: 96, bottom: 34, left: 56 };

//: Six decades below the maximum: dynamic range typical of dN/dlogDp without drowning in floor.
const DECADES = 6;

/** Paper -> navy ramp, interpolated in-channel. Good enough for a magnitude ramp of one hue. */
function rampColor(fraction: number): string {
  const f = Math.max(0, Math.min(1, fraction));
  const from = [236, 229, 219]; // --sunken paper
  const to = [9, 24, 52]; // --navy
  const channel = (i: number) => Math.round(from[i]! + (to[i]! - from[i]!) * f);
  return `rgb(${channel(0)}, ${channel(1)}, ${channel(2)})`;
}

export function Heatmap({
  timeDays,
  diameterUm,
  values,
  xLabel = "days",
  yLabel = "dry Dp (µm)",
  legendLabel = "dN/dlogDp (cm⁻³)",
  height = 300,
  maxColumns = 120,
}: {
  timeDays: number[];
  diameterUm: number[];
  values: number[][]; // [time][bin]
  xLabel?: string;
  yLabel?: string;
  legendLabel?: string;
  height?: number;
  maxColumns?: number;
}) {
  if (!timeDays.length || !diameterUm.length || !values.length) {
    return <p className="empty">no time-resolved spectrum in this summary (re-run to get one)</p>;
  }
  // Uniform column decimation for the DOM's sake (<= maxColumns * bins rects). Uniform, so no
  // aliasing bias beyond what any decimation has; the stride is visual only.
  const stride = Math.max(1, Math.ceil(timeDays.length / maxColumns));
  const columns: number[] = [];
  for (let i = 0; i < timeDays.length; i += stride) columns.push(i);

  const flat = values.flat().filter((v) => Number.isFinite(v) && v > 0);
  const peak = flat.length ? Math.max(...flat) : 1;
  const logPeak = Math.log10(peak);
  const logFloor = logPeak - DECADES;

  const x = linearScale(
    [timeDays[0] ?? 0, timeDays[timeDays.length - 1] ?? 1],
    [PAD.left, WIDTH - PAD.right],
  );
  const dpLow = diameterUm[0] ?? 1e-3;
  const dpHigh = diameterUm[diameterUm.length - 1] ?? 10;
  const y = logScale([dpLow, dpHigh], [height - PAD.bottom, PAD.top]);

  const fraction = (v: number) =>
    v > 0 ? Math.max(0, Math.min(1, (Math.log10(v) - logFloor) / DECADES)) : 0;

  const cells: React.ReactElement[] = [];
  for (let c = 0; c < columns.length; c++) {
    const i = columns[c]!;
    const x0 = c === 0 ? PAD.left : x((timeDays[columns[c - 1]!]! + timeDays[i]!) / 2);
    const next = columns[c + 1];
    const x1 = next === undefined ? WIDTH - PAD.right : x((timeDays[i]! + timeDays[next]!) / 2);
    const row = values[i]!;
    for (let b = 0; b < diameterUm.length; b++) {
      const value = row[b] ?? 0;
      if (!(value > 0)) continue; // paper background IS the zero colour; skip the rect
      // Bin edges approximated by midpoints between mid-diameters -- display geometry only.
      const yTop =
        b === diameterUm.length - 1
          ? PAD.top
          : y(Math.sqrt(diameterUm[b]! * diameterUm[b + 1]!));
      const yBottom = b === 0 ? height - PAD.bottom : y(Math.sqrt(diameterUm[b - 1]! * diameterUm[b]!));
      cells.push(
        <rect
          key={`${c}-${b}`}
          x={x0}
          width={Math.max(0.5, x1 - x0)}
          y={yTop}
          height={Math.max(0.5, yBottom - yTop)}
          fill={rampColor(fraction(value))}
        />,
      );
    }
  }

  const legendSteps = Array.from({ length: DECADES + 1 }, (_, i) => logFloor + i);

  return (
    <figure className="chart heatmap">
      <svg viewBox={`0 0 ${WIDTH} ${height}`} role="img" aria-label={`${legendLabel} over time and diameter`}>
        {cells}
        {y.ticks(5).map((tick) => (
          <text key={`y-${tick}`} className="chart-tick" x={PAD.left - 7} y={y(tick)} textAnchor="end" dy="0.32em">
            {tickLabel(tick)}
          </text>
        ))}
        {x.ticks(6).map((tick) => (
          <text key={`x-${tick}`} className="chart-tick" x={x(tick)} y={height - PAD.bottom + 15} textAnchor="middle">
            {tickLabel(tick)}
          </text>
        ))}
        <text className="chart-axis-label" x={PAD.left} y={height - 4}>
          {xLabel}
        </text>
        <text className="chart-axis-label" transform={`translate(11 ${PAD.top + 4}) rotate(-90)`} textAnchor="end">
          {yLabel}
        </text>
        {/* Colourbar: one decade per band, labelled -- a log colour scale must say it is one. */}
        {legendSteps.slice(0, -1).map((_, i) => {
          const barTop = PAD.top + 12;
          const barHeight = height - PAD.bottom - PAD.top - 24;
          const bandHeight = barHeight / DECADES;
          return (
            <rect
              key={`band-${i}`}
              x={WIDTH - PAD.right + 14}
              width={12}
              y={barTop + barHeight - (i + 1) * bandHeight}
              height={bandHeight}
              fill={rampColor((i + 0.5) / DECADES)}
            />
          );
        })}
        {legendSteps.map((step, i) => {
          const barTop = PAD.top + 12;
          const barHeight = height - PAD.bottom - PAD.top - 24;
          return (
            <text
              key={`bandlab-${i}`}
              className="chart-tick"
              x={WIDTH - PAD.right + 30}
              y={barTop + barHeight - (i * barHeight) / DECADES}
              dy="0.32em"
            >
              10{superscriptOf(Math.round(step))}
            </text>
          );
        })}
        <text className="chart-axis-label" x={WIDTH - PAD.right + 14} y={PAD.top + 4}>
          {legendLabel}
        </text>
      </svg>
    </figure>
  );
}

function superscriptOf(power: number): string {
  const map: Record<string, string> = {
    "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
  };
  return String(power).split("").map((ch) => map[ch] ?? ch).join("");
}
