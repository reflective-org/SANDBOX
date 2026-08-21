// Copyright (C) 2026 University Corporation for Atmospheric Research
// SPDX-License-Identifier: Apache-2.0
/**
 * The one chart the preview panels are drawn with.
 *
 * Deliberately plain: thin marks, a recessive grid, direct labels rather than a legend box where
 * there is room, and a crosshair that reads out real values -- these panels exist to be
 * interrogated ("what is V/V0 at day 3?"), not admired.
 *
 * One series is usually the subject and the rest are context (the chosen dilution regime against
 * the five others). Context is drawn muted rather than in its own hue, which keeps the reader's eye
 * on the choice being made and sidesteps needing six categorical colours that stay distinguishable
 * under colour-vision deficiency.
 */

import { useCallback, useRef, useState } from "react";
import { type Scale, linearScale, linePath, logScale, nearestIndex, tickLabel } from "./scales";

export interface Series {
  name: string;
  xs: number[];
  ys: number[];
  /** Muted series are context; exactly one series is normally the subject. */
  muted?: boolean;
  dashed?: boolean;
  /** Drawn at the series' right-hand end. Omit to leave the series unlabelled. */
  label?: string;
  color?: string;
}

export interface Band {
  from: number;
  to: number;
  label?: string;
}

export interface Marker {
  x: number;
  y: number;
  label: string;
}

interface Props {
  series: Series[];
  xLabel: string;
  yLabel: string;
  xLog?: boolean;
  yLog?: boolean;
  height?: number;
  bands?: Band[];
  markers?: Marker[];
  xDomain?: [number, number];
  yDomain?: [number, number];
  /**
   * Draw the y axis with the LARGER value at the bottom. For pressure as a vertical coordinate:
   * pressure falls with altitude, so a profile reads correctly only when 300 hPa sits at the
   * bottom of the frame and 5 hPa at the top -- the atmosphere the way anyone pictures it.
   */
  yReverse?: boolean;
  /**
   * A second LABELING of the y axis on the right edge -- the same positions in different units
   * (altitude for a pressure axis), never an independent second scale. Each tick names a position
   * in the PRIMARY y domain.
   */
  rightTicks?: { y: number; label: string }[];
  rightLabel?: string;
  /** How a hovered value is written in the tooltip. */
  format?: (value: number) => string;
  caption?: string;
}

const WIDTH = 680;
const PAD = { top: 12, right: 92, bottom: 34, left: 56 };

function domainOf(values: number[], log: boolean, override?: [number, number]): [number, number] {
  if (override) return override;
  const usable = values.filter((v) => Number.isFinite(v) && (!log || v > 0));
  if (!usable.length) return log ? [1, 10] : [0, 1];
  let min = Math.min(...usable);
  let max = Math.max(...usable);
  if (min === max) {
    if (log) return [min / 2, max * 2];
    return [min - 1, max + 1];
  }
  if (!log) {
    // A little headroom, so a peak does not sit on the frame.
    const pad = (max - min) * 0.05;
    min -= pad;
    max += pad;
  }
  return [min, max];
}

export function Chart({
  series,
  xLabel,
  yLabel,
  xLog = false,
  yLog = false,
  height = 240,
  bands = [],
  markers = [],
  xDomain,
  yDomain,
  yReverse = false,
  rightTicks = [],
  rightLabel,
  format = (v) => tickLabel(v),
  caption,
}: Props) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [hoverX, setHoverX] = useState<number | null>(null);

  const allX = series.flatMap((s) => s.xs).concat(markers.map((m) => m.x));
  const allY = series.flatMap((s) => s.ys).concat(markers.map((m) => m.y));
  const xd = domainOf(allX, xLog, xDomain);
  const yd = domainOf(allY, yLog, yDomain);

  const makeScale = (
    domain: [number, number],
    range: [number, number],
    log: boolean,
  ): Scale | null => {
    try {
      return log ? logScale(domain, range) : linearScale(domain, range);
    } catch {
      // A log axis was asked for over a non-positive domain. Say so instead of drawing nothing.
      return null;
    }
  };

  const x = makeScale(xd, [PAD.left, WIDTH - PAD.right], xLog);
  // The default puts the domain minimum at the bottom (SVG y grows downward, so the pixel range is
  // inverted). yReverse swaps the pixel range instead of the domain, so ticks stay ascending.
  const y = makeScale(
    yd,
    yReverse ? [PAD.top, height - PAD.bottom] : [height - PAD.bottom, PAD.top],
    yLog,
  );
  if (!x || !y) {
    return (
      <p className="chart-error">
        Cannot draw {yLabel} against {xLabel} on a logarithmic axis: the data spans zero or negative
        values.
      </p>
    );
  }

  const onMove = useCallback(
    (event: React.MouseEvent<SVGSVGElement>) => {
      const svg = svgRef.current;
      if (!svg) return;
      const box = svg.getBoundingClientRect();
      // The SVG is scaled to its container, so pointer pixels must be mapped back to viewBox units.
      const px = ((event.clientX - box.left) / box.width) * WIDTH;
      setHoverX(px < PAD.left || px > WIDTH - PAD.right ? null : px);
    },
    [],
  );

  const hover =
    hoverX === null
      ? null
      : (() => {
          const value = x.invert(hoverX);
          const readouts = series
            .map((s) => {
              const index = nearestIndex(s.xs, value);
              if (index < 0) return null;
              const yv = s.ys[index];
              if (yv === undefined || !Number.isFinite(yv)) return null;
              return { name: s.label ?? s.name, value: yv, muted: s.muted === true };
            })
            .filter((r): r is { name: string; value: number; muted: boolean } => r !== null);
          return { at: value, readouts };
        })();

  return (
    <figure className="chart">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${WIDTH} ${height}`}
        role="img"
        aria-label={`${yLabel} against ${xLabel}`}
        onMouseMove={onMove}
        onMouseLeave={() => setHoverX(null)}
      >
        {bands.map((band, i) => (
          <rect
            key={`band-${i}`}
            className="chart-band"
            x={Math.min(x(band.from), x(band.to))}
            width={Math.abs(x(band.to) - x(band.from))}
            y={PAD.top}
            height={height - PAD.bottom - PAD.top}
          />
        ))}

        {y.ticks(5).map((tick) => (
          <g key={`y-${tick}`}>
            <line
              className="chart-grid"
              x1={PAD.left}
              x2={WIDTH - PAD.right}
              y1={y(tick)}
              y2={y(tick)}
            />
            <text className="chart-tick" x={PAD.left - 7} y={y(tick)} textAnchor="end" dy="0.32em">
              {tickLabel(tick)}
            </text>
          </g>
        ))}

        {x.ticks(6).map((tick) => (
          <text
            key={`x-${tick}`}
            className="chart-tick"
            x={x(tick)}
            y={height - PAD.bottom + 15}
            textAnchor="middle"
          >
            {tickLabel(tick)}
          </text>
        ))}

        <line
          className="chart-axis"
          x1={PAD.left}
          x2={WIDTH - PAD.right}
          y1={height - PAD.bottom}
          y2={height - PAD.bottom}
        />

        {series.map((s) => (
          <path
            key={s.name}
            className={`chart-line${s.muted ? " muted" : ""}${s.dashed ? " dashed" : ""}`}
            d={linePath(s.xs, s.ys, x, y)}
            {...(s.color ? { style: { stroke: s.color } } : {})}
          />
        ))}

        {/* Direct labels at each series' right-hand end, nudged apart so they never overlap. */}
        {labelPositions(series, y, height).map(({ series: s, py }) => (
          <text
            key={`label-${s.name}`}
            className={`chart-series-label${s.muted ? " muted" : ""}`}
            x={WIDTH - PAD.right + 6}
            y={py}
            dy="0.32em"
            {...(s.color ? { style: { fill: s.color } } : {})}
          >
            {s.label}
          </text>
        ))}

        {rightTicks.map((tick) => (
          <g key={`right-${tick.label}`}>
            <line
              className="chart-axis"
              x1={WIDTH - PAD.right}
              x2={WIDTH - PAD.right + 5}
              y1={y(tick.y)}
              y2={y(tick.y)}
            />
            <text
              className="chart-tick"
              x={WIDTH - PAD.right + 8}
              y={y(tick.y)}
              dy="0.32em"
            >
              {tick.label}
            </text>
          </g>
        ))}
        {rightLabel && rightTicks.length ? (
          <text
            className="chart-axis-label"
            transform={`translate(${WIDTH - 10} ${PAD.top + 4}) rotate(90)`}
          >
            {rightLabel}
          </text>
        ) : null}

        {markers.map((marker) => (
          <g key={marker.label}>
            <circle className="chart-marker" cx={x(marker.x)} cy={y(marker.y)} r={4.5} />
            <text
              className="chart-marker-label"
              x={x(marker.x)}
              y={y(marker.y) - 10}
              textAnchor="middle"
            >
              {marker.label}
            </text>
          </g>
        ))}

        {hoverX !== null ? (
          <line
            className="chart-crosshair"
            x1={hoverX}
            x2={hoverX}
            y1={PAD.top}
            y2={height - PAD.bottom}
          />
        ) : null}

        <text className="chart-axis-label" x={PAD.left} y={height - 4}>
          {xLabel}
        </text>
        <text
          className="chart-axis-label"
          transform={`translate(11 ${PAD.top + 4}) rotate(-90)`}
          textAnchor="end"
        >
          {yLabel}
        </text>
      </svg>

      {hover && hover.readouts.length ? (
        <div className="chart-readout">
          <span className="at">
            {xLabel.split(" ")[0]} {format(hover.at)}
          </span>
          {hover.readouts
            .filter((r) => !r.muted)
            .slice(0, 4)
            .map((r) => (
              <span key={r.name}>
                <strong>{r.name}</strong> {format(r.value)}
              </span>
            ))}
        </div>
      ) : caption ? (
        <figcaption>{caption}</figcaption>
      ) : (
        <figcaption>&nbsp;</figcaption>
      )}
    </figure>
  );
}

/** Right-hand label positions, pushed apart so two close series stay readable. */
function labelPositions(series: Series[], y: Scale, height: number) {
  const placed = series
    .filter((s) => s.label)
    .map((s) => {
      let py = height / 2;
      for (let i = s.ys.length - 1; i >= 0; i--) {
        const yv = s.ys[i];
        const xv = s.xs[i];
        if (yv !== undefined && xv !== undefined && Number.isFinite(yv) && Number.isFinite(xv)) {
          py = y(yv);
          break;
        }
      }
      return { series: s, py };
    })
    .sort((a, b) => a.py - b.py);

  const MIN_GAP = 12;
  for (let i = 1; i < placed.length; i++) {
    const previous = placed[i - 1];
    const current = placed[i];
    if (previous && current && current.py - previous.py < MIN_GAP) {
      current.py = previous.py + MIN_GAP;
    }
  }
  return placed;
}
