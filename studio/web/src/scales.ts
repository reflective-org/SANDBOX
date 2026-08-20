// Copyright (C) 2026 University Corporation for Atmospheric Research
// SPDX-License-Identifier: Apache-2.0
/**
 * Scales and ticks for the preview panels.
 *
 * Hand-rolled rather than pulled from a plotting library. ADR-007 defers the library choice to "the
 * first interactive panel", and these five panels are small, static-domain line charts: a library
 * would add ~100 kB and its own idioms to draw what is a few dozen lines of arithmetic. The choice
 * genuinely arrives with the comparison view (Phase 7), where many series update interactively.
 *
 * The arithmetic is the part that silently lies -- a log scale that quietly clamps a zero, an axis
 * whose ticks imply a precision the data does not have -- so it lives here, separately, and is
 * tested directly.
 */

export interface Scale {
  (value: number): number;
  /** Domain [min, max] in data units. */
  domain: [number, number];
  /** Range [min, max] in pixels. */
  range: [number, number];
  ticks: (count?: number) => number[];
  /** Sub-decade positions (2..9 per decade) for a log scale; absent on linear scales. */
  minorTicks?: () => number[];
  /** Inverse, for turning a pointer position back into a data value. */
  invert: (pixel: number) => number;
}

/** Round a number to a "nice" step: 1, 2, 2.5 or 5 times a power of ten. */
function niceStep(rough: number): number {
  const power = Math.pow(10, Math.floor(Math.log10(rough)));
  const scaled = rough / power;
  if (scaled <= 1) return power;
  if (scaled <= 2) return 2 * power;
  if (scaled <= 2.5) return 2.5 * power;
  if (scaled <= 5) return 5 * power;
  return 10 * power;
}

export function linearScale(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  // A zero-width domain would divide by zero; hold the midpoint instead of producing Infinity.
  const span = d1 - d0 || 1;
  const scale = ((value: number) => r0 + ((value - d0) / span) * (r1 - r0)) as Scale;
  scale.domain = domain;
  scale.range = range;
  scale.invert = (pixel: number) => d0 + ((pixel - r0) / (r1 - r0 || 1)) * span;
  scale.ticks = (count = 5) => {
    const step = niceStep(Math.abs(span) / Math.max(1, count));
    const start = Math.ceil(Math.min(d0, d1) / step) * step;
    const out: number[] = [];
    for (let v = start; v <= Math.max(d0, d1) + step * 1e-9; v += step) {
      // Re-round each tick: accumulating `+= step` drifts into 0.30000000000000004 territory.
      out.push(Number((Math.round(v / step) * step).toPrecision(12)));
    }
    return out;
  };
  return scale;
}

/**
 * A base-10 log scale.
 *
 * Non-positive domain bounds are a caller error, not something to silently clamp: a size
 * distribution whose lower bound has become 0 means the data is wrong, and clamping would draw a
 * confident axis over it.
 */
export function logScale(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain;
  if (!(d0 > 0) || !(d1 > 0)) {
    throw new Error(`log scale needs a positive domain, got [${d0}, ${d1}]`);
  }
  const [r0, r1] = range;
  const l0 = Math.log10(d0);
  const l1 = Math.log10(d1);
  const span = l1 - l0 || 1;
  const scale = ((value: number) =>
    r0 + ((Math.log10(Math.max(value, Number.MIN_VALUE)) - l0) / span) * (r1 - r0)) as Scale;
  scale.domain = domain;
  scale.range = range;
  scale.invert = (pixel: number) => Math.pow(10, l0 + ((pixel - r0) / (r1 - r0 || 1)) * span);
  scale.minorTicks = () => {
    // 2..9 within each decade of the domain -- the sub-decade gridlines a log axis is read by.
    // Lines only, no labels: labelled minors would crowd the axis into noise.
    const out: number[] = [];
    for (let power = Math.floor(l0) - 1; power <= Math.ceil(l1); power++) {
      for (let mantissa = 2; mantissa <= 9; mantissa++) {
        const value = mantissa * Math.pow(10, power);
        if (value >= d0 && value <= d1) out.push(value);
      }
    }
    return out;
  };
  scale.ticks = (count = 6) => {
    const first = Math.floor(l0);
    const last = Math.ceil(l1);
    const decades: number[] = [];
    // Thin the decades rather than the labels: every decade drawn but every nth labelled would put
    // unlabelled lines on the axis, which read as data.
    const stride = Math.max(1, Math.ceil((last - first) / Math.max(1, count)));
    for (let power = first; power <= last; power += stride) {
      const value = Math.pow(10, power);
      if (value >= d0 * 0.999 && value <= d1 * 1.001) decades.push(value);
    }
    return decades;
  };
  return scale;
}

/** An SVG path through the points, skipping any the scales cannot place. */
export function linePath(
  xs: readonly number[],
  ys: readonly number[],
  x: Scale,
  y: Scale,
): string {
  const parts: string[] = [];
  let penDown = false;
  for (let i = 0; i < Math.min(xs.length, ys.length); i++) {
    const xv = xs[i];
    const yv = ys[i];
    if (xv === undefined || yv === undefined || !Number.isFinite(xv) || !Number.isFinite(yv)) {
      // A gap, not a straight line across it: interpolating over missing data invents data.
      penDown = false;
      continue;
    }
    const px = x(xv);
    const py = y(yv);
    if (!Number.isFinite(px) || !Number.isFinite(py)) {
      penDown = false;
      continue;
    }
    parts.push(`${penDown ? "L" : "M"}${px.toFixed(2)},${py.toFixed(2)}`);
    penDown = true;
  }
  return parts.join("");
}

/** Format a number for an axis tick: compact, and never implying precision it does not have. */
export function tickLabel(value: number): string {
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude >= 1e4 || magnitude < 1e-2) {
    const power = Math.round(Math.log10(magnitude));
    // Exact powers of ten are the common case on a log axis, and 10³ beats 1.0e+3 there.
    if (Math.abs(magnitude - Math.pow(10, power)) < magnitude * 1e-9) {
      return `10${superscript(power)}`;
    }
    return value.toExponential(1);
  }
  return String(Number(value.toPrecision(4)));
}

const SUPERSCRIPTS: Record<string, string> = {
  "-": "⁻",
  "0": "⁰",
  "1": "¹",
  "2": "²",
  "3": "³",
  "4": "⁴",
  "5": "⁵",
  "6": "⁶",
  "7": "⁷",
  "8": "⁸",
  "9": "⁹",
};

export function superscript(power: number): string {
  return String(power)
    .split("")
    .map((ch) => SUPERSCRIPTS[ch] ?? ch)
    .join("");
}

/** Index of the point nearest `value` in a sorted array -- for the hover crosshair. */
export function nearestIndex(sorted: readonly number[], value: number): number {
  if (!sorted.length) return -1;
  let low = 0;
  let high = sorted.length - 1;
  while (high - low > 1) {
    const mid = (low + high) >> 1;
    if ((sorted[mid] ?? 0) < value) low = mid;
    else high = mid;
  }
  const lowValue = sorted[low] ?? 0;
  const highValue = sorted[high] ?? 0;
  return Math.abs(value - lowValue) <= Math.abs(value - highValue) ? low : high;
}

/** Domain covering the finite values, padded, and safe to hand to a log scale when asked. */
export function extent(values: readonly number[], positiveOnly = false): [number, number] {
  const usable = values.filter((v) => Number.isFinite(v) && (!positiveOnly || v > 0));
  if (!usable.length) return positiveOnly ? [1, 10] : [0, 1];
  const min = Math.min(...usable);
  const max = Math.max(...usable);
  if (min === max) return positiveOnly ? [min / 2, max * 2] : [min - 1, max + 1];
  return [min, max];
}
