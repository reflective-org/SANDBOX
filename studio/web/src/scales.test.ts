// Copyright (C) 2026 University Corporation for Atmospheric Research
// SPDX-License-Identifier: Apache-2.0
/**
 * The chart arithmetic, which is where a plot lies quietly if it lies at all: an axis that clamps a
 * zero on a log scale, a line drawn straight through missing data, a tick label implying precision
 * the data does not have.
 */

import { describe, expect, it } from "vitest";
import {
  extent,
  linearScale,
  linePath,
  logScale,
  nearestIndex,
  tickLabel,
} from "./scales";

describe("linearScale", () => {
  it("maps the domain onto the range, including inverted ranges", () => {
    const x = linearScale([0, 10], [0, 100]);
    expect(x(0)).toBe(0);
    expect(x(5)).toBe(50);
    expect(x(10)).toBe(100);
    // SVG y grows downward, so a y scale is built with an inverted range.
    const y = linearScale([0, 1], [200, 0]);
    expect(y(0)).toBe(200);
    expect(y(1)).toBe(0);
  });

  it("holds steady on a zero-width domain instead of dividing by zero", () => {
    const x = linearScale([5, 5], [0, 100]);
    expect(Number.isFinite(x(5))).toBe(true);
  });

  it("produces round ticks, free of floating-point dust", () => {
    expect(linearScale([0, 1], [0, 1]).ticks(5)).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1]);
    expect(linearScale([0, 24], [0, 1]).ticks(6)).toEqual([0, 5, 10, 15, 20]);
  });

  it("inverts, so a pointer position becomes a data value", () => {
    const x = linearScale([0, 24], [40, 640]);
    expect(x.invert(x(13.5))).toBeCloseTo(13.5, 10);
  });
});

describe("logScale", () => {
  it("maps decades to equal pixel spans", () => {
    const x = logScale([1, 1000], [0, 300]);
    expect(x(1)).toBeCloseTo(0);
    expect(x(10)).toBeCloseTo(100);
    expect(x(1000)).toBeCloseTo(300);
  });

  it("refuses a non-positive domain rather than silently clamping it", () => {
    // Clamping would draw a confident axis over data that is wrong -- fail loud (ADR-005).
    expect(() => logScale([0, 10], [0, 1])).toThrow(/positive domain/);
    expect(() => logScale([-1, 10], [0, 1])).toThrow(/positive domain/);
  });

  it("ticks on decades inside the domain", () => {
    expect(logScale([1, 10000], [0, 1]).ticks(4)).toEqual([1, 10, 100, 1000, 10000]);
  });

  it("thins decades on a very wide domain rather than crowding the axis", () => {
    const ticks = logScale([1e-9, 1e9], [0, 1]).ticks(6);
    expect(ticks.length).toBeLessThanOrEqual(8);
    expect(ticks.length).toBeGreaterThan(2);
  });
});

describe("linePath", () => {
  const x = linearScale([0, 10], [0, 100]);
  const y = linearScale([0, 10], [100, 0]);

  it("draws a path through the points", () => {
    expect(linePath([0, 5, 10], [0, 5, 10], x, y)).toBe("M0.00,100.00L50.00,50.00L100.00,0.00");
  });

  it("breaks the line at missing data instead of interpolating across it", () => {
    const path = linePath([0, 5, 10], [0, NaN, 10], x, y);
    expect(path).toBe("M0.00,100.00M100.00,0.00");
    expect(path).not.toContain("L");
  });

  it("ignores trailing points with no partner", () => {
    expect(linePath([0, 5, 10], [0, 5], x, y)).toBe("M0.00,100.00L50.00,50.00");
  });
});

describe("tickLabel", () => {
  it("writes powers of ten as powers of ten", () => {
    expect(tickLabel(1000000)).toBe("10⁶");
    expect(tickLabel(1e-9)).toBe("10⁻⁹");
  });

  it("keeps ordinary numbers ordinary", () => {
    expect(tickLabel(0)).toBe("0");
    expect(tickLabel(24)).toBe("24");
    expect(tickLabel(0.12)).toBe("0.12");
  });

  it("falls back to an exponent for large non-decade values", () => {
    expect(tickLabel(3.4e7)).toBe("3.4e+7");
  });
});

describe("nearestIndex", () => {
  const xs = [0, 1, 2, 3, 4, 5];
  it("finds the closest point for a crosshair", () => {
    expect(nearestIndex(xs, 0)).toBe(0);
    expect(nearestIndex(xs, 2.4)).toBe(2);
    expect(nearestIndex(xs, 2.6)).toBe(3);
    expect(nearestIndex(xs, 99)).toBe(5);
    expect(nearestIndex(xs, -99)).toBe(0);
  });
  it("says -1 for no data rather than pointing at a point that is not there", () => {
    expect(nearestIndex([], 1)).toBe(-1);
  });
});

describe("extent", () => {
  it("covers the finite values", () => {
    expect(extent([3, 1, 2])).toEqual([1, 3]);
    expect(extent([1, NaN, 5])).toEqual([1, 5]);
  });
  it("drops non-positive values when the axis will be logarithmic", () => {
    expect(extent([0, 1, 10], true)).toEqual([1, 10]);
  });
  it("widens a single-valued domain so it can still be drawn", () => {
    expect(extent([5, 5])).toEqual([4, 6]);
    expect(extent([5, 5], true)).toEqual([2.5, 10]);
  });
  it("gives a usable domain for no data at all", () => {
    expect(extent([])).toEqual([0, 1]);
    expect(extent([], true)).toEqual([1, 10]);
  });
});
