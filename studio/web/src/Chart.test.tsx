// Copyright (C) 2026 University Corporation for Atmospheric Research
// SPDX-License-Identifier: Apache-2.0
/**
 * Axis orientation, pinned by rendered pixel positions.
 *
 * Reported from use: the ERA5 profile drew 300 hPa at the TOP of the frame — an upside-down
 * atmosphere — while its own caption claimed otherwise. The default y mapping (domain minimum at
 * the bottom) is right for every quantity that grows upward and exactly wrong for pressure as a
 * vertical coordinate, which falls with altitude. `yReverse` exists for that case, and this file
 * asserts which way is up in both modes, by reading marker positions out of the rendered SVG.
 */

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { Chart } from "./Chart";

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

/** cy of the marker labelled `label`, in SVG user units. */
function markerY(label: string): number {
  const labels = [...container.querySelectorAll("text.chart-marker-label")];
  const hit = labels.find((el) => el.textContent === label);
  if (!hit) throw new Error(`no marker labelled ${label}`);
  const circle = hit.parentElement?.querySelector("circle.chart-marker");
  return Number(circle?.getAttribute("cy"));
}

function render(yReverse: boolean) {
  act(() => {
    root.render(
      <Chart
        series={[{ name: "profile", xs: [200, 220], ys: [5, 300] }]}
        xLabel="temperature (K)"
        yLabel="pressure (hPa)"
        yLog
        yReverse={yReverse}
        yDomain={[5, 300]}
        markers={[
          { x: 200, y: 5, label: "high-altitude" },
          { x: 220, y: 300, label: "low-altitude" },
        ]}
      />,
    );
  });
}

describe("pressure as a vertical coordinate", () => {
  it("yReverse puts high pressure at the bottom -- the atmosphere the right way up", () => {
    render(true);
    // SVG y grows downward, so "at the bottom" means the LARGER cy.
    expect(markerY("low-altitude")).toBeGreaterThan(markerY("high-altitude"));
  });

  it("the default keeps the domain minimum at the bottom, for quantities that grow upward", () => {
    render(false);
    expect(markerY("high-altitude")).toBeGreaterThan(markerY("low-altitude"));
  });

  it("ticks are drawn in both modes -- reversal swaps the pixel range, not the domain", () => {
    render(true);
    const ticks = [...container.querySelectorAll("text.chart-tick")].map((el) => el.textContent);
    expect(ticks.length).toBeGreaterThan(3);
    // The log-decade y ticks: plain numbers below 1e4 (tickLabel's threshold for superscripts).
    expect(ticks).toContain("10");
    expect(ticks).toContain("100");
  });
});
