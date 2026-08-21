// Copyright (C) 2026 University Corporation for Atmospheric Research
// SPDX-License-Identifier: Apache-2.0
/**
 * The results view, against a fixture shaped like a REAL summary.
 *
 * The first version invented its own shape -- headline scalars on the summary, a key named
 * `size_distribution` -- and rendered four zeros over two charts. The fixture here mirrors what
 * `/api/runs/{id}/summary` and `/api/runs` actually return (headline on the RUN, spectrum under
 * `final_size_distribution`), so a drift in either direction fails a test instead of rendering
 * confident zeros.
 */

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Results, type RunSummaryPayload } from "./Results";
import type { RunBrief } from "./types";

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

const run: RunBrief = {
  run_id: "abcd1234efgh",
  label: "first run from the wizard",
  config_hash: "d".repeat(64),
  created_at: "2026-08-21T00:00:00Z",
  reproducible: true,
  state: "succeeded",
  exit_code: 0,
  detail: "completed",
  termination: "completed",
  headline: {
    final_so2_pptv: 1.7e6,
    peak_h2so4_pptv: 15.05,
    peak_number_cm3: 3.1e6,
    final_surface_area: 211.6,
    flags: ["open_system_dilution"],
  },
};

const summary: RunSummaryPayload = {
  time_days: [0, 0.5, 1],
  termination: "completed",
  flags: ["open_system_dilution"],
  series: {
    SO2: { values: [3.3e9, 1e8, 1.7e6], unit: "pptv", basis: "n/a", description: "" },
    H2SO4: { values: [0.001, 15.05, 2.0], unit: "pptv", basis: "n/a", description: "" },
    total_n: { values: [0.03, 3.1e6, 9.9e3], unit: "cm^-3", basis: "dry", description: "" },
  },
  final_size_distribution: {
    diameter_um: [0.01, 0.05, 0.2],
    dn_dlogdp_cm3: [100, 20000, 50],
    total_number_cm3: 11227.3,
    basis: "dry",
  },
};

describe("Results", () => {
  it("shows the headline from the RUN and all three charts from the summary", () => {
    act(() => root.render(<Results run={run} summary={summary} onClose={() => {}} />));
    const tiles = [...container.querySelectorAll(".stat-value")].map((el) => el.textContent);
    expect(tiles.join(" | ")).toContain("15.05 pptv");
    expect(tiles.join(" | ")).not.toContain("0 pptv"); // the confident-zeros regression
    expect(container.querySelectorAll(".results-charts figure.chart")).toHaveLength(3);
    expect(container.textContent).toContain("open_system_dilution");
    expect(container.textContent).toContain("11227.3");
  });

  it("says what is happening while there is no summary, instead of an empty shell", () => {
    act(() =>
      root.render(
        <Results run={{ ...run, state: "running", headline: null }} summary={null} onClose={() => {}} />,
      ),
    );
    expect(container.textContent).toContain("the run is running");
    expect(container.querySelectorAll("figure.chart")).toHaveLength(0);
  });

  it("close goes back to configuring", () => {
    const onClose = vi.fn();
    act(() => root.render(<Results run={run} summary={summary} onClose={onClose} />));
    act(() => {
      (
        [...container.querySelectorAll("button")].find((b) =>
          b.textContent?.includes("back to configure"),
        ) as HTMLButtonElement
      ).click();
    });
    expect(onClose).toHaveBeenCalledOnce();
  });
});
