// Copyright (C) 2026 University Corporation for Atmospheric Research
// SPDX-License-Identifier: Apache-2.0
/**
 * The review diff.
 *
 * The first test is a regression. The diff originally compared against the JSON Schema's `default`
 * keys, and Pydantic does not emit those for fields built by a `default_factory` -- so an untouched
 * configuration reported "2 fields differ from the defaults", one of them displaying `—` against
 * `—`. A review screen that invents two changes is worse than no review screen: it teaches the
 * reader to ignore the count, which is the one number on that page that has to be trusted.
 */

import { describe, expect, it } from "vitest";
import { diffRows } from "./Review";
import type { FieldSpec } from "./schema";
import type { LayoutManifest, ResolvedPayload } from "./types";

const layout: LayoutManifest = {
  first_stage: "one",
  hidden_fields: ["schema_version"],
  stages: [
    {
      id: "one",
      number: 1,
      title: "One",
      blurb: "b",
      spec_ref: "5.1",
      blocked_on: [],
      sections: [
        {
          title: "S",
          note: "",
          fields: [
            "site.temperature_k",
            "background.gas_pptv",
            "injection.plume_volume_cm3",
          ],
        },
      ],
    },
  ],
};

function spec(path: string, over: Partial<FieldSpec> = {}): FieldSpec {
  return {
    path,
    label: path,
    description: "",
    unit: "",
    kind: "number",
    choices: [],
    default: undefined,
    provenance: "model_default",
    source: "",
    cite: "",
    derivedFrom: [],
    derived: false,
    nullable: false,
    ...over,
  };
}

const specs = new Map<string, FieldSpec>([
  ["site.temperature_k", spec("site.temperature_k")],
  ["background.gas_pptv", spec("background.gas_pptv", { kind: "number_map" })],
  [
    "injection.plume_volume_cm3",
    spec("injection.plume_volume_cm3", { derived: true, nullable: true }),
  ],
]);

/** What the reference config looks like: a real dict default, and a computed volume. */
const reference = {
  site: { temperature_k: 210 },
  background: { gas_pptv: { O2: 2.1e11, O3: 1.18e6 } },
  injection: { plume_volume_cm3: 1.5e12 },
};

function payloadOf(
  config: Record<string, unknown>,
  overrides: Record<string, { value: unknown; inputs: Record<string, unknown> }> = {},
): ResolvedPayload {
  return {
    config,
    config_hash: "0".repeat(64),
    overrides,
    consistent: true,
    stale_fields: [],
    stale: [],
    derived: {},
  };
}

describe("diffRows", () => {
  it("reports no changes for a config identical to the reference", () => {
    const rows = diffRows(layout, specs, payloadOf(structuredClone(reference)), reference);
    expect(rows).toHaveLength(3);
    expect(rows.filter((r) => r.changed)).toEqual([]);
  });

  it("does not mistake a default_factory dict for a change (the regression)", () => {
    // The exact shape that broke: a field with a real dict value whose schema carries no `default`.
    const rows = diffRows(layout, specs, payloadOf(structuredClone(reference)), reference);
    const gas = rows.find((r) => r.path === "background.gas_pptv");
    expect(gas?.changed).toBe(false);
    expect(gas?.defaultValue).toEqual({ O2: 2.1e11, O3: 1.18e6 });
  });

  it("reports exactly the field that moved", () => {
    const config = structuredClone(reference);
    config.site.temperature_k = 216.8;
    const changed = diffRows(layout, specs, payloadOf(config), reference).filter((r) => r.changed);
    expect(changed.map((r) => r.path)).toEqual(["site.temperature_k"]);
    expect(changed[0]?.defaultValue).toBe(210);
    expect(changed[0]?.value).toBe(216.8);
  });

  it("counts a derived field only when it is pinned", () => {
    // Auto: the value moved because an input moved. That is the machinery working, not a decision,
    // and counting it would bury the real edit among its own consequences.
    const moved = structuredClone(reference);
    moved.injection.plume_volume_cm3 = 2.0e12;
    const auto = diffRows(layout, specs, payloadOf(moved), reference);
    expect(auto.find((r) => r.path === "injection.plume_volume_cm3")?.changed).toBe(false);

    const pinned = diffRows(
      layout,
      specs,
      payloadOf(moved, { "injection.plume_volume_cm3": { value: 2.0e12, inputs: {} } }),
      reference,
    );
    const row = pinned.find((r) => r.path === "injection.plume_volume_cm3");
    expect(row?.changed).toBe(true);
    expect(row?.overridden).toBe(true);
  });

  it("ignores a laid-out field the schema could not describe rather than crashing", () => {
    const thin = new Map([["site.temperature_k", spec("site.temperature_k")]]);
    const rows = diffRows(layout, thin, payloadOf(structuredClone(reference)), reference);
    expect(rows.map((r) => r.path)).toEqual(["site.temperature_k"]);
  });
});
