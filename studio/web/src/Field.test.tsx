// Copyright (C) 2026 University Corporation for Atmospheric Research
// SPDX-License-Identifier: Apache-2.0
/**
 * Typing into a field, keystroke by keystroke.
 *
 * This is the test that was missing. Every other check set a whole value at once -- the unit tests
 * call `coerce("15000")`, and even the browser smoke script assigns the finished string -- so
 * nothing exercised the thing a person actually does, which is press five keys in a row. In use,
 * the first keypress committed "1" to the server, the form re-rendered, the in-flight request
 * disabled the input, and focus was gone before the second digit. Reported from the wizard, not
 * caught here, which is exactly why it is here now.
 *
 * The rule these tests pin: a text field is a draft until the user says otherwise. Keystrokes cost
 * nothing; Enter or blur costs one request.
 */

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Field } from "./Field";
import type { FieldSpec } from "./schema";

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

const numberSpec: FieldSpec = {
  path: "injection.plume_length_m",
  label: "Plume length",
  description: "",
  unit: "m",
  kind: "number",
  choices: [],
  default: 15000,
  provenance: "paper_ensemble",
  source: "",
  cite: "",
  derivedFrom: [],
  derived: false,
  nullable: false,
};

const enumSpec: FieldSpec = {
  ...numberSpec,
  path: "microphysics.n_bins",
  label: "Size bins",
  kind: "enum",
  choices: [40, 80, 160],
  default: 80,
};

function render(spec: FieldSpec, value: unknown, onChange: (p: string, v: unknown) => void) {
  act(() => {
    root.render(
      <Field
        spec={spec}
        value={value}
        overridden={false}
        stale={undefined}
        disabled={false}
        onChange={onChange}
        onAccept={() => {}}
        onKeep={() => {}}
      />,
    );
  });
}

/** One keystroke, the way a browser delivers it: the whole new value plus an input event. */
function type(input: HTMLInputElement, text: string) {
  act(() => {
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
    setter?.call(input, text);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

function press(input: HTMLInputElement, key: string) {
  act(() => {
    input.dispatchEvent(new KeyboardEvent("keydown", { key, bubbles: true }));
  });
}

describe("typing a multi-digit number", () => {
  it("sends nothing until the user is done", () => {
    const onChange = vi.fn();
    render(numberSpec, 15000, onChange);
    const input = container.querySelector("input") as HTMLInputElement;

    // "20000", one key at a time.
    for (const text of ["2", "20", "200", "2000", "20000"]) type(input, text);

    expect(onChange).not.toHaveBeenCalled();
    expect(input.value).toBe("20000");
  });

  it("commits once, with the whole number, on Enter", () => {
    const onChange = vi.fn();
    render(numberSpec, 15000, onChange);
    const input = container.querySelector("input") as HTMLInputElement;

    for (const text of ["2", "20", "200", "2000", "20000"]) type(input, text);
    press(input, "Enter");

    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange).toHaveBeenCalledWith("injection.plume_length_m", 20000);
  });

  it("commits on blur, for the user who clicks away instead", () => {
    const onChange = vi.fn();
    render(numberSpec, 15000, onChange);
    const input = container.querySelector("input") as HTMLInputElement;

    type(input, "12500");
    // React binds onBlur to `focusout`, not `blur` -- `blur` does not bubble, so React never sees a
    // synthetic one. A real browser fires both; the smoke script covers that end.
    act(() => input.dispatchEvent(new FocusEvent("focusout", { bubbles: true })));

    expect(onChange).toHaveBeenCalledExactlyOnceWith("injection.plume_length_m", 12500);
  });

  it("abandons the edit on Escape", () => {
    const onChange = vi.fn();
    render(numberSpec, 15000, onChange);
    const input = container.querySelector("input") as HTMLInputElement;

    type(input, "999");
    press(input, "Escape");

    expect(onChange).not.toHaveBeenCalled();
    expect(input.value).toBe("15000");
  });

  it("says nothing when the value is committed unchanged", () => {
    const onChange = vi.fn();
    render(numberSpec, 15000, onChange);
    const input = container.querySelector("input") as HTMLInputElement;

    type(input, "15000");
    press(input, "Enter");

    expect(onChange).not.toHaveBeenCalled();
  });

  it("shows the reader that an edit is pending", () => {
    render(numberSpec, 15000, vi.fn());
    const input = container.querySelector("input") as HTMLInputElement;
    expect(container.querySelector(".draft-hint")).toBeNull();
    type(input, "20000");
    expect(container.querySelector(".draft-hint")?.textContent).toMatch(/Enter/);
  });
});

describe("values arriving from elsewhere", () => {
  it("takes a new server value while the field is idle", () => {
    const onChange = vi.fn();
    render(numberSpec, 15000, onChange);
    let input = container.querySelector("input") as HTMLInputElement;
    expect(input.value).toBe("15000");

    // e.g. the user accepted a recomputed value on another field.
    render(numberSpec, 30000, onChange);
    input = container.querySelector("input") as HTMLInputElement;
    expect(input.value).toBe("30000");
  });

  it("does not yank the text out from under someone mid-edit", () => {
    const onChange = vi.fn();
    render(numberSpec, 15000, onChange);
    const input = container.querySelector("input") as HTMLInputElement;

    type(input, "123");
    render(numberSpec, 30000, onChange); // a re-render lands while they are typing
    expect((container.querySelector("input") as HTMLInputElement).value).toBe("123");
  });
});

describe("discrete controls stay immediate", () => {
  it("commits a dropdown as soon as it changes, with the schema's type", () => {
    const onChange = vi.fn();
    render(enumSpec, 80, onChange);
    const select = container.querySelector("select") as HTMLSelectElement;

    act(() => {
      const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value")?.set;
      setter?.call(select, "40");
      select.dispatchEvent(new Event("change", { bubbles: true }));
    });

    // One interaction is one decision, so there is nothing to draft -- and 40, not "40" (#89).
    expect(onChange).toHaveBeenCalledExactlyOnceWith("microphysics.n_bins", 40);
  });
});
