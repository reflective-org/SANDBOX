// Copyright (C) 2026 University Corporation for Atmospheric Research
// SPDX-License-Identifier: Apache-2.0
//
// Manual smoke test: drives the real wizard in a real browser over the DevTools Protocol.
//
// The unit tests cover the generator and the diff; the Python tests cover the endpoints. Neither
// covers the thing a user actually does -- type into a field, watch a derived value follow, pin it,
// watch it go stale, and click accept. This does, against a running server, and it is how the
// missing upper bound in issue #91 was found (typing 9999 K and getting no refusal).
//
// Not in CI: it needs a built bundle, a running uvicorn and a real Chrome. Run it before merging a
// change to the wizard's state flow.
//
//   .venv/bin/uvicorn studio.api.app:app --port 8765 &
//   "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless \
//       --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-smoke about:blank &
//   node studio/web/scripts/smoke.mjs /tmp/shots
//
// Node 22's global WebSocket keeps this dependency-free -- no Playwright, no Puppeteer.
import { writeFileSync } from "node:fs";

const OUT = process.argv[2] ?? ".";
const PORT = 9222;
const base = `http://127.0.0.1:${PORT}`;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// --- find the page target -------------------------------------------------------------------
let targets = [];
for (let i = 0; i < 30; i++) {
  try {
    targets = await (await fetch(`${base}/json/list`)).json();
    if (targets.some((t) => t.type === "page")) break;
  } catch {}
  await sleep(500);
}
const page = targets.find((t) => t.type === "page");
if (!page) throw new Error("no page target; is chrome running with --remote-debugging-port?");

const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((res, rej) => {
  ws.onopen = res;
  ws.onerror = rej;
});

let nextId = 1;
const pending = new Map();
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.id && pending.has(msg.id)) {
    const { resolve, reject } = pending.get(msg.id);
    pending.delete(msg.id);
    msg.error ? reject(new Error(JSON.stringify(msg.error))) : resolve(msg.result);
  }
};
const send = (method, params = {}) =>
  new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ id, method, params }));
  });

/** Evaluate in the page, returning the value (throws on an in-page exception). */
async function evaluate(expression) {
  const result = await send("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  if (result.exceptionDetails) {
    throw new Error(`in-page error: ${JSON.stringify(result.exceptionDetails.exception?.value ?? result.exceptionDetails.text)}`);
  }
  return result.result.value;
}

async function shot(name) {
  const { data } = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
  writeFileSync(`${OUT}/${name}.png`, Buffer.from(data, "base64"));
  console.log(`  screenshot -> ${name}.png`);
}

/**
 * Set a React-controlled input. Assigning `.value` does not notify React, so this uses the native
 * setter and dispatches a bubbling input event -- which is what a real keystroke produces.
 */
const setValue = (selector, value) => `(() => {
  const el = document.querySelector(${JSON.stringify(selector)});
  if (!el) throw new Error('no element ' + ${JSON.stringify(selector)});
  const proto = el.tagName === 'SELECT' ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, ${JSON.stringify(String(value))});
  el.dispatchEvent(new Event('input', { bubbles: true }));
  el.dispatchEvent(new Event('change', { bubbles: true }));
  return el.value;
})()`;

/** Click by visible text, so the test depends on what the user sees, not on a CSS hook. */
const clickText = (text) => `(() => {
  const wanted = ${JSON.stringify(text)};
  const button = [...document.querySelectorAll('button')].find((b) => b.textContent.includes(wanted));
  if (!button) throw new Error('no button containing ' + wanted + '; buttons: ' + [...document.querySelectorAll('button')].map(b => b.textContent).join(' | '));
  button.click();
  return button.textContent.trim();
})()`;

const state = `(() => ({
  stageTabs: [...document.querySelectorAll('.stage-tab')].length,
  current: document.querySelector('.stage-tab.current .name')?.textContent,
  staleTabs: [...document.querySelectorAll('.stage-tab.has-stale .name')].map(e => e.textContent),
  badges: [...document.querySelectorAll('.badge')].map(e => e.textContent),
  staleBoxes: [...document.querySelectorAll('.stale-box')].map(e => e.textContent.replace(/\\s+/g, ' ').trim()),
  volume: document.querySelector('#injection\\\\.plume_volume_cm3')?.value,
  length: document.querySelector('#injection\\\\.plume_length_m')?.value,
  headerStale: document.querySelector('.header-stale')?.textContent?.trim() ?? null,
  error: document.querySelector('.error')?.textContent ?? null,
}))()`;

await send("Page.enable");
await send("Runtime.enable");

// A hash-only change does not remount the app, and re-running this script against an already-open
// page would otherwise silently start from whatever stage the last run left behind. Navigate, then
// reload, so every run starts from the same place.
console.log("1. load stage 2 (plume volume)");
await send("Page.navigate", { url: "http://127.0.0.1:8765/app/#plume_volume" });
await sleep(600);
await send("Page.reload", { ignoreCache: true });
await sleep(2500);
console.log("  ", JSON.stringify(await evaluate(state)));

console.log("2. type a new plume length -> derived volume must follow");
await evaluate(setValue("#injection\\.plume_length_m", "20000"));
await sleep(1200);
let s = await evaluate(state);
console.log("   length:", s.length, "volume:", s.volume, "badges:", s.badges.join(","));

console.log("3. type into the DERIVED volume box -> becomes a pinned override");
await evaluate(setValue("#injection\\.plume_volume_cm3", "9.9e11"));
await sleep(1200);
s = await evaluate(state);
console.log("   volume:", s.volume, "badges:", s.badges.join(","), "stale:", s.staleBoxes.length);
await shot("drive-2-override");

console.log("4. move the input underneath it -> the override must go stale");
await evaluate(setValue("#injection\\.plume_length_m", "12000"));
await sleep(1400);
s = await evaluate(state);
console.log("   header:", s.headerStale, "| stale tabs:", s.staleTabs.join(","), "| badges:", s.badges.join(","));
console.log("   stale box:", s.staleBoxes[0]?.slice(0, 200));
await shot("drive-3-stale");

console.log("5. click the accept button (its label carries the recomputed value)");
const clicked = await evaluate(clickText("Use "));
await sleep(1400);
s = await evaluate(state);
console.log("   clicked:", JSON.stringify(clicked));
console.log("   volume now:", s.volume, "| stale boxes:", s.staleBoxes.length, "| header:", s.headerStale);
await shot("drive-4-accepted");

console.log("6. navigate to review and read the diff");
await evaluate(clickText("Review"));
await sleep(1200);
const review = await evaluate(`(() => ({
  changedCount: document.querySelector('.review-summary .big')?.textContent,
  consistent: document.querySelector('.review-summary .big.ok, .review-summary .big.bad')?.textContent,
  rows: [...document.querySelectorAll('table.diff tbody tr')].map(r => r.textContent.replace(/\\s+/g,' ').trim()),
  submitDisabled: document.querySelector('button.primary')?.disabled,
}))()`);
console.log("  ", JSON.stringify(review, null, 2).slice(0, 700));
await shot("drive-5-review");

// 9999 K is NOT refused -- the schema declares only gt=0 and no upper bound (issue #91). Use a
// value that genuinely violates the declared constraint, so this step tests error surfacing
// rather than quietly asserting a bound that does not exist.
console.log("7. a value the schema refuses must surface its message, not fail silently");
await evaluate(clickText("Environment"));
await sleep(900);
await evaluate(setValue("#site\\.temperature_k", "-5"));
await sleep(1300);
s = await evaluate(state);
console.log("   error shown:", s.error ? s.error.replace(/\s+/g, " ").slice(0, 200) : "NONE (BAD)");
await shot("drive-6-validation");

console.log("8. and 9999 K is accepted, which is issue #91 -- recorded here, not asserted as good");
await evaluate(setValue("#site\\.temperature_k", "9999"));
await sleep(1300);
s = await evaluate(state);
console.log("   error for 9999 K:", s.error ? "shown" : "NONE -- unbounded above, see #91");

ws.close();
console.log("done");
