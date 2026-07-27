# Headless checks for inverse_lab.html

`inverse_lab.html` is a measuring instrument: someone will read a number off it and put it in a
homework answer key, so its values are verified against the source `state.npz` rather than eyeballed.

These scripts run the page's **own JavaScript** (no copy of the math) under a stub DOM in Node,
including a small PNG decoder so the size-distribution raster path is genuinely exercised. No
browser, no npm dependencies.

```bash
# 1. drive the page, dump what it reports
node coupled/viz/tests_headless/harness.js coupled/viz/inverse_lab.html /tmp/probe.json

# 2. compare against the model output the bake read
python coupled/viz/tests_headless/check_page.py /tmp/probe.json

# 3. instructor mode, URL hash, controls, CSV shapes
node coupled/viz/tests_headless/harness2.js coupled/viz/inverse_lab.html

# 4. the "t =" time entry: parsing, snapping, clamping, write-back
node coupled/viz/tests_headless/harness3.js coupled/viz/inverse_lab.html
```

`check_page.py` needs the run outputs (`coupled/paper_ensemble/runs_bgstop*/`) on disk, same as the
bake. Expect every series to match the npz to ~3e-8 (float32 round-off), the tracer identity to
~1e-16, and the per-bin distribution to sit inside the raster's ±4.2% quantization bound.

For a visual pass, any browser will do; headless Chrome works too, but **not** with
`--force-dark-mode` (it drops the page CSS entirely — a headless quirk, not a page bug). To check
the dark palette, screenshot a copy with `data-theme="dark"` on the `<html>` element.
