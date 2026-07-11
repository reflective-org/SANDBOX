# D1 plume globe — interactive visualization

`d1_globe.html` is a single, self-contained web page that tells the story of the **D1
clean-stratosphere dilution run** (`coupled/run_dilution_d1_clean.py`) for a non-specialist
audience: an SO₂ plume injected at 20 km / 30°N, drifting and shearing over a stylized Earth while
its chemistry and aerosol microphysics play out in synchronized panels.

It is the interactive "v2" of the matplotlib GIF in `coupled/animate_d1_plume.py` (v1).

## Open it

Just open the file — no build step, no server, no network:

```
open coupled/viz/d1_globe.html          # macOS
```

It also works as a hosted page (GitHub Pages, any static host) and as a Claude Artifact. The whole
thing is ~160 KB with all data, coastlines, and code inlined.

### Controls
- **Play / pause** ▶, speed **0.5–4×**, draggable **scrubber** (time is warped so the first-day
  nucleation burst gets proportional room; diamonds mark narrative beats).
- **Keyboard:** Space = play/pause, ←/→ step (Shift = bigger), Home/End = jump to start/end.
- **Drag** on any time-axis chart to scrub.
- Respects `prefers-reduced-motion` (starts paused).
- URL hash for deep links / testing: `#t=<day>&paused=1` (e.g. `d1_globe.html#t=2&paused=1`).

## Re-bake the data

`bake_d1_globe.py` reads a saved run and injects a minified JSON blob between the
`/*__D1_DATA_BEGIN__*/ … /*__D1_DATA_END__*/` markers in the HTML. Edit the HTML freely; re-baking
only rewrites that one line.

```
python coupled/viz/bake_d1_globe.py [npz_path] [--html d1_globe.html] [--no-net]
```

Default `npz_path` is the 80-bin run in the sibling working copy
(`../SANDBOX/coupled/analyses/d1_clean_80bin/d1_clean.npz`). On first run it downloads Natural Earth
110 m coastlines into `cache/` (gitignored); pass `--no-net` to require the cache. The bake prints a
per-section byte report and hard-fails if the file would exceed 800 KB.

## What is real, and what is illustrative

**Real model output** (from the coupled gas-chemistry + TUV-x photolysis + TOMAS aerosol run):
- the size distribution dN/dlogDₚ, total particle number, effective radius;
- the sulfur budget (SO₂ gas → particulate sulfate), dilution factor V(t)/V₀;
- day/night for the **chemistry** (from the run's photolysis rates), shown as the chart night bands
  and the "daylight/night (model)" readout.

**Illustrative, not model output** — a climatological sketch to give the box model a place on Earth:
- **Drift**: the plume centroid is advected by a hand-specified June ~50 hPa easterly climatology
  (subtropical summer easterlies, ~8 m/s at 30°N), carrying it to the mid-Atlantic (~70°W) by day 10.
- **Shear / filament**: vertical wind shear (~2 m/s per km) acting over an assumed layer depth
  growing 10 m → ~1 km stretches the 30 km track into a ~1,100 km along-wind filament. Plume **width
  is exaggerated ~×120** so it is visible at globe scale; **length is to scale**. The altitude inset
  shows the classic shear picture (a layer tilting into a sloped sliver).
- **Globe terminator**: physically correct solar geometry for the date/UTC — but note the model's
  photolysis was computed at the *fixed* injection point (0°E), so late in the run the drifting
  plume's local solar time differs from the globe terminator by up to ~4.5 h. The legend states this.

These caveats are surfaced in the on-screen legend, not buried here.

## How it's built (for the next editor)

One HTML file, three parts: a `<style>` block, a `<script type="application/json">` data island, and
one IIFE. The IIFE is sectioned:

- **§A** decode (base64 int16 → typed arrays)
- **§B** geo/solar math (orthographic projection, subsolar point)
- **§C** wind/plume (centroid path + shear filament — the illustrative layer)
- **§D** time-warp + interpolation helpers
- **§E** globe renderer (Canvas 2D: disc, graticule, coastlines, terminator, sun, plume ribbon)
- **§F** altitude side-view inset
- **§G** microphysics charts (size distribution, sulfur budget, particle count)
- **§H** state fan-out (`setSimDay`) + `Player` (rAF, camera easing) + controls
- **§I** debug: `window.__d1` ({ ready, simDay, subsolarLon, endLon, filamentKm, sample(), bench() })
  for headless verification.

Verify headlessly with Chrome/Playwright: screenshot at `#t=…&paused=1`, or read `window.__d1`.
`__d1.bench()` returns full-update timing (≈0.4 ms p95 — comfortably 60 fps).

## Out of scope (future ideas)

- Horizontal stretching-and-folding (the schematic's 2-D turbulence row) — needs a real turbulence
  model; would blur the honest/illustrative line.
- Real ERA5 monthly-mean winds baked in (spatially varying, path curves with the anticyclone).
- A D1/D2/D3 dilution-regime "race" using the same layout.
- A photoreal (three.js / Blue Marble) globe.
