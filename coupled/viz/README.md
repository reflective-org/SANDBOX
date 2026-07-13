# Plume globe — interactive visualization

`d1_globe.html` is a single, self-contained web page that tells the story of an SO₂ plume in the
stratosphere for a non-specialist audience: injected as a 10 m-wide trail, drifting and shearing
over a stylized Earth while its chemistry and aerosol microphysics play out in synchronized panels.

It shows a **grid of paper-ensemble runs** selectable from three dropdowns:

- **Dilution parameterization** — D1 low Kz, D2 medium (default), D3 high, D5 very high, and the
  transient turbulence burst (Schumann-type V(t)/V₀ expansions; see
  `paper_ensemble/TABLE_dilution_parameters.md` on the `paper/simulations` branch).
- **Background aerosol** — SABR 220 (N₂O-aged air, default), SABR 330 (N₂O-young air), and the
  AER-2D geoengineered stratosphere (Pierce fig. 2; N = 120 cm⁻³, Dg = 0.30 µm — background SO₂
  100 pptv instead of 20).
- **Start location** — 30°N / 20 km (55 hPa) and 60°N / 15 km (120 hPa).

Every case runs until the plume is indistinguishable from the background — the timeline ends at
the first time the wet aerosol surface area has stayed within 10% of the background SA for 24 h
straight (the run-time stop criterion of `paper_ensemble/run_60day.py` / `run_bgstop.py`). That
lifetime depends on regime and background (≈3 days for a fast-mixing plume in the geoengineered
background up to ≈5 weeks for D1 in clean aged air), so each case has its own timeline length;
the header states it.

All cases share one scenario family (`paper_ensemble/run_ensemble.py` ICs,
1 t SO₂ into 10 m × 10 m × 15 km, release 06:00 local, doy 172, α×1 / nuc×1 / coag×1, 80 bins,
aerosol→J off). It is the interactive "v2" of the matplotlib GIF in `coupled/animate_d1_plume.py`.

## Open it

Just open the file — no build step, no server, no network:

```
open coupled/viz/d1_globe.html          # macOS
```

It also works as a hosted page (GitHub Pages, any static host) and as a Claude Artifact. The whole
thing stays under the 1.6 MB bake limit with all thirty cases, coastlines, and code inlined.

### Controls
- **Dropdowns** for dilution regime, background aerosol, and start location (state is kept across
  switches: the current sim day carries over, clamped to the new case's lifetime).
- **Play / pause** ▶, speed **0.5–4×**, draggable **scrubber** (time is warped so the first-day
  nucleation burst gets proportional room; diamonds mark narrative beats).
- **Keyboard:** Space = play/pause, ←/→ step (Shift = bigger), Home/End = jump to start/end.
- **Drag** on any time-axis chart to scrub.
- Respects `prefers-reduced-motion` (starts paused).
- URL hash for deep links / testing: `#t=<day>&paused=1&case=<site>|<background>|<regime>`
  (e.g. `d1_globe.html#t=2&paused=1&case=60N_15km|aergeo|D5vhigh`).

## Re-bake the data

`bake_d1_globe.py` reads the saved runs and injects a minified JSON blob between the
`/*__D1_DATA_BEGIN__*/ … /*__D1_DATA_END__*/` markers in the HTML. Edit the HTML freely; re-baking
only rewrites that one line.

```
python coupled/viz/bake_d1_globe.py [--runs-root ...] [--html d1_globe.html] [--no-net]
```

Default `--runs-root` is `coupled/paper_ensemble/` in the sibling working copy (the
`paper/simulations` branch), reading:

- `runs_bgstop/<site>__<bg>__<regime>__h06_bgstop/state.npz` — 60-day-max runs with the run-time
  background stop (`run_bgstop.py`; all cases except the two below);
- `runs_start_time/<site>__sabr220__{D3high,D5vhigh}__h06/state.npz` — 10-day runs that reach the
  criterion in-window; the bake truncates them at it;
- `runs_bgstop_ctrl/<site>__<bg>__<regime>__h06_ctrl/state.npz` — paired no-injection control
  runs (`run_bgstop_control.py`), one per case, for the dilution-corrected sulfur budget.

On first run it downloads Natural Earth 110 m coastlines into `cache/` (gitignored); pass
`--no-net` to require the cache. The bake prints per-case lifetimes and a per-section byte report,
and hard-fails if the file would exceed 800 KB.

## What is real, and what is illustrative

**Real model output** (from the coupled gas-chemistry + TUV-x photolysis + TOMAS aerosol runs):
- the size distribution dN/dlogDₚ, total particle number, effective radius;
- the sulfur panel, two toggleable views: **"vs control · tonnes"** (default) is the
  dilution-corrected budget against a PAIRED no-injection control run — same site, same dilution
  regime, so the dilution terms cancel in (plume − control)×V and the SO₂/particle split decays
  only through genuine chemistry differences (total conserved at exactly the injected 1 t; a
  naive excess-over-static-background budget instead claims 100% conversion the moment plume SO₂
  blends down to background). **"share of plume air"** is the raw, uncorrected per-cm³ partition,
  which converges to the background mix (dashed line);
- dilution factor V(t)/V₀ and each case's background-relaxation endpoint;
- day/night for the **chemistry** (from the run's photolysis rates), shown as the chart night bands
  and the "daylight/night (model)" readout.

**Illustrative, not model output** — a climatological sketch to give the box model a place on Earth:
- **Drift**: the plume centroid is advected by a hand-specified June ~50 hPa easterly climatology
  (~8 m/s at 30°N, ~2 m/s at 60°N).
- **Shear / filament**: vertical wind shear (~2 m/s per km) acting over an assumed layer depth
  growing 10 m → ~1 km stretches the 15 km track into a long along-wind filament. Plume **width is
  exaggerated per case** (the case's final width draws ~8 px, never below real scale — the legend
  states the factor); **length is to scale**. Ribbon brightness tracks the plume's
  distinguishability from background (SO₂ excess early, excess aerosol surface area late), so the
  veil fades out exactly where the timeline ends. The filament stays in the injection latitude
  band — consistent with the real interhemispheric mixing barrier, a NH plume does not cross into
  the SH on these timescales. The altitude inset shows the classic shear picture (a layer tilting
  into a sloped sliver).
- **Globe terminator**: physically correct solar geometry for the date/UTC — but note the model's
  photolysis was computed at the *fixed* injection point (0°E), so late in a run the drifting
  plume's local solar time differs from the globe terminator. The legend states this.

These caveats are surfaced in the on-screen legend, not buried here.

## How it's built (for the next editor)

One HTML file, three parts: a `<style>` block, a `<script type="application/json">` data island, and
one IIFE. The IIFE is sectioned:

- **§A** decode (base64 int16 coastlines, base64 int8 log-quantized size distributions)
- **§B** geo/solar math (orthographic projection, subsolar point)
- **§C** wind/plume (centroid path + shear filament — the illustrative layer)
- **§D** time-warp + interpolation helpers
- **§E** globe renderer (Canvas 2D: disc, graticule, coastlines, terminator, plume ribbon)
- **§F** altitude side-view inset
- **§G** microphysics charts (size distribution, sulfur budget, particle count; axes rescale
  per case)
- **§H** state fan-out (`setSimDay`) + `Player` (rAF, camera easing) + controls
- **§J** case switching (`applyCase` rebuilds the per-case derived state: path, time-warp,
  chart scales, track ticks, captions)
- debug: `window.__d1` ({ ready, caseKey, daysTotal, setCase(), listCases(), simDay, subsolarLon,
  endLon, filamentKm, sample(), bench() }) for headless verification.

Verify headlessly with Chrome/Playwright: screenshot at `#t=…&paused=1&case=…`, or read
`window.__d1`. `__d1.bench()` returns full-update timing (≈0.5 ms p95 — comfortably 60 fps).

## Out of scope (future ideas)

- Horizontal stretching-and-folding (the schematic's 2-D turbulence row) — needs a real turbulence
  model; would blur the honest/illustrative line.
- Real ERA5 monthly-mean winds baked in (spatially varying, path curves with the anticyclone).
- Side-by-side regime "race" (same layout, two cases in sync).
- A photoreal (three.js / Blue Marble) globe.
