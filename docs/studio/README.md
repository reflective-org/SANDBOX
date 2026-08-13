# Plume Studio — documentation

Plume Studio is an interactive front-end for configuring, running and comparing Lagrangian box-model
simulations of aerosol microphysics and chemistry in a dispersing stratospheric plume. It
**configures and orchestrates**; it does not reimplement any physics. The model is `coupled/` and its
three submodules.

Code lives in [`../../studio/`](../../studio/). This directory is namespaced under the repository's
existing `docs/` tree, which belongs to the model (ASSUMPTION-6).

## Start here

| If you want to… | Read |
|---|---|
| Understand why it is built this way | [`adr/`](adr/) — start with the divergences table in [`adr/README.md`](adr/README.md) |
| Know what is undecided | [`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md) |
| Know what was assumed to unblock work | [`ASSUMPTIONS.md`](ASSUMPTIONS.md) |
| Know what the results do *not* mean | [`CAVEATS.md`](CAVEATS.md) |
| Look up a term | [`GLOSSARY.md`](GLOSSARY.md) |
| See what has shipped | [`PROGRESS.md`](PROGRESS.md) |
| Follow the current phase | [`plan/PHASE_0.md`](plan/PHASE_0.md) |
| Work on it as an agent | [`../../studio/CLAUDE.md`](../../studio/CLAUDE.md) |

## What it is for, in priority order

1. **Correct, provenanced defaults.** Nobody should hand-look-up a tropopause height, stratospheric
   water vapour, or a background ionisation rate. Every default carries its source.
2. **Ensembles as a first-class concept.** The motivating pain point is configuring many related
   runs, so `RunSet` is the primary object and a single run is the N = 1 case of it.
3. **Reproducibility.** Any figure traces to an exact configuration, model version and input dataset
   checksum.
4. **Interactive intuition.** Dilution curves, size distributions and photolysis/OH diurnal cycles
   that update as parameters change, *before* committing compute.
5. **An education layer.** In-app explanation of the chemistry scheme, TUV-x and TOMAS, so a new user
   can make defensible choices.

## Explicit non-goals for v1

- Not a replacement for the model.
- Not a 3-D or multi-parcel plume model — single Lagrangian box.
- Not a public multi-tenant service (BLOCKING-2).
- **No new science.** v1 must reproduce existing trusted runs within a documented tolerance.

## The two things designed in depth up front

Deliberately, only two: the **configuration schema** and the **execution model**. Everything else is
planned one phase ahead, against working code. The failure mode being avoided is a large set of
planning documents that diverge from the code within two weeks.

## Relationship to `coupled/viz/`

None, by decision. The existing self-contained HTML pages (`plume_dynamics.html`, `inverse_lab.html`,
`paper_story.html`) and their Node headless-verification harness are untouched and share no code with
Studio. Their hard-won *lessons* — the sampling-aliasing trap, the wet/dry diameter trap, the time-axis
drift — are recorded in [`CAVEATS.md`](CAVEATS.md) and enforced by Studio's own tests.
