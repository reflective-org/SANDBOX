# ADR-006 — Every run records its provenance, immutably

**Status:** Accepted (2026-08-13)

## Context

Requirement: any figure can be traced to an exact configuration, model version, and input dataset
checksum.

What exists today falls short of that in a specific, fixable way. Run identity is a **hand-composed
string** — `"__".join([...])` at `coupled/paper_ensemble/run_ensemble.py:84`, giving e.g.
`30N_20km__sabr220__D2med__a1p0__nuc1__cg1`. This is genuinely good design for a fixed factorial:
the name *is* the parameter set, it is greppable, it round-trips (`_tokens()` at
`make_paper_candidate_plots.py:92`), and figure scripts construct paths from tokens rather than
looking them up.

But it breaks down for an app:

- It only works because every axis has a short symbolic level. Continuous parameters have no token.
- It captures the *axes*, not the resolved configuration. Everything fixed across the ensemble —
  `ion_pair_rate=30.0`, `so2_ho2_rate=1e-18`, `day_of_year=172`, the background gas composition — is
  invisible in the identity, so two ensembles differing only in a "fixed" value collide.
- **There is no hashing anywhere in the repository.** No `hashlib`, no content-addressed run IDs, no
  git SHA capture, no version on the coupling layer.
- Idempotence is by filename: `run_ensemble.py:194` skips a case if `<case_id>/state.npz` exists. If
  the model changed underneath, the stale result is silently kept.

`manifest.csv` (`run_ensemble.py:110,124`) is the closest existing thing — 27 columns of resolved
parameters, one row per case — and is the natural seed for the provenance record.

## Decision

Every run records, at submit time and immutably:

- **`config_hash`** — stable SHA-256 over the canonical JSON serialisation of the resolved config.
- **App version** — `studio.__version__`, and the SANDBOX commit SHA.
- **Model version** — the SANDBOX commit SHA plus the **three submodule SHAs** (`tuvx-jax`,
  `stratchem-jax`, `tomas-jax`), which together pin the model exactly (ADR-001). Dirty working trees
  are recorded as such and flag the run.
- **Dataset identifiers and checksums** for every input dataset consulted.
- **The resolved, post-derivation parameter set** — not just the user's inputs. What the model
  actually received.

This record is written before execution begins and is never mutated. Configs are immutable once
submitted (ADR-004); an edit produces a new config and a new run.

`config_hash` doubles as the cache key: identical hash + identical model version + identical dataset
checksums means the existing result may be offered instead of recomputing. **A cached result is
always shown as cached**, with its original run date — never passed off as fresh.

## Consequences

- Human-readable case IDs do not go away; they remain as a *label*, because they are genuinely useful
  and the existing ensemble depends on them. Identity is the hash; the label is for people.
- Hash stability is a tested property — across dict insertion order and across Python versions —
  because a hash that drifts silently invalidates every cache and every golden fixture.
- Recording submodule SHAs requires reading them at submit time (`git rev-parse HEAD` per submodule).
  This is the one place `studio` shells out to git, and it must handle "not a git checkout" by
  raising rather than recording an empty SHA.
- The existing 810-run ensemble **has no provenance record** beyond `manifest.csv` and cannot be
  retrofitted with one. Golden fixtures derived from it therefore record the SHA at which the
  reference was *measured*, not the SHA at which it was originally produced — an honest limitation,
  documented in `ASSUMPTIONS.md`.

## Alternatives rejected

**Hash the user's input config rather than the resolved one.** Cheaper and stable across derivation
changes, but two different resolved configurations could then share a hash after a derivation is
fixed — exactly the collision the cache must not have.

**Record a version string instead of git SHAs.** Requires a release process the model does not have
(ADR-001), and version strings drift from what was actually executed.

**Trust the case-ID convention and extend it.** Rejected on the continuous-parameter and
invisible-fixed-value grounds above; the convention is retained as a label, not as identity.
