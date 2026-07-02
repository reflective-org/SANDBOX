# Phase 1 sub-plan — Gas-phase SO2→SO3→H2SO4 (+ JPL 19-5 cross-check)

Tracking issue: #2. Milestone: "Phase 1". Each subtask below is its own `phase1/<task>` branch → PR.

## Goal
Give the frank-model gas chemistry a real gas-phase sulfuric-acid source (SO2 → SO3 → H2SO4) so a
later phase can hand gas H2SO4 to TOMAS. Keep it faithful: new species appended, new chemistry gated
to non-reference modes (reference mode stays MATLAB byte-identical for the original 34 species),
mirrored in the JAX backend, and every rate/product cross-checked against NASA-JPL Evaluation 19-5.

## Design
New species (appended, so indices 1–34 unchanged): `SO3` (35), `H2SO4` (36).

Gate: `Env.sulfur_chain` (True when `photolysis != "reference"`), set in `build_env`.

Reactions:
| reaction | rate | active when |
|---|---|---|
| `SO2 + OH -> HO2` (legacy) | `_k68` | reference only |
| `SO2 + HO2 ->` (legacy null sink) | `1e-18` | reference only |
| `SO2 + OH -> SO3 + HO2` | `_k68` | non-reference |
| `SO2 + HO2 -> SO3 + OH` | `1e-18` | non-reference |
| `SO3 + H2O -> H2SO4` | `k(T)·[H2O]` (Lovejoy/JPL, 2nd-order in H2O) | non-reference |

In reference mode the new reactions contribute 0 and SO3/H2SO4 stay 0 → species 1–34 tendencies are
byte-identical to today. Only the O(1D) split is unchanged; H2SO4 is a new terminal gas (no sink yet
— condensation arrives in Phase 3, so H2SO4 accumulates; expected).

## Subtasks (branch → PR each)
1. **`phase1/species`** — add SO3/H2SO4 to `config.SPECIES`; update `tests/test_config.py`
   (count 36, positions), add 2 zero rows to `tests/fixtures/rhs.csv`, `test_rhs.py` shape → `N_SPECIES`.
   No chemistry change; all tests green.
2. **`phase1/sulfur-chain-numpy`** — `Env.sulfur_chain` + `build_env`; the 3 gated reactions + `_k_so3_h2o`
   helper; gate legacy SO2 reactions to reference. Tests: sulfur mass conservation; gate off in
   reference (SO3=H2SO4=0, species 1–34 identical); gate on in tuvx.
3. **`phase1/sulfur-chain-jax`** — mirror the 3 reactions' coefficients in `jaxmodel/rates.py`
   (same REACTIONS order) + sulfur gate in `jaxmodel/chem.py` `build_params`; keep `test_jax_dcdt` parity.
4. **`phase1/jpl-crosscheck`** — verify each sulfur reaction's **rate constant AND product channel**
   vs `frank-model/references/NASA-JPL_Evaluation_19-5.pdf`: SO2+OH(+M)→HOSO2 (Table 2-1 I4), the
   lumped HOSO2+O2→SO3+HO2 step, SO2+HO2 (upper limit), SO3+H2O(+H2O)→H2SO4 (confirm Lovejoy/JPL value).
   Document exact table/section beside each rate in `reactions.py`; record in `docs/VALIDATION.md` +
   `docs/ASSUMPTIONS.md`; flag SO2+HO2 (no JPL recommendation).
5. **`phase1/run-and-verify`** — tuvx run showing H2SO4 production; sulfur-budget conservation
   diagnostic + plot; update `docs/PROGRESS.md`/`VALIDATION.md`; **3-agent verification gate**; tag
   `phase1-complete`.

## Open items to confirm during the phase
- **SO3 + H2O → H2SO4 rate value** vs JPL 19-5 (subtask 4). Result is insensitive (SO3 lifetime ≪ 1 s),
  but the number must be documented, not silent.
- Keep SO3 as an explicit intermediate (per the "full chain" decision) vs lumping SO2+OH→H2SO4 directly
  — decided: keep SO3 explicit.

## Verification
Per-subtask pytest (chemistry + `test_jax_dcdt` parity + new sulfur tests). Phase gate: 3-agent
independent verification (correctness vs JPL; tests real; conservation/no-undocumented-assumptions),
verdicts in `docs/VALIDATION.md`.
