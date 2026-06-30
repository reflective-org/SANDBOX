# Reaction reference

Stable reference numbers for the box-model mechanism (`src-python/reactions.py`).

- **R#** — stable position in the mechanism table.
- **k-label** — the original MATLAB rate-constant name (`concs_het.m`), incl. a/b branches.

In code: `reactions.BY_RNUMBER["R26"]` or `reactions.BY_KLABEL["k22"]`; list everything with
`reactions.MECHANISM.describe()`.

| R# | k-label | type | reaction | note |
|----|---------|------|----------|------|
| R1 | k1 | gas | ClO + NO -> NO2 + Cl | JPL 19-5 (F130) |
| R2 | k2 | gas | ClO + ClO + M -> ClOOCl + M | JPL 19-5 (F10), 298 K ref |
| R3 | k3 | gas | ClOOCl + M -> ClO + ClO + M | JPL 19-5 (k2 / Keq, Table 3-1 #20) |
| R4 | k4 | gas | ClO + NO2 + M -> ClONO2 + M | JPL 19-5 (F8), 298 K ref |
| R5 | k5 | gas | ClONO2 + M -> ClO + NO2 + M | Fahey (not in JPL; kept) |
| R6 | k6 | gas | Cl + O3 -> ClO + O2 | JPL 19-5 (F68) |
| R7 | k7 | gas | Cl + CH4 -> HCl + CH3 | JPL 19-5 (F75) |
| R8 | k8 | het | ClONO2 + HCl -> Cl2 + HNO3aq |  |
| R9 | k9 | het | ClONO2 + H2O -> HOCl + HNO3aq |  |
| R10 | k10 | het | HOCl + HCl -> Cl2 + H2O |  |
| R11 | k11 | het | N2O5 + H2O -> 2 HNO3aq |  |
| R12 | k12a | photo | ClONO2 -> Cl + NO3 |  |
| R13 | k12b | photo | ClONO2 -> ClO + NO2 |  |
| R14 | k13a | photo | ClOOCl -> 2 Cl + O2 |  |
| R15 | k13b | photo | ClOOCl -> 2 ClO |  |
| R16 | k14 | photo | Cl2 -> 2 Cl |  |
| R17 | k15 | photo | HOCl -> OH + Cl |  |
| R18 | k16 | photo | HNO3 -> OH + NO2 | main OH production |
| R19 | k17a | photo | NO3 -> NO2 + O |  |
| R20 | k17b | photo | NO3 -> NO + O2 |  |
| R21 | k18 | photo | NO2 -> NO + O |  |
| R22 | k19 | photo | N2O5 -> NO2 + NO3 |  |
| R23 | k20a | photo | O3 -> O2 + O | O3 photolysis (opt-dependent) |
| R24 | k20b | photo | O3 -> O2 + O1D | O3 photolysis (opt-dependent) |
| R25 | k21 | gas | NO2 + NO3 + M -> N2O5 + M | JPL 19-5 (C7), 298 K ref |
| R26 | k22 | gas | OH + HNO3 -> H2O + NO3 | JPL 19-5 (K2), chemical-activation form |
| R27 | k23 | gas | OH + NO2 + M -> HNO3 + M | JPL 19-5 (C4), 298 K ref |
| R28 | k24 | gas | O + NO + M -> NO2 + M | JPL 19-5 (C1), 298 K ref |
| R29 | k25 | gas | O + NO2 + M -> NO3 + M | JPL 19-5 (K1), association channel |
| R30 | k26 | gas | O + NO2 -> NO + O2 | JPL 19-5 (K1), chemical-activation channel |
| R31 | k27 | gas | NO + O3 -> NO2 + O2 | JPL 19-5 (C19) |
| R32 | k28 | gas | NO2 + O3 -> NO3 + O2 | JPL 19-5 (C21) |
| R33 | k29 | gas | OH + NO + M -> HONO + M | JPL 19-5 (C3), 298 K ref |
| R34 | k30 | gas | OH + HONO -> H2O + NO2 | JPL 19-5 (C7-bimol) |
| R35 | k31 | gas | NO + NO3 -> 2 NO2 | JPL 19-5 (C20) |
| R36 | k32 | gas | OH + HNO4 -> H2O + NO2 + O2 | JPL 19-5 (C8) |
| R37 | k33 | gas | HO2 + NO -> NO2 + OH | JPL 19-5 (C10), main OH prod |
| R38 | k34 | gas | HO2 + NO2 + M -> HNO4 + M | JPL 19-5 (C6), 298 K ref |
| R39 | k35 | gas | O + O2 + M -> O3 + M | JPL 19-5 (A1), 298 K ref |
| R40 | k36 | gas | O + O3 -> 2 O2 | JPL 19-5 |
| R41 | k37 | gas | OH + O3 -> HO2 + O2 | JPL 19-5 |
| R42 | k38 | gas | OH + HO2 -> H2O + O2 | JPL 19-5 |
| R43 | k39 | gas | HO2 + O3 -> OH + 2 O2 | JPL 19-5 |
| R44 | k40 | gas | O1D + O2 -> O + O2 | JPL 19-5 (off if opt=1) |
| R45 | k41 | gas | O1D -> O | O1D + N2 -> O + N2; JPL 19-5 (off if opt=1) |
| R46 | k42 | gas | O1D + H2O -> 2 OH | JPL 19-5 (OH source) |
| R47 | k43 | gas | O1D + CH4 -> CH3 + OH | JPL 19-5 |
| R48 | k44 | gas | O + ClO -> Cl + O2 | JPL 19-5 |
| R49 | k45 | gas | OH + ClO -> Cl + HO2 | JPL 19-5 |
| R50 | k46 | gas | OH + ClO -> HCl + O2 | JPL 19-5 |
| R51 | k47 | gas | OH + HCl -> Cl + H2O | JPL 19-5 |
| R52 | k48 | gas | HO2 + ClO -> HOCl + O2 | JPL 19-5 |
| R53 | k49 | gas | HO2 + ClO -> HCl + O3 | not a listed JPL channel -> disabled [disabled] |
| R54 | k50 | gas | O + BrO -> Br + O2 | JPL 19-5 |
| R55 | k51 | gas | Br + O3 -> BrO + O2 | JPL 19-5 |
| R56 | k52 | gas | BrO + NO -> NO2 + Br | JPL 19-5 |
| R57 | k53 | gas | BrO + ClO -> Br + Cl + O2 | JPL 19-5 |
| R58 | k54 | gas | BrO + ClO -> BrCl + O2 | JPL 19-5 |
| R59 | k55 | gas | BrO + ClO -> Br + OClO | JPL 19-5 |
| R60 | k56 | gas | BrO + NO2 + M -> BrONO2 + M | JPL 19-5 (G2), 298 K ref |
| R61 | k57a | photo | HNO4 -> NO2 + HO2 |  |
| R62 | k57b | photo | HNO4 -> NO3 + OH |  |
| R63 | k58 | photo | OClO -> O + ClO |  |
| R64 | k59 | photo | BrO -> Br + O |  |
| R65 | k60a | photo | BrONO2 -> Br + NO3 |  |
| R66 | k60b | photo | BrONO2 -> BrO + NO2 |  |
| R67 | k61 | photo | BrCl -> Br + Cl |  |
| R68 | k62 | photo | O2 -> 2 O | ~5% O3/day in unperturbed conditions -> disabled [disabled] |
| R69 | k63 | photo | HONO -> OH + NO | important for OH |
| R70 | k64 | gas | HNO3aq -> HNO3 | re-evaporation from aerosol |
| R71 | k65 | gas | Cl + C2H6 -> HCl | disabled [disabled] |
| R72 | k66 | het | BrONO2 + H2O -> HOBr + HNO3aq | gamma fixed at 0.8 (JPL) |
| R73 | k67 | photo | HOBr -> OH + Br |  |
| R74 | k68 | gas | SO2 + OH -> HO2 | JPL 19-5 (I4) termolecular; product lumped to HO2 |
| R75 | k69 | gas | CH4 + OH -> HO2 | JPL 19-5 (D14, OH+CH4); product lumped to HO2 |
| R76 | k70 | gas | HO2 + HO2 -> H2O2 | JPL 19-5 (B13): bimol + termol[M] + H2O enhancement |
| R77 | k71 | gas | H2O2 -> 2 OH | FK: constant, not day/night gated |
| R78 | k72 | gas | SO2 + HO2 -> | UPPER LIMIT (Graham 1979 / JPL 19-5 I34); sensitivity test needed |
