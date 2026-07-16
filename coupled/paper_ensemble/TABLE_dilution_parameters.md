# Simulation parameters: dilution sensitivity set

Baseline configuration and varied dilution regimes (5 simulations). All non-dilution
parameters are held at the microphysics-table defaults (condensation α = 1, nucleation ×1,
coagulation ×1). The four constant-Kz regimes share the common form (Schumann et al. 1998)

V(t)/V₀ = max(1, t⁰·⁸) for t ≤ 10⁴ s; 1585 · exp[k (t − 10⁴)^(3/2)] for t > 10⁴ s

and differ only in the turbulent-growth coefficient k; the burst regime replaces the single
exponential with a three-stage sequence representing a transient burst of turbulence
(Kz = 10 m² s⁻¹ for ≈14 h), continuous at the breakpoints.

|                             | Parameter                | Value                                                                                                                                                                           |
| --------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Injection**               | SO₂ released             | 1 t into V₀ = 10 m × 10 m × 15 km (6.27×10¹⁵ molec cm⁻³)                                                                                                                        |
|                             | Release time             | 00:00 local solar time, day of year 172 (≈ 21 June)                                                                                                                             |
| **Site**                    | Location, altitude       | 30°N, ~20 km                                                                                                                                                                    |
|                             | Temperature, pressure    | 210 K, 55 hPa                                                                                                                                                                   |
|                             | Relative humidity        | 3% (6.91 ppmv H₂O)                                                                                                                                                              |
| **Background**              | Aerosol                  | SABRE (aged air, 220-230 ppbv of N₂O): lognormal, D_g = 0.12 µm, σ_g = 1.6, N = 3.5 cm⁻³ ambient                                                                                |
|                             | SO₂                      | 20 pptv (entrained air carries full stratospheric background composition)                                                                                                       |
| **Chemistry**               | Gas phase                | 36-species stratospheric mechanism (HOₓ/NOₓ/Cl/Br + sulfur), JPL 19-5; TUV-x photolysis                                                                                         |
| **Microphysics**            |                          | TOMAS, 80 bins; Dunne et al. (2016) nucleation (ion pair rate 30 cm⁻³ s⁻¹), Fuchs–Sutugin condensation (α = 1), Brownian–Fuchs coagulation (×1), Tabazadeh water uptake         |
| **Varied: dilution regime** | Low Kz (D1)              | k = 2.811×10⁻⁹                                                                                                                                                                  |
|                             | **Med Kz (D2, default)** | k = 8.89×10⁻⁹                                                                                                                                                                   |
|                             | High Kz (D3)             | k = 2.811×10⁻⁸                                                                                                                                                                  |
|                             | Very high (D5)           | k = 5.33×10⁻⁸                                                                                                                                                                   |
|                             | Burst of turbulence      | t⁰·⁸ to 10⁴ s; 1585·exp[2.811×10⁻⁹(t−10⁴)^(3/2)] to 1.728×10⁵ s; 1906·exp[2.811×10⁻⁷(t−1.728×10⁵)^(3/2)] to 2.238×10⁵ s; 4.83×10⁴·exp[2.811×10⁻⁹(t−2.238×10⁵)^(3/2)] thereafter |
