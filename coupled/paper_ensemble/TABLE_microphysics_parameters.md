# Simulation parameters: microphysical sensitivity set

Baseline configuration and varied microphysical factors (full 2 × 3 × 3 factorial = 18
simulations). Bold marks the default level of each varied factor.

| | Parameter | Value |
|---|---|---|
| **Injection** | SO₂ released | 1 t into V₀ = 10 m × 10 m × 15 km (6.27×10¹⁵ molec cm⁻³) |
| | Release time | 00:00 local solar time, day of year 172 (≈ 21 June) |
| **Site** | Location, altitude | 30°N, ~20 km |
| | Temperature, pressure | 210 K, 55 hPa |
| | Relative humidity | 3% (6.91 ppmv H₂O) |
| **Background** | Aerosol | SABR-220 (aged air, low N₂O): lognormal, D_g = 0.12 µm, σ_g = 1.6, N = 3.5 cm⁻³ ambient |
| | SO₂ | 20 pptv (entrained air carries full stratospheric background composition) |
| **Dilution** | Regime | Med Kz (Schumann et al. 1998) |
| | V(t)/V₀ | max(1, t⁰·⁸) for t ≤ 10⁴ s; 1585 · exp[8.89×10⁻⁹ (t − 10⁴)^(3/2)] for t > 10⁴ s |
| **Chemistry** | Gas phase | 36-species stratospheric mechanism (HOₓ/NOₓ/Cl/Br + sulfur), JPL 19-5; TUV-x photolysis |
| **Microphysics** | Scheme | TOMAS, 80 size bins (dry D_p 1.7 nm – 17.5 µm) |
| | Nucleation | Dunne et al. (2016) binary H₂SO₄–H₂O, neutral + ion-induced |
| | Ion pair production rate | 30 ion pairs cm⁻³ s⁻¹ (galactic cosmic rays at ~20 km) |
| | Condensation | Fuchs–Sutugin mass transfer, accommodation coefficient α |
| | Coagulation | Brownian kernel with Fuchs non-continuum correction |
| | Water uptake | Tabazadeh et al. (1997) H₂SO₄–H₂O equilibrium |
| **Varied factors** | Condensation α | 0.5, **1.0** |
| | Nucleation rate scale | 0.01×, **1×**, 100× |
| | Coagulation kernel scale | 0.5×, **1×**, 2× |
