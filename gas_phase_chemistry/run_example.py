"""Run a scenario and print a short summary.

By default this reproduces the MATLAB default setup (T=210 K, P=68 mbar, SA=2 um^2/cm^3,
H2O=5 ppm, opt=1) over several day/night cycles. A YAML/JSON scenario file can be supplied.

Usage:
    python3 src-python/run_example.py
    python3 src-python/run_example.py --config scenarios/default.yaml
"""

from __future__ import annotations

import argparse

from config import IDX
from driver import integrate, integrate_sza
from scenario import Scenario


def main():
    parser = argparse.ArgumentParser(description="Run the gas-phase chemistry box model.")
    parser.add_argument("--config", help="Path to a YAML/JSON scenario file")
    args = parser.parse_args()

    scenario = Scenario.load(args.config) if args.config else Scenario()
    cfg = scenario.to_config()
    x0 = scenario.initial_state()
    if scenario.photolysis == "reference":
        # prescribed day/night schedule (photolysis on by day / off by night)
        t, states = integrate(cfg, x0, td=scenario.td, tn=scenario.tn,
                              days=scenario.days, DT=scenario.DT)
    else:
        # 'sza'/'tuvx': the real sun drives photolysis, so integrate continuously
        t, states = integrate_sza(cfg, x0, days=scenario.days, DT=scenario.DT)

    hours = t / 3600.0
    o3 = states[:, IDX["O3"]] / cfg.M * 1e12
    o3_loss_pct = (1.0 - o3[-1] / o3[0]) * 100.0

    print(f"Scenario: T={scenario.T} K, P={scenario.P} mbar, SA={scenario.SA} um^2/cm^3, "
          f"H2O={scenario.WTR} ppm, opt={scenario.opt}, photolysis={scenario.photolysis}")
    print(f"Integrated {hours[-1]:.1f} hours ({len(t)} output points), {cfg.count} RHS calls.")
    print(f"O3: {o3[0]/1000:.1f} -> {o3[-1]/1000:.1f} ppbv ({o3_loss_pct:+.2f}% over the run)")
    print()
    print("Final mixing ratios (pptv):")
    for name in ["ClO", "ClONO2", "HCl", "Cl2", "ClOOCl", "NO", "NO2",
                 "OH", "HO2", "BrO", "HOCl", "N2O5"]:
        print(f"  {name:8s} {states[-1, IDX[name]] / cfg.M * 1e12:12.3f}")


if __name__ == "__main__":
    main()
