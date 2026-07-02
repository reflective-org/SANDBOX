"""Phase 1.5 diagnostic: gas-phase H2SO4 production and sulfur-atom conservation.

Runs the coupled TUV-x scenario and shows (a) SO2 -> SO3 -> H2SO4 evolution and (b) that total
sulfur (SO2 + SO3 + H2SO4) is conserved -- there is no sulfur sink yet (condensation arrives with
the TOMAS coupling in Phase 3), so H2SO4 accumulates while total S stays constant.

    python sulfur_budget.py --config scenarios/model_input.yaml --out sulfur_budget
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import IDX
from driver import integrate_sza
from scenario import Scenario

p = argparse.ArgumentParser()
p.add_argument("--config", default="scenarios/model_input.yaml")
p.add_argument("--out", default="sulfur_budget")
args = p.parse_args()

sc = Scenario.load(args.config)
cfg = sc.to_config()
x0 = sc.initial_state()
t, states = integrate_sza(cfg, x0, days=sc.days, DT=sc.DT)
days = t / 86400.0

def ppt(name):
    return states[:, IDX[name]] / cfg.M * 1e12

so2, so3, h2so4 = ppt("SO2"), ppt("SO3"), ppt("H2SO4")
# total sulfur in molec/cm^3 (atom-conserving quantity)
tot_S = states[:, IDX["SO2"]] + states[:, IDX["SO3"]] + states[:, IDX["H2SO4"]]
drift = (tot_S[-1] - tot_S[0]) / tot_S[0]

os.makedirs(args.out, exist_ok=True)

# (a) sulfur species evolution
fig, ax = plt.subplots(figsize=(8, 4.5))
for y, name in [(so2, "SO2"), (so3, "SO3"), (h2so4, "H2SO4")]:
    ax.plot(days, np.where(y > 0, y, np.nan), label=name, lw=1.3)
ax.set_yscale("log"); ax.set_xlabel("day"); ax.set_ylabel("mixing ratio (pptv)")
ax.set_title(f"Gas-phase sulfur oxidation ({sc.photolysis}, P={sc.P} mbar, lat {sc.latitude})")
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig(os.path.join(args.out, "sulfur_species.png"), dpi=110)
plt.close(fig)

# (b) total-sulfur conservation
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(days, tot_S / tot_S[0], lw=1.5, color="C3")
ax.set_xlabel("day"); ax.set_ylabel("total S / initial  (SO2+SO3+H2SO4)")
ax.set_title(f"Sulfur-atom conservation  (drift = {drift:+.2e} over {days[-1]:.1f} days)")
ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig(os.path.join(args.out, "sulfur_conservation.png"), dpi=110)
plt.close(fig)

print(f"scenario: {args.config}  ({sc.photolysis}, {sc.days} days)")
print(f"SO2   : {so2[0]:.1f} -> {so2[-1]:.1f} pptv  ({(so2[-1]/so2[0]-1)*100:+.3f}%)")
print(f"SO3   : {so3[0]:.3e} -> {so3[-1]:.3e} pptv")
print(f"H2SO4 : {h2so4[0]:.3e} -> {h2so4[-1]:.3e} pptv  (produced)")
print(f"total sulfur drift over run: {drift:+.3e}  (should be ~0; no sulfur sink yet)")
print(f"wrote plots to {args.out}/")
