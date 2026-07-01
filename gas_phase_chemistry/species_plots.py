"""Write one time-series plot per species into an output folder.

``write_species_plots`` saves ``<output_dir>/<species>.png`` for every modelled species (mixing
ratio in pptv vs. time in days), creating the folder if needed and overwriting the files each call,
so the folder always reflects the latest run. The folder name comes from the scenario's
``output_dir`` (set in the input file).
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import IDX, SPECIES


def write_species_plots(output_dir, t, states, cfg, species=None, title=""):
    """Write one PNG per species to ``output_dir``. Returns (folder, list of species written).

    ``t`` is time (s), ``states`` is (n_time, n_species) in molec/cm^3, ``cfg`` provides the air
    number density ``M`` for the mixing-ratio conversion.
    """
    os.makedirs(output_dir, exist_ok=True)
    days = np.asarray(t) / 86400.0
    species = species or SPECIES
    written = []
    for name in species:
        y = states[:, IDX[name]] / cfg.M * 1e12  # pptv
        fig, ax = plt.subplots(figsize=(8, 4.5))
        pos = y[y > 0]
        if pos.size and pos.max() > 0:
            # log scale for the positive signal; non-positive points (e.g. tiny night values) drop out
            ax.plot(days, np.where(y > 0, y, np.nan), lw=1.2, color="C0")
            ax.set_yscale("log")
        else:
            ax.plot(days, y, lw=1.2, color="C0")  # identically zero -> linear
        ax.set_xlabel("day")
        ax.set_ylabel(f"{name} mixing ratio (pptv)")
        ax.set_title(f"{name}" + (f"  —  {title}" if title else ""))
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=110)
        plt.close(fig)
        written.append(name)
    return output_dir, written
