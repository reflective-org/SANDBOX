# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Phase 5.2: box temperature tendency from O3 photochemical + aerosol SW-absorption heating."""

import numpy as np
import pytest

from coupled import CoupledScenario
from coupled import tomas_bridge as tb
from coupled import aerosol_optics as ao
from coupled import heating as ht
from coupled.model_bridge import to_model_config, initial_state
from tuvx_photolysis_adapter import calculator_grids
from config import IDX

IDX_O3 = IDX["O3"]


def _cfg_conc(**kw):
    params = dict(T=210.0, P=68.0, WTR=5.0, latitude=0.0, day_of_year=80, start_utc_hour=12.0,
                  days=1, DT=3600.0, dt_couple=3600.0, photolysis="tuvx",
                  concentrations={"O2": 2.1e11, "O3": 1.18e6, "HCl": 777.0, "ClONO2": 127.0})
    params.update(kw)
    sc = CoupledScenario(**params)
    return to_model_config(sc), initial_state(sc), sc


@pytest.mark.slow
def test_gas_heating_positive_daytime_and_scales_with_o3():
    cfg, conc, _ = _cfg_conc()
    dT = ht.box_dTdt(cfg, conc, 0.0)               # noon, no aerosol
    assert dT > 0.0
    conc2 = conc.copy(); conc2[IDX_O3] = conc[IDX_O3] * 2.0
    dT2 = ht.box_dTdt(cfg, conc2, 0.0)
    assert abs(dT2 / dT - 2.0) < 1e-6              # gas heating linear in [O3] (no aerosol term)


@pytest.mark.slow
def test_night_zero():
    cfg, conc, _ = _cfg_conc(start_utc_hour=0.0)   # midnight at lon 0 -> sun down
    assert ht.box_dTdt(cfg, conc, 0.0) == 0.0


@pytest.mark.slow
def test_aerosol_absorption_adds_heating():
    cfg, conc, sc = _cfg_conc()
    st = tb.initial_tomas_state(sc)._replace()      # aerosol present
    st = st._replace(Nk=st.Nk * 1.0e4)              # scale up so absorption is appreciable
    wl_nm, height_edges = calculator_grids(cfg)
    mie = ao.MieTable(wl_nm)
    aer = ao.aerosol_optical_props(st, wl_nm, height_edges, (10.0, 30.0), mie=mie)
    dT_gas = ht.box_dTdt(cfg, conc, 0.0)
    dT_both = ht.box_dTdt(cfg, conc, 0.0, aerosol_props=aer, tstate=st, mie=mie)
    assert dT_both > dT_gas                          # aerosol SW absorption adds heating

