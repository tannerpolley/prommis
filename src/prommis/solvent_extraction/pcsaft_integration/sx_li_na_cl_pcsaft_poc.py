#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2024 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""
One-stage Li/Na/Cl solvent extraction POC using an ePC-SAFT-derived surrogate.

This script demonstrates:
1) Building a one-stage MSContactor-based SolventExtraction model.
2) Binding Li extraction efficiency from a surrogate-style PCSAFTBlock.
3) Mapping surrogate output to Li partition behavior in the stage equations.
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path
from typing import Any

from pyomo.environ import Constraint, ConcreteModel, Expression, Var, value
from pyomo.environ import units as pyunits

from idaes.core import FlowDirection, FlowsheetBlock
from idaes.core.initialization.block_triangularization import BlockTriangularizationInitializer
from idaes.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom

_SRC_ROOT = Path(__file__).resolve().parents[3]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from prommis.solvent_extraction.pcsaft_integration._epcsaft_pse_bridge import ensure_epcsaft_pse_imports

from prommis.solvent_extraction.pcsaft_integration.li_na_cl_aq_properties import LiNaClAqParameters
from prommis.solvent_extraction.pcsaft_integration.li_na_cl_og_properties import LiNaClOgParameters
from prommis.solvent_extraction.solvent_extraction import SolventExtraction


def _neutral_cl_mg_L(li_mg_L: float, na_mg_L: float) -> float:
    mw_li = 6.94e-3
    mw_na = 22.98976928e-3
    mw_cl = 35.45e-3
    li_mol_L = li_mg_L * 1e-6 / mw_li
    na_mol_L = na_mg_L * 1e-6 / mw_na
    return (li_mol_L + na_mol_L) * mw_cl * 1e6


def _compute_nominal_calibration(run_lle_stage, StageInputs, *, li_mg_L: float, na_mg_L: float, o_to_a_ratio: float, top_mol_L: float) -> dict[str, float]:
    try:
        out = run_lle_stage(StageInputs(li_mg_L=li_mg_L, na_mg_L=na_mg_L, o_to_a_ratio=o_to_a_ratio, top_mol_L=top_mol_L))
        return {
            'kappa_li': float(out.kappa_li),
            'kappa_na': float(out.kappa_na),
            'kappa_cl': float(out.kappa_cl),
            'eta_li_pct': float(out.eta_li_pct),
            'eta_na_pct': float(out.eta_na_pct),
            'li_na_selectivity': float(out.li_na_selectivity),
            'd_li': float(out.D_li),
        }
    except Exception as exc:
        warnings.warn(f'Direct ePC-SAFT nominal calibration failed; falling back to defaults. Cause: {exc}', RuntimeWarning, stacklevel=2)
        return {
            'kappa_li': 0.52,
            'kappa_na': 0.01,
            'kappa_cl': 0.01,
            'eta_li_pct': 52.0,
            'eta_na_pct': 1.0,
            'li_na_selectivity': 100.0,
            'd_li': 1.08,
        }


def _constant_surrogate_model(SurrogateModel, *, eta_li_pct: float, li_mg_L: float, na_mg_L: float, o_to_a_ratio: float, top_mol_L: float):
    eta_frac = min(0.999999, max(1e-6, eta_li_pct / 100.0))
    intercept = math.log(eta_frac / (1.0 - eta_frac))
    coeff = [intercept] + [0.0] * 14
    return SurrogateModel.from_dict(
        {
            'feature_names': ['li_mg_L', 'na_mg_L', 'o_to_a_ratio', 'top_mol_L'],
            'term_names': ['intercept', 'li_mg_L', 'na_mg_L', 'o_to_a_ratio', 'top_mol_L', 'li_mg_L^2', 'na_mg_L^2', 'o_to_a_ratio^2', 'top_mol_L^2', 'li_mg_L*na_mg_L', 'li_mg_L*o_to_a_ratio', 'li_mg_L*top_mol_L', 'na_mg_L*o_to_a_ratio', 'na_mg_L*top_mol_L', 'o_to_a_ratio*top_mol_L'],
            'mean': [li_mg_L, na_mg_L, o_to_a_ratio, top_mol_L],
            'scale': [1.0, 1.0, 1.0, 1.0],
            'targets': [
                {
                    'name': 'eta_li_pct',
                    'transform': 'logit_0_100',
                    'coefficients': coeff,
                    'fit_metrics': {},
                }
            ],
            'metadata': {'source': 'constant_epcsaft_bridge_adapter'},
        }
    )


def build_li_na_cl_pcsaft_poc_model(*, surrogate_json: str | Path | None = None, calibration: dict[str, float] | None = None) -> tuple[ConcreteModel, dict[str, float]]:
    PCSAFTBlock, run_lle_stage, SurrogateModel, deserialize_surrogate, StageInputs = ensure_epcsaft_pse_imports()

    li_feed_mg_L = 60.0
    na_feed_mg_L = 10900.0
    cl_feed_mg_L = _neutral_cl_mg_L(li_feed_mg_L, na_feed_mg_L)
    aq_flow_L_h = 1000.0
    oa_ratio = 1.0
    top_mol_L = 0.015
    hexane_mol_L = 0.6548 / 0.08617848

    calibration_data = calibration or _compute_nominal_calibration(run_lle_stage, StageInputs, li_mg_L=li_feed_mg_L, na_mg_L=na_feed_mg_L, o_to_a_ratio=oa_ratio, top_mol_L=top_mol_L)
    if surrogate_json is None:
        surrogate_model = _constant_surrogate_model(SurrogateModel, eta_li_pct=calibration_data['eta_li_pct'], li_mg_L=li_feed_mg_L, na_mg_L=na_feed_mg_L, o_to_a_ratio=oa_ratio, top_mol_L=top_mol_L)
    else:
        surrogate_model = deserialize_surrogate(surrogate_json)

    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.prop_aq = LiNaClAqParameters()
    m.fs.prop_og = LiNaClOgParameters()

    m.fs.solex = SolventExtraction(
        number_of_finite_elements=1,
        dynamic=False,
        aqueous_stream={'property_package': m.fs.prop_aq, 'flow_direction': FlowDirection.forward, 'has_energy_balance': False, 'has_pressure_balance': False},
        organic_stream={'property_package': m.fs.prop_og, 'flow_direction': FlowDirection.forward, 'has_energy_balance': False, 'has_pressure_balance': False},
        aqueous_to_organic=True,
    )

    stage = m.fs.solex.mscontactor.elements.first()
    m.fs.solex.partition_coefficient[stage, 'aqueous', 'organic', 'Li'] = float(calibration_data['kappa_li'])
    m.fs.solex.partition_coefficient[stage, 'aqueous', 'organic', 'Na'] = float(calibration_data['kappa_na'])
    m.fs.solex.partition_coefficient[stage, 'aqueous', 'organic', 'Cl'] = float(calibration_data['kappa_cl'])

    aq_in = m.fs.solex.mscontactor.aqueous_inlet_state[0]
    aq_in.conc_mass_comp['Li'].fix(li_feed_mg_L)
    aq_in.conc_mass_comp['Na'].fix(na_feed_mg_L)
    aq_in.conc_mass_comp['Cl'].fix(cl_feed_mg_L)
    aq_in.flow_vol.fix(aq_flow_L_h)

    og_in = m.fs.solex.mscontactor.organic_inlet_state[0]
    top_mg_L = top_mol_L * value(m.fs.prop_og.mw['TOP']) * 1e6
    hexane_mg_L = hexane_mol_L * value(m.fs.prop_og.mw['Hexane']) * 1e6
    og_in.conc_mass_comp['TOP'].fix(top_mg_L)
    og_in.conc_mass_comp['Hexane'].fix(hexane_mg_L)
    og_in.conc_mass_comp['Li'].fix(1e-9)
    og_in.conc_mass_comp['Na'].fix(1e-9)
    og_in.conc_mass_comp['Cl'].fix(1e-9)
    og_in.flow_vol.fix(aq_flow_L_h * oa_ratio)
    m.fs.solex.mscontactor.organic[0, stage].flow_vol.fix(value(og_in.flow_vol))

    m.fs.o_to_a_ratio = Var(initialize=oa_ratio, bounds=(0.1, None))
    m.fs.o_to_a_ratio_constraint = Constraint(expr=m.fs.o_to_a_ratio == og_in.flow_vol / aq_in.flow_vol)
    m.fs.top_mol_L = Expression(expr=pyunits.convert(og_in.conc_mass_comp['TOP'] / m.fs.prop_og.mw['TOP'], to_units=pyunits.mol / pyunits.litre))
    m.fs.li_extraction_efficiency = Var(initialize=float(calibration_data['eta_li_pct']), bounds=(0.0, 100.0))
    m.fs.li_distribution_ratio = Var(initialize=max(1e-6, float(calibration_data['d_li'])), bounds=(0.0, None))

    m.fs.pcsaft = PCSAFTBlock(
        aqueous_phase={'composition': {'Li+': aq_in.conc_mass_comp['Li'], 'Na+': aq_in.conc_mass_comp['Na']}},
        organic_phase={'composition': {'TOP': m.fs.top_mol_L}},
        system_state={'o_to_a_ratio': m.fs.o_to_a_ratio},
        outputs={('li_extraction_efficiency', None): m.fs.li_extraction_efficiency, ('li_distribution_ratio', None): m.fs.li_distribution_ratio},
        mode='surrogate',
        surrogate_model=surrogate_model,
    )

    m.fs.k_li = Var(initialize=float(calibration_data['kappa_li']), bounds=(0.0, 0.999999))
    m.fs.k_li_from_eta_constraint = Constraint(expr=m.fs.k_li == m.fs.li_extraction_efficiency / 100.0)

    for t in m.fs.time:
        m.fs.solex.mass_transfer_constraint[t, stage, 'aqueous', 'organic', 'Li'].deactivate()

    @m.fs.Constraint(m.fs.time)
    def li_mass_transfer_constraint(b, t):
        return b.solex.mscontactor.material_transfer_term[t, stage, ('aqueous', 'organic', 'Li')] == -b.solex.mscontactor.aqueous_inlet_state[t].get_material_flow_terms(b.solex.mscontactor.aqueous.phase_list, 'Li') * b.k_li

    return m, calibration_data


def solve_li_na_cl_pcsaft_poc(*, surrogate_json: str | Path | None = None, calibration: dict[str, float] | None = None, tee: bool = False) -> tuple[ConcreteModel, dict[str, float], Any]:
    m, calibration_data = build_li_na_cl_pcsaft_poc_model(surrogate_json=surrogate_json, calibration=calibration)
    dof = degrees_of_freedom(m)
    if dof != 0:
        raise RuntimeError(f'Expected 0 degrees of freedom, found {dof}.')
    initializer = BlockTriangularizationInitializer(constraint_tolerance=1e-6)
    initializer.initialize(m)
    solver = get_solver()
    results = solver.solve(m, tee=tee)
    return m, calibration_data, results


def summarize_results(m: ConcreteModel) -> dict[str, float]:
    stage = m.fs.solex.mscontactor.elements.first()
    aq_out = m.fs.solex.mscontactor.aqueous[0, stage]
    return {
        'eta_li_pct': value(m.fs.li_extraction_efficiency),
        'd_li': value(m.fs.li_distribution_ratio),
        'k_li': value(m.fs.k_li),
        'li_aq_out_mg_L': value(aq_out.conc_mass_comp['Li']),
        'na_aq_out_mg_L': value(aq_out.conc_mass_comp['Na']),
        'cl_aq_out_mg_L': value(aq_out.conc_mass_comp['Cl']),
    }


if __name__ == '__main__':
    model, calib, _ = solve_li_na_cl_pcsaft_poc(tee=False)
    summary = summarize_results(model)
    print('Calibration:', calib)
    print('Summary:', summary)
