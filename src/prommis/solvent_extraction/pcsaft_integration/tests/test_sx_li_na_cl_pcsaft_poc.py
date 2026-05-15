#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2024 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
from pyomo.environ import ConcreteModel, value

from idaes.core.util.model_statistics import degrees_of_freedom

import pytest

from prommis.solvent_extraction.pcsaft_integration.li_na_cl_aq_properties import LiNaClAqParameters
from prommis.solvent_extraction.pcsaft_integration.li_na_cl_og_properties import LiNaClOgParameters
from prommis.solvent_extraction.pcsaft_integration.sx_li_na_cl_pcsaft_poc import build_li_na_cl_pcsaft_poc_model


@pytest.mark.unit
def test_property_packages_include_na_and_li():
    model = ConcreteModel()
    model.aq = LiNaClAqParameters()
    model.og = LiNaClOgParameters()
    assert 'Li' in model.aq.component_list
    assert 'Na' in model.aq.component_list
    assert 'Cl' in model.aq.component_list
    assert 'TOP' in model.og.component_list
    assert 'Hexane' in model.og.component_list
    assert 'Li' in model.og.component_list
    assert 'Na' in model.og.component_list
    assert 'Cl' in model.og.component_list


@pytest.mark.component
def test_build_one_stage_pcsaft_poc_model():
    calibration = {'kappa_li': 0.52, 'kappa_na': 0.01, 'kappa_cl': 0.01, 'eta_li_pct': 52.0, 'eta_na_pct': 1.0, 'li_na_selectivity': 100.0, 'd_li': 1.08}
    m, calib = build_li_na_cl_pcsaft_poc_model(calibration=calibration)
    assert calib['kappa_li'] == pytest.approx(0.52, rel=1e-12)
    assert calib['kappa_li'] > calib['kappa_na']
    assert degrees_of_freedom(m) == 0

    stage = m.fs.solex.mscontactor.elements.first()
    li_default = m.fs.solex.mass_transfer_constraint[0, stage, 'aqueous', 'organic', 'Li']
    assert not li_default.active
    assert m.fs.li_mass_transfer_constraint[0].active

    m.fs.k_li.fix(0.52)
    m.fs.li_extraction_efficiency.fix(52.0)
    assert value(m.fs.k_li_from_eta_constraint.body) == pytest.approx(0.0, abs=1e-10)
