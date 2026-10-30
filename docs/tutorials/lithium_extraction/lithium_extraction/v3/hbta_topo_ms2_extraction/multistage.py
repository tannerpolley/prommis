"""Coupled IDAES stages closed by the loaded-organic ePC-SAFT surrogate."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from idaes.core import FlowDirection, FlowsheetBlock
from idaes.core.solvers import get_solver
from idaes.models.unit_models import MSContactor
from pyomo.environ import Block, ConcreteModel, Constraint, Expression, Var, exp, value
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.visitor import identify_variables
from pyomo.environ import units as pyunits
from pyomo.opt import check_optimal_termination

from lithium_extraction.common.thermodynamics.hbta_topo.constants import (
    EFFECTIVE_DILUENT_INVENTORY_MOL_L,
    LI_MOLAR_MASS_G_MOL,
    NA_MOLAR_MASS_G_MOL,
)
from lithium_extraction.common.thermodynamics.hbta_topo.properties import (
    HbtaTopoAqueousParameters,
    HbtaTopoOrganicParameters,
)
from .loaded_state_surrogate import (
    LoadedOrganicStateSurrogate,
    attach_loaded_state_surrogate,
)


MS2_LI_MG_L = 168.0
MS2_NA_MG_L = 64_100.0
NOMINAL_CARRIER_MOL_L = 0.015
NOMINAL_PH = 5.77
NOMINAL_TEMPERATURE_K = 298.15
NOMINAL_AQUEOUS_FLOW_L_H = 1000.0
BALANCE_TOLERANCE_MOL_H = 1.0e-7


@dataclass(frozen=True)
class MultistageCase:
    case_id: str
    stage_count: int
    li_feed_mol_l: float
    na_feed_mol_l: float
    feed_ph: float
    carrier_mol_l_organic: float
    organic_to_aqueous_ratio: float
    temperature_k: float
    aqueous_flow_l_h: float


# Stage counts the surrogate screens enumerate. Five stages are out of scope (ePC-SAFT #63) and took most of the
# screening solver time; the stage model itself still builds one to five stages for the live campaign.
SCREEN_STAGE_COUNTS = (1, 2, 3, 4)


def nominal_ms2_case(
    stage_count: int,
    *,
    organic_to_aqueous_ratio: float = 1.0,
    carrier_mol_l_organic: float = NOMINAL_CARRIER_MOL_L,
    feed_ph: float = NOMINAL_PH,
    temperature_k: float = NOMINAL_TEMPERATURE_K,
    aqueous_flow_l_h: float = NOMINAL_AQUEOUS_FLOW_L_H,
    li_feed_mg_l: float = MS2_LI_MG_L,
    na_feed_mg_l: float = MS2_NA_MG_L,
) -> MultistageCase:
    if stage_count not in range(1, 6):
        raise ValueError("stage_count must be an integer from 1 through 5")
    return MultistageCase(
        case_id=f"ms2_{stage_count}_stage",
        stage_count=stage_count,
        li_feed_mol_l=li_feed_mg_l / 1000.0 / LI_MOLAR_MASS_G_MOL,
        na_feed_mol_l=na_feed_mg_l / 1000.0 / NA_MOLAR_MASS_G_MOL,
        feed_ph=feed_ph,
        carrier_mol_l_organic=carrier_mol_l_organic,
        organic_to_aqueous_ratio=organic_to_aqueous_ratio,
        temperature_k=temperature_k,
        aqueous_flow_l_h=aqueous_flow_l_h,
    )


def _aqueous_stage_inlet(unit: MSContactor, stage: int):
    if stage == unit.elements.first():
        return unit.aqueous_inlet_state[0]
    return unit.aqueous[0, unit.elements.prev(stage)]


def _organic_stage_inlet(unit: MSContactor, stage: int):
    if stage == unit.elements.last():
        return unit.organic_inlet_state[0]
    return unit.organic[0, unit.elements.next(stage)]


def _surrogate_stage_response(model: ConcreteModel):
    surrogate = model._hbta_topo_loaded_surrogate

    def response(inputs: dict[str, float]) -> np.ndarray:
        output = surrogate.evaluate_surrogate(pd.DataFrame([inputs])).iloc[0]
        return np.asarray(
            [
                output["equilibrium_li_loading_fraction"],
                output["equilibrium_na_loading_fraction"],
            ],
            dtype=float,
        )

    return response


def _set_initial_values(model: ConcreteModel, stage_response=None, feed_loading=(0.0, 0.0)) -> None:
    """Countercurrent successive substitution on the stage loadings, then consistent starting values.

    ``stage_response(inputs) -> (Li, Na) equilibrium loading`` defaults to the loaded-state surrogate;
    ``feed_loading`` is the (Li, Na) loading of the organic feed (zero for fresh organic).
    """
    fs = model.fs
    unit = fs.extractor
    stages = tuple(unit.elements)
    count = len(stages)
    ratio = value(fs.organic_to_aqueous_ratio)
    carrier = value(fs.carrier_mol_l_organic)
    feed_li = value(unit.aqueous_inlet_state[0].conc_mol_comp["Li"])
    feed_na = value(unit.aqueous_inlet_state[0].conc_mol_comp["Na"])
    feed_h = 10.0 ** (-value(fs.feed_ph))
    feed_oh = 1.0e-14 / feed_h
    aqueous_in = unit.aqueous_inlet_state[0]
    aqueous_in.conc_mol_comp["H"].set_value(feed_h)
    aqueous_in.conc_mol_comp["OH"].set_value(feed_oh)
    aqueous_in.conc_mol_comp["Cl"].set_value(feed_li + feed_na + feed_h - feed_oh)
    organic_in = unit.organic_inlet_state[0]
    organic_in.flow_vol.set_value(value(aqueous_in.flow_vol) * ratio)
    organic_in.conc_mol_comp["HBTA"].set_value(carrier)
    organic_in.conc_mol_comp["TOPO"].set_value(carrier)
    organic_in.conc_mol_comp["Diluent"].set_value(EFFECTIVE_DILUENT_INVENTORY_MOL_L)
    organic_in.conc_mol_comp["Li"].set_value(carrier * feed_loading[0])
    organic_in.conc_mol_comp["Na"].set_value(carrier * feed_loading[1])
    response = _surrogate_stage_response(model) if stage_response is None else stage_response
    feed_loading = np.asarray(feed_loading, dtype=float)
    profiles = np.tile(feed_loading, (count, 1))  # every stage starts at the feed organic's loading
    for _iteration in range(200):
        predicted = []
        aqueous = np.asarray([feed_li, feed_na, feed_h], dtype=float)
        for offset, _stage in enumerate(stages):
            incoming = profiles[offset + 1] if offset + 1 < count else feed_loading
            equilibrium = response(
                {
                    "organic_to_aqueous_ratio": ratio,
                    "li_mol_l_aqueous": max(aqueous[0], 1.0e-8),
                    "na_mol_l_aqueous": max(aqueous[1], 1.0e-8),
                    "hydrogen_mol_l_aqueous": max(aqueous[2], 1.0e-8),
                    "carrier_mol_l_organic": carrier,
                    "incoming_li_loading_fraction": incoming[0],
                    "incoming_na_loading_fraction": incoming[1],
                    "temperature_k": value(fs.temperature / pyunits.K),
                }
            )
            predicted.append(equilibrium)
            transfer = ratio * carrier * (equilibrium - incoming)
            aqueous[:2] -= transfer
            aqueous[2] += transfer.sum()
        predicted_array = np.asarray(predicted)
        error = float(np.max(np.abs(predicted_array - profiles)))
        profiles = 0.5 * profiles + 0.5 * predicted_array
        if error <= 1.0e-10:
            break
    profiles_by_stage = {
        int(stage): tuple(profiles[offset]) for offset, stage in enumerate(stages)
    }
    aqueous = np.asarray([feed_li, feed_na, feed_h], dtype=float)
    for stage in stages:
        li_loading, na_loading = profiles_by_stage[int(stage)]
        org = unit.organic[0, stage]
        org.flow_vol.set_value(value(unit.organic_inlet_state[0].flow_vol))
        org.conc_mol_comp["HBTA"].set_value(carrier)
        org.conc_mol_comp["TOPO"].set_value(carrier)
        org.conc_mol_comp["Diluent"].set_value(EFFECTIVE_DILUENT_INVENTORY_MOL_L)
        org.conc_mol_comp["Li"].set_value(carrier * li_loading)
        org.conc_mol_comp["Na"].set_value(carrier * na_loading)
        stage_block = fs.stage_equilibrium[stage]
        stage_block.equilibrium_li_loading_fraction.set_value(li_loading)
        stage_block.equilibrium_na_loading_fraction.set_value(na_loading)
        incoming_li, incoming_na = (
            tuple(feed_loading)
            if stage == unit.elements.last()
            else profiles_by_stage[int(unit.elements.next(stage))]
        )
        stage_block.incoming_li_loading_fraction.set_value(incoming_li)
        stage_block.incoming_na_loading_fraction.set_value(incoming_na)
        stage_block.organic_to_aqueous_ratio.set_value(ratio)
        stage_block.carrier_mol_l_organic.set_value(carrier)
        stage_block.temperature_k.set_value(value(fs.temperature / pyunits.K))
        delta_li = ratio * carrier * (li_loading - incoming_li)
        delta_na = ratio * carrier * (na_loading - incoming_na)
        stage_aqueous_in = _aqueous_stage_inlet(unit, stage)
        aqueous = unit.aqueous[0, stage]
        aqueous.flow_vol.set_value(value(unit.aqueous_inlet_state[0].flow_vol))
        aqueous_values = np.asarray(
            [
                value(stage_aqueous_in.conc_mol_comp["Li"]) - delta_li,
                value(stage_aqueous_in.conc_mol_comp["Na"]) - delta_na,
                value(stage_aqueous_in.conc_mol_comp["H"]) + delta_li + delta_na,
            ]
        )
        aqueous.conc_mol_comp["Li"].set_value(max(aqueous_values[0], 1.0e-10))
        aqueous.conc_mol_comp["Na"].set_value(max(aqueous_values[1], 1.0e-10))
        aqueous.conc_mol_comp["H"].set_value(max(aqueous_values[2], 1.0e-10))
        aqueous.conc_mol_comp["Cl"].set_value(value(unit.aqueous_inlet_state[0].conc_mol_comp["Cl"]))
        aqueous.conc_mol_comp["OH"].set_value(value(unit.aqueous_inlet_state[0].conc_mol_comp["OH"]))
        stage_block.li_mol_l_aqueous.set_value(value(stage_aqueous_in.conc_mol_comp["Li"]))
        stage_block.na_mol_l_aqueous.set_value(value(stage_aqueous_in.conc_mol_comp["Na"]))
        stage_block.hydrogen_mol_l_aqueous.set_value(value(stage_aqueous_in.conc_mol_comp["H"]))
        unit.material_transfer_term[0, stage, ("aqueous", "organic", "Li")].set_value(
            -case_flow(model) * delta_li
        )
        unit.material_transfer_term[0, stage, ("aqueous", "organic", "Na")].set_value(
            -case_flow(model) * delta_na
        )


def case_flow(model: ConcreteModel) -> float:
    return float(value(model.fs.extractor.aqueous_inlet_state[0].flow_vol))


def build_multistage_model(
    case: MultistageCase,
    *,
    surrogate_path: Path | None = None,
) -> ConcreteModel:
    surrogate = LoadedOrganicStateSurrogate.from_file(surrogate_path)
    model = ConcreteModel()
    model.fs = FlowsheetBlock(dynamic=False)
    fs = model.fs
    fs.aqueous_properties = HbtaTopoAqueousParameters()
    fs.organic_properties = HbtaTopoOrganicParameters()
    fs.extractor = MSContactor(
        number_of_finite_elements=case.stage_count,
        streams={
            "aqueous": {
                "property_package": fs.aqueous_properties,
                "flow_direction": FlowDirection.forward,
                "has_energy_balance": False,
                "has_pressure_balance": False,
            },
            "organic": {
                "property_package": fs.organic_properties,
                "flow_direction": FlowDirection.backward,
                "has_energy_balance": False,
                "has_pressure_balance": False,
            },
        },
    )
    unit = fs.extractor
    fs.organic_to_aqueous_ratio = Var(
        initialize=case.organic_to_aqueous_ratio,
        bounds=tuple(surrogate.payload["input_bounds"]["organic_to_aqueous_ratio"]),
        units=pyunits.dimensionless,
    )
    fs.carrier_mol_l_organic = Var(
        initialize=case.carrier_mol_l_organic,
        bounds=tuple(surrogate.payload["input_bounds"]["carrier_mol_l_organic"]),
        units=pyunits.mol / pyunits.L,
    )
    fs.feed_ph = Var(initialize=case.feed_ph, bounds=(5.5, 8.0), units=pyunits.dimensionless)
    fs.temperature = Var(
        initialize=case.temperature_k,
        bounds=tuple(surrogate.payload["input_bounds"]["temperature_k"]),
        units=pyunits.K,
    )
    for variable in (
        fs.organic_to_aqueous_ratio,
        fs.carrier_mol_l_organic,
        fs.feed_ph,
        fs.temperature,
    ):
        variable.fix()
    aqueous_in = unit.aqueous_inlet_state[0]
    aqueous_in.flow_vol.fix(case.aqueous_flow_l_h)
    aqueous_in.conc_mol_comp["Li"].fix(case.li_feed_mol_l)
    aqueous_in.conc_mol_comp["Na"].fix(case.na_feed_mol_l)
    fs.feed_hydrogen = Constraint(
        expr=aqueous_in.conc_mol_comp["H"]
        == exp(-math.log(10.0) * fs.feed_ph) * pyunits.mol / pyunits.L
    )
    fs.feed_hydroxide = Constraint(
        expr=aqueous_in.conc_mol_comp["OH"]
        * aqueous_in.conc_mol_comp["H"]
        == 1.0e-14 * (pyunits.mol / pyunits.L) ** 2
    )
    fs.feed_charge = Constraint(
        expr=aqueous_in.conc_mol_comp["Cl"]
        == aqueous_in.conc_mol_comp["Li"]
        + aqueous_in.conc_mol_comp["Na"]
        + aqueous_in.conc_mol_comp["H"]
        - aqueous_in.conc_mol_comp["OH"]
    )
    organic_in = unit.organic_inlet_state[0]
    fs.organic_flow = Constraint(
        expr=organic_in.flow_vol == aqueous_in.flow_vol * fs.organic_to_aqueous_ratio
    )
    fs.organic_hbta = Constraint(
        expr=organic_in.conc_mol_comp["HBTA"] == fs.carrier_mol_l_organic
    )
    fs.organic_topo = Constraint(
        expr=organic_in.conc_mol_comp["TOPO"] == fs.carrier_mol_l_organic
    )
    organic_in.conc_mol_comp["Diluent"].fix(EFFECTIVE_DILUENT_INVENTORY_MOL_L)
    organic_in.conc_mol_comp["Li"].fix(0.0)
    organic_in.conc_mol_comp["Na"].fix(0.0)

    @unit.Constraint(unit.elements)
    def constant_organic_flow(block, stage):
        return block.organic[0, stage].flow_vol == organic_in.flow_vol

    fs.stage_equilibrium = Block(unit.elements)
    input_bounds = surrogate.payload["input_bounds"]
    for stage in unit.elements:
        block = fs.stage_equilibrium[stage]
        aqueous_stage_in = _aqueous_stage_inlet(unit, stage)
        organic_stage_in = _organic_stage_inlet(unit, stage)
        organic_stage_out = unit.organic[0, stage]
        block.organic_to_aqueous_ratio = Var(
            initialize=case.organic_to_aqueous_ratio,
            bounds=tuple(input_bounds["organic_to_aqueous_ratio"]),
        )
        block.li_mol_l_aqueous = Var(
            initialize=case.li_feed_mol_l,
            bounds=tuple(input_bounds["li_mol_l_aqueous"]),
        )
        block.na_mol_l_aqueous = Var(
            initialize=case.na_feed_mol_l,
            bounds=tuple(input_bounds["na_mol_l_aqueous"]),
        )
        block.hydrogen_mol_l_aqueous = Var(
            initialize=10.0**-case.feed_ph,
            bounds=tuple(input_bounds["hydrogen_mol_l_aqueous"]),
        )
        block.carrier_mol_l_organic = Var(
            initialize=case.carrier_mol_l_organic,
            bounds=tuple(input_bounds["carrier_mol_l_organic"]),
        )
        block.incoming_li_loading_fraction = Var(
            initialize=0.0,
            bounds=tuple(input_bounds["incoming_li_loading_fraction"]),
        )
        block.incoming_na_loading_fraction = Var(
            initialize=0.0,
            bounds=tuple(input_bounds["incoming_na_loading_fraction"]),
        )
        block.temperature_k = Var(
            initialize=case.temperature_k,
            bounds=tuple(input_bounds["temperature_k"]),
        )
        block.equilibrium_li_loading_fraction = Var(initialize=0.8, bounds=(0.0, 1.0))
        block.equilibrium_na_loading_fraction = Var(initialize=0.01, bounds=(0.0, 1.0))
        block.ratio_link = Constraint(
            expr=block.organic_to_aqueous_ratio
            == organic_stage_in.flow_vol / aqueous_stage_in.flow_vol
        )
        block.li_link = Constraint(
            expr=block.li_mol_l_aqueous
            == aqueous_stage_in.conc_mol_comp["Li"] / (pyunits.mol / pyunits.L)
        )
        block.na_link = Constraint(
            expr=block.na_mol_l_aqueous
            == aqueous_stage_in.conc_mol_comp["Na"] / (pyunits.mol / pyunits.L)
        )
        block.hydrogen_link = Constraint(
            expr=block.hydrogen_mol_l_aqueous
            == aqueous_stage_in.conc_mol_comp["H"] / (pyunits.mol / pyunits.L)
        )
        block.carrier_link = Constraint(
            expr=block.carrier_mol_l_organic
            == fs.carrier_mol_l_organic / (pyunits.mol / pyunits.L)
        )
        block.incoming_li_link = Constraint(
            expr=block.incoming_li_loading_fraction
            * fs.carrier_mol_l_organic
            == organic_stage_in.conc_mol_comp["Li"]
        )
        block.incoming_na_link = Constraint(
            expr=block.incoming_na_loading_fraction
            * fs.carrier_mol_l_organic
            == organic_stage_in.conc_mol_comp["Na"]
        )
        block.temperature_link = Constraint(
            expr=block.temperature_k == fs.temperature / pyunits.K
        )
        block.incoming_loading_domain = Constraint(
            expr=block.incoming_li_loading_fraction
            + block.incoming_na_loading_fraction
            <= surrogate.payload["maximum_incoming_total_loading_fraction"]
        )
        attach_loaded_state_surrogate(
            block,
            input_variables={label: getattr(block, label) for label in surrogate.input_labels()},
            output_variables={label: getattr(block, label) for label in surrogate.output_labels()},
            path=surrogate_path,
        )
        block.organic_li_equilibrium = Constraint(
            expr=organic_stage_out.conc_mol_comp["Li"]
            == fs.carrier_mol_l_organic * block.equilibrium_li_loading_fraction
        )
        block.organic_na_equilibrium = Constraint(
            expr=organic_stage_out.conc_mol_comp["Na"]
            == fs.carrier_mol_l_organic * block.equilibrium_na_loading_fraction
        )
        unit.aqueous_material_balance[0, stage, "H"].deactivate()
        aqueous_stage_out = unit.aqueous[0, stage]
        block.hydrogen_response = Constraint(
            expr=aqueous_stage_out.flow_vol * aqueous_stage_out.conc_mol_comp["H"]
            == aqueous_stage_in.flow_vol * aqueous_stage_in.conc_mol_comp["H"]
            + organic_stage_in.flow_vol
            * fs.carrier_mol_l_organic
            * (
                block.equilibrium_li_loading_fraction
                + block.equilibrium_na_loading_fraction
                - block.incoming_li_loading_fraction
                - block.incoming_na_loading_fraction
            )
        )

    # The last stage receives the metal-free organic feed, so its incoming loadings are zero by construction.
    # Left free at their zero bounds and pinned by equalities, they violate LICQ; Ipopt then drifts them by
    # ~1e-8, which moves the surrogate's loaded-Li share feature from 0.5 to 0.27 and its predicted loading.
    fresh = fs.stage_equilibrium[unit.elements.last()]
    for fraction, link in (
        (fresh.incoming_li_loading_fraction, fresh.incoming_li_link),
        (fresh.incoming_na_loading_fraction, fresh.incoming_na_link),
    ):
        fraction.fix(0.0)
        link.deactivate()
    fresh.incoming_loading_domain.deactivate()

    final_aqueous = unit.aqueous[0, unit.elements.last()]
    loaded_organic = unit.organic[0, unit.elements.first()]
    fs.li_extraction_fraction = Expression(
        expr=1.0 - final_aqueous.conc_mol_comp["Li"] / aqueous_in.conc_mol_comp["Li"]
    )
    fs.na_extraction_fraction = Expression(
        expr=1.0 - final_aqueous.conc_mol_comp["Na"] / aqueous_in.conc_mol_comp["Na"]
    )
    fs.loaded_organic_li_fraction = Expression(
        expr=loaded_organic.conc_mol_comp["Li"]
        / (loaded_organic.conc_mol_comp["Li"] + loaded_organic.conc_mol_comp["Na"])
    )
    fs.li_distribution_ratio = Expression(
        expr=loaded_organic.conc_mol_comp["Li"]
        / final_aqueous.conc_mol_comp["Li"]
    )
    fs.na_distribution_ratio = Expression(
        expr=loaded_organic.conc_mol_comp["Na"]
        / final_aqueous.conc_mol_comp["Na"]
    )
    fs.li_na_separation_factor = Expression(
        expr=fs.li_distribution_ratio / fs.na_distribution_ratio
    )
    model._hbta_topo_multistage_case = case
    model._hbta_topo_loaded_surrogate = surrogate
    _set_initial_values(model)
    return model


def degrees_of_freedom(model: ConcreteModel) -> int:
    """Unfixed variables in active equalities minus active equalities.

    Same count as IDAES ``degrees_of_freedom`` for these models (no grey-box blocks), but each named surrogate expression is
    walked once instead of once per constraint that references it.
    """
    cache: dict = {}
    variables, equalities = ComponentSet(), 0
    for constraint in model.component_data_objects(Constraint, active=True, descend_into=True):
        if constraint.equality:
            equalities += 1
            variables.update(identify_variables(constraint.body, include_fixed=False, named_expression_cache=cache))
    return len(variables) - equalities


def solve_square_model(model: ConcreteModel, *, tee: bool = False):
    dof = degrees_of_freedom(model)
    if dof != 0:
        raise RuntimeError(f"square multistage model required, found DOF={dof}")
    solver = get_solver()
    solver.options["tol"] = 1.0e-8
    solver.options["constr_viol_tol"] = 1.0e-8
    solver.options["hessian_approximation"] = "limited-memory"
    solver.options["max_iter"] = 10_000
    results = solver.solve(model, tee=tee)
    if not check_optimal_termination(results):
        raise RuntimeError(f"multistage solve failed: {results.solver.termination_condition}")
    return results


@lru_cache(maxsize=32)  # about 7 MB per five-stage model
def _solved_square(case: MultistageCase) -> ConcreteModel:
    model = build_multistage_model(case)
    solve_square_model(model)
    return model


def solved_square_model(case: MultistageCase) -> ConcreteModel:
    """A clone of the square-solved default-surrogate model of ``case``.

    Studies that optimize the same start under several cost assumptions reuse one build and
    square solve; the clone starts every optimization from the same point as a fresh build.
    """
    return _solved_square(case).clone()


def maximum_balance_error_mol_h(model: ConcreteModel) -> float:
    unit = model.fs.extractor
    aq_phase = model.fs.aqueous_properties.phase_list.first()
    org_phase = model.fs.organic_properties.phase_list.first()
    maximum = 0.0
    for component in ("Li", "Na"):
        inlet = value(unit.aqueous_inlet_state[0].get_material_flow_terms(aq_phase, component))
        inlet += value(unit.organic_inlet_state[0].get_material_flow_terms(org_phase, component))
        outlet = value(unit.aqueous[0, unit.elements.last()].get_material_flow_terms(aq_phase, component))
        outlet += value(unit.organic[0, unit.elements.first()].get_material_flow_terms(org_phase, component))
        maximum = max(maximum, abs(inlet - outlet))
    charge_in = value(
        unit.aqueous_inlet_state[0].flow_vol
        * (
            unit.aqueous_inlet_state[0].conc_mol_comp["Li"]
            + unit.aqueous_inlet_state[0].conc_mol_comp["Na"]
            + unit.aqueous_inlet_state[0].conc_mol_comp["H"]
            - unit.aqueous_inlet_state[0].conc_mol_comp["Cl"]
            - unit.aqueous_inlet_state[0].conc_mol_comp["OH"]
        )
    )
    final = unit.aqueous[0, unit.elements.last()]
    charge_out = value(
        final.flow_vol
        * (
            final.conc_mol_comp["Li"]
            + final.conc_mol_comp["Na"]
            + final.conc_mol_comp["H"]
            - final.conc_mol_comp["Cl"]
            - final.conc_mol_comp["OH"]
        )
    )
    return max(maximum, abs(charge_in), abs(charge_out))


def stage_rows(model: ConcreteModel, results: Any) -> list[dict[str, Any]]:
    unit = model.fs.extractor
    case = model._hbta_topo_multistage_case
    rows = []
    for stage in unit.elements:
        block = model.fs.stage_equilibrium[stage]
        aq_in = _aqueous_stage_inlet(unit, stage)
        aq_out = unit.aqueous[0, stage]
        rows.append(
            {
                "case_id": case.case_id,
                "stage_count": case.stage_count,
                "stage": int(stage),
                "termination_condition": str(results.solver.termination_condition),
                "li_in_mol_l": value(aq_in.conc_mol_comp["Li"]),
                "li_out_mol_l": value(aq_out.conc_mol_comp["Li"]),
                "na_in_mol_l": value(aq_in.conc_mol_comp["Na"]),
                "na_out_mol_l": value(aq_out.conc_mol_comp["Na"]),
                "hydrogen_in_mol_l": value(aq_in.conc_mol_comp["H"]),
                "hydrogen_out_mol_l": value(aq_out.conc_mol_comp["H"]),
                "incoming_li_loading_fraction": value(block.incoming_li_loading_fraction),
                "incoming_na_loading_fraction": value(block.incoming_na_loading_fraction),
                "equilibrium_li_loading_fraction": value(block.equilibrium_li_loading_fraction),
                "equilibrium_na_loading_fraction": value(block.equilibrium_na_loading_fraction),
                "cumulative_li_extraction_fraction": 1.0
                - value(aq_out.conc_mol_comp["Li"])
                / value(unit.aqueous_inlet_state[0].conc_mol_comp["Li"]),
                "cumulative_na_extraction_fraction": 1.0
                - value(aq_out.conc_mol_comp["Na"])
                / value(unit.aqueous_inlet_state[0].conc_mol_comp["Na"]),
                "organic_to_aqueous_ratio": value(model.fs.organic_to_aqueous_ratio),
                "carrier_mol_l_organic": value(model.fs.carrier_mol_l_organic),
                "feed_ph": value(model.fs.feed_ph),
                "temperature_k": value(model.fs.temperature),
                "maximum_balance_error_mol_h": maximum_balance_error_mol_h(model),
            }
        )
    return rows
