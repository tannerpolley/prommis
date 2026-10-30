"""IDAES-owned optimization of the local MS-2 extraction-stage model."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any

from idaes.core import register_idaes_currency_units
from idaes.core.solvers import get_solver
from pyomo.environ import Block, Constraint, Expression, Objective, minimize, value
from pyomo.environ import units as pyunits
from pyomo.opt import check_optimal_termination

from lithium_extraction.common.economics.hbta_topo import load_screening_cost_assumptions
from .multistage import LI_MOLAR_MASS_G_MOL, maximum_balance_error_mol_h


HBTA_MOLAR_MASS_KG_MOL = 0.21616
TOPO_MOLAR_MASS_KG_MOL = 0.3866349
REFERENCE_CARRIER_MOL_L_ORGANIC = 0.015
DEFAULT_MARGINAL_EXTRACTANT_PRICE_USD_KG = 100.0
# Zhang et al. (2017, Section 3.5) state 1.5 g/L Li, 0.72 per pair, as the maximum loading of 0.3 M HBTA/TOPO in
# kerosene. It is consistent with their plotted isotherm trend and slightly above the 0.645-0.705 per pair near
# saturation in equimolar HBTA/TOPO sources (Zhang 2020, 2021); Sun (2025, HTTA/P113) plateaus at about 0.70-0.73.
# The crystal structure is a 1:1:1 dimer (Zhang 2021), and the 2:1:1 reading rests on a printed slope the plotted
# data of Zhang (2017) do not support. 0.72 is an assumption carried from source conditions, not a limit measured
# at MS-2 extractant concentration, pH, sodium level or diluent (analyses/v3/hbta_topo_li_capacity_evidence).
# Designs keep every stage's outlet organic Li at or below 0.72; Li only, sodium not capped.
MAXIMUM_ORGANIC_LITHIUM_LOADING_FRACTION = 0.72

register_idaes_currency_units()
if not hasattr(pyunits, "source_period_nominal_USD"):
    pyunits.load_definitions_from_strings(["source_period_nominal_USD = USD_2023"])
SOURCE_PERIOD_NOMINAL_USD = pyunits.source_period_nominal_USD


@dataclass(frozen=True)
class OptimizationResult:
    status: str
    termination_condition: str
    stage_count: int
    organic_to_aqueous_ratio: float
    carrier_mol_l_organic: float
    feed_ph: float
    temperature_k: float
    li_extraction_fraction: float
    na_extraction_fraction: float
    loaded_organic_li_fraction: float
    li_distribution_ratio: float
    na_distribution_ratio: float
    li_na_separation_factor: float
    extraction_screening_cost_usd_y: float | None
    extraction_screening_cost_usd_kg_li: float | None
    organic_formulation_cost_usd_l: float | None
    annual_organic_makeup_l: float | None
    annual_hbta_makeup_kg: float | None
    annual_topo_makeup_kg: float | None
    hbta_marginal_price_usd_kg: float | None
    topo_marginal_price_usd_kg: float | None
    maximum_balance_error_mol_h: float
    maximum_outlet_li_loading_fraction: float


def reference_anchored_organic_cost_usd_l(
    carrier_mol_l_organic: Any,
    *,
    formulated_organic_density_kg_l: float,
    reference_formulated_organic_price_usd_kg: float,
    hbta_marginal_price_usd_kg: float,
    topo_marginal_price_usd_kg: float,
) -> Any:
    """Return formulation replacement cost per litre.

    The registered blended-organic analogue anchors the 0.015 mol/L HBTA plus
    0.015 mol/L TOPO formulation.  Concentration changes are priced from the
    exact added or removed HBTA and TOPO masses.  The marginal component
    prices remain explicit screening assumptions until bulk quotes exist.
    """

    if isinstance(carrier_mol_l_organic, Real):
        concentration_delta = carrier_mol_l_organic - REFERENCE_CARRIER_MOL_L_ORGANIC
        density = formulated_organic_density_kg_l
        reference_price = reference_formulated_organic_price_usd_kg
        hbta_mass = HBTA_MOLAR_MASS_KG_MOL
        topo_mass = TOPO_MOLAR_MASS_KG_MOL
        hbta_price = hbta_marginal_price_usd_kg
        topo_price = topo_marginal_price_usd_kg
    else:
        concentration_delta = carrier_mol_l_organic - (
            REFERENCE_CARRIER_MOL_L_ORGANIC * pyunits.mol / pyunits.L
        )
        density = formulated_organic_density_kg_l * pyunits.kg / pyunits.L
        reference_price = (
            reference_formulated_organic_price_usd_kg
            * SOURCE_PERIOD_NOMINAL_USD
            / pyunits.kg
        )
        hbta_mass = HBTA_MOLAR_MASS_KG_MOL * pyunits.kg / pyunits.mol
        topo_mass = TOPO_MOLAR_MASS_KG_MOL * pyunits.kg / pyunits.mol
        hbta_price = hbta_marginal_price_usd_kg * SOURCE_PERIOD_NOMINAL_USD / pyunits.kg
        topo_price = topo_marginal_price_usd_kg * SOURCE_PERIOD_NOMINAL_USD / pyunits.kg
    return (
        density
        * reference_price
        + concentration_delta
        * (
            hbta_mass * hbta_price
            + topo_mass * topo_price
        )
    )


def annual_extraction_electricity_kwh(
    case: Any,
    assumptions: Any,
    *,
    aqueous_flow_l_h: Any | None = None,
    organic_to_aqueous_ratio: Any | None = None,
) -> Any:
    """Return annual mixer and pump electricity for an extraction case."""

    aqueous_flow = case.aqueous_flow_l_h * pyunits.L / pyunits.h if aqueous_flow_l_h is None else aqueous_flow_l_h
    ratio = case.organic_to_aqueous_ratio if organic_to_aqueous_ratio is None else organic_to_aqueous_ratio
    total_flow_m3_h = pyunits.convert(
        aqueous_flow * (1.0 + ratio),
        to_units=pyunits.m**3 / pyunits.h,
    )
    mixing_holdup_m3 = total_flow_m3_h * (0.1 * pyunits.h) * case.stage_count
    mixer_power_kw = assumptions.mixer_power_density * pyunits.kW / pyunits.m**3 * mixing_holdup_m3
    pump_power_kw = pyunits.convert(
        1000.0
        * pyunits.kg
        / pyunits.m**3
        * 9.80665
        * pyunits.m
        / pyunits.s**2
        * assumptions.pump_head
        * pyunits.m
        * total_flow_m3_h
        / assumptions.pump_efficiency
        * case.stage_count,
        to_units=pyunits.kW,
    )
    return pyunits.convert(
        (mixer_power_kw + pump_power_kw) * assumptions.annual_operating_hours * pyunits.h / pyunits.year,
        to_units=pyunits.kWh / pyunits.year,
    )


def _unfix_decisions(model: Any, block: Block, *, optimize_ph: bool) -> None:
    """Free the design decisions and hold every stage's outlet Li below the measured organic maximum."""
    model.fs.organic_to_aqueous_ratio.unfix()
    model.fs.carrier_mol_l_organic.unfix()
    if optimize_ph:
        model.fs.feed_ph.unfix()
    block.maximum_organic_lithium_loading = Constraint(
        model.fs.extractor.elements,
        rule=lambda _b, stage: model.fs.stage_equilibrium[stage].equilibrium_li_loading_fraction
        <= MAXIMUM_ORGANIC_LITHIUM_LOADING_FRACTION,
    )


def add_pareto_optimization(model: Any, *, minimum_li_extraction: float) -> Block:
    """Minimize Na co-extraction at an exact Li extraction level."""

    if not 0.0 < minimum_li_extraction < 1.0:
        raise ValueError("minimum Li extraction must be between zero and one")
    block = model.fs.performance_optimization = Block()
    _unfix_decisions(model, block, optimize_ph=True)
    block.target_li_extraction = Constraint(
        expr=model.fs.li_extraction_fraction == minimum_li_extraction
    )
    block.objective = Objective(expr=model.fs.na_extraction_fraction, sense=minimize)
    block.target_li_extraction_value = minimum_li_extraction
    return block


def add_minimum_cost_optimization(
    model: Any,
    *,
    minimum_li_extraction: float = 0.95,
    maximum_na_extraction: float = 0.02,
    hbta_marginal_price_usd_kg: float = DEFAULT_MARGINAL_EXTRACTANT_PRICE_USD_KG,
    topo_marginal_price_usd_kg: float = DEFAULT_MARGINAL_EXTRACTANT_PRICE_USD_KG,
    total_organic_loss_fraction_per_pass: float | None = None,
    stage_operation_allowance_usd_m3_feed: float | None = None,
) -> Block:
    """Minimize the source-bounded extraction-section screening cost.

    The objective includes concentration-dependent organic makeup,
    stage-operation allowance, mixer power, and pump power.  The registered
    blended-organic analogue anchors the 0.015 mol/L formulation; explicit
    HBTA and TOPO marginal prices value concentration changes.
    """

    if not 0.0 < minimum_li_extraction < 1.0:
        raise ValueError("minimum Li extraction must be between zero and one")
    if not 0.0 <= maximum_na_extraction < 1.0:
        raise ValueError("maximum Na extraction must be between zero and one")
    if hbta_marginal_price_usd_kg < 0.0 or topo_marginal_price_usd_kg < 0.0:
        raise ValueError("marginal extractant prices must be nonnegative")
    for label, override in (
        ("total organic-loss fraction per pass", total_organic_loss_fraction_per_pass),
        ("stage-operation allowance", stage_operation_allowance_usd_m3_feed),
    ):
        if override is not None and (not isfinite(override) or override < 0.0):
            raise ValueError(f"{label} must be finite and nonnegative")
    fs = model.fs
    unit = fs.extractor
    case = model._hbta_topo_multistage_case
    assumptions = load_screening_cost_assumptions()
    block = fs.cost_optimization = Block()
    _unfix_decisions(model, block, optimize_ph=False)
    block.minimum_li_extraction = Constraint(
        expr=fs.li_extraction_fraction >= minimum_li_extraction
    )
    block.maximum_na_extraction = Constraint(
        expr=fs.na_extraction_fraction <= maximum_na_extraction
    )
    aqueous_flow_l_h = unit.aqueous_inlet_state[0].flow_vol
    annual_feed_m3 = pyunits.convert(
        aqueous_flow_l_h
        * assumptions.annual_operating_hours
        * pyunits.h
        / pyunits.year,
        to_units=pyunits.m**3 / pyunits.year,
    )
    organic_flow_l_h = aqueous_flow_l_h * fs.organic_to_aqueous_ratio
    loss_fraction = (
        total_organic_loss_fraction_per_pass
        if total_organic_loss_fraction_per_pass is not None
        else assumptions.organic_entrainment_loss_fraction
        + assumptions.organic_degradation_loss_fraction
        + assumptions.organic_purge_fraction
    )
    stage_operation_allowance = (
        stage_operation_allowance_usd_m3_feed
        if stage_operation_allowance_usd_m3_feed is not None
        else assumptions.extraction_stage_operation_auxiliary
    )
    annual_organic_makeup_l = (
        organic_flow_l_h
        * assumptions.annual_operating_hours
        * pyunits.h
        / pyunits.year
        * loss_fraction
    )
    organic_formulation_cost_usd_l = reference_anchored_organic_cost_usd_l(
        fs.carrier_mol_l_organic,
        formulated_organic_density_kg_l=assumptions.formulated_organic_density,
        reference_formulated_organic_price_usd_kg=assumptions.formulated_organic_price,
        hbta_marginal_price_usd_kg=hbta_marginal_price_usd_kg,
        topo_marginal_price_usd_kg=topo_marginal_price_usd_kg,
    )
    annual_electricity_kwh = annual_extraction_electricity_kwh(
        case,
        assumptions,
        aqueous_flow_l_h=aqueous_flow_l_h,
        organic_to_aqueous_ratio=fs.organic_to_aqueous_ratio,
    )
    block.organic_formulation_cost_usd_l = Expression(
        expr=organic_formulation_cost_usd_l
    )
    block.annual_organic_makeup_l = Expression(expr=annual_organic_makeup_l)
    block.annual_hbta_makeup_kg = Expression(
        expr=annual_organic_makeup_l
        * fs.carrier_mol_l_organic
        * HBTA_MOLAR_MASS_KG_MOL
        * pyunits.kg
        / pyunits.mol
    )
    block.annual_topo_makeup_kg = Expression(
        expr=annual_organic_makeup_l
        * fs.carrier_mol_l_organic
        * TOPO_MOLAR_MASS_KG_MOL
        * pyunits.kg
        / pyunits.mol
    )
    block.annual_hbta_concentration_increment_kg = Expression(
        expr=annual_organic_makeup_l
        * (
            fs.carrier_mol_l_organic
            - REFERENCE_CARRIER_MOL_L_ORGANIC * pyunits.mol / pyunits.L
        )
        * HBTA_MOLAR_MASS_KG_MOL
        * pyunits.kg
        / pyunits.mol
    )
    block.annual_topo_concentration_increment_kg = Expression(
        expr=annual_organic_makeup_l
        * (
            fs.carrier_mol_l_organic
            - REFERENCE_CARRIER_MOL_L_ORGANIC * pyunits.mol / pyunits.L
        )
        * TOPO_MOLAR_MASS_KG_MOL
        * pyunits.kg
        / pyunits.mol
    )
    block.annual_organic_makeup_cost = Expression(expr=annual_organic_makeup_l * block.organic_formulation_cost_usd_l)
    block.annual_stage_operation_cost = Expression(
        expr=annual_feed_m3 * stage_operation_allowance * SOURCE_PERIOD_NOMINAL_USD / pyunits.m**3 * case.stage_count
    )
    block.annual_electricity_cost = Expression(
        expr=annual_electricity_kwh * assumptions.electricity_price * SOURCE_PERIOD_NOMINAL_USD / pyunits.kWh
    )
    block.annual_extraction_screening_cost = Expression(
        expr=block.annual_organic_makeup_cost + block.annual_stage_operation_cost + block.annual_electricity_cost
    )
    feed_li_mol_h = aqueous_flow_l_h * unit.aqueous_inlet_state[0].conc_mol_comp["Li"]
    block.annual_extracted_li_kg = Expression(
        expr=pyunits.convert(
            feed_li_mol_h
            * fs.li_extraction_fraction
            * assumptions.annual_operating_hours
            * pyunits.h
            / pyunits.year
            * LI_MOLAR_MASS_G_MOL
            * pyunits.g
            / pyunits.mol,
            to_units=pyunits.kg / pyunits.year,
        )
    )
    block.screening_cost_usd_kg_li = Expression(
        expr=block.annual_extraction_screening_cost / block.annual_extracted_li_kg
    )
    block.scaled_objective = Objective(
        expr=block.screening_cost_usd_kg_li / 1000.0,
        sense=minimize,
    )
    block.minimum_li_extraction_value = minimum_li_extraction
    block.maximum_na_extraction_value = maximum_na_extraction
    block.hbta_marginal_price_usd_kg = hbta_marginal_price_usd_kg
    block.topo_marginal_price_usd_kg = topo_marginal_price_usd_kg
    block.total_organic_loss_fraction_per_pass = loss_fraction
    block.stage_operation_allowance_usd_m3_feed = stage_operation_allowance
    block.cost_scope = "reference_anchored_component_resolved_organic_replacement"
    return block


def solve_optimization(model: Any, *, tee: bool = False) -> Any:
    solver = get_solver()
    solver.options["tol"] = 1.0e-8
    solver.options["constr_viol_tol"] = 1.0e-8
    # Exact Hessian: limited-memory steps stalled at dual infeasibility ~4e-5 on flat cost directions.
    solver.options["max_iter"] = 1_500
    results = solver.solve(model, tee=tee)
    if not check_optimal_termination(results):
        raise RuntimeError(f"multistage optimization failed: {results.solver.termination_condition}")
    return results


def optimization_result(model: Any, results: Any) -> OptimizationResult:
    cost_block = getattr(model.fs, "cost_optimization", None)
    return OptimizationResult(
        status="optimized",
        termination_condition=str(results.solver.termination_condition),
        stage_count=model._hbta_topo_multistage_case.stage_count,
        organic_to_aqueous_ratio=float(value(model.fs.organic_to_aqueous_ratio)),
        carrier_mol_l_organic=float(value(model.fs.carrier_mol_l_organic)),
        feed_ph=float(value(model.fs.feed_ph)),
        temperature_k=float(value(model.fs.temperature)),
        li_extraction_fraction=float(value(model.fs.li_extraction_fraction)),
        na_extraction_fraction=float(value(model.fs.na_extraction_fraction)),
        loaded_organic_li_fraction=float(value(model.fs.loaded_organic_li_fraction)),
        li_distribution_ratio=float(value(model.fs.li_distribution_ratio)),
        na_distribution_ratio=float(value(model.fs.na_distribution_ratio)),
        li_na_separation_factor=float(value(model.fs.li_na_separation_factor)),
        extraction_screening_cost_usd_y=(
            float(value(cost_block.annual_extraction_screening_cost)) if cost_block is not None else None
        ),
        extraction_screening_cost_usd_kg_li=(
            float(value(cost_block.screening_cost_usd_kg_li)) if cost_block is not None else None
        ),
        organic_formulation_cost_usd_l=(
            float(value(cost_block.organic_formulation_cost_usd_l)) if cost_block is not None else None
        ),
        annual_organic_makeup_l=(float(value(cost_block.annual_organic_makeup_l)) if cost_block is not None else None),
        annual_hbta_makeup_kg=(float(value(cost_block.annual_hbta_makeup_kg)) if cost_block is not None else None),
        annual_topo_makeup_kg=(float(value(cost_block.annual_topo_makeup_kg)) if cost_block is not None else None),
        hbta_marginal_price_usd_kg=(float(cost_block.hbta_marginal_price_usd_kg) if cost_block is not None else None),
        topo_marginal_price_usd_kg=(float(cost_block.topo_marginal_price_usd_kg) if cost_block is not None else None),
        maximum_balance_error_mol_h=maximum_balance_error_mol_h(model),
        maximum_outlet_li_loading_fraction=max(
            float(value(model.fs.stage_equilibrium[stage].equilibrium_li_loading_fraction))
            for stage in model.fs.extractor.elements
        ),
    )
