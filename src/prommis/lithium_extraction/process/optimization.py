"""Lithium #201 scope objectives and released extraction decisions; chemistry stays source-owned.

Design bounds, the Li extraction target and the sodium limit belong to the solvent record (`records.SOLVENTS`, #341).
"""

from pyomo.environ import (
    Block,
    Constraint,
    Expression,
    Objective,
    Var,
    units as pyunits,
    value,
)

from prommis.lithium_extraction.process.screening_costs import (
    load_screening_cost_assumptions,
)

from prommis.lithium_extraction.process import (
    waters as brines,
    costing,
    whole_process as full_process,
    feeds as records,
)
from prommis.lithium_extraction.extraction import fresh_organic as fresh


def decisions(model, solvent):
    """Fixed design first; boundaries and downstream accounts share these decision quantities."""
    fs = model.fs
    fs.design = Block()
    b = fs.design
    record = records.SOLVENTS[solvent]
    values = dict(
        model._design,
        saponification_ratio=model._design.get("saponification_ratio", 0.0),
    )
    names = ("organic_to_aqueous_ratio", "extractant_mol_l", "saponification_ratio")
    model._decisions = {}
    for name, bounds in records.bounds(
        solvent, getattr(model, "_end", None), getattr(model, "_rule", False)
    ).items():
        variable = Var(bounds=bounds, initialize=values[name])
        b.add_component(name, variable)
        variable.fix(values[name])
        model._decisions[name] = variable
    ratio, c, sap = (model._decisions[n] for n in names)
    inlet = fs.boundary
    initial_ratio = model._design["organic_to_aqueous_ratio"]
    inlet["diluent"].unfix()
    b.diluent = Constraint(
        expr=inlet["diluent"] == ratio * model._feeds[1]["diluent"] / initial_ratio
    )
    free = record["free_extractant"]
    inlet[free].unfix()
    b.free_extractant = Constraint(expr=inlet[free] == ratio * c * (1 - sap))
    if record["saponified"]:
        inlet["sodium-carrier"].unfix()
        b.saponified_extractant = Constraint(
            expr=inlet["sodium-carrier"] == ratio * c * sap
        )
        inlet["topo"].unfix()
        b.free_topo = Constraint(expr=inlet["topo"] == inlet[free])
    if (
        "extractant_mol_l" in fs.parameters
    ):  # the stage adapter carries the concentration as an Engine parameter
        fs.parameters["extractant_mol_l"].unfix()
        b.engine_extractant = Constraint(expr=fs.parameters["extractant_mol_l"] == c)
    b.li_extraction = Expression(
        expr=1
        - fs.stage[len(fs.stages)].outlet["lithium-cation"]
        / fs.boundary["lithium-cation"]
    )
    model._solvent = solvent
    costing.add_extraction(
        model
    )  # full_process.connect_units replaces it with the full-process accounts


def add_rule(model, objective):
    """#339 decision 1 per m3 raw water: q kg Li2CO3, C_r USD outside the active set, L kg active-set loss, P USD/kg.

    'affordable': max theta subject to C_r + theta L = P q (theta of either sign); 'uniform': min (C_r + theta L)/q at
    a fixed uniform price theta (the priced form of check 1); 'headroom': max P q - C_r. Replaces any earlier objective.
    """
    fs, b = model.fs, model.fs.design
    usd_kg = costing.usd_per(pyunits.kg)
    price = carbonate_price(model, objective) * usd_kg
    q, cost, loss = (
        fs.costing.li_kg_h * costing.LI2CO3_PER_LI,
        fs.costing.operating_cost,
        fs.costing.active_loss_kg_h,
    )
    for name in ("rule", "minimum_cost"):
        if hasattr(model, name):
            model.del_component(name)
    for name in ("theta", "affordable"):
        if hasattr(b, name):
            b.del_component(name)
    kind = objective["kind"]
    if kind == "levelized_cost":
        model.rule = Objective(expr=fs.plant_costing.cost_of_production)
    elif kind == "affordable":
        start = value(
            loss, exception=False
        )  # units may not be initialized before the solve
        b.theta = Var(
            initialize=value((price * q - cost) / loss) if start else 0.0, units=usd_kg
        )
        b.affordable = Constraint(expr=cost + b.theta * loss == price * q)
        model.rule = Objective(expr=-b.theta)
    elif kind == "uniform":
        model.rule = Objective(expr=(cost + objective["theta"] * usd_kg * loss) / q)
    elif kind == "headroom":
        model.rule = Objective(expr=cost - price * q)
    elif kind == "priced_margin":
        purchased = (
            costing.ligand_cost(
                fs.costing.active_loss, objective["ligand_prices_usd_kg"]
            )
            * usd_kg
        )
        equipment = objective.get("concentration_equipment")
        if equipment is None:
            utility = objective.get("heat_price_usd_mj", 0.0) * evaporation_heat_rate(
                fs
            )
        else:
            if equipment["kind"] == "ro_mvr" and not hasattr(fs, "reverse_osmosis"):
                raise ValueError(
                    "RO followed by MVR requires a connected membrane stage"
                )
            _, heat, power = costing.evaporation_purchases(
                1000 * evaporated_water(fs), equipment
            )
            if equipment["kind"] == "ro_mvr":
                power += 1000 * fs.reverse_osmosis.electricity_kwh_per_raw_l
            heat_price, power_price = costing.utility_price(equipment)
            utility = heat_price * heat + power_price * power
        model.rule = Objective(
            expr=cost + purchased + utility * costing.usd_per(pyunits.hr) - price * q
        )
    else:
        raise ValueError(f"unknown objective {kind}")
    model._objective = dict(objective)


def carbonate_price(model, objective):
    if objective and objective["kind"] == "priced_margin":
        return objective["carbonate_price_usd_t_lce"] / 1000
    return costing.CARBONATE_USD_KG[model._case]


def evaporated_water(fs):
    """Gross kg evaporated per raw litre, including any returned condensate."""
    if hasattr(fs, "concentration"):
        return (
            fs.concentration.recycle_state[0].flow_mol_comp["H2O"]
            * brines.WATER_MOLAR_MASS_KG
        )
    if hasattr(getattr(fs, "cleanup", None), "water"):
        return fs.cleanup.water["concentrator_vapor_kg"]
    return 0.0


def evaporation_heat_rate(fs):
    """Single-effect MJ per raw m³, on the gross evaporated water basis."""
    return costing.WATER_LATENT_HEAT_MJ_KG * evaporated_water(fs) * 1000


def rule_values(model, objective=None):
    """q, C_r, L, P and the affordable price at the solved state, per m3 raw water."""
    fs = model.fs
    q = value(fs.costing.li_kg_h) * costing.LI2CO3_PER_LI
    cost, loss = value(fs.costing.operating_cost), value(fs.costing.active_loss_kg_h)
    price = carbonate_price(model, objective or getattr(model, "_objective", None))
    return {
        "carbonate_kg_m3": q,
        "cost_outside_active_usd_m3": cost,
        "active_loss_kg_m3": loss,
        "gross_evaporated_water_kg_m3": 1000 * value(evaporated_water(fs)),
        "evaporation_heat_mj_m3": value(evaporation_heat_rate(fs)),
        "active_loss_by_component_kg_m3": {
            k: value(n) for k, n in fs.costing.active_loss.items()
        },
        "carbonate_price_usd_kg": price,
        "headroom_usd_m3": price * q - cost,
        "affordable_price_usd_kg": (price * q - cost) / loss if loss > 0 else None,
    }


def add_optimization(
    model,
    scope,
    *,
    minimum_li_extraction=None,
    sodium_limit=None,
    objective=None,
    hold=(),
):
    """The extraction target remains on extraction, including in the full-process scope. `objective` selects a
    #342 rule objective (`add_rule`); without it the scope cost per kg Li is minimized. Decisions named in `hold`
    stay at their current values (the held optimum of #342 check 8)."""
    record, b, fs = records.SOLVENTS[model._solvent], model.fs.design, model.fs
    design = record["design"]
    for name, variable in model._decisions.items():
        if variable.lb != variable.ub and name not in hold:
            variable.unfix()
    if (
        getattr(model, "_full_process", {})
        .get("assumptions", {})
        .get("optimize_mother_liquor_return")
    ):
        fs.mother_liquor_return.fraction.unfix()
    target = (
        design["li_target"] if minimum_li_extraction is None else minimum_li_extraction
    )
    if target is not None:
        b.li_target = Constraint(expr=b.li_extraction >= target)
    ratio, c = (
        model._decisions[n] for n in ("organic_to_aqueous_ratio", "extractant_mol_l")
    )
    b.loading = Constraint(
        fs.stages,
        rule=lambda _, k: fs.stage[k].outlet["lithium-carrier"] <= 0.72 * ratio * c,
    )
    if not record["saponified"]:  # net Na extracted against feed Na
        limit = design["sodium_limit"] if sodium_limit is None else sodium_limit
        b.sodium = Constraint(
            expr=fs.stage[1].outlet["sodium-carrier"]
            <= limit * model._feeds[0]["sodium-cation"]
        )
    elif sodium_limit is not None:
        li, na = (fs.stage[1].outlet[s] for s in ("lithium-carrier", "sodium-carrier"))
        if scope == "full_process":
            li, na = (fs.strip.metals[s] for s in ("Li", "Na"))
        b.sodium = Constraint(expr=na <= sodium_limit * (li + na))
    feed = model._physical_feed
    volume = fs.washed_feed_l_per_raw_l if scope == "full_process" else 1.0
    deficits = {
        s: feed["model_feed"][records.SPECIES_LABELS[s]]
        - feed["physical_feed"].get(records.SPECIES_LABELS[s], 0.0)
        for s in model._chemistry.supports[0]
    }
    b.physical_raffinate = Constraint(
        tuple(s for s, n in deficits.items() if n > 0),
        rule=lambda _, s: volume * fs.stage[len(fs.stages)].outlet[s] >= deficits[s],
    )
    model.feasibility.deactivate()
    if objective is None:
        model.minimum_cost = Objective(expr=fs.costing.usd_kg_li)
    else:
        add_rule(model, objective)


def synchronize(model):
    """Report the solved boundaries and decision values, never the initialization design."""
    model._design.update({k: value(v) for k, v in model._decisions.items()})
    model._feeds = tuple(
        {s: value(model.fs.boundary[s]) for s in feed} for feed in model._feeds
    )


def nominal_to_raw_litre(result, v):
    """Label nominal washed-feed amounts and scale conserved residuals to the raw litre."""
    result["outlet_mol_per_nominal_l"] = result.pop("outlet_mol_per_raw_l")
    residuals = result["balance_residuals_mol_per_raw_l"]
    result["balance_residuals_mol_per_nominal_l"] = residuals
    result["balance_residuals_mol_per_raw_l"] = {k: n * v for k, n in residuals.items()}
    for row in result["stages"]:
        for name in ("inlet", "outlet", "balance_residuals"):
            row[name + "_mol_per_nominal_l"] = row.pop(name + "_mol_per_raw_l")
        row["balance_residuals_mol_per_raw_l"] = {
            k: n * v for k, n in row["balance_residuals_mol_per_nominal_l"].items()
        }
        row["phase_charge_eq_per_nominal_l"] = row["phase_charge_eq_per_raw_l"]
        row["phase_charge_eq_per_raw_l"] = [
            n * v for n in row["phase_charge_eq_per_nominal_l"]
        ]
        difference = row["certificate"]["outlet_max_difference_mol_per_raw_l"]
        row["certificate"]["outlet_max_difference_mol_per_nominal_l"] = difference
        row["certificate"]["outlet_max_difference_mol_per_raw_l"] = (
            None if difference is None else difference * v
        )
    return result


def report(model, scope, softening=None, *, stages=None):
    synchronize(model)
    solvent = model._solvent
    record = records.SOLVENTS[solvent]
    result = (
        records.stage_adapter(record["stage_adapter"]).report(model)
        if stages is None
        else stages
    )
    if (
        scope == "full_process" and "outlet_mol_per_raw_l" in result
    ):  # an adapter reporting the nominal washed litre
        nominal_to_raw_litre(result, model._washed["feed_l"])
    result["solvent"] = solvent
    if scope == "full_process":
        result = full_process.report(model, result, softening)
    else:
        if "costs" not in result:
            result["costs"] = fresh.extraction_costs(
                model,
                result["li_extraction_fraction"],
                model._feeds[1]["sodium-carrier"],
            )
        owner = result["costs"]
        hours = (
            load_screening_cost_assumptions().annual_operating_hours
        )  # owner's annual m3 at 1 m3/h
        result["idaes_costing_reconciliation"] = costing.reconcile(
            model.fs.costing,
            {k: line["usd_y"] / hours for k, line in owner["lines"].items()},
            owner["priced_subtotal_usd_kg_li"],
            owner["priced_subtotal_usd_m3_raw_brine"]
            / owner["priced_subtotal_usd_kg_li"],
        )
        feed = result["feed"]
        result["calibration_coverage"] = [
            {
                "quantity": name,
                "value": feed["model_feed"][key],
                "unit": unit,
                "basis": "projected nominal aqueous inlet",
                "comparison": "physical calibration coverage unverified",
                "source_owner": records.SOLVENTS[solvent]["feed_owner"],
            }
            for name, key, unit in (
                ("projected Li", "Li", "mol/L aqueous"),
                ("projected Na", "Na", "mol/L aqueous"),
            )
        ]
    ratio, c = (
        model._design[n] for n in ("organic_to_aqueous_ratio", "extractant_mol_l")
    )
    result["outlet_li_loading_fractions"] = [
        value(model.fs.stage[k].outlet["lithium-carrier"]) / (ratio * c)
        for k in model.fs.stages
    ]
    li, na = (
        value(model.fs.stage[1].outlet[s])
        for s in ("lithium-carrier", "sodium-carrier")
    )
    result["na_metric"] = {
        "name": (
            "net Na extracted / feed Na"
            if not record["saponified"]
            else (
                "strip-liquor Na/(Li+Na)"
                if scope == "full_process"
                else "loaded-organic Na/(Li+Na)"
            )
        ),
        "value": (
            na / model._feeds[0]["sodium-cation"]
            if not record["saponified"]
            else (
                result["full_process"]["strip_na_fraction"]
                if scope == "full_process"
                else na / (li + na)
            )
        ),
    }
    result["cost_usd_kg_li"] = (
        result["full_process"]["operating_usd_kg_li"]
        if scope == "full_process"
        else value(model.fs.costing.usd_kg_li)
    )
    result["cost_usd_m3_raw_brine"] = (
        sum(result["full_process"]["cost_usd_m3"].values())
        if scope == "full_process"
        else value(model.fs.costing.operating_cost)
    )
    result["cost_basis"] = records.SCOPES[scope]["li_denominator"]
    result["hydraulics"] = costing.hydraulic_report(
        value(model.fs.washed_feed_l_per_raw_l), ratio, len(model.fs.stages), scope
    )
    result["product_recovery_fraction"] = (
        result["full_process"]["raw_li_recovery"] if scope == "full_process" else None
    )
    if hasattr(model.fs.costing, "active_loss_kg_h"):
        result["rule"] = rule_values(model) | {
            "objective": getattr(model, "_objective", None)
        }
    result["labels"] = [
        s
        for s in result["labels"]
        if "fixed-design" not in s and "no full-process" not in s
    ]
    result["labels"] += [
        "model prediction; no physical validation; local solution, no global optimum claim"
    ]
    return result


def target_failures(result, solvent, minimum_li_extraction=None):
    failures = list(result["R7"]["failures"])
    record = records.SOLVENTS[solvent]
    target, sodium = record["design"]["li_target"], record["design"]["sodium_limit"]
    target = target if minimum_li_extraction is None else minimum_li_extraction
    if target is not None and result["li_extraction_fraction"] < target - 1e-9:
        failures.append("Li extraction target not met")
    if max(result["outlet_li_loading_fractions"], default=0.0) > 0.72 + 1e-9:
        failures.append("Li loading exceeds 0.72 per pair")
    if sodium is not None and result["na_metric"]["value"] > sodium + 1e-9:
        failures.append(f"Na extraction exceeds {sodium:g} feed Na")
    return failures
