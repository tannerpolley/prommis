"""Connected #176 scrub, strip and Ge carbonation units on the extraction stages.

Scrub and strip are fixed-fraction PrOMMiS SolventExtraction stages; carbonation is an IDAES Mixer,
PrOMMiS Precipitator and IDAES Separator recycle loop. The historical owner remains unchanged and
independently evaluates every solved quantity and cost account. Downstream flows are mol/h at 1 L/h
raw brine (mol per raw litre); water is kg/raw L.
"""

import csv
import math

from idaes.core.util.scaling import constraint_scaling_transform
from pyomo.common.collections import ComponentSet
from pyomo.contrib.incidence_analysis import solve_strongly_connected_components
from pyomo.environ import (
    Constraint,
    Expression,
    NonNegativeReals,
    Var,
    units as pyunits,
    value,
)
from pyomo.network import Arc
from pyomo.util.check_units import assert_units_consistent
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.util.subsystems import TemporarySubsystemManager, create_subsystem_block

from prommis.lithium_extraction.thermodynamics.hbta_topo_constants import (
    LI_MOLAR_MASS_G_MOL,
)

from prommis.lithium_extraction.process import (
    waters as brines,
    carbonation,
    acid_recovery as chain,
    costing,
    strip_and_carbonation as downstream,
    flowsheet,
    mother_liquor_return,
    feeds as records,
    retained_references as references,
)
from prommis.lithium_extraction.extraction import stage

OWNER_COMPARISON_LIMIT = (
    1e-9  # connected quantities against the independent #176 owner recomputation
)


def reconciled_nacl(raw, bounded):
    """The solved, bounded NaCl coordinate for a recomputed value that differs from it only by rounding (#267)."""
    if (
        not math.isfinite(raw)
        or not 0 <= bounded <= 3
        or abs(raw - bounded) > OWNER_COMPARISON_LIMIT
    ):
        raise ValueError("NaCl differs from the bounded connected state")
    return bounded


def connected_nacl(raw, model, washed, loaded, assumptions, ratio):
    """Only the unchanged state of the last accepted solve, with its own inputs, supplies the endpoint coordinate."""
    stored = {
        k: n
        for k, n in model._full_process["assumptions"].items()
        if k != "extractant_mol_l"
    }
    if hasattr(getattr(model.fs, "strip", None), "aqueous_to_organic"):
        stored["strip_aqueous_to_organic"] = value(model.fs.strip.aqueous_to_organic)
    supplied = {
        k: n for k, n in assumptions.items() if k != "extractant_mol_l"
    }  # enumerate_units never reads it
    if (
        getattr(model, "_solved_values", None) != stage.solved_values(model)
        or washed is not model._washed
        or loaded != extraction(model)["loaded_organic"]
        or supplied != stored
        or ratio != model._design["organic_to_aqueous_ratio"]
    ):
        raise ValueError(
            "NaCl endpoint requires the unchanged state and inputs of the last accepted solve"
        )
    return reconciled_nacl(raw, value(model.fs.carbonation.nacl_mol_kg))


def extraction(model):
    """Transport the generic boundary amounts to the historical cost owner's named inputs."""
    v = (
        value(model.fs.washed_feed_l_per_raw_l)
        if hasattr(model.fs, "mother_liquor_return")
        else model._washed["feed_l"]
    )
    aq = model.fs.stage[len(model.fs.stages)].outlet
    org = model.fs.stage[1].outlet
    free = "free-carrier" if "free-carrier" in org else "hbta"
    return {
        "raffinate": {
            "li": value(aq["lithium-cation"]),
            "na": value(aq["sodium-cation"]),
            "bta": value(aq["bta-anion"]) if "bta-anion" in aq else 0.0,
        },
        "loaded_organic": {
            "li": value(org["lithium-carrier"]),
            "na": value(org["sodium-carrier"]),
            "hbta": sum(
                value(org[s]) for s in (free, "lithium-carrier", "sodium-carrier")
            ),
        },
        "diluent_mol": value(org["diluent"]) * v,
        "saponification_mol_per_raw_l": model._feeds[1].get("sodium-carrier", 0.0) * v,
        "organic_volume_l_per_raw_l": model._design["organic_to_aqueous_ratio"] * v,
        "mixed_feed_l_per_raw_l": v,
    } | (
        {
            "organic_entrained_lithium": value(
                model.fs.organic_loss.loss_state[0].flow_mol_comp["Li"]
            )
        }
        if hasattr(getattr(model.fs, "design", None), "returned_lithium")
        else {}
    )


def enumerate_units(
    washed, loaded, assumptions, *, organic_to_aqueous_ratio=1.0, model=None
):
    """Enumerate both scrub bounds and every measured Ge interval before calling IPOPT.

    Fixed-design screening discards only analytically inadmissible alternatives;
    released extraction decisions require solving those alternatives in step 7.
    """
    v = washed["feed_l"]
    li, na = (v * loaded[k] for k in ("li", "na"))
    source = assumptions["scrub_hcl_mol_l"] * organic_to_aqueous_ratio * v / 10
    demand = assumptions["scrub_li_loss"] * li + assumptions["scrub_na_efficiency"] * na
    strip_eff = assumptions.get("strip_li_fraction", flowsheet.strip_li_fraction())
    stripped_li = (1 - assumptions["scrub_li_loss"]) * li * strip_eff
    stripped_na = (1 - assumptions["scrub_na_efficiency"]) * na
    gross = (
        assumptions.get("strip_hcl_mol_l", flowsheet.STRIP_HCL_MOL_L)
        * organic_to_aqueous_ratio
        * v
        * assumptions.get("strip_aqueous_to_organic", 0.10)
    )
    fresh = (
        stripped_li
        + stripped_na
        + (1 - assumptions["acid_recovery"]) * (gross - stripped_li - stripped_na)
    )
    salt = (
        fresh / (stripped_li / assumptions["concentrate_li_mol_kg"])
        if stripped_li > 0
        else None
    )
    evaluated = salt
    if salt is not None and salt > 3 and model is not None:
        evaluated = connected_nacl(
            salt, model, washed, loaded, assumptions, organic_to_aqueous_ratio
        )
    with carbonation.GE_TABLE.open() as stream:
        knots = [
            float(r["nacl_mol_kg"])
            for r in csv.DictReader(stream)
            if r["series"] == "NaCl"
        ]
    candidates = []
    for regime, acid, other in (
        ("source", source, demand),
        ("stoichiometric", demand, source),
    ):
        for lo, hi in zip(knots, knots[1:]):
            reasons = []
            if acid < other:
                reasons.append("scrub acid below the other bound")
            if gross < stripped_li + stripped_na:
                reasons.append("strip acid inventory below metal proton demand")
            if (
                evaluated is None
                or not math.isfinite(evaluated)
                or not 0 <= evaluated <= 3
            ):
                reasons.append("carbonation outside the measured solubility data")
            elif not lo <= evaluated <= hi:
                reasons.append("NaCl outside this interpolation interval")
            else:
                s = carbonation.empirical_solubility(
                    assumptions["carbonation_temperature_k"], evaluated
                )
                if (
                    stripped_li
                    - 2 * s * stripped_li / assumptions["concentrate_li_mol_kg"]
                    < 0
                ):
                    reasons.append("concentration input is below Li2CO3 saturation")
            candidates.append(
                {
                    "scrub_regime": regime,
                    "carbonation_interval_mol_kg": [lo, hi],
                    "mother_nacl_mol_kg": salt,
                    "scrub_acid_mol": acid,
                    **(
                        {"evaluated_nacl_mol_kg": evaluated}
                        if evaluated is not salt
                        else {}
                    ),
                    "scrub_other_bound_mol": other,
                    "feasible": not reasons,
                    "reasons": reasons,
                }
            )
    return candidates


def strip_ratio_interval(
    loaded, assumptions, organic_to_aqueous_ratio, *, water_tolerance=0.0
):
    """Admissible strip A/O at the entering organic state, per nominal aqueous litre."""
    li = (
        (1 - assumptions["scrub_li_loss"])
        * loaded["li"]
        * assumptions.get("strip_li_fraction", flowsheet.strip_li_fraction())
    )
    na = (1 - assumptions["scrub_na_efficiency"]) * loaded["na"]
    needed_water = li / assumptions["concentrate_li_mol_kg"]
    credit = assumptions["acid_recovery"]
    dose = assumptions.get("strip_hcl_mol_l", flowsheet.STRIP_HCL_MOL_L)
    lower, upper = assumptions["strip_aqueous_to_organic_bounds"]
    acid_and_design_min = max(lower, (li + na) / (dose * organic_to_aqueous_ratio))
    hi = upper
    lo = max(acid_and_design_min, needed_water / organic_to_aqueous_ratio)
    if credit < 1.0:
        hi = min(
            hi,
            (3 * needed_water - credit * (li + na))
            / ((1 - credit) * dose * organic_to_aqueous_ratio),
        )
    if li <= 0:
        return None
    if lo <= hi:
        return lo, hi
    # A replayed state can touch the water endpoint within the retained
    # fixed-control Jacobian bound on water per nominal litre.
    if (
        acid_and_design_min <= hi
        and needed_water - hi * organic_to_aqueous_ratio <= water_tolerance
    ):
        return hi, hi
    # Allow roundoff in the scrub/strip products and divisions at a touching
    # endpoint. The connected solve still enforces the registered bounds.
    if math.isfinite(lo + hi) and lo - hi <= 4 * (math.ulp(lo) + math.ulp(hi)):
        touching = min(upper, max(lower, hi))
        return touching, touching
    return None


def strip_ratio_start(loaded, assumptions, organic_to_aqueous_ratio):
    interval = strip_ratio_interval(loaded, assumptions, organic_to_aqueous_ratio)
    return sum(interval) / 2 if interval else None


def carbonation_interval_constraint(carbon, assumptions, interval):
    lo, hi = interval
    # With zero recovery, fresh chloride is dose times gross strip water.
    # Available strip water therefore already enforces m >= dose.
    dose = assumptions.get("strip_hcl_mol_l", flowsheet.STRIP_HCL_MOL_L)
    if (
        "strip_aqueous_to_organic_bounds" in assumptions
        and assumptions["acid_recovery"] == 0
        and lo == dose
    ):
        lo = None
    carbon.interval.set_value((lo, carbon.nacl_mol_kg, hi))


def connect_units(model, case, assumptions, candidate, *, initial_accounts=None):
    """Write the owner's algebra with an explicit active scrub bound and fixed Ge interval."""
    washed, v = model._washed, model._washed["feed_l"]
    initial = extraction(model)
    assumptions = dict(assumptions, extractant_mol_l=model._design["extractant_mol_l"])
    owner = (
        flowsheet.costs_and_balances(washed, initial, case, assumptions)
        if initial_accounts is None
        else initial_accounts
    )
    owner = dict(
        owner, mass_kg_per_l=owner["mass_kg_per_l"] | flowsheet.softening_masses(washed)
    )
    if "organic_loss" not in assumptions and "active_mass_kg_per_l" not in owner:
        # A retained replay starts a #342 solve: its values initialize the units, but HBTA and TOPO are the unpriced
        # active set, never costed accounts. Every other price is the same on both bases.
        masses = owner["mass_kg_per_l"]
        owner = dict(
            owner,
            mass_kg_per_l={
                k: m for k, m in masses.items() if k not in ("HBTA", "TOPO")
            },
            active_mass_kg_per_l={k: masses[k] for k in ("HBTA", "TOPO")},
        )
    model._full_process = {
        "case": case,
        "assumptions": assumptions,
        "candidate": candidate,
        "initial_accounts": owner,
    }
    fs = model.fs
    fs.washed_feed_l_per_raw_l.set_value(v)
    if "mother_liquor_return" in assumptions:
        mother_liquor_return.mix(model, assumptions["mother_liquor_return"], owner)
        loop = fs.mother_liquor_return
        v = fs.washed_feed_l_per_raw_l
        owner["mass_kg_per_l"]["mother_conditioning_HCl31"] = value(
            loop.conditioned["HCl31_kg"]
        )
        owner["prices_usd_kg"] = owner["prices_usd_kg"] | {
            "mother_conditioning_HCl31": owner["prices_usd_kg"]["strip_HCl31"]
        }
    org = fs.stage[1].outlet
    free = "free-carrier" if "free-carrier" in org else "hbta"
    ratio = (
        model._decisions["organic_to_aqueous_ratio"]
        if hasattr(model, "_decisions")
        else model._design["organic_to_aqueous_ratio"]
    )
    # Scrub and strip: one PrOMMiS stage each; the source fraction of entering metal-loaded organic is the only transfer closure.
    metals = {"Li": "lithium", "Na": "sodium"}
    fractions = {
        "scrub": {
            "Li": assumptions["scrub_li_loss"],
            "Na": assumptions["scrub_na_efficiency"],
        },
        "strip": {
            "Li": assumptions.get("strip_li_fraction", flowsheet.strip_li_fraction()),
            "Na": 1.0,
        },
    }
    for name, fraction in fractions.items():
        downstream.fixed_fraction_strip(
            fs,
            name,
            {e: f"{m}-carrier" for e, m in metals.items()},
            {e: f"{m}-cation" for e, m in metals.items()},
            dict.fromkeys(metals, 1.0),
            {
                e: {free: 1.0} | ({"topo": 1.0} if free == "hbta" else {})
                for e in metals
            },
            fraction,
            spectators=("diluent",),
        )
    scrub, strip = fs.scrub, fs.strip
    scrub.loaded_organic = Constraint(
        fs.loaded_properties.component_list,
        rule=lambda b, s: scrub.unit.mscontactor.organic_inlet_state[0].flow_mol_comp[s]
        == v * org[s],
    )
    scrub.waste = Expression(
        tuple(metals), rule=lambda b, e: b.aqueous_out[f"{metals[e]}-cation"]
    )
    source = assumptions["scrub_hcl_mol_l"] * ratio * v / 10
    demand = sum(scrub.waste.values())
    scrub.acid_mol = Var(domain=NonNegativeReals, initialize=owner["scrub_acid_mol"])
    scrub.acid = Constraint(
        ("hydrogen-cation", "chloride-anion"),
        rule=lambda b, s: scrub.unit.mscontactor.aqueous_inlet_state[0].flow_mol_comp[s]
        == b.acid_mol,
    )
    active, other = (
        (source, demand) if candidate["scrub_regime"] == "source" else (demand, source)
    )
    scrub.active_bound = Constraint(expr=scrub.acid_mol == active)
    scrub.other_bound = Constraint(expr=scrub.acid_mol >= other)
    strip.scrubbed_organic = Arc(
        source=scrub.unit.organic_outlet, destination=strip.unit.organic_inlet
    )
    strip_ratio = assumptions.get("strip_aqueous_to_organic", 0.10)
    bounds = assumptions.get("strip_aqueous_to_organic_bounds")
    if bounds is not None:
        if not 0.08 <= bounds[0] <= strip_ratio <= bounds[1] <= 0.10:
            raise ValueError("HBTA strip A/O target must be within [0.08, 0.10]")
        strip.aqueous_to_organic = Var(bounds=bounds, initialize=strip_ratio)
        strip.aqueous_to_organic.fix()
        strip_ratio = model._decisions["strip_aqueous_to_organic"] = (
            strip.aqueous_to_organic
        )
    strip.gross_water_kg = Expression(expr=ratio * v * strip_ratio)
    strip.gross_acid_mol = Expression(
        expr=assumptions.get("strip_hcl_mol_l", flowsheet.STRIP_HCL_MOL_L)
        * strip.gross_water_kg
    )
    strip.acid = Constraint(
        ("hydrogen-cation", "chloride-anion"),
        rule=lambda b, s: strip.unit.mscontactor.aqueous_inlet_state[0].flow_mol_comp[s]
        == b.gross_acid_mol,
    )
    strip.metals = Expression(
        tuple(metals), rule=lambda b, e: b.aqueous_out[f"{metals[e]}-cation"]
    )
    strip.acid_sufficiency = Constraint(expr=strip.aqueous_out["hydrogen-cation"] >= 0)
    downstream.liquor_properties(fs)
    concentrate = chain.aqueous(
        fs, assumptions, strip.gross_water_kg, strip.metals["Li"]
    )
    chain.organic(
        fs, model, assumptions, v, free, strip_ratio, fractions["strip"]["Li"]
    )
    downstream.connect(fs, concentrate, assumptions, candidate, owner)
    if hasattr(fs, "mother_liquor_return"):
        mother_liquor_return.split(fs)
    carbon = fs.carbonation
    carbonation_interval_constraint(
        carbon, assumptions, candidate["carbonation_interval_mol_kg"]
    )
    _initialize_units(fs, owner, assumptions["mother_recycle"])
    if hasattr(getattr(fs, "design", None), "returned_lithium"):
        # The unit initialization has evaluated the strip outlet at the prepared stage profile.
        # Seed the coupled organic inlet with that returned inventory before the NLP closes it.
        for species, equation in (
            ("lithium-carrier", fs.design.returned_lithium),
            ("hbta", fs.design.free_extractant),
            ("topo", fs.design.free_topo),
        ):
            calculate_variable_from_constraint(fs.boundary[species], equation)
    # The original owner supplies all prices and kg/mol conversions, including 31% HCl.
    quantities = {
        "saponification_NaOH": fs.saponification.naoh_mol,
        "scrub_HCl31": scrub.acid_mol,
        "strip_HCl31": strip.fresh_acid_mol,
        "neutralization_NaOH": fs.neutralization.naoh_mol,
        "carbonation_Na2CO3": carbon.carbonate_mol,
        "HBTA": fs.organic_loss.hbta_mol,
        "TOPO": fs.organic_loss.topo_mol,
        "diluent": fs.organic_loss.diluent_mol,
    }
    coefficients = {
        k: flowsheet.MASS_KG_PER_MOL[
            (
                "NaOH"
                if "NaOH" in k
                else "Na2CO3" if "Na2CO3" in k else "HCl31" if "HCl31" in k else k
            )
        ]
        for k in quantities
    }
    if hasattr(fs, "mother_liquor_return"):
        quantities["mother_conditioning_HCl31"] = fs.mother_liquor_return.conditioned[
            "HCl31_kg"
        ]
        coefficients["mother_conditioning_HCl31"] = 1.0
    kg_li_per_mol_product = (
        owner["li_product_kg_m3"] / 1000 / owner["product_li2co3_mol"]
        if owner["product_li2co3_mol"]
        else 2 * LI_MOLAR_MASS_G_MOL / 1000
    )
    costing.attach(
        fs,
        {
            k: (
                (quantities[k] * coefficients[k] if k in quantities else mass)
                * costing.RAW_L_H
                * pyunits.kg,
                owner["prices_usd_kg"][k] * costing.usd_per(pyunits.kg),
            )
            for k, mass in owner["mass_kg_per_l"].items()
        },
        carbon.product_mol * kg_li_per_mol_product * costing.RAW_L_H * pyunits.kg,
        active={
            k: quantities[k] * coefficients[k] * costing.RAW_L_H * pyunits.kg
            for k in owner.get("active_mass_kg_per_l", {})
        },
    )
    assert_units_consistent(model)

    # #457: five serial Li equalities in mol/s would each allow 3600 times the IPOPT
    # constraint tolerance in mol/raw L at 1 raw L/h. Use the mol/h basis on all five.
    for constraint in (
        fs.strip_liquid.mapping["Li"],
        fs.neutralization.feed_expanded.flow_mol_phase_comp_equality[0, "Liq", "Li"],
        fs.neutralization.material_mixing_equations[0, "Li"],
        fs.neutralization.inherent_reaction_constraint[0, "Liq", "Li"],
        fs.neutral_projection.feed_expanded.flow_mol_phase_comp_equality[
            0, "Liq", "Li"
        ],
    ):
        constraint_scaling_transform(constraint, 3600)


def _initialize_units(fs, owner, recycle):
    """Scrub, strip and concentrate follow the live extraction; the recycle loop starts at the owner's closed form."""
    # References repeat unit variables, so the set keeps each once.
    units = tuple(
        getattr(fs, name)
        for name in (
            "scrub",
            "strip",
            "acid_recovery",
            "strip_liquid",
            "neutralization",
            "neutral_projection",
            "concentration",
            "regeneration",
            "regenerated",
            "organic_loss",
            "organic_makeup",
            "unloaded_organic",
            "saponification",
        )
    )
    variables = ComponentSet(
        v
        for b in (*units, fs.carbonation.mixer.concentrate_state[0])
        for v in b.component_data_objects(Var)
        if not v.fixed
    )
    equalities = [
        c
        for b in (*units, fs.carbonation.concentrate_feed.expanded_block)
        for c in b.component_data_objects(Constraint, active=True)
        if c.equality
    ]
    block = create_subsystem_block(equalities, list(variables))
    with TemporarySubsystemManager(to_fix=list(block.input_vars.values())):
        solve_strongly_connected_components(block)
    downstream.initialize(fs, owner, recycle)


def select_units(model, candidate):
    """Set the active bound and measured segment after inspecting a converged extraction."""
    fs, a = model.fs, model._full_process["assumptions"]
    source = (
        a["scrub_hcl_mol_l"]
        * model._decisions["organic_to_aqueous_ratio"]
        * fs.washed_feed_l_per_raw_l
        / 10
    )
    demand = sum(fs.scrub.waste.values())
    active, other = (
        (source, demand) if candidate["scrub_regime"] == "source" else (demand, source)
    )
    fs.scrub.active_bound.set_value(fs.scrub.acid_mol == active)
    fs.scrub.other_bound.set_value(fs.scrub.acid_mol >= other)
    fs.scrub.other_bound.activate()
    interval = candidate["carbonation_interval_mol_kg"]
    dose = a.get("strip_hcl_mol_l", flowsheet.STRIP_HCL_MOL_L)
    omit_lower = (
        "strip_aqueous_to_organic_bounds" in a
        and a["acid_recovery"] == 0
        and interval[0] == dose
    )
    downstream.set_ge_interval(
        fs.carbonation,
        a["carbonation_temperature_k"],
        interval,
        lower_bound=not omit_lower,
    )
    fs.carbonation.interval.activate()


def report(model, result, softening):
    """Retain solved unit inventories, historical comparisons, complete physical R7 and costs."""
    data, fs = model._full_process, model.fs
    washed, v = model._washed, value(fs.washed_feed_l_per_raw_l)
    assumptions = dict(
        data["assumptions"], extractant_mol_l=model._design["extractant_mol_l"]
    )
    if "strip_aqueous_to_organic" in model._design:
        assumptions["strip_aqueous_to_organic"] = model._design[
            "strip_aqueous_to_organic"
        ]
    extracted = extraction(model)
    aqueous_return = (
        mother_liquor_return.report(fs) if hasattr(fs, "mother_liquor_return") else None
    )
    owner = flowsheet.costs_and_balances(
        washed,
        extracted,
        data["case"],
        assumptions,
        nacl_coordinate=lambda raw: connected_nacl(
            raw,
            model,
            washed,
            extracted["loaded_organic"],
            assumptions,
            model._design["organic_to_aqueous_ratio"],
        ),
        mother_return=aqueous_return,
    )
    owner["organic_volume_l_per_raw_l"] = extracted["organic_volume_l_per_raw_l"]
    owner["strip_water_recycle_fraction"] = 1 - owner["fresh_water_kg"] / value(
        fs.strip.gross_water_kg
    )
    if model._design["organic_to_aqueous_ratio"] != 1.0:
        owner["evaporation_alternative_mj_m3_latent_only"] = None
    feed_li = (
        washed["front_end"]["original_raw_elements_mol"]["Li"]
        if washed.get("front_end")
        else washed["raw_mol"]["Li"]
    )
    quantities = {
        "scrub_acid_mol": fs.scrub.acid_mol,
        "gross_strip_acid_mol": fs.strip.gross_acid_mol,
        "fresh_strip_acid_mol": fs.strip.fresh_acid_mol,
        "neutralization_naoh_mol": fs.neutralization.naoh_mol,
        "carbonation_na2co3_mol": fs.carbonation.carbonate_mol,
        "product_li2co3_mol": fs.carbonation.product_mol,
        "ge_li2co3_mol_kg": fs.carbonation.solubility_mol_kg,
        "mother_nacl_mol_kg": fs.carbonation.nacl_mol_kg,
        "reactor_water_kg": fs.carbonation.water_kg,
        "fresh_water_kg": fs.concentration.net_water_kg,
        "raw_li_recovery": 2 * fs.carbonation.product_mol / feed_li,
        "li_product_kg_m3": fs.costing.li_kg_h,
    }
    if (
        "active_mass_kg_per_l" in owner
    ):  # #342: the unpriced active-set loss against the independent owner
        quantities["active_loss_kg_m3"] = fs.costing.active_loss_kg_h
    actual = {k: value(n) for k, n in quantities.items()}
    actual["operating_usd_kg_li"] = (
        value(fs.costing.usd_kg_li)
        if owner["li_product_kg_m3"] and actual["li_product_kg_m3"]
        else None
    )
    prepared = max(
        abs(value(n)) for n in fs.saponification.prepared_difference.values()
    )
    if prepared > 1e-9:
        raise RuntimeError("Prepared organic differs from the extraction inlet")
    differences = {k: actual[k] - owner[k] for k in actual if actual[k] is not None}
    # Convert the #219 post-purge product Li mass allowance to mol/raw L and divide by raw-feed Li.
    recovery_limit = OWNER_COMPARISON_LIMIT / (LI_MOLAR_MASS_G_MOL * feed_li)
    if any(
        not math.isfinite(n)
        or abs(n)
        > (recovery_limit if k == "raw_li_recovery" else OWNER_COMPARISON_LIMIT)
        for k, n in differences.items()
        if k != "operating_usd_kg_li"
    ):
        raise RuntimeError(f"Connected units differ from #176 owner: {differences}")
    reconciliation = costing.reconcile(
        fs.costing,
        owner["cost_usd_m3"],
        owner["operating_usd_kg_li"],
        owner["li_product_kg_m3"],
        owner["mass_kg_per_l"],
    )
    actual["cost_usd_m3"] = reconciliation["usd_m3"]
    balances = dict(result["balance_residuals_mol_per_raw_l"])
    balances.update({"softening/" + k: n for k, n in washed["closure_mol"].items()})
    balances.update({"chain/" + k: n for k, n in owner["closure_mol"].items()})
    retained_li = (
        {
            "lithium returned in organic": value(fs.boundary["lithium-carrier"]) * v,
            "lithium lost with organic": extracted["organic_entrained_lithium"],
        }
        if "organic_entrained_lithium" in extracted
        else {}
    )
    residual_balance = (
        "residual organic lithium" if retained_li else "regeneration lithium discharge"
    )
    balances[residual_balance] = value(
        fs.strip.regeneration_li - fs.strip.organic_out["lithium-carrier"]
    ) + math.fsum(retained_li.values())
    balances["regeneration lithium separator"] = value(
        fs.regenerated.lithium_discharge_state[0].flow_mol_comp["Li"]
        - fs.strip.regeneration_li
    )
    if aqueous_return is not None:
        balances.update(aqueous_return["residuals_mol_per_raw_l"])
    for b in (fs.scrub, fs.strip, fs.carbonation):
        for c in b.component_data_objects(Constraint, active=True):
            if c.equality:
                balances[c.name] = value(c.body - c.lower)
    final = result["outlet_mol_per_nominal_l"]
    raffinate = dict(result["feed"]["model_feed"])
    raffinate.update(
        {records.SPECIES_LABELS[s]: final[s] * v for s in model._chemistry.supports[0]}
    )
    removed_acid = actual["gross_strip_acid_mol"] - actual["fresh_strip_acid_mol"]
    recovered_acid = removed_acid if data["assumptions"]["acid_recovery"] else 0.0
    balances["strip acid recovery"] = removed_acid - recovered_acid
    r7 = references.check_r7(
        result["feed"],
        raffinate,
        balance_residuals_mol_per_raw_l=balances,
        strict_certificate=result["strict_certificate"],
    )
    stripped = {e: value(fs.strip.metals[e]) for e in ("Li", "Na")}
    physical = {
        "washed extraction feed": result["feed"]["physical_feed"],
        "physical raffinate": r7["physical_raffinate"],
        "saponified organic input": {s: n * v for s, n in model._feeds[1].items()},
        "loaded organic": {s: final[s] * v for s in model._chemistry.supports[1]},
        "saponification NaOH input": {
            "Na": value(fs.saponification.naoh_mol),
            "OH": value(fs.saponification.naoh_mol),
        },
        "scrub HCl input": {
            "H": actual["scrub_acid_mol"],
            "Cl": actual["scrub_acid_mol"],
        },
        "scrub waste": {e: value(fs.scrub.waste[e]) for e in ("Li", "Na")}
        | {
            "H": (
                0.0
                if data["candidate"]["scrub_regime"] == "stoichiometric"
                else actual["scrub_acid_mol"]
                - sum(value(n) for n in fs.scrub.waste.values())
            ),
            "Cl": actual["scrub_acid_mol"],
        },
        "gross strip HCl input": {
            "H": actual["gross_strip_acid_mol"],
            "Cl": actual["gross_strip_acid_mol"],
        },
        "recovered unused strip HCl": {"H": recovered_acid, "Cl": recovered_acid},
        "strip liquor before neutralization": stripped
        | {
            "H": actual["neutralization_naoh_mol"],
            "Cl": actual["fresh_strip_acid_mol"],
        },
        "regeneration lithium discharge": {"Li": value(fs.strip.regeneration_li)},
        "neutralization NaOH input": {
            "Na": actual["neutralization_naoh_mol"],
            "OH": actual["neutralization_naoh_mol"],
        },
        "carbonation soda ash input": {
            "Na": 2 * value(fs.carbonation.carbonate_mol),
            "C": value(fs.carbonation.carbonate_mol),
        },
        "Li2CO3 product": {
            "Li": 2 * actual["product_li2co3_mol"],
            "C": actual["product_li2co3_mol"],
        },
        "mother liquor": {
            e: value(fs.carbonation.mother[e]) for e in ("Li", "Na", "C")
        },
        "mother liquor purge": {
            e: value(fs.carbonation.purge[e]) for e in ("Li", "Na", "C")
        },
        "mother liquor recycle": {
            e: value(fs.carbonation.mother[e] - fs.carbonation.purge[e])
            for e in ("Li", "Na", "C")
        },
    }
    if washed.get("front_end"):
        physical.update(
            {
                "front-end " + ("internal wash return" if s == "wash" else s): a
                for s, a in washed["front_end"]["streams_mol_per_raw_l"].items()
            }
        )
    for name, state in (
        ("organic loss", fs.organic_loss.loss_state[0]),
        ("regenerated organic", fs.organic_loss.retained_state[0]),
    ):
        physical[name] = {
            j: value(state.flow_mol_comp[j])
            for j in ("BTA_total", "TOPO_total", "diluent")
        }
    if retained_li:
        physical["organic loss"]["Li"] = retained_li["lithium lost with organic"]
        physical["regenerated organic"]["Li"] = retained_li[
            "lithium returned in organic"
        ]
    physical["organic make-up input"] = {
        label: value(fs.organic_makeup.fresh_state[0].flow_mol_comp[j])
        for label, j in (
            ("HBTA", "BTA_total"),
            ("TOPO", "TOPO_total"),
            ("diluent", "diluent"),
        )
    }
    margins = {
        c.name: {
            "lower": None if c.lower is None else value(c.body - c.lower),
            "upper": None if c.upper is None else value(c.upper - c.body),
        }
        for c in model.component_data_objects(Constraint, active=True)
        if not c.equality
    }
    if any(
        n < -1e-9 for row in margins.values() for n in row.values() if n is not None
    ):
        raise RuntimeError(f"Full-process inequality violated: {margins}")
    if aqueous_return is not None:
        result["mother_liquor_return"] = aqueous_return
        physical["external mother liquor bleed"] = aqueous_return["bleed_mol_per_raw_l"]
        physical["mother liquor returned"] = aqueous_return["returned_mol_per_raw_l"]
        physical["mother liquor conditioning CO2"] = {
            "C": aqueous_return["conditioned"]["CO2"]
        }
        returned = aqueous_return["returned_mol_per_raw_l"]
        conditioned = aqueous_return["conditioned"]
        bleed = aqueous_return["bleed_mol_per_raw_l"]
        balances["external chloride after softening"] = math.fsum(
            (
                model._physical_feed["physical_feed"]["Cl"],
                actual["scrub_acid_mol"],
                actual["fresh_strip_acid_mol"],
                2 * returned["C"],
                -r7["physical_raffinate"]["Cl"],
                -physical["scrub waste"]["Cl"],
                -bleed["Cl"],
            )
        )
        balances["recycle water after conditioning"] = (
            v - washed["feed_l"]
        ) / brines.WATER_MOLAR_MASS_KG - conditioned["H2O"]
        r7 = references.check_r7(
            result["feed"],
            raffinate,
            balance_residuals_mol_per_raw_l=balances,
            strict_certificate=result["strict_certificate"],
        )
        result["uncharacterized_recycle"] = {
            "scrub waste": {
                "li_upper_bound_mol_per_raw_l": physical["scrub waste"]["Li"],
                "missing": "scrub dilution water and the resulting conditioning volume",
            },
            "regeneration discharge": {
                "li_upper_bound_mol_per_raw_l": value(fs.strip.regeneration_li),
                "missing": "counterions, carrier water and conditioning reagent inventory",
            },
        }
    return complete_reporting(
        result
        | {
            "softening": softening,
            "case": data["case"],
            "full_process": owner | actual,
            "R7": r7,
            "balance_residuals_mol_per_raw_l": balances,
            "physical_streams_mol_per_raw_l": physical,
            "softening_streams_elements_mol_per_raw_l": {
                k: washed[k]
                for k in ("raw_mol", "cake_mol", "solid_mol", "wash_elements_mol")
            }
            | {"reagent_elements_mol": washed["reagent_elements_mol"]},
            "owner_differences": differences,
            "prepared_organic_max_difference_mol_per_raw_l": prepared,
            "idaes_costing_reconciliation": reconciliation,
            "inequality_margins": margins,
            "unit_enumeration_selected": data["candidate"],
            "labels": [s for s in result["labels"] if "no full-process solve" not in s]
            + [
                "fixed-design model prediction and numerical replay; no physical validation",
                "#176 accounts per kg Li in Li2CO3; capital, labour and energy excluded",
                "scrub/strip laboratory fractions transferred to produced water; no finished-product assay",
                "downstream H/O and regeneration counterions unquantified in the #176 accounting",
                data["assumptions"].get("organic_loss_provenance")
                or data["assumptions"]["entrainment_provenance"],
            ],
        },
        fs,
    )


def nominal_water(fs, washed, gross_strip_water, concentrate_water, *, gap_bound=0.0):
    """Named #447 water accounts on the existing HBTA strip and Ge basis."""
    mw = brines.WATER_MOLAR_MASS_KG
    carrier = flowsheet.HCL_MOLAR_MASS_KG * (1 / flowsheet.hcl_stock_fraction() - 1)
    g, c = gross_strip_water, concentrate_water
    s = carrier * value(fs.strip.fresh_acid_mol)
    projected = c > g and c - g <= gap_bound
    if projected:
        c = g
    r, bleed, trim = g - c, max(0.0, s - c), max(0.0, c - s)
    if not 0 <= c <= g or not 0 <= bleed <= r or not 0 <= s < g:
        raise ValueError(
            f"Incompatible nominal HBTA strip water preparation: gross={g}, Ge={c}, HCl water={s}, "
            f"gap bound={gap_bound} kg/raw L"
        )
    neutral = value(fs.neutralization.excluded_reaction_water_kg)
    regeneration = value(fs.saponification.naoh_mol) * mw
    softening_stock = carrier * washed["reagent_mol"]["HCl"]
    scrub_stock = carrier * value(fs.scrub.acid_mol)
    numeric = {
        "softening HCl stock carrier": softening_stock,
        "washed-liquor to raffinate water projection": softening_stock,
        "scrub HCl stock carrier": scrub_stock,
        "fresh strip HCl stock carrier": s,
        "fresh strip trim": trim,
        "concentrator feed water": g,
        "concentrator separated water": r,
        "concentrator clean-water bleed": bleed,
        "concentrator water returned to strip": r - bleed,
        "concentrate water to Ge": c,
        "neutralization mixer outlet water": g + neutral,
        "neutralization reaction-water projection": neutral,
        "saponification reaction-water discharge": regeneration,
        "strip mixer water closure": math.fsum((s, trim, r, -bleed, -g)),
        "external downstream water closure": math.fsum(
            (
                softening_stock,
                scrub_stock,
                s,
                trim,
                neutral,
                regeneration,
                -softening_stock,
                -scrub_stock,
                -neutral,
                -regeneration,
                -bleed,
                -c,
            )
        ),
    }
    if projected:
        numeric["concentrator feed minus concentrate before projection"] = (
            gross_strip_water - concentrate_water
        )
        numeric["concentrator gap numerical bound"] = gap_bound
    symbolic = {
        "fresh scrub dilution": (0.0, "U_s"),
        "scrub wastewater": (scrub_stock, "U_s"),
        "recovered strip acid carrier to mixer": (0.0, "U_r"),
        "recovered strip acid carrier from separator": (0.0, "U_r"),
        "gross strip aqueous water to recovery": (g, "U_r"),
    }
    return {
        k: {"kg_per_raw_l": n, "basis": "#447 nominal water accounting"}
        for k, n in numeric.items()
    } | {
        k: {
            "kg_per_raw_l": None,
            "known_kg_per_raw_l": n,
            "unquantified_kg_per_raw_l": symbol,
            "basis": "#447 unmeasured, nonnegative water amount",
        }
        for k, (n, symbol) in symbolic.items()
    }


def complete_reporting(result, fs):
    """Expose known #176 inventories and washed-feed coverage without rerunning chemistry."""
    washed = result["softening"][result["case"]]
    unit = result["full_process"]
    physical = result["physical_streams_mol_per_raw_l"]
    purge_cl = unit["fresh_strip_acid_mol"]
    mother_cl = unit["mother_nacl_mol_kg"] * unit["reactor_water_kg"]
    for name, chloride in (
        ("mother liquor", mother_cl),
        ("mother liquor purge", purge_cl),
        ("mother liquor recycle", mother_cl - purge_cl),
    ):
        physical[name]["Cl"] = chloride
    negative = {
        f"{name}/{s}": n
        for name, stream in physical.items()
        for s, n in stream.items()
        if not math.isfinite(n) or n < 0
    }
    if negative:
        raise RuntimeError(
            f"Negative or nonfinite physical full-process stream: {negative}"
        )
    net, circulating = unit["fresh_water_kg"], unit["reactor_water_kg"]
    result["water_streams"] = {
        "raw feed": {
            "kg_per_raw_l": result["softening"]["raw_feed"]["water_kg"],
            "basis": "source-record water mass; brines.raw_feed",
        },
        "softening wash input": {
            "kg_per_raw_l": washed["wash_water_mol"] * brines.WATER_MOLAR_MASS_KG,
            "basis": "pure wash water, 1 kg/L; Reaktoro H2O molar mass",
        },
        "washed aqueous H2O species": {
            "kg_per_raw_l": washed["aqueous_species_mol"]["H2O"]
            * brines.WATER_MOLAR_MASS_KG,
            "basis": "retained Reaktoro H2O species; distinct from analytical water component",
        },
        "gross strip liquor": {
            "kg_per_raw_l": value(
                fs.strip_liquid.properties_out[0].flow_mol_phase_comp["Liq", "H2O"]
            )
            * 3600
            * brines.WATER_MOLAR_MASS_KG,
            "basis": "#176 nominal strip-water accounting, 1 kg/L at organic/aqueous 10",
        },
        "strip water recycle": {
            "kg_per_raw_l": value(
                fs.concentration.recycle_state[0].flow_mol_comp["H2O"]
            )
            * brines.WATER_MOLAR_MASS_KG,
            "basis": "#176 declared strip-water recycle; energy/equipment excluded",
        },
        "net water to carbonation": {
            "kg_per_raw_l": net,
            "basis": "#176 fresh stripped Li / concentrate_li_mol_kg; not external fresh water",
        },
        "mother liquor": {
            "kg_per_raw_l": circulating,
            "basis": "#176 reactor water at fixed mother recycle",
        },
        "mother liquor purge": {
            "kg_per_raw_l": net,
            "basis": "#176 net water leaves in the mother purge",
        },
        "mother liquor recycle": {
            "kg_per_raw_l": circulating - net,
            "basis": "#176 circulating water minus purge water",
        },
        **{
            name: {"kg_per_raw_l": None, "basis": "not quantified by #176"}
            for name in (
                "scrub water",
                "recovered acid water",
                "cake liquid H2O species",
                "solid H2O species",
                "Li2CO3 product moisture",
                "organic water",
            )
        },
    }
    if result["water_streams"]["gross strip liquor"]["kg_per_raw_l"] > 0.0:
        result["water_streams"].update(
            nominal_water(
                fs,
                washed,
                result["water_streams"]["gross strip liquor"]["kg_per_raw_l"],
                net,
                gap_bound=result["solve"].get(
                    "strip_water_gap_bound_kg_per_raw_l", 0.0
                ),
            )
        )
    feed = result["feed"]
    if washed.get("front_end"):
        result["water_streams"]["front-end fresh water"] = dict(
            kg_per_raw_l=washed["front_end"]["water_input_kg_per_raw_l"],
            basis="H-D0 V1/V7; wash return is internal",
        )
    record = records.SOLVENTS[result.get("solvent", "C")]
    mixed_feed = "mother_liquor_return" in result
    volume = feed["washed_feed_l_per_raw_l"] if mixed_feed else washed["feed_l"]
    inlet_basis = (
        "mixed extraction feed after mother-purge return"
        if mixed_feed
        else "softened, washed aqueous inlet"
    )
    molarity_basis = "mol/L mixed aqueous" if mixed_feed else "mol/L washed aqueous"
    result["calibration_coverage"] = [
        {
            "quantity": quantity,
            "value": number,
            "unit": unit_name,
            "basis": (
                inlet_basis
                if quantity != "process pH"
                else "softened, washed aqueous inlet"
            )
            + "; "
            + basis,
            "source_owner": (
                record["feed_owner"]
                if quantity != "process pH"
                else "analyses.prototypes.hbta_topo_ms2_prototype.scripts.countercurrent:acidity_at_ph"
            ),
            "chemistry_owner": record["chemistry_owner"],
            "coverage_owner": record["domain_owner"],
            "comparison": "physical calibration coverage unverified",
            "comparison_basis": "existing owner establishes no quantitative physical-calibration bounds for arbitrary brine; "
            "source-data extrema and numerical interpolation are separate",
        }
        for quantity, number, unit_name, basis in (
            (
                "projected Li",
                feed["model_feed"]["Li"] / volume,
                molarity_basis,
                "owner model projection divided by the inlet volume",
            ),
            (
                "projected Na",
                feed["model_feed"]["Na"] / volume,
                molarity_basis,
                "owner model projection divided by the inlet volume",
            ),
            (
                "process pH",
                washed["pH"],
                "dimensionless",
                "retained softening pH conditioning; separate from raw measured pH",
            ),
        )
    ]
    return result
