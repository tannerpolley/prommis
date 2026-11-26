"""Raiguel strip plus the declared ideal cleanup and Ge carbonation (Lithium #214)."""

import csv
import math

from pyomo.environ import (
    Block,
    Constraint,
    Expression,
    Expr_if,
    Var,
    units as pyunits,
    value,
)
from pyomo.common.collections import ComponentSet
from pyomo.contrib.incidence_analysis import solve_strongly_connected_components
from pyomo.util.subsystems import TemporarySubsystemManager, create_subsystem_block
from pyomo.util.calc_var_value import calculate_variable_from_constraint

from prommis.lithium_extraction.process import (
    waters as brines,
    costing,
    strip_and_carbonation as downstream,
    flowsheet,
    mother_liquor_return,
    feeds as records,
    retained_references as references,
)
from prommis.lithium_extraction.extraction import d2ehdtpa_buphen_stage as d2_stage


def strip_preparation(acid, volume_l):
    """Prepared dilute HCl at the #447 nominal density of 1 kg/L."""
    active = acid * flowsheet.HCL_MOLAR_MASS_KG
    stock = active / flowsheet.hcl_stock_fraction()
    dilution = volume_l - stock
    if value(dilution) < 0:
        raise ValueError("HCl stock mass exceeds prepared strip solution mass")
    return {
        "feed_volume_l": volume_l,
        "outlet_volume_l": volume_l,
        "acid_mol": acid,
        "hcl_mol_l": acid / volume_l,
        "active_hcl_kg": active,
        "hcl_stock_kg": stock,
        "stock_water_kg": stock - active,
        "fresh_dilution_water_kg": dilution,
        "strip_liquor_water_kg": dilution + stock - active,
    }


def cleanup(stripped, acid, assumptions, *, preparation=None):
    """Exact-stoichiometry complete Mg/Ca/Sr removal; Li carbonate co-precipitation is excluded by assumption."""
    mg, ca, sr = (stripped[e] for e in ("Mg", "Ca", "Sr"))
    free_h = acid - sum(d2_stage.rc.ION[e][2] * n for e, n in stripped.items())
    naoh, soda = free_h + 2 * mg, ca + sr
    before = {"Li": stripped["Li"], "Na": stripped["Na"] + naoh + 2 * soda, "Cl": acid}
    water = (
        stripped["Li"] / assumptions["concentrate_li_mol_kg"]
        if preparation is None
        else preparation["strip_liquor_water_kg"] + free_h * brines.WATER_MOLAR_MASS_KG
    )
    mass = water + sum(n * d2_stage.owner.MASSES[e] for e, n in before.items())
    atomic = brines.MOLAR_MASS
    dry = (
        mg * (atomic["Mg"] + 2 * (atomic["O"] + atomic["H"]))
        + ca * (atomic["Ca"] + atomic["C"] + 3 * atomic["O"])
        + sr * (atomic["Sr"] + atomic["C"] + 3 * atomic["O"])
    ) / 1000
    washes = assumptions["wash_volumes"]
    held, fraction, loss = downstream.held_liquid_loss(
        dry, mass, assumptions["moisture"], washes
    )
    cake = {e: loss * n for e, n in before.items()}
    clean = {e: (1 - loss) * n for e, n in before.items()}
    solid = {"Mg": mg, "Ca": ca, "Sr": sr, "C": soda}
    incoming = dict(stripped, Cl=acid, C=soda)
    incoming["Na"] += naoh + 2 * soda
    closure = {
        e: n - clean.get(e, 0.0) - cake.get(e, 0.0) - solid.get(e, 0.0)
        for e, n in incoming.items()
    }
    charge = clean["Li"] + clean["Na"] - clean["Cl"]
    if max(abs(value(n)) for n in (*closure.values(), charge)) > 1e-12:
        raise RuntimeError(
            f"Ideal cleanup closure/charge exceeds 1e-12 mol/raw L: {closure}; {charge}"
        )
    concentrated = (
        water
        if preparation is None
        else clean["Li"] / assumptions["concentrate_li_mol_kg"]
    )
    water_streams = {}
    if preparation is not None:
        # Separate screening densities (cleanup liquor and pure wash water), both 1 kg/L.
        held_l = held / 1.0
        wash = washes * held_l * 1.0
        cake_water = loss * water + (fraction - loss) * water
        wash_return = fraction * water + wash - cake_water
        filtrate = (1 - fraction) * water + wash_return
        excess = filtrate - concentrated

        def positive(x):
            return (
                Expr_if(IF=x > 0.0, THEN=x, ELSE=0.0)
                if hasattr(x, "is_expression_type")
                else max(x, 0.0)
            )

        water_streams = {
            "post_precipitation_water_kg": water,
            "neutralization_reaction_water_kg": free_h * brines.WATER_MOLAR_MASS_KG,
            "post_precipitation_liquor_l": mass / 1.0,
            "held_liquor_l": held_l,
            "held_liquor_fraction": fraction,
            "fresh_wash_water_kg": wash,
            "final_cake_water_kg": cake_water,
            "recovered_wash_water_kg": wash_return,
            "recovered_wash_mol": {e: (fraction - loss) * n for e, n in before.items()},
            "combined_filtrate_water_kg": filtrate,
            "concentrator_vapor_kg": positive(excess),
            "concentrator_trim_water_kg": positive(-excess),
        }
    return {
        "naoh_mol": naoh,
        "na2co3_mol": soda,
        "unconsumed_hcl_mol": free_h,
        "dry_sludge_kg": dry,
        "wet_sludge_kg": dry / (1 - assumptions["moisture"]),
        "concentrate_water_kg": concentrated,
        "liquor_kg": mass,
        "held_liquid_kg": held,
        "solute_loss_fraction": loss,
        "cake_mol": cake,
        "solid_mol": solid,
        "cleaned_mol": clean,
        "closure_mol": closure,
        "charge_mol": charge,
        "li_coprecipitation_bound_fraction": 2 * soda / stripped["Li"],
        **water_streams,
    }


ACTIVE = {
    "BuPhen": "BuPhen_loss_kg_m3",
    "D2EHDTPA": "D2EHDTPA_loss_kg_m3",
}  # #339 decision 1 active set
ACCOUNT_RELATIVE_LIMIT = (
    1e-9  # #339 check 8: connected accounts against the independent recomputation
)


def accounts(strip, assumptions, feed_li, *, aqueous_return=None, nacl_coordinate=None):
    """Operating cost outside the active set (C_r) on product Li, and the unpriced BuPhen and D2EHDTPA losses."""
    stripped = {
        e: strip[f"strip_{e}_mol_L_feed"] for e in ("Li", "Na", "Mg", "Ca", "Sr")
    }
    acid = strip["strip_HCl_mol_L_feed"]
    preparation = (
        strip_preparation(acid, strip["strip_outlet_volume_l_per_raw_l"])
        if "strip_aqueous_to_organic" in assumptions
        else None
    )
    cleaned = cleanup(stripped, acid, assumptions, preparation=preparation)
    carbon = downstream.accounts(
        dict(cleaned["cleaned_mol"], H=0.0, C=0.0),
        assumptions,
        nacl_coordinate=nacl_coordinate,
    )
    case = strip["case"]
    octanol = float(
        flowsheet._row(
            "data/reference/tea/extractant_price_register.csv",
            "component_id",
            "1_octanol",
        )["price_usd_kg"]
    )
    masses = {
        "regeneration_NaOH": strip["NaOH_mol_L_feed"]
        * flowsheet.MASS_KG_PER_MOL["NaOH"],
        "strip_HCl31": (
            preparation["hcl_stock_kg"]
            if preparation
            else acid * flowsheet.MASS_KG_PER_MOL["HCl31"]
        ),
        "octanol_dissolution": strip["octanol_dissolved_kg_m3"] / 1000,
        "octanol_entrainment": strip["octanol_entrained_kg_m3"] / 1000,
        "diluent": strip["dodecane_loss_kg_m3"] / 1000,
        "cleanup_NaOH": cleaned["naoh_mol"] * flowsheet.MASS_KG_PER_MOL["NaOH"],
        "cleanup_Na2CO3": cleaned["na2co3_mol"] * flowsheet.MASS_KG_PER_MOL["Na2CO3"],
        "sludge": cleaned["wet_sludge_kg"],
        "carbonation_Na2CO3": carbon["carbonation_na2co3_mol"]
        * flowsheet.MASS_KG_PER_MOL["Na2CO3"],
    }
    if aqueous_return is not None:
        masses["mother_conditioning_HCl31"] = aqueous_return["conditioned"]["HCl31_kg"]
    prices = {
        k: (
            0.285
            if "NaOH" in k
            else (
                0.100
                if "HCl" in k
                else (
                    (0.35 if case == "base" else 0.20)
                    if "Na2CO3" in k
                    else (
                        (0.05 if case == "base" else 0.03)
                        if k == "sludge"
                        else (
                            assumptions["diluent_usd_kg"] if k == "diluent" else octanol
                        )
                    )
                )
            )
        )
        for k in masses
    }
    costs = {k: 1000 * m * prices[k] for k, m in masses.items()}
    li_kg_m3 = 2 * carbon["product_li2co3_mol"] * d2_stage.owner.MASSES["Li"] * 1000
    total = sum(costs.values())
    active = {k: strip[key] / 1000 for k, key in ACTIVE.items()}
    return (
        carbon
        | ({"strip_preparation": preparation} if preparation else {})
        | {
            "case": case,
            "assumptions": assumptions,
            "cleanup": cleaned,
            "stripped_mol": stripped,
            "mass_kg_per_l": masses,
            "prices_usd_kg": prices,
            "price_source": "Lithium #214 price table; source-period nominal USD; EIA diluent analogue unrounded",
            "cost_usd_m3": costs,
            "cost_usd_kg_li": {
                k: n / li_kg_m3 if li_kg_m3 else None for k, n in costs.items()
            },
            "operating_usd_kg_li": total / li_kg_m3 if li_kg_m3 else None,
            "li_product_kg_m3": li_kg_m3,
            "raw_li_recovery": 2 * carbon["product_li2co3_mol"] / feed_li,
            **flowsheet.rule_accounts(case, active, total / 1000, li_kg_m3 / 1000),
            "claim": "conditional on ideal complete cleanup and no Li2CO3 co-precipitation; no purity claim; "
            "BuPhen and D2EHDTPA unpriced; capital, labour and energy excluded",
        }
    )


def carbonation_interval(nacl):
    """The measured Ge 2025 NaCl interval holding `nacl` (enumerated outside the solver), or None outside the data."""
    with downstream.carbonation.GE_TABLE.open() as stream:
        knots = [
            float(r["nacl_mol_kg"])
            for r in csv.DictReader(stream)
            if r["series"] == "NaCl"
        ]
    return next(
        ([lo, hi] for lo, hi in zip(knots, knots[1:]) if lo <= nacl <= hi), None
    )


def set_interval(model, interval):
    """Move the connected carbonation to another measured interval (its bounds and linear interpolation)."""
    carbon, (lo, hi) = model.fs.carbonation, interval
    s0, s1 = (
        downstream.carbonation.empirical_solubility(
            model._d2_full_process["carbonation_temperature_k"], salt
        )
        for salt in (lo, hi)
    )
    carbon.interval.set_value((lo, carbon.nacl_mol_kg, hi))
    carbon.interpolation.set_value(
        carbon.solubility_mol_kg
        == s0 + (s1 - s0) * (carbon.nacl_mol_kg - lo) / (hi - lo)
    )
    model._carbonation_interval = list(interval)


def connect_strip(model, case, recycle_assumptions=None, *, ratio_bounds=None):
    """Strip the loaded organic and, when requested, return the residual complexes."""
    fs, rc = model.fs, d2_stage.rc
    n = len(fs.stages)
    org = fs.stage[1].outlet
    complexes = {e: s for e, s in dict(rc.COMPLEX, Na="naa").items() if s in org}
    ions = {e: rc.ION[e][0] for e in complexes}
    charges = {e: rc.ION[e][2] for e in complexes}
    released = {
        e: {
            "ha": rc.MOIETIES["A"][s],
            **({"buphen": rc.MOIETIES["B"][s]} if e == "Li" else {}),
        }
        for e, s in complexes.items()
    }
    fractions = d2_stage.owner.BASIS["strip_fractions"][case] | (
        recycle_assumptions or {}
    ).get("strip_fractions", {})
    if any(not 0 <= f <= 1 for f in fractions.values()):
        raise ValueError("Strip transfer fractions must be between zero and one")
    spectators = ("diluent",) if recycle_assumptions is not None else ()
    strip = downstream.fixed_fraction_strip(
        fs,
        "strip",
        complexes,
        ions,
        charges,
        released,
        fractions,
        spectators=spectators,
    )
    raffinate_loss = getattr(
        fs,
        "raffinate_entrained_organic",
        dict.fromkeys(model._chemistry.supports[1], 0.0),
    )
    strip_feed = {
        s: org[s] - (raffinate_loss[s] if n == 1 else 0.0)
        for s in model._chemistry.supports[1]
    }
    inlet = strip.unit.mscontactor.organic_inlet_state[0].flow_mol_comp
    strip.loaded = Constraint(
        fs.loaded_properties.component_list,
        rule=lambda b, s: inlet[s] == strip_feed.get(s, 0.0),
    )
    strip.acid_mol = Expression(
        expr=sum(charges[e] * strip_feed[s] for e, s in complexes.items())
    )
    strip.acid = Constraint(
        ("hydrogen-cation", "chloride-anion"),
        rule=lambda b, s: strip.unit.mscontactor.aqueous_inlet_state[0].flow_mol_comp[s]
        == strip.acid_mol,
    )
    strip.metals = Expression(
        tuple(complexes), rule=lambda b, e: strip.aqueous_out[ions[e]]
    )
    # Strip precedes the cleanup expressions, so initialize its linear acid exchange first.
    variables = ComponentSet(
        v for v in strip.component_data_objects(Var) if not v.fixed
    )
    equations = [
        c for c in strip.component_data_objects(Constraint, active=True) if c.equality
    ]
    block = create_subsystem_block(equations, list(variables))
    with TemporarySubsystemManager(to_fix=list(block.input_vars.values())):
        solve_strongly_connected_components(block)
    if recycle_assumptions is None:
        return
    fs.organic_recycle = recycle = Block()
    entrainment = recycle_assumptions["entrainment_l_per_l"]
    raffinate_l = entrainment * fs.aqueous_volume
    volume = fs.design.organic_volume - raffinate_l
    recycle.loaded_volume_l = Expression(expr=volume)
    ratio_name = "strip_aqueous_to_organic"
    nominal = ratio_name in recycle_assumptions
    ratio = recycle_assumptions.get(ratio_name)
    if ratio_bounds is not None:
        if ratio is None or not 0 < ratio_bounds[0] <= ratio <= ratio_bounds[1]:
            raise ValueError("Initial strip A/O must lie within its positive bounds")
        recycle.strip_aqueous_to_organic = Var(bounds=ratio_bounds, initialize=ratio)
        recycle.strip_aqueous_to_organic.fix()
        ratio = model._decisions[ratio_name] = recycle.strip_aqueous_to_organic
    outlet_l = (
        volume * ratio
        if nominal
        else d2_stage.owner.historical_strip_volume(
            recycle_assumptions, strip.metals["Li"]
        )
    )
    recycle.strip_entrained_volume_l = Expression(expr=entrainment * outlet_l)
    strip_l = recycle.strip_entrained_volume_l
    if nominal:
        model._d2_full_process = recycle_assumptions
        prepared = strip_preparation(strip.acid_mol, outlet_l)
        strip.preparation = Expression(tuple(prepared), initialize=prepared)
        strip.dilution_nonnegative = Constraint(
            expr=strip.preparation["fresh_dilution_water_kg"] >= 0.0
        )
        recycle.entrainment_feasible = Constraint(expr=strip_l <= volume)
    recycle.strip_loss = Expression(
        strip.organic_out.index_set(),
        rule=lambda b, s: strip.organic_out[s] * strip_l / volume,
    )
    recycle.returned = Expression(
        strip.organic_out.index_set(),
        rule=lambda b, s: strip.organic_out[s] - b.strip_loss[s],
    )
    recycle.regeneration_naoh = Expression(
        expr=fs.boundary["naa"] - recycle.returned["naa"]
    )
    if nominal:
        recycle.fresh_acid_form_sites = Expression(
            expr=sum(
                rc.MOIETIES["A"].get(s, 0.0) * fs.boundary[s]
                for s in model._chemistry.supports[1]
            )
            - recycle.returned["ha"]
            - sum(charges[e] * recycle.returned[s] for e, s in complexes.items())
        )
        recycle.regeneration_water_kg = Expression(
            expr=recycle.regeneration_naoh * brines.WATER_MOLAR_MASS_KG
        )
    recirculated = [s for e, s in complexes.items() if e != "Na" and fractions[e] < 1]
    for s in recirculated:
        fs.boundary[s].unfix()
        fs.boundary[s].set_value(value(recycle.returned[s]))
    recycle.metals = Constraint(
        recirculated, rule=lambda b, s: fs.boundary[s] == b.returned[s]
    )
    for s in ("naa", "buphen"):
        calculate_variable_from_constraint(fs.boundary[s], fs.design.organic_inlet[s])


def connect_units(model, initial, assumptions, interval):
    """Connect ideal cleanup and common carbonation in the selected measured Ge NaCl interval."""
    if not hasattr(model.fs, "strip"):
        connect_strip(model, initial["case"])
    fs, strip = model.fs, model.fs.strip
    if model._aqueous_return:
        mother_liquor_return.mix(
            model,
            assumptions["mother_liquor_return"],
            initial,
            stages_per_nominal_litre=False,
        )
        if model._raiguel_recipe:
            ratio = model._decisions["O_A"]
            ratio.unfix()  # S is the independent control. O/A also follows the mixed Li in fixed replays.
            fs.design.site_concentration.activate()
            calculate_variable_from_constraint(ratio, fs.design.site_concentration)
    fs.cleanup = Block()
    amounts = d2_stage.strip_accounts(
        {f"strip_{e}_mol_L_feed": n for e, n in strip.metals.items()}, model._solvent
    )
    preparation = getattr(strip, "preparation", None)
    cleaned = cleanup(
        {e: amounts[f"strip_{e}_mol_L_feed"] for e in ("Li", "Na", "Mg", "Ca", "Sr")},
        strip.acid_mol,
        assumptions,
        preparation=preparation,
    )
    for key in ("naoh_mol", "na2co3_mol", "wet_sludge_kg", "solute_loss_fraction"):
        fs.cleanup.add_component(key, Expression(expr=cleaned[key]))
    fs.cleanup.liquor = Expression(
        ("Li", "Na", "Cl"), rule=lambda b, e: cleaned["cleaned_mol"][e]
    )
    if preparation is not None:
        water = {
            k: n
            for k, n in cleaned.items()
            if k.endswith("_water_kg") or k == "concentrator_vapor_kg"
        }
        fs.cleanup.water = Expression(tuple(water), initialize=water)
    model._d2_full_process = assumptions
    downstream.connect(
        fs,
        dict(fs.cleanup.liquor.items(), H=0.0, C=0.0),
        assumptions,
        {"carbonation_interval_mol_kg": interval},
        initial,
    )
    if model._aqueous_return:
        mother_liquor_return.split(fs)
    model._carbonation_interval = list(interval)
    for e, n in cleaned["cleaned_mol"].items():
        fs.carbonation.mixer.concentrate_state[0].flow_mol_comp[e].set_value(value(n))
    fs.carbonation.mixer.concentrate_state[0].flow_mol_comp["H2O"].set_value(
        value(cleaned["cleaned_mol"]["Li"])
        / assumptions["concentrate_li_mol_kg"]
        / brines.WATER_MOLAR_MASS_KG
    )
    fs.carbonation.mixer.concentrate_state[0].flow_mol_comp["C"].set_value(0.0)
    downstream.initialize(fs, initial, assumptions["mother_recycle"])
    # Design-dependent accounts (#342): the shared entrainment on raffinate and strip liquor, octanol, BuPhen,
    # D2EHDTPA and dodecane losses and the saponification NaOH follow the solved design and stage outlets.
    d = model._decisions if hasattr(model, "_decisions") else model._design
    ratio = fs.design.organic_volume if hasattr(fs, "design") else d["O_A"]
    n = len(fs.stages)
    recycle = getattr(fs, "organic_recycle", None)
    buphen = d["B"] if recycle is not None else fs.boundary["buphen"] / ratio
    naoh = recycle.regeneration_naoh if recycle is not None else fs.boundary["naa"]
    free_b = fs.stage[n].outlet["buphen"] / ratio
    if preparation is not None:
        lost = d2_stage.owner.separated_losses(
            getattr(fs, "raffinate_entrained_organic", {}),
            recycle.strip_loss,
            free_b,
            assumptions["entrainment_l_per_l"] * fs.aqueous_volume
            + recycle.strip_entrained_volume_l,
            aqueous_volume=fs.aqueous_volume,
            modifier_fractions=assumptions.get(
                "modifier_volume_fractions", d2_stage.owner.HISTORICAL_OCTANOL_FRACTIONS
            ),
        )
    else:
        old_volume = d2_stage.owner.historical_strip_volume(
            assumptions, strip.metals["Li"]
        )
        lost = d2_stage.owner.losses_kg_per_l(
            d2_stage.owner.entrained_l(assumptions, old_volume),
            d["S"],
            ratio,
            buphen,
            0.0,
            free_b,
            fs.boundary["diluent"] / ratio,
            site_basis_li_mol_l=model._design["site_basis_li_mol_l"],
        )
    fs.losses = Expression(tuple(lost), rule=lambda b, k: lost[k])
    live = {
        "regeneration_NaOH": naoh * flowsheet.MASS_KG_PER_MOL["NaOH"],
        "strip_HCl31": (
            preparation["hcl_stock_kg"]
            if preparation is not None
            else strip.acid_mol * flowsheet.MASS_KG_PER_MOL["HCl31"]
        ),
        "octanol_dissolution": fs.losses["octanol_dissolution"],
        "octanol_entrainment": fs.losses["octanol_entrainment"],
        "diluent": fs.losses["diluent"],
        "cleanup_NaOH": fs.cleanup.naoh_mol * flowsheet.MASS_KG_PER_MOL["NaOH"],
        "cleanup_Na2CO3": fs.cleanup.na2co3_mol * flowsheet.MASS_KG_PER_MOL["Na2CO3"],
        "sludge": fs.cleanup.wet_sludge_kg,
        "carbonation_Na2CO3": fs.carbonation.carbonate_mol
        * flowsheet.MASS_KG_PER_MOL["Na2CO3"],
    }
    if model._aqueous_return:
        live["mother_conditioning_HCl31"] = fs.mother_liquor_return.conditioned[
            "HCl31_kg"
        ]
        initial = initial | {
            "mass_kg_per_l": initial["mass_kg_per_l"]
            | {"mother_conditioning_HCl31": 0.0},
            "prices_usd_kg": initial["prices_usd_kg"]
            | {"mother_conditioning_HCl31": initial["prices_usd_kg"]["strip_HCl31"]},
        }
    active = {
        "BuPhen": fs.losses["BuPhen_dissolution"] + fs.losses["BuPhen_entrainment"],
        "D2EHDTPA": fs.losses["D2EHDTPA_entrainment"],
    }
    if set(live) != set(initial["mass_kg_per_l"]) or set(active) != set(ACTIVE):
        raise ValueError("every account must be a live expression")
    kg = costing.RAW_L_H * pyunits.kg
    costing.attach(
        fs,
        {
            k: (m * kg, initial["prices_usd_kg"][k] * costing.usd_per(pyunits.kg))
            for k, m in live.items()
        },
        2 * fs.carbonation.product_mol * d2_stage.owner.MASSES["Li"] * kg,
        active={k: m * kg for k, m in active.items()},
    )


def recycle_report(model, result, owner, actual):
    fs, rc = model.fs, d2_stage.rc
    recycle = fs.organic_recycle
    returned = {s: value(n) for s, n in recycle.returned.items()}
    strip_loss = {s: value(n) for s, n in recycle.strip_loss.items()}
    raff_loss = {
        s: value(n) for s, n in getattr(fs, "raffinate_entrained_organic", {}).items()
    }
    recycled = {
        e: returned[s] for e, s in dict(rc.COMPLEX, Na="naa").items() if s in returned
    }
    boundary = result["organic_inlet_mol_per_raw_l"]
    residuals = {
        s: boundary[s] - returned[s] for s in rc.COMPLEX.values() if s in boundary
    }
    limit = result["solve"]["options"]["constr_viol_tol"]
    if max(map(abs, residuals.values())) > limit:
        raise RuntimeError(f"Organic recycle exceeds {limit} mol/raw L: {residuals}")
    naoh = value(recycle.regeneration_naoh)
    external = {}
    for e, s in dict(rc.COMPLEX, Na="naa").items():
        if e not in recycled:
            continue
        ion = rc.ION[e][0]
        external[e] = (
            result["aqueous_inlet_mol_per_raw_l"][ion]
            + (naoh if e == "Na" else 0.0)
            - result["outlet_mol_per_raw_l"][ion]
            - actual[e]
            - raff_loss.get(s, 0.0)
            - strip_loss[s]
        )
    if (
        max(map(abs, external.values()))
        > references.load_reference("R7")["balance_mol_per_raw_l_limit"]
    ):
        raise RuntimeError(f"External metal balance exceeds R7: {external}")
    mm = d2_stage.owner.MOLAR_MASS_KG_MOL
    modifiers = owner["assumptions"].get(
        "modifier_volume_fractions", d2_stage.owner.HISTORICAL_OCTANOL_FRACTIONS
    )
    makeup = {
        "A": owner["active_mass_kg_per_l"]["D2EHDTPA"] / mm["D2EHDTPA"],
        "B": owner["active_mass_kg_per_l"]["BuPhen"] / mm["BuPhen"],
        "diluent": owner["mass_kg_per_l"]["diluent"]
        / ((1 - sum(modifiers.values())) * mm["dodecane"]),
    }
    dissolved_b = result["accounts"]["BuPhen_dissolved_kg_m3"] / (1000 * mm["BuPhen"])
    ligand_balance = {}
    for name, fresh in makeup.items():
        row = rc.MOIETIES[name] | ({"ha": 1.0} if name == "A" else {})
        ligand_balance[name] = (
            fresh
            - (dissolved_b if name == "B" else 0.0)
            - sum(
                c * (boundary.get(s, 0.0) - returned.get(s, 0.0))
                for s, c in row.items()
            )
        )
    if (
        max(map(abs, ligand_balance.values()))
        > references.load_reference("R7")["balance_mol_per_raw_l_limit"]
    ):
        raise RuntimeError(f"Organic makeup balance exceeds R7: {ligand_balance}")
    nominal = {}
    if hasattr(fs.strip, "preparation"):
        row = rc.MOIETIES["A"] | {"ha": 1.0}
        bought = value(recycle.fresh_acid_form_sites)
        returned_ha = returned["ha"]
        errors = {
            "acid_form_sites": bought
            - sum(
                c * (raff_loss.get(s, 0.0) + strip_loss[s])
                for s, c in row.items()
                if s in strip_loss
            ),
            "newly_saponified_sites": naoh - returned_ha - bought,
        }
        if max(map(abs, errors.values())) > limit:
            raise RuntimeError(
                f"Regeneration balance exceeds {limit} mol/raw L: {errors}"
            )
        incoming = fs.strip.unit.mscontactor.organic_inlet_state[0].flow_mol_comp
        nominal = {
            "loaded_to_strip_mol_per_raw_l": {s: value(n) for s, n in incoming.items()},
            "post_reaction_organic_mol_per_raw_l": {
                s: value(n) for s, n in fs.strip.organic_out.items()
            },
            "loaded_organic_l_per_raw_l": value(recycle.loaded_volume_l),
            "raffinate_entrained_l_per_raw_l": owner["assumptions"][
                "entrainment_l_per_l"
            ]
            * value(fs.aqueous_volume),
            "strip_entrained_l_per_raw_l": value(recycle.strip_entrained_volume_l),
            "fresh_acid_form_sites_mol_per_raw_l": bought,
            "returned_HA_mol_per_raw_l": returned_ha,
            "regeneration_water_kg_per_raw_l": value(recycle.regeneration_water_kg),
            "regeneration_residuals_mol_per_raw_l": errors,
            "buphen_exposed_mol_per_raw_l": sum(
                c * value(incoming[s])
                for s, c in rc.MOIETIES["B"].items()
                if s in incoming
            ),
            "raffinate_dissolved_and_replaced_buphen_mol_per_raw_l": dissolved_b,
            "strip_chemical_buphen_loss_mol_per_raw_l": 0.0,
            "organic_purge_l_per_raw_l": 0.0,
        }
    return {
        "strip_li_recovery_fraction": result["accounts"]["R_rec"],
        "organic_recycle": {
            "returned_species_mol_per_raw_l": returned,
            "returned_metals_mol_per_raw_l": recycled,
            "raffinate_entrainment_mol_per_raw_l": raff_loss,
            "strip_entrainment_mol_per_raw_l": strip_loss,
            "regeneration_naoh_mol_per_raw_l": naoh,
            "residual_mol_per_raw_l": residuals,
            "organic_makeup_mol_per_raw_l": makeup,
            "makeup_balance_mol_per_raw_l": ligand_balance,
            "external_metal_balance_mol_per_raw_l": external,
            "constraint_tolerance_mol_per_raw_l": limit,
            **nominal,
        },
    }


def carbonation_salt_check(fs, owner, assumptions, constraint_tolerance):
    """Convert the raw-litre equality tolerance into the independent Ge coordinate."""
    carbon = fs.carbonation
    recycle = assumptions["mother_recycle"]
    purge_fraction = 1 - recycle
    observed = value(carbon.nacl_mol_kg)
    expected = owner["mother_nacl_mol_kg"]
    water = purge_fraction * value(carbon.water_kg)
    independent = owner["cleanup"]["cleaned_mol"]
    # For conserved Cl and H2O, (1-r) times reactor outlet equals fresh input
    # plus the residuals of the concentrate equality, mixer, reactor, inlet
    # arc, recycle split and recycle arc, and r times the splitter inlet arc.
    flow_error = (6 + recycle) * constraint_tolerance
    cl = value(fs.cleanup.liquor["Cl"])
    fresh_water = value(fs.cleanup.liquor["Li"]) / assumptions["concentrate_li_mol_kg"]
    chloride_error = abs(cl - independent["Cl"]) + flow_error
    water_error = (
        abs(fresh_water - owner["fresh_water_kg"])
        + flow_error * brines.WATER_MOLAR_MASS_KG
    )
    # The salt equality contributes (1-r) times its mol/raw L tolerance.
    # Subtract m_expected * W from m_connected * W, then divide by W.
    bound = (
        chloride_error
        + purge_fraction * constraint_tolerance
        + abs(expected) * water_error
    ) / water
    bound += math.ulp(observed) + math.ulp(expected)
    return {
        "difference_mol_kg": observed - expected,
        "bound_mol_kg": bound,
        "connected_net_water_kg_per_raw_l": water,
        "chloride_error_mol_per_raw_l": chloride_error,
        "water_error_kg_per_raw_l": water_error,
        "constraint_tolerance": constraint_tolerance,
    }


def report(model, result):
    """Recompute cleanup closure, strip ligand inventories and independent operating accounts after solve."""
    fs, rc = model.fs, d2_stage.rc
    assumptions = model._d2_full_process
    if "strip_aqueous_to_organic" in model._design:
        assumptions = dict(
            assumptions,
            strip_aqueous_to_organic=model._design["strip_aqueous_to_organic"],
        )
    aqueous_return = mother_liquor_return.report(fs) if model._aqueous_return else None
    owner = accounts(
        d2_stage.strip_accounts(result["accounts"], model._solvent),
        assumptions,
        model._physical_feed["model_feed"]["Li"],
        aqueous_return=aqueous_return,
        nacl_coordinate=lambda _: value(fs.carbonation.nacl_mol_kg),
    )
    strip = fs.strip
    actual = {e: value(n) for e, n in strip.metals.items()}
    differences = {e: actual[e] - owner["stripped_mol"][e] for e in actual}
    for e, n in differences.items():
        if abs(n) > 1e-12:
            raise RuntimeError(f"Strip fraction differs by {n} mol/raw L for {e}")
    incoming = strip.unit.mscontactor.organic_inlet_state[0].flow_mol_comp
    inventories = {
        name: value(
            sum(row.get(s, 0.0) * n for s, n in incoming.items())
            - sum(
                (row.get(s, 0.0) + (s == "ha" and name == "A")) * n
                for s, n in strip.organic_out.items()
            )
        )
        for name, row in (("A", rc.MOIETIES["A"]), ("B", rc.MOIETIES["B"]))
    }
    proton_error = value(strip.acid_mol - strip.aqueous_out["hydrogen-cation"]) - sum(
        rc.ION[e][2] * n for e, n in actual.items()
    )
    if max(abs(n) for n in (*inventories.values(), proton_error)) > 1e-12:
        raise RuntimeError(
            f"Strip ligand/proton closure exceeds 1e-12 mol/raw L: {inventories}; {proton_error}"
        )
    reconciliation = costing.reconcile(
        fs.costing,
        owner["cost_usd_m3"],
        owner["operating_usd_kg_li"],
        owner["li_product_kg_m3"],
        owner["mass_kg_per_l"],
    )
    unit_errors = {
        k: value(n) - owner[k]
        for k, n in {
            "product_li2co3_mol": fs.carbonation.product_mol,
            "mother_nacl_mol_kg": fs.carbonation.nacl_mol_kg,
            "fresh_water_kg": fs.carbonation.mixer.concentrate_state[0].flow_mol_comp[
                "H2O"
            ]
            * brines.WATER_MOLAR_MASS_KG,
            "carbonation_na2co3_mol": fs.carbonation.carbonate_mol,
        }.items()
    }
    tau = result["solve"]["options"]["constr_viol_tol"]
    salt_check = carbonation_salt_check(fs, owner, assumptions, tau)
    if (
        not math.isfinite(salt_check["difference_mol_kg"])
        or abs(salt_check["difference_mol_kg"]) > salt_check["bound_mol_kg"]
    ):
        raise RuntimeError(
            f"Connected carbonation salt differs from arithmetic: {salt_check}"
        )
    if max(abs(n) for k, n in unit_errors.items() if k != "mother_nacl_mol_kg") > 1e-12:
        raise RuntimeError(
            f"Connected carbonation differs from arithmetic: {unit_errors}"
        )
    # #339 check 8: every reagent, loss and product account of the connected model against the independent
    # recomputation from the solved stage outlets at the solved design, relative.
    pairs = {
        **{
            k: (value(getattr(fs.costing, "aggregate_flow_" + k)) / 1000, m)
            for k, m in owner["mass_kg_per_l"].items()
        },
        **{
            k: (value(fs.costing.active_loss[k]) / 1000, m)
            for k, m in owner["active_mass_kg_per_l"].items()
        },
        "li_product_kg_m3": (value(fs.costing.li_kg_h), owner["li_product_kg_m3"]),
    }
    conditioning = None
    if aqueous_return is not None:
        a, b = pairs.pop("mother_conditioning_HCl31")
        # Conditioning stock is proportional to returned C. Propagate the splitter's
        # mol/raw L tolerance and the costing equality's kg/h tolerance at 1,000 raw L/h.
        bound = (
            2 * flowsheet.MASS_KG_PER_MOL["HCl31"] * tau
            + tau / 1000
            + math.ulp(a)
            + math.ulp(b)
        )
        conditioning = {"difference_kg_per_raw_l": a - b, "bound_kg_per_raw_l": bound}
        if not math.isfinite(a - b) or abs(a - b) > bound:
            raise RuntimeError(
                f"Conditioning stock reconciliation failed: {conditioning}"
            )
    account_differences = {
        k: abs(a / b - 1) if b else abs(a) for k, (a, b) in pairs.items()
    }
    if max(account_differences.values()) > ACCOUNT_RELATIVE_LIMIT:
        raise RuntimeError(
            f"Connected accounts differ from the independent recomputation: {account_differences}"
        )
    balances = {
        c.name: value(c.body - c.lower)
        for b in (strip, fs.carbonation)
        for c in b.component_data_objects(Constraint, active=True)
        if c.equality
    }
    if max(map(abs, balances.values())) > 1e-12:
        raise RuntimeError(
            f"Connected downstream closure exceeds 1e-12 mol/raw L: {balances}"
        )
    physical = {
        "strip liquor": dict(
            actual,
            H=value(strip.aqueous_out["hydrogen-cation"]),
            Cl=value(strip.acid_mol),
        ),
        "strip HCl input": {"H": value(strip.acid_mol), "Cl": value(strip.acid_mol)},
        "cleanup NaOH input": {
            "Na": owner["cleanup"]["naoh_mol"],
            "OH": owner["cleanup"]["naoh_mol"],
        },
        "cleanup soda ash input": {
            "Na": 2 * owner["cleanup"]["na2co3_mol"],
            "C": owner["cleanup"]["na2co3_mol"],
        },
        "carbonation soda ash input": {
            "Na": 2 * owner["carbonation_na2co3_mol"],
            "C": owner["carbonation_na2co3_mol"],
        },
        "strip mixer organic outlet": {
            s: value(n) for s, n in strip.organic_out.items()
        },
        "cleaned liquor": owner["cleanup"]["cleaned_mol"],
        "cleanup cake liquid": owner["cleanup"]["cake_mol"],
        "cleanup precipitates": owner["cleanup"]["solid_mol"],
        "Li2CO3 product": {
            "Li": 2 * owner["product_li2co3_mol"],
            "C": owner["product_li2co3_mol"],
        },
        "mother liquor": dict(
            owner["mother_liquor_mol"],
            Cl=owner["mother_nacl_mol_kg"] * owner["reactor_water_kg"],
        ),
        "mother liquor purge": dict(
            owner["mother_purge_mol"], Cl=owner["cleanup"]["cleaned_mol"]["Cl"]
        ),
    }
    if aqueous_return is not None:
        aqueous_return["conditioning_stock_reconciliation"] = conditioning
        result["mother_liquor_return"] = aqueous_return
        physical["external mother liquor bleed"] = aqueous_return["bleed_mol_per_raw_l"]
        physical["mother liquor returned"] = aqueous_return["returned_mol_per_raw_l"]
        physical["mother liquor conditioning CO2"] = {
            "C": aqueous_return["conditioned"]["CO2"]
        }
        balances.update(aqueous_return["residuals_mol_per_raw_l"])
    recycle = {}
    if hasattr(fs, "organic_recycle"):
        recycle = recycle_report(model, result, owner, actual)
        organic = recycle["organic_recycle"]
        physical["separated organic return"] = organic["returned_species_mol_per_raw_l"]
        physical["strip organic entrainment"] = organic[
            "strip_entrainment_mol_per_raw_l"
        ]
    water = {}
    if hasattr(strip, "preparation"):
        cleaned = owner["cleanup"]
        water = {k: value(n) for k, n in fs.cleanup.water.items()} | {
            "strip HCl stock carrier": owner["strip_preparation"]["stock_water_kg"],
            "strip fresh dilution": owner["strip_preparation"][
                "fresh_dilution_water_kg"
            ],
            "strip discharge": owner["strip_preparation"]["strip_liquor_water_kg"],
            "regeneration discharge": value(fs.organic_recycle.regeneration_water_kg),
        }
        water["external_closure_kg"] = math.fsum(
            (
                water["strip HCl stock carrier"],
                water["strip fresh dilution"],
                water["fresh_wash_water_kg"],
                water["neutralization_reaction_water_kg"],
                water["concentrator_trim_water_kg"],
                -water["final_cake_water_kg"],
                -water["concentrator_vapor_kg"],
                -owner["fresh_water_kg"],
            )
        )
        physical["cleanup recovered wash filtrate"] = cleaned["recovered_wash_mol"]
    return result | {
        "full_process": owner,
        "product_recovery_fraction": owner["raw_li_recovery"],
        **recycle,
        "cost_usd_kg_li": owner["operating_usd_kg_li"],
        "cost_usd_m3_raw_brine": sum(owner["cost_usd_m3"].values()),
        "cost_basis": records.SCOPES["full_process"]["li_denominator"],
        "physical_streams_mol_per_raw_l": physical,
        "idaes_costing_reconciliation": reconciliation,
        "strip_ligand_closure_mol": inventories,
        "strip_proton_closure_mol": proton_error,
        "strip_fraction_difference_mol": differences,
        "owner_differences": unit_errors,
        "carbonation_salt_reconciliation": salt_check,
        "account_relative_differences": account_differences,
        "carbonation_interval_mol_kg": model._carbonation_interval,
        "downstream_balance_residuals_mol_per_raw_l": balances,
        "water_streams_kg_per_raw_l": {
            "cleanup concentrate": owner["cleanup"]["concentrate_water_kg"],
            "carbonation concentrate after wash return and trim": owner[
                "fresh_water_kg"
            ],
            "mother liquor": owner["reactor_water_kg"],
            "mother liquor purge": owner["fresh_water_kg"],
            "mother liquor recycle": owner["reactor_water_kg"]
            - owner["fresh_water_kg"],
            **water,
        },
        "labels": [
            s for s in result["labels"] if "no optimizer or full process" not in s
        ]
        + [owner["claim"]],
    }
