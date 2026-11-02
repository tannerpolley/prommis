"""Raiguel strip plus the declared ideal cleanup and Ge carbonation (Lithium #214)."""

import csv

from pyomo.environ import Block, Constraint, Expression, Var, units as pyunits, value
from pyomo.common.collections import ComponentSet
from pyomo.contrib.incidence_analysis import solve_strongly_connected_components
from pyomo.util.subsystems import TemporarySubsystemManager, create_subsystem_block

from prommis.lithium_extraction.process import (
    waters as brines,
    costing,
    strip_and_carbonation as downstream,
    flowsheet,
    feeds as records,
)
from prommis.lithium_extraction.extraction import d2ehdtpa_buphen_stage as d2_stage


def cleanup(stripped, acid, assumptions):
    """Exact-stoichiometry complete Mg/Ca/Sr removal; Li carbonate co-precipitation is excluded by assumption."""
    mg, ca, sr = (stripped[e] for e in ("Mg", "Ca", "Sr"))
    free_h = acid - sum(d2_stage.rc.ION[e][2] * n for e, n in stripped.items())
    naoh, soda = free_h + 2 * mg, ca + sr
    before = {"Li": stripped["Li"], "Na": stripped["Na"] + naoh + 2 * soda, "Cl": acid}
    water = stripped["Li"] / assumptions["concentrate_li_mol_kg"]
    mass = water + sum(n * d2_stage.owner.MASSES[e] for e, n in before.items())
    atomic = brines.MOLAR_MASS
    dry = (
        mg * (atomic["Mg"] + 2 * (atomic["O"] + atomic["H"]))
        + ca * (atomic["Ca"] + atomic["C"] + 3 * atomic["O"])
        + sr * (atomic["Sr"] + atomic["C"] + 3 * atomic["O"])
    ) / 1000
    held, _, loss = downstream.held_liquid_loss(
        dry, mass, assumptions["moisture"], assumptions["wash_volumes"]
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
    return {
        "naoh_mol": naoh,
        "na2co3_mol": soda,
        "unconsumed_hcl_mol": free_h,
        "dry_sludge_kg": dry,
        "wet_sludge_kg": dry / (1 - assumptions["moisture"]),
        "concentrate_water_kg": water,
        "liquor_kg": mass,
        "held_liquid_kg": held,
        "solute_loss_fraction": loss,
        "cake_mol": cake,
        "solid_mol": solid,
        "cleaned_mol": clean,
        "closure_mol": closure,
        "charge_mol": charge,
        "li_coprecipitation_bound_fraction": 2 * soda / stripped["Li"],
    }


def accounts(strip, assumptions, feed_li):
    """Priced operating subtotal on product Li; D2EHDTPA and BuPhen remain unpriced."""
    stripped = {
        e: strip[f"strip_{e}_mol_L_feed"] for e in ("Li", "Na", "Mg", "Ca", "Sr")
    }
    cleaned = cleanup(stripped, strip["strip_HCl_mol_L_feed"], assumptions)
    carbon = downstream.accounts(
        dict(cleaned["cleaned_mol"], H=0.0, C=0.0), assumptions
    )
    case = strip["case"]
    octanol = float(
        flowsheet._row(
            "data/reference/tea/d2ehdtpa_buphen_price_register.csv",
            "component_id",
            "1_octanol",
        )["delivered_price_usd_kg"]
    )
    masses = {
        "regeneration_NaOH": strip["NaOH_mol_L_feed"]
        * flowsheet.MASS_KG_PER_MOL["NaOH"],
        "strip_HCl31": strip["strip_HCl_mol_L_feed"]
        * flowsheet.MASS_KG_PER_MOL["HCl31"],
        "octanol_dissolution": strip["octanol_dissolved_kg_m3"] / 1000,
        "octanol_entrainment": strip["octanol_entrained_kg_m3"] / 1000,
        "diluent": strip["dodecane_loss_kg_m3"] / 1000,
        "cleanup_NaOH": cleaned["naoh_mol"] * flowsheet.MASS_KG_PER_MOL["NaOH"],
        "cleanup_Na2CO3": cleaned["na2co3_mol"] * flowsheet.MASS_KG_PER_MOL["Na2CO3"],
        "sludge": cleaned["wet_sludge_kg"],
        "carbonation_Na2CO3": carbon["carbonation_na2co3_mol"]
        * flowsheet.MASS_KG_PER_MOL["Na2CO3"],
    }
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
    ceilings = []
    for price in (17.5, 20.9):
        ceiling = (li_kg_m3 * costing.LI2CO3_PER_LI * price - total) / strip[
            "BuPhen_loss_kg_m3"
        ]
        prior = (
            strip["li_recovered_kg_m3"] * costing.LI2CO3_PER_LI * price
            - strip["priced_subtotal_usd_m3"]
        ) / strip["BuPhen_loss_kg_m3"]
        if ceiling >= prior:
            raise RuntimeError(
                "F7 product-basis BuPhen ceiling is not lower than its matching strip basis"
            )
        ceilings.append(
            dict(
                li2co3_usd_kg=price,
                buphen_ceiling_usd_kg=ceiling,
                strip_basis_ceiling_usd_kg=prior,
            )
        )
    return carbon | {
        "case": case,
        "assumptions": assumptions,
        "cleanup": cleaned,
        "stripped_mol": stripped,
        "mass_kg_per_l": masses,
        "prices_usd_kg": prices,
        "price_source": "Lithium #214 price table; source-period nominal USD; EIA diluent analogue unrounded",
        "cost_usd_m3": costs,
        "cost_usd_kg_li": {k: n / li_kg_m3 for k, n in costs.items()},
        "operating_usd_kg_li": total / li_kg_m3,
        "li_product_kg_m3": li_kg_m3,
        "raw_li_recovery": 2 * carbon["product_li2co3_mol"] / feed_li,
        "buphen_ceilings": ceilings,
        "claim": "conditional on ideal complete cleanup and no Li2CO3 co-precipitation; no purity claim; "
        "BuPhen and D2EHDTPA unpriced; capital, labour and energy excluded",
    }


def connect_units(model, initial, assumptions):
    """Connect the charge-weighted strip, ideal cleanup and common carbonation to live extraction."""
    fs, rc = model.fs, d2_stage.rc
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
    strip = downstream.fixed_fraction_strip(
        fs,
        "strip",
        complexes,
        ions,
        charges,
        released,
        d2_stage.owner.BASIS["strip_fractions"][initial["case"]],
    )
    inlet = strip.unit.mscontactor.organic_inlet_state[0].flow_mol_comp
    strip.loaded = Constraint(
        fs.loaded_properties.component_list,
        rule=lambda b, s: inlet[s] == (org[s] if s in org else 0.0),
    )
    strip.acid_mol = Expression(
        expr=sum(charges[e] * org[s] for e, s in complexes.items())
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
    fs.cleanup = Block()
    amounts = d2_stage.strip_accounts(
        {f"strip_{e}_mol_L_feed": n for e, n in strip.metals.items()}, model._solvent
    )
    cleaned = cleanup(
        {e: amounts[f"strip_{e}_mol_L_feed"] for e in ("Li", "Na", "Mg", "Ca", "Sr")},
        strip.acid_mol,
        assumptions,
    )
    for key in ("naoh_mol", "na2co3_mol", "wet_sludge_kg", "solute_loss_fraction"):
        fs.cleanup.add_component(key, Expression(expr=cleaned[key]))
    fs.cleanup.liquor = Expression(
        ("Li", "Na", "Cl"), rule=lambda b, e: cleaned["cleaned_mol"][e]
    )
    with downstream.carbonation.GE_TABLE.open() as stream:
        knots = [
            float(r["nacl_mol_kg"])
            for r in csv.DictReader(stream)
            if r["series"] == "NaCl"
        ]
    salt = initial["mother_nacl_mol_kg"]
    interval = next([lo, hi] for lo, hi in zip(knots, knots[1:]) if lo <= salt <= hi)
    downstream.connect(
        fs,
        dict(fs.cleanup.liquor.items(), H=0.0, C=0.0),
        assumptions,
        {"carbonation_interval_mol_kg": interval},
        initial,
    )
    for e, n in cleaned["cleaned_mol"].items():
        fs.carbonation.mixer.concentrate_state[0].flow_mol_comp[e].set_value(value(n))
    fs.carbonation.mixer.concentrate_state[0].flow_mol_comp["H2O"].set_value(
        value(cleaned["cleaned_mol"]["Li"])
        / assumptions["concentrate_li_mol_kg"]
        / brines.WATER_MOLAR_MASS_KG
    )
    fs.carbonation.mixer.concentrate_state[0].flow_mol_comp["C"].set_value(0.0)
    downstream.initialize(fs, initial, assumptions["mother_recycle"])
    live = {
        "strip_HCl31": strip.acid_mol * flowsheet.MASS_KG_PER_MOL["HCl31"],
        "cleanup_NaOH": fs.cleanup.naoh_mol * flowsheet.MASS_KG_PER_MOL["NaOH"],
        "cleanup_Na2CO3": fs.cleanup.na2co3_mol * flowsheet.MASS_KG_PER_MOL["Na2CO3"],
        "sludge": fs.cleanup.wet_sludge_kg,
        "carbonation_Na2CO3": fs.carbonation.carbonate_mol
        * flowsheet.MASS_KG_PER_MOL["Na2CO3"],
    }
    costing.attach(
        fs,
        {
            k: (
                live.get(k, mass) * costing.RAW_L_H * pyunits.kg,
                initial["prices_usd_kg"][k] * costing.usd_per(pyunits.kg),
            )
            for k, mass in initial["mass_kg_per_l"].items()
        },
        2
        * fs.carbonation.product_mol
        * d2_stage.owner.MASSES["Li"]
        * costing.RAW_L_H
        * pyunits.kg,
    )
    model._d2_full_process = assumptions


def report(model, result):
    """Recompute cleanup closure, strip ligand inventories and independent operating accounts after solve."""
    fs, rc = model.fs, d2_stage.rc
    owner = accounts(
        d2_stage.strip_accounts(result["accounts"], model._solvent),
        model._d2_full_process,
        result["feed"]["model_feed"]["Li"],
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
    if max(abs(n) for n in unit_errors.values()) > 1e-12:
        raise RuntimeError(
            f"Connected carbonation differs from arithmetic: {unit_errors}"
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
        "retained organic": {s: value(n) for s, n in strip.organic_out.items()},
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
    return result | {
        "full_process": owner,
        "product_recovery_fraction": owner["raw_li_recovery"],
        "cost_usd_kg_li": owner["operating_usd_kg_li"],
        "cost_usd_m3_raw_brine": sum(owner["cost_usd_m3"].values()),
        "cost_basis": records.SCOPES["full_process"]["li_denominator"],
        "physical_streams_mol_per_raw_l": physical,
        "idaes_costing_reconciliation": reconciliation,
        "strip_ligand_closure_mol": inventories,
        "strip_proton_closure_mol": proton_error,
        "strip_fraction_difference_mol": differences,
        "owner_differences": unit_errors,
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
        },
        "labels": [
            s for s in result["labels"] if "no optimizer or full process" not in s
        ]
        + [owner["claim"]],
    }
