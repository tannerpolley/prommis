"""#218 E1–E8 and accepted H-D0: transferred sorbent balances and conditional cost floors."""

import csv

from idaes.core import FlowsheetBlock, MomentumBalanceType
from idaes.models.unit_models import Feed, Mixer, Separator
from idaes.models.unit_models.mixer import MixingType, MomentumMixingType
from idaes.models.unit_models.separator import EnergySplittingType, SplittingType
from pyomo.environ import ConcreteModel, Constraint, units as u, value
from pyomo.util.check_units import assert_units_consistent

from prommis.lithium_extraction.process.divalent_softening import (
    NAOH_KG_MOL,
    NA2CO3_KG_MOL,
    NA2CO3_USD_KG,
    MOLAR_MASS_G_MOL,
    SLUDGE_KG_MOL,
)
from prommis.lithium_extraction.process.screening_costs import (
    load_screening_cost_assumptions,
)
from prommis.lithium_extraction.process import (
    waters as brines,
    costing,
    flowsheet,
    softening,
)

ROOT = flowsheet.REPO_ROOT
TABLE = list(
    csv.DictReader(
        (ROOT / "data/reference/sorbent/aleynikov2025_li_al_ldh_tables.csv").open()
    )
)
PRICES = {
    r["quantity"]: float(r["value"])
    for r in csv.DictReader(
        (ROOT / "data/reference/tea/front_end_price_register.csv").open()
    )
}
SOURCE = {
    s: {
        r["column_label"].replace("_labelled_BO3", ""): float(r["value"])
        for r in TABLE
        if r["table"] == "1" and r["quantity"] == "mass_concentration"
    }
    for s in ("initial_brine",)
}
for sample in ("wash_2_BV", "eluate_6_BV"):
    SOURCE[sample] = {
        r["row_label_printed"]
        .split("+")[0]
        .replace("2", "")
        .replace("BO3^3-", "B"): float(r["value"])
        for r in TABLE
        if r["table"] == "4"
        and r["quantity"] == "mass_concentration"
        and r["sample_or_condition"] == sample
    }
UPTAKE = next(
    float(r["value"]) / 100
    for r in TABLE
    if r["table"] == "3" and r["quantity"] == "li_extraction" and r["Q_BV_per_h"] == "2"
)
CAPACITY_G_KG = next(
    float(r["value"])
    for r in TABLE
    if r["table"] == "3"
    and r["quantity"] == "full_dynamic_exchange_capacity_mass"
    and r["Q_BV_per_h"] == "2"
)
FW = (
    2
    * SOURCE["wash_2_BV"]["Li"]
    / (2 * SOURCE["wash_2_BV"]["Li"] + 6 * SOURCE["eluate_6_BV"]["Li"])
)
KAPPA = SOURCE["eluate_6_BV"]["Li"] / SOURCE["initial_brine"]["Li"]
MASSES = {j: m / 1000 for j, m in brines.MOLAR_MASS.items()} | {
    "CO2": 0.044009,
    "H2O": brines.WATER_MOLAR_MASS_KG,
}


def amounts(state):
    return {j: value(n) * 3600 for (p, j), n in state.flow_mol_phase_comp.items()}


def density(amount, volume):
    rho = 1000 * sum(MASSES[j] * n for j, n in amount.items()) / volume
    if not 500 <= rho <= 2000:
        raise ValueError(f"H-D0 accounting density outside MCAS bounds: {rho}")
    return rho


def build(
    raw, kind="sorbent", *, kappa=KAPPA, recycle=True, recovery=0.5, rejection=0.0
):
    """One raw L/h. H-D0 V1–V7 per-state densities, with no liquid-density correlation claim."""
    n = dict(raw["mol"], H2O=raw["water_kg"] / MASSES["H2O"])
    charge = {j: brines.CHARGE.get(j, 0) for j in n}
    model = ConcreteModel()
    model.fs = fs = FlowsheetBlock(dynamic=False)
    fs.properties = softening.reactive_properties(n, {j: MASSES[j] for j in n}, charge)
    fs.feed = Feed(property_package=fs.properties)
    softening.set_state(fs.feed.properties[0], n, density(n, 1.0))
    d = UPTAKE / (1 - UPTAKE * FW) if recycle else UPTAKE
    ve = (1 - FW) * d / kappa
    vw = ve / 3
    if kind == "none":
        streams, volumes, water = {"outlet": n}, {"outlet": 1.0}, 0.0
    else:
        if kind == "sorbent":
            e, w = {}, {}
            for j in raw["mol"]:
                source_j = "Ca" if j in ("Sr", "Ba") else j
                for out, sample, volume in (
                    (e, "eluate_6_BV", ve),
                    (w, "wash_2_BV", vw),
                ):
                    out[j] = (
                        n[j] * d * ((1 - FW) if out is e else FW)
                        if j == "Li"
                        else (
                            n[j]
                            * SOURCE[sample][source_j]
                            / SOURCE["initial_brine"][source_j]
                            * volume
                            if source_j in SOURCE[sample]
                            else 0.0
                        )
                    )
            for out in (e, w):
                fraction = sum(charge[j] * out[j] for j in out if charge[j] > 0) / sum(
                    charge[j] * n[j] for j in n if charge[j] > 0
                )
                out.update({j: fraction * n[j] for j in n if charge[j] < 0})
            e["H2O"], w["H2O"] = 0.0, 0.0
            fs.return_wash = Mixer(
                property_package=fs.properties,
                inlet_list=["raw", "solutes", "water"],
                energy_mixing_type=MixingType.none,
                momentum_mixing_type=MomentumMixingType.none,
            )
            returned = {
                j: w[j] + (vw / MASSES["H2O"] if j == "H2O" else 0.0) for j in n
            }
            column = {j: n[j] + (returned[j] if recycle else 0.0) for j in n}
            vc = 1 + (vw if recycle else 0.0)
            for state, a, rho in (
                (fs.return_wash.raw_state[0], n, density(n, 1.0)),
                (fs.return_wash.solutes_state[0], w if recycle else {}, 1000.0),
                (
                    fs.return_wash.water_state[0],
                    {"H2O": vw / MASSES["H2O"]} if recycle else {},
                    1000.0,
                ),
                (fs.return_wash.mixed_state[0], column, density(column, vc)),
            ):
                softening.set_state(state, a, rho)
            split_outputs = {"eluate_solutes": e, "wash_solutes": w}
        elif kind == "nf":
            # H-D0 N1/N2 only: a target balance, never a connected membrane prediction.
            e = {
                j: recovery
                * n[j]
                * (1 - rejection if j in ("Ca", "Mg", "Sr", "Ba") else 1.0)
                for j in n
            }
            e["Cl"] = sum(charge[j] * v for j, v in e.items() if j != "Cl")
            column, split_outputs, vc = n, {"permeate": e}, 1.0
        else:
            raise ValueError(kind)
        split_outputs["raffinate"] = {
            j: column[j] - sum(a[j] for a in split_outputs.values()) for j in n
        }
        if min(v for a in split_outputs.values() for v in a.values()) < 0:
            raise ValueError("Front-end split requires a negative component inventory")
        fs.splitter = Separator(
            property_package=fs.properties,
            outlet_list=list(split_outputs),
            split_basis=SplittingType.componentFlow,
            energy_split_basis=EnergySplittingType.none,
            momentum_balance_type=MomentumBalanceType.none,
        )
        softening.set_state(fs.splitter.mixed_state[0], column, density(column, vc))
        for name, a in split_outputs.items():
            for j in n:
                fs.splitter.split_fraction[0, name, j].fix(
                    a[j] / column[j] if column[j] else float(name == "raffinate")
                )
            rho = (
                density(a, vc) if name == "raffinate" and kind == "sorbent" else 1000.0
            )
            softening.set_state(getattr(fs.splitter, name + "_state")[0], a, rho)
        if kind == "sorbent":
            fs.elute = Mixer(
                property_package=fs.properties,
                inlet_list=["solutes", "water"],
                energy_mixing_type=MixingType.none,
                momentum_mixing_type=MomentumMixingType.none,
            )
            collected = dict(e, H2O=ve / MASSES["H2O"])
            for state, a, rho in (
                (
                    fs.elute.solutes_state[0],
                    amounts(fs.splitter.eluate_solutes_state[0]),
                    1000.0,
                ),
                (fs.elute.water_state[0], {"H2O": ve / MASSES["H2O"]}, 1000.0),
                (fs.elute.mixed_state[0], collected, density(collected, ve)),
            ):
                softening.set_state(state, a, rho)
            streams = {
                "outlet": amounts(fs.elute.mixed_state[0]),
                "wash": dict(w, H2O=vw / MASSES["H2O"]),
                "raffinate": split_outputs["raffinate"],
            }
            fs.wash_solution = fs.properties.build_state_block([0], defined_state=True)
            softening.set_state(
                fs.wash_solution[0], streams["wash"], density(streams["wash"], vw)
            )
            for state in (fs.elute.mixed_state[0], fs.wash_solution[0]):
                state.conc_mass_phase_comp
            volumes, water = {"outlet": ve, "wash": vw, "raffinate": vc}, ve + vw
        else:
            streams, volumes, water = (
                {"outlet": e, "raffinate": split_outputs["raffinate"]},
                {"outlet": recovery, "raffinate": 1 - recovery},
                0.0,
            )
            for name, volume in (("permeate", recovery), ("raffinate", 1 - recovery)):
                softening.set_state(
                    getattr(fs.splitter, name + "_state")[0],
                    split_outputs[name],
                    density(split_outputs[name], volume),
                )
    softening.evaluate(model)
    assert_units_consistent(model)
    residuals = {
        c.name: value(c.body - c.lower) * 3600
        for c in model.component_data_objects(Constraint, active=True)
        if c.equality and ("material" in c.name or "split_fraction_eq" in c.name)
    }
    external = (
        ["outlet"]
        if kind == "none"
        else ["outlet", "raffinate"]
        + (["wash"] if kind == "sorbent" and not recycle else [])
    )
    closure = {
        j: n[j]
        + (water / MASSES["H2O"] if j == "H2O" else 0.0)
        - sum(streams[s][j] for s in external)
        for j in n
    }
    charges = {s: sum(charge[j] * v for j, v in a.items()) for s, a in streams.items()}
    if (
        max(map(abs, (*residuals.values(), *closure.values(), *charges.values())))
        > 1e-9
    ):
        raise RuntimeError(f"Front-end balances: {closure}, {charges}, {residuals}")
    outlet = streams["outlet"]
    alternative_c = (
        n.get("CO2", 0.0) * outlet["Cl"] / n["Cl"] if kind == "sorbent" else None
    )
    record = dict(
        kind=kind,
        original_raw_feed=raw,
        streams_mol_per_raw_l=streams,
        solution_l_per_raw_l=volumes,
        inorganic_carbon_transfer=(
            "neutral CO2 remains in raffinate; unvalidated transfer; Aleynikov reports no carbon"
            if kind == "sorbent"
            else "unchanged component passage"
        ),
        common_anion_carbon_arithmetic_bound=(
            None
            if alternative_c is None
            else dict(
                mol_per_raw_l=alternative_c,
                fraction_of_raw_soda_floor=alternative_c
                / sum(v for j, v in n.items() if j in ("Ca", "Sr", "Ba")),
                fraction_of_eluate_soda_floor=alternative_c
                / sum(v for j, v in outlet.items() if j in ("Ca", "Sr", "Ba")),
            )
        ),
        water_input_kg_per_raw_l=water,
        component_closure=closure,
        charge_eq_per_raw_l=charges,
        unit_residuals=residuals,
        recovery=outlet["Li"] / n["Li"],
        gross_uptake=d if kind == "sorbent" else None,
        kappa=kappa,
        wash_fraction=FW,
        fresh_water_l_per_raw_l=water,
        make_up_kg_per_raw_l={
            case: (
                (
                    1 - PRICES["dong_retention_30_cycles"] ** (1 / 30)
                    if case == "base"
                    else PRICES["dong_dissolution_per_cycle"]
                )
                * d
                * n["Li"]
                * MASSES["Li"]
                * 1000
                / CAPACITY_G_KG
                if kind == "sorbent"
                else 0.0
            )
            for case in ("base", "optimistic")
        },
    )
    return (
        model,
        dict(
            raw,
            mol={j: v for j, v in outlet.items() if j != "H2O"},
            water_kg=outlet["H2O"] * MASSES["H2O"],
            front_end=record,
        ),
        record,
    )


def prices(case):
    from prommis.costing.ree_costing_dictionaries import load_default_resource_prices

    p = load_default_resource_prices()
    return {
        "NaOH": load_screening_cost_assumptions().naoh_price,
        "Na2CO3": (
            float(
                flowsheet._row(
                    "data/reference/tea/pretreatment_cost_assumptions.csv",
                    "parameter",
                    "pretreatment_soda_ash_usd_kg",
                )["base_value"]
            )
            if case == "base"
            else NA2CO3_USD_KG
        ),
        "sorbent": PRICES["dong_sorbent_price"] / 1000 / PRICES["cny_per_usd_2025"],
        "water": value(u.convert(p["water"], to_units=u.USD_2021 / u.L)),
        "CaCO3": value(u.convert(p["CACO3"], to_units=u.USD_2020 / u.kg)),
        "MgOH2": PRICES["mgoh2_credit_upper"],
    }


def floor(feed, case, *, tolerance=0.0):
    """E5 screening-conversion inlet; P's native floor units own doses and mineral balances."""
    from prommis.lithium_extraction.process import precipitative_softening as ps

    system, _ = ps._system(ps.PHASES)
    state = ps._state(
        system, 25.0, feed["water_kg"], {j: n for j, n in feed["mol"].items() if n != 0}
    )
    before = ps._record(system, state, ps.PHASES)
    units = softening.Softening(ps, before["aq"])
    doses = {}
    for reagent in ("NaOH", "Na2CO3"):
        unit, liquid, _, _ = units.floor(
            reagent,
            before,
            reagent,
            0.0,
            0.0,
            density=1000.0,
            carbonate_allowance=tolerance,
        )
        doses[reagent] = value(unit.flow_mol_reagent[reagent]) * 3600
        mass = sum(v * units.mass[j] for j, v in liquid.items())
        before = {
            "aq": {
                e: sum(
                    v * ps.DB.species(units.species[j]).elements().coefficient(e)
                    for j, v in liquid.items()
                )
                for e in before["aq"]
            },
            "liquor_kg": mass,
            "liquor_l": mass,
        }
    residuals = units.check()
    masses = {
        "NaOH": doses["NaOH"] * NAOH_KG_MOL,
        "Na2CO3": doses["Na2CO3"] * NA2CO3_KG_MOL,
    }
    fe = feed["front_end"]
    masses.update(
        sorbent=fe["make_up_kg_per_raw_l"][case], water=fe["fresh_water_l_per_raw_l"]
    )
    p = prices(case)
    cost = costing.attach(
        units.model.fs,
        {
            j: (
                m * costing.RAW_L_H * (u.L if j == "water" else u.kg),
                p[j] * costing.usd_per(u.L if j == "water" else u.kg),
            )
            for j, m in masses.items()
        },
        feed["mol"]["Li"] * MASSES["Li"] * costing.RAW_L_H * u.kg,
    )
    accounts = {j: value(v) for j, v in cost.aggregate_flow_costs.items()}
    return dict(
        F=sum(accounts[j] for j in ("NaOH", "Na2CO3")),
        A_fe=sum(accounts[j] for j in ("sorbent", "water")),
        accounts_usd_m3=accounts,
        mass_per_raw_l=masses,
        dose_mol_per_raw_l=doses,
        maximum_unit_balance_mol=max(map(abs, residuals.values()), default=0.0),
        solids_kg_per_raw_l={
            j: sum(
                value(unit.flow_mol_precipitate[j]) * 3600 * SLUDGE_KG_MOL[e]
                for unit in units.stages
                if hasattr(unit, "flow_mol_precipitate")
                and j in unit.flow_mol_precipitate
            )
            for j, e in (("Calcite", "Ca"), ("Brucite", "Mg"))
        },
    )


def screening_feed(site):
    raw = brines.raw_feed(site)
    for j in ("Ca", "Mg", "Sr", "Ba"):
        old = raw["mol"][j]
        raw["mol"][j] = raw["printed_mg_l"][j] / 1000 / MOLAR_MASS_G_MOL[j]
        raw["mol"]["Cl"] += 2 * (raw["mol"][j] - old)
    return raw


def break_even(row, case):
    p, dose = prices(case), row["dose_mol_per_raw_l"]
    solids = row["solids_kg_per_raw_l"]
    return dict(
        co2_carbonate_supply_parity_usd_kg=(
            p["Na2CO3"] * NA2CO3_KG_MOL - 2 * p["NaOH"] * NAOH_KG_MOL
        )
        / 0.044009,
        naoh_carbonate_supply_parity_usd_kg=p["Na2CO3"]
        * NA2CO3_KG_MOL
        / (2 * NAOH_KG_MOL),
        common_solids_zero_headroom_usd_kg=-row["H"] / (1000 * sum(solids.values())),
        credit_bounds_usd_m3=[
            0.0,
            1000 * (solids["Calcite"] * p["CaCO3"] + solids["Brucite"] * p["MgOH2"]),
        ],
        credit_price_bounds_usd_kg={
            "CaCO3": [0.0, p["CaCO3"]],
            "MgOH2": [0.0, p["MgOH2"]],
        },
        counterfactual_caustic_mol_per_raw_l=dose["NaOH"] + 2 * dose["Na2CO3"],
        counterfactual_caustic_solution_l_per_raw_l=(dose["NaOH"] + 2 * dose["Na2CO3"])
        / PRICES["bmed_solution_concentration"],
        declared_electricity_cost_scenario_usd_m3=1000
        * (dose["NaOH"] + 2 * dose["Na2CO3"])
        * NAOH_KG_MOL
        * PRICES["bmed_sec_scenario"]
        * PRICES["bmed_electricity_scenario"],
        co2_free_floor_usd_m3=1000
        * (dose["NaOH"] + 2 * dose["Na2CO3"])
        * NAOH_KG_MOL
        * p["NaOH"],
        basis="floor; parity is not zero headroom; credit only CaCO3/Mg(OH)2; BMED counterfactual pending D4",
    )


def nf_target(value_li, softening_floor, recovery, expenditure):
    """H-D0 N4/N5; None preserves an unattainable target rather than clipping it to one."""
    x = 1 - value_li / softening_floor + expenditure / (recovery * softening_floor)
    return dict(
        required_divalent_rejection=None if x > 1 else max(0.0, x),
        allowable_expenditure_at_zero_rejection=recovery * (value_li - softening_floor),
        maximum_expenditure=recovery * value_li,
    )
