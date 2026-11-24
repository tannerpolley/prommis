"""Feed-dependent prototype: #166 softening/set C, labelled acid cycle and empirical carbonation."""

import csv
import importlib.metadata
import json
from pathlib import Path
from urllib.parse import unquote, urlparse

from prommis.lithium_extraction.paths import ENGINE_CONDA_RELEASES, REPO_ROOT
from prommis.lithium_extraction.process import strip_and_carbonation as downstream

# Engine source commit of each declared wheel, keyed by wheel sha256: the wheel itself carries no commit. Only wheels
# with the (T, P, *controls) equilibrium callback of Engine #1098 run this code (Lithium #354).
ENGINE_COMMITS = {
    "393e2f0f36e037cba039c5ceb21d6bd2b3786af490a6ba5baeeb33646200a7a1": "80e7ae869f4a82764dc70b4f324f7462a17240b0",
    "e2668bb17d6e3468586802f77b30dc2e8953b38a9b3603cbce02ed9a6ebe5860": "5cb4867fd9e690bff74043a32eaec4a0ec97306d",
    "e904494fbc17fb18837e2d9725a2f685bc484fb03e45db5c56a62d83237efdb3": "d180b2e24f9f71b0cd103eea13fbc307c90904cd",
    "16f8d54caf704d235db6dd093634161c4c1028c96285be44ad247a0ba2f08025": "ce6b4fcbd91d77de638089b1631b0b53b1322fc1",
    "7e7ef16f989913a4618b0eb50a345a71e645fe4d648c2556df9e07ea55663bba": "7316df21d03dc406e235e92d1abf14353a7a2e02",
}


HCL_MOLAR_MASS_KG = 0.036461


def hcl_stock_fraction():
    return float(
        _row(
            "analyses/prototypes/modular_process/ligand_prices/revision_prices.csv",
            "component",
            "HCl",
        )["active_mass_fraction"]
    )


MASS_KG_PER_MOL = {
    "NaOH": 0.039997,
    "Na2CO3": 0.105988,
    "HCl31": HCL_MOLAR_MASS_KG / 0.31,
    "HBTA": 0.216157,
    "TOPO": 0.38663,
    "diluent": 0.17033484,
}
REAGENT_REGISTER = "data/reference/tea/hbta_topo_screening_cost_assumptions.csv"
PRETREATMENT_REGISTER = "data/reference/tea/pretreatment_cost_assumptions.csv"


def installed_engine():
    """Identify a declared wheel or a conda release allowed by the tutorial export."""
    from prommis.lithium_extraction.thermodynamics.reaction_topology import (
        _sha256,
        _verify_wheel,
    )

    distribution = importlib.metadata.distribution("epcsaft")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    if "archive_info" in direct:  # installed from a wheel file
        wheel = Path(unquote(urlparse(direct["url"]).path))
        sha256 = _sha256(wheel)
        if sha256 not in ENGINE_COMMITS:
            raise ValueError(
                f"Installed Engine wheel {wheel} ({sha256}) is not a declared wheel; wheels before "
                "Engine #1098, including epcsaft 1.2.0, lack the (T, P, *controls) equilibrium "
                "callback this code calls (Lithium #354)"
            )
        return _verify_wheel(wheel, sha256, ENGINE_COMMITS[sha256]) | {
            "identified_by": "wheel sha256"
        }
    version = distribution.version
    # A conda package's direct URL points to its source build directory.
    if (
        distribution.read_text("INSTALLER") or ""
    ).strip() == "conda" and version in ENGINE_CONDA_RELEASES:
        return {
            "filename": f"epcsaft-{version}",
            "version": version,
            "source_commit": ENGINE_CONDA_RELEASES[version],
            "identified_by": "conda release version",
        }
    raise ValueError(
        f"Installed epcsaft {version} needs a wheel declared by SHA-256 or a supported conda release. "
        f"Allowed conda versions: {sorted(ENGINE_CONDA_RELEASES)}"
    )


def washed_feed(
    feed,
    moisture,
    wash_volumes,
    *,
    unit_model=None,
    soda_multiple=1.2,
    omit_carbonate=False,
    residual_limits=None,
    record_steps=False,
):
    """Equilibrium dosing and cake washing per raw litre, with optional Mg and Ca targets in mg/raw L."""
    from prommis.lithium_extraction.process import precipitative_softening as ps
    from prommis.lithium_extraction.process.softening import (
        Softening,
        components,
        set_state,
    )
    from pyomo.environ import value

    system, solver = ps._system(ps.PHASES)
    state = ps._state(
        system, 25.0, feed["water_kg"], {j: n for j, n in feed["mol"].items() if n != 0}
    )
    initial = ps._elements(system, state.elementAmounts())
    tracked = tuple(initial)  # The Reaktoro state owns the complete element support.
    units = Softening(ps, initial, unit_model)
    reagent_mol = {
        "NaOH": 2.4 * feed["mol"]["Mg"],
        "Na2CO3": (0.0 if omit_carbonate else soda_multiple)
        * sum(feed["mol"][e] for e in ("Ca", "Sr", "Ba")),
    }
    cake, solid = {e: 0.0 for e in tracked}, {e: 0.0 for e in tracked}
    wash_water_mol = 0.0
    dry_kg = wet_kg = 0.0
    steps, residual_acceptance = {}, {}
    if residual_limits is not None and omit_carbonate:
        raise ValueError("Ca target requires the carbonate step")
    for reagent in (*reagent_mol, "HCl"):
        if reagent == "Na2CO3" and omit_carbonate:
            continue
        before = ps._record(system, state, ps.PHASES)
        if reagent == "HCl":
            unconditioned = state
            dose, state = ps._to_ph(
                system,
                solver,
                unconditioned,
                "HCl",
                feed["pH"],
                upper=2 * before["aq"]["C"] + 1,
            )
            ph_acceptance = ph_limit(ps, system, solver, unconditioned, dose)
        elif residual_limits is not None:
            element = "Mg" if reagent == "NaOH" else "Ca"
            dose, state, residual_acceptance[element] = ps._to_residual(
                system,
                solver,
                state,
                reagent,
                element,
                residual_limits[element],
                reagent_mol[reagent],
            )
        else:
            dose = reagent_mol[reagent]
            state = ps._dosed(system, solver, state, reagent, dose)
        step = ps._record(system, state, ps.PHASES)
        if record_steps or residual_limits is not None:
            steps[reagent] = step
        water_before = float(state.speciesAmount("H2O"))
        unit, next_liquid, cake_liquid, wash = units.stage(
            reagent,
            before,
            step,
            reagent,
            dose,
            moisture,
            0.0 if reagent == "HCl" else wash_volumes,
            water_before,
        )
        reagent_mol[reagent] = value(unit.flow_mol_reagent[reagent]) * 3600
        if not hasattr(unit, "flow_mass_precipitate"):
            continue  # No equilibrium solids: the native dissolution outlet is the next inlet, with no cake wash.
        dry_kg += sum(value(n) for n in unit.flow_mass_precipitate.values()) * 3600
        wet_kg += (
            sum(value(n) for n in unit.flow_mass_precipitate.values()) * 3600
            + sum(
                value(n)
                for n in unit.separator.waste_state[0].flow_mass_phase_comp.values()
            )
            * 3600
        )
        for mineral, amount in unit.flow_mol_precipitate.items():
            composition = ps.DB.species(mineral).elements()
            for e in tracked:
                solid[e] += value(amount) * 3600 * composition.coefficient(e)
        for e in tracked:
            cake[e] += sum(
                n * ps.DB.species(units.species[j]).elements().coefficient(e)
                for j, n in cake_liquid.items()
            )
        wash_water_mol += wash
        fs = units.model.fs
        retained = 1 - value(unit.separator.split_fraction[0, "waste", "Liq"]) * value(
            getattr(fs, reagent + "_filter").split_fraction[0, "cake", "Li"]
        )
        analytical, _ = components(ps, step["aq"])
        ps._strip(state, ps.PHASES)
        for species in system.phases().get("AqueousPhase").species():
            name = species.name()
            state.set(name, float(state.speciesAmount(name)) * retained, "mol")
        state.set(
            "H2O",
            next_liquid["H2O"] - retained * (analytical["H2O"] - water_before),
            "mol",
        )
        post = ps._record(system, state, ps.PHASES)
        set_state(
            getattr(fs, reagent + "_recombine").mixed_state[0],
            next_liquid,
            post["liquor_kg"] / (post["liquor_l"] / 1000),
        )
    fin = ps._record(system, state, ps.PHASES)
    outlet = (
        units.model.fs.HCl_recombine.mixed_state[0]
        if hasattr(units.model.fs, "HCl_recombine")
        else units.model.fs.HCl.dissolution_reactor.properties_out[0]
    )
    units.check()
    aqueous = {e: fin["aq"][e] for e in tracked}
    reagent = {
        e: sum(n * ps.REAGENT_ELEMENTS[k].get(e, 0) for k, n in reagent_mol.items())
        for e in tracked
    }
    wash = {
        e: wash_water_mol * ps.DB.species("H2O").elements().coefficient(e)
        for e in tracked
    }
    closure = {
        e: initial[e] + reagent[e] + wash[e] - aqueous[e] - cake[e] - solid[e]
        for e in tracked
    }
    if (
        max(map(abs, closure.values())) > ps.CLOSURE_MOL
        or min(aqueous.values()) < 0
        or abs(fin["charge_mol"]) > ps.CLOSURE_MOL
    ):
        raise RuntimeError(f"Washed softening balance: {closure}")
    species = system.phases().get("AqueousPhase").species()
    physical, mapped_species = components(ps, aqueous)
    physical["water"] = physical.pop("H2O")
    if physical["OH"] == 0.0:
        physical.pop("OH")
    component_species = {
        ("water" if j == "H2O" else j): name for j, name in mapped_species.items()
    }
    reported_species = {j: name for j, name in component_species.items() if j != "OH"}
    component_charge = {
        j: ps.DB.species(component_species[j]).charge() for j in physical
    }
    reconstruction = {
        e: sum(
            physical[k] * ps.DB.species(s).elements().coefficient(e)
            for k, s in component_species.items()
            if k in physical
        )
        - aqueous[e]
        for e in tracked
    }
    if min(physical.values()) < 0 or max(map(abs, reconstruction.values())) > 1e-9:
        raise RuntimeError(f"Washed aqueous component reconstruction: {reconstruction}")
    result = {
        "front_end": feed.get("front_end"),
        "aqueous_mol": aqueous,
        "feed_l": value(outlet.flow_vol_phase["Liq"]) * 3600 * 1000,
        "cake_mol": cake,
        "solid_mol": solid,
        "raw_mol": {e: initial[e] for e in tracked},
        "reagent_mol": reagent_mol,
        "wash_water_mol": wash_water_mol,
        "wash_elements_mol": wash,
        "aqueous_species_mol": {
            s.name(): float(state.speciesAmount(s.name())) for s in species
        },
        "aqueous_species_charge": {s.name(): s.charge() for s in species},
        "physical_component_mol": physical,
        "physical_component_charge": component_charge,
        "component_species": reported_species | {"OH": "OH-"},
        "component_reconstruction_mol": reconstruction,
        "aqueous_charge_eq_per_raw_l": fin["charge_mol"],
        "aqueous_charge_tolerance_eq_per_raw_l": ps.CLOSURE_MOL,
        "closure_mol": closure,
        "dry_sludge_kg": dry_kg,
        "wet_sludge_kg": wet_kg,
        "moisture": moisture,
        "wash_volumes": wash_volumes,
        "pH": fin["pH"],
        "pH_acceptance": ph_acceptance,
        "ca_mg_mg_l": (
            fin["aq"]["Ca"] * ps.MOLAR_MASS["Ca"]
            + fin["aq"]["Mg"] * ps.MOLAR_MASS["Mg"]
        )
        * 1000,
    }
    if record_steps or residual_limits is not None:
        result["steps"] = steps
    if residual_limits is not None:
        ph_acceptance["limit_ph"] = max(1e-6, ph_acceptance["limit_ph"])
        ph_acceptance[
            "basis"
        ] += ". Target mode also applies the owner pH resolution of 1e-6."
        if abs(fin["pH"] - feed["pH"]) > ph_acceptance["limit_ph"]:
            raise RuntimeError(f"HCl: washed pH {fin['pH']:.12g}, target {feed['pH']}")
        result.update(
            residual_limits_mg_per_raw_l=dict(residual_limits),
            residual_acceptance=residual_acceptance,
        )
    return result


def ph_limit(ps, system, solver, state, dose):
    """How far the reached washed pH may lie from the process pH, from the two solvers' own precision (#342; owner
    rule: acceptance limits come from upstream precision, #219).

    1. Reaktoro's Optima solver stops when the optimality error of the Gibbs-energy/RT minimization is below its
       convergence tolerance (Reaktoro 2.13 default 1e-8). That residual is in units of mu/RT = ln a for an aqueous
       species, so ln a(H+) is within the tolerance and pH = -log10 a(H+) within tolerance / ln 10 (4.34e-9).
    2. brentq returns the HCl dose within xtol + rtol * dose (mol) of the root; the pH error is that times |dpH/dn| at
       the root, taken by a forward difference of 1e-4 of the dose.
    The limit is their sum; it changes no equilibrium calculation.
    """
    import math

    equilibrium = ps.rkt.EquilibriumOptions().optima.convergence.tolerance / math.log(
        10
    )
    step = max(1e-4 * dose, 1e-9)
    ph = [
        float(ps.rkt.AqueousProps(ps._dosed(system, solver, state, "HCl", n)).pH())
        for n in (dose, dose + step)
    ]
    slope, root = abs(ph[1] - ph[0]) / step, ps.PH_XTOL + ps.PH_RTOL * dose
    return {
        "equilibrium_tolerance_ph": equilibrium,
        "dose_tolerance_mol": root,
        "slope_ph_per_mol": slope,
        "limit_ph": equilibrium + slope * root,
        "basis": "Optima optimality tolerance / ln 10 plus |dpH/dn| times the brentq dose tolerance",
    }


def extract(feed, constants, extractant):
    """Three countercurrent strict Engine stages at O/A 1, R 70%, actual washed Li/Na feed."""
    from prommis.lithium_extraction.extraction.countercurrent import (
        acidity_at_ph,
        countercurrent,
    )

    aq, v = feed["aqueous_mol"], feed["feed_l"]
    result = countercurrent(
        constants,
        reaction_set="C",
        stage_count=3,
        organic_to_aqueous=1.0,
        extractant=extractant,
        saponification_fraction=0.7,
        feed_li=aq["Li"] / v,
        feed_na=aq["Na"] / v,
        feed_acidity=acidity_at_ph(feed["pH"]),
    )
    # Engine's fixed organic diluent amount is retained for component replacement costing.
    result["diluent_mol"] = result["start"]["guesses"][0][1]["diluent"] * v
    result.pop("start")
    return result


def _row(path, key, value):
    with (REPO_ROOT / path).open() as stream:
        return next(r for r in csv.DictReader(stream) if r[key] == value)


def softening_masses(feed):
    """Reagent and disposed sludge masses in kg per raw litre."""
    dose = feed["reagent_mol"]
    return {
        "softening_NaOH": dose["NaOH"] * MASS_KG_PER_MOL["NaOH"],
        "softening_Na2CO3": dose["Na2CO3"] * MASS_KG_PER_MOL["Na2CO3"],
        "softening_HCl31": dose["HCl"] * MASS_KG_PER_MOL["HCl31"],
        "sludge": feed["wet_sludge_kg"],
    }


def softening_prices(case):
    """The screen's reagent and wet-sludge prices in USD/kg."""
    price_column = "base_value" if case == "base" else "low_value"
    return {
        "softening_NaOH": float(
            _row(REAGENT_REGISTER, "assumption_id", "naoh_price")["base"]
        ),
        "softening_Na2CO3": float(
            _row(PRETREATMENT_REGISTER, "parameter", "pretreatment_soda_ash_usd_kg")[
                price_column
            ]
        ),
        "softening_HCl31": float(
            _row(REAGENT_REGISTER, "assumption_id", "hcl_price")["base"]
        ),
        "sludge": float(
            _row(PRETREATMENT_REGISTER, "parameter", "sludge_disposal_usd_kg")[
                price_column
            ]
        ),
    }


def strip_li_fraction():
    """Measured Li strip fraction at 2.5 M HCl (Zhang 2017, Fig. 6)."""
    observation = "data/reference/extraction_models/hbta_topo_li_na/observations.csv"
    return (
        float(
            _row(observation, "observation_id", "zhang2017_fig6_li_strip_2p5m_hcl")[
                "value"
            ]
        )
        / 100
    )


def costs_and_balances(
    feed, extraction, case, assumptions, scrub_na=None, *, nacl_coordinate=None
):
    """Labelled mass balances. Carbonation uses Ge measurements after the failed Pitzer falsifier.

    Basis: one L raw brine. Strip water recycle and unused-acid recovery are separate declared operations.
    Mother liquor recycles at fixed composition; its external purge closes steady-state salt/carbonate flows.
    """
    v, aq = feed["feed_l"], feed["aqueous_mol"]
    organic_volume = extraction.get("organic_volume_l_per_raw_l", v)
    x, loaded = extraction["raffinate"], extraction["loaded_organic"]
    c = assumptions["extractant_mol_l"]
    li, na, bta = (v * loaded[k] for k in ("li", "na", "hbta"))
    scrub_eff = assumptions["scrub_na_efficiency"] if scrub_na is None else scrub_na
    scrub_li = assumptions["scrub_li_loss"]
    strip_eff = strip_li_fraction()
    scrub_waste = {"Li": scrub_li * li, "Na": scrub_eff * na}
    scrub_acid = max(
        assumptions["scrub_hcl_mol_l"] * organic_volume / 10, sum(scrub_waste.values())
    )  # source O/A or stoichiometric acid floor
    stripped = {
        "Li": (li - scrub_waste["Li"]) * strip_eff,
        "Na": na - scrub_waste["Na"],
    }
    regen_waste = {"Li": (li - scrub_waste["Li"]) * (1 - strip_eff), "Na": 0.0}
    gross_strip_acid = 2.5 * organic_volume / 10
    bound = sum(stripped.values())
    if gross_strip_acid < bound:
        raise ValueError("Source strip acid inventory below metal proton demand")
    free_acid = (1 - assumptions["acid_recovery"]) * (gross_strip_acid - bound)
    strip_acid = bound + free_acid
    carbon = downstream.accounts(
        {
            "Li": stripped["Li"],
            "Na": stripped["Na"],
            "Cl": strip_acid,
            "C": 0.0,
            "H": free_acid,
        },
        assumptions,
        nacl_coordinate=nacl_coordinate,
    )
    net_water, water, nacl, s, mother, purge, product, carbonate = (
        carbon[k]
        for k in (
            "fresh_water_kg",
            "reactor_water_kg",
            "mother_nacl_mol_kg",
            "ge_li2co3_mol_kg",
            "mother_liquor_mol",
            "mother_purge_mol",
            "product_li2co3_mol",
            "carbonation_na2co3_mol",
        )
    )
    sap = extraction.get("saponification_mol_per_raw_l", 0.7 * c * v)
    retained = (
        "organic_loss" in assumptions
    )  # a retained #176 record replays on its own loss and price basis
    # #342: shared entrainment on the raffinate (v) and the scrub and strip liquors (organic/10 each), per circulated
    # organic litre; no purge (owner decision, 12 Nov 2026).
    solvent_loss = (
        assumptions["organic_loss"]
        if retained
        else assumptions["entrainment_l_per_l"] * (v / organic_volume + 2 / 10)
    )
    hbta_loss_mol = v * x["bta"] + solvent_loss * bta
    masses = softening_masses(feed) | {
        "saponification_NaOH": sap * MASS_KG_PER_MOL["NaOH"],
        "scrub_HCl31": scrub_acid * MASS_KG_PER_MOL["HCl31"],
        "strip_HCl31": strip_acid * MASS_KG_PER_MOL["HCl31"],
        "neutralization_NaOH": free_acid * MASS_KG_PER_MOL["NaOH"],
        "carbonation_Na2CO3": carbonate * MASS_KG_PER_MOL["Na2CO3"],
        "HBTA": hbta_loss_mol * MASS_KG_PER_MOL["HBTA"],
        "TOPO": solvent_loss * c * organic_volume * MASS_KG_PER_MOL["TOPO"],
        "diluent": solvent_loss
        * extraction["diluent_mol"]
        * MASS_KG_PER_MOL["diluent"],
    }
    front = feed.get("front_end")
    if front and front["kind"] == "sorbent":
        from prommis.lithium_extraction.process.front_end import prices as front_prices

        masses.update(
            front_sorbent=front["make_up_kg_per_raw_l"][case],
            front_water=front["fresh_water_l_per_raw_l"],
        )
    # #342: the active set (HBTA, TOPO) is lost mass, priced only from the register and kept outside the cost C_r.
    active = {} if retained else {k: masses.pop(k) for k in ("HBTA", "TOPO")}
    reg, pret = REAGENT_REGISTER, PRETREATMENT_REGISTER
    soft_prices = softening_prices(case)
    naoh, soda, hcl, sludge = (
        soft_prices[k]
        for k in ("softening_NaOH", "softening_Na2CO3", "softening_HCl31", "sludge")
    )
    prices = {
        k: (
            naoh
            if "NaOH" in k
            else soda if "Na2CO3" in k else hcl if "HCl" in k else sludge
        )
        for k in masses
    }
    prices["diluent"] = assumptions["diluent_usd_kg"]
    if retained:
        organic = _row(
            "data/reference/tea/hbta_topo_organic_cost_scenarios.csv",
            "scenario_id",
            "base_screening_case" if case == "base" else "low_screening_case",
        )
        prices.update(
            HBTA=float(organic["hbta_marginal_price_usd_kg"]),
            TOPO=float(organic["topo_marginal_price_usd_kg"]),
        )
    if front and front["kind"] == "sorbent":
        prices.update(
            front_sorbent=front_prices(case)["sorbent"],
            front_water=front_prices(case)["water"],
        )
    source_rows = {
        k: (
            f"{reg}:naoh_price"
            if "NaOH" in k
            else (
                f"{pret}:pretreatment_soda_ash_usd_kg:base_value"
                if "Na2CO3" in k and case == "base"
                else (
                    "Zhang 2026 Table 8:Na2CO3 200 USD/t"
                    if "Na2CO3" in k
                    else (
                        f"{reg}:hcl_price"
                        if "HCl" in k
                        else f"{pret}:sludge_disposal_usd_kg"
                    )
                )
            )
        )
        for k in masses
    }
    source_rows["diluent"] = (
        "EIA EER_EPJK_PF4_RGC_DPG Aug 2026:3.724 USD/US gal; jet-fuel analogue"
    )
    if retained:
        source_rows.update(
            HBTA=f"hbta_topo_organic_cost_scenarios.csv:{organic['scenario_id']}:HBTA; not a vendor quote",
            TOPO=f"hbta_topo_organic_cost_scenarios.csv:{organic['scenario_id']}:TOPO; not a vendor quote",
        )
    if front and front["kind"] == "sorbent":
        source_rows.update(
            front_sorbent="front_end_price_register.csv:Dong and Federal Reserve 2025",
            front_water="PrOMMiS water; USD_2021/L at 1 kg/L fresh water; face value",
        )
    breakdown = {k: masses[k] * prices[k] for k in masses}
    kg_li = 2 * product * 0.00694
    total = sum(breakdown.values())
    lce_factor = 0.07389 / (2 * 0.00694)
    outputs = {
        "Li": feed["cake_mol"]["Li"]
        + feed["solid_mol"]["Li"]
        + v * x["li"]
        + scrub_waste["Li"]
        + regen_waste["Li"]
        + 2 * product
        + purge["Li"],
        "Na": feed["cake_mol"]["Na"]
        + feed["solid_mol"]["Na"]
        + v * x["na"]
        + scrub_waste["Na"]
        + purge["Na"]
        + sap,
        "Ca": feed["cake_mol"]["Ca"] + feed["solid_mol"]["Ca"] + aq["Ca"],
        "C": feed["cake_mol"]["C"]
        + feed["solid_mol"]["C"]
        + aq["C"]
        + product
        + purge["C"],
        "BTA": v * x["bta"] + bta * solvent_loss + bta * (1 - solvent_loss),
    }
    # This retained downstream accounting still covers its original Li/Na/Ca/inorganic-C quantities.
    inputs = {e: feed["raw_mol"][e] for e in ("Li", "Na", "Ca", "C")} | {
        "BTA": c * organic_volume
    }
    original_li = feed["raw_mol"]["Li"]
    if front and front["kind"] == "sorbent":
        original = front["original_raw_elements_mol"]
        original_li = original["Li"]
        for e in ("Li", "Na", "Ca", "C"):
            outputs[e] += original[e] - feed["raw_mol"][e]
            inputs[e] = original[e]
    inputs["Na"] += (
        2 * feed["reagent_mol"]["Na2CO3"]
        + feed["reagent_mol"]["NaOH"]
        + 2 * sap
        + free_acid
        + 2 * carbonate
    )
    inputs["C"] += feed["reagent_mol"]["Na2CO3"] + carbonate
    closure = {e: inputs[e] - outputs[e] for e in inputs}
    if max(map(abs, closure.values())) > 1e-9:
        raise RuntimeError(f"Chain balances exceed 1e-9 mol/L raw brine: {closure}")
    soda_mass = sum(m for k, m in masses.items() if "Na2CO3" in k)
    breaks = []
    for lce_price in (17.5, 20.9, 25.0) if retained else ():
        extractant_mass = masses["HBTA"] + masses["TOPO"]
        revenue = kg_li * lce_factor * lce_price
        other_soda = total - soda_mass * soda
        other_extractant = total - breakdown["HBTA"] - breakdown["TOPO"]
        breaks.append(
            {
                "lce_price_usd_kg": lce_price,
                "soda_ash_usd_kg": (revenue - other_soda) / soda_mass,
                "equal_hbta_topo_usd_kg": (revenue - other_extractant)
                / extractant_mass,
                "hbta_loss_kg_m3_at_fixed_topo_loss": 1000
                * (revenue - total + breakdown["HBTA"])
                / prices["HBTA"],
            }
        )
    return {
        **carbon,
        "case": case,
        "basis": "one L raw brine; multiply stream mol and cost by 1000 for one m3",
        "inputs_mol": inputs,
        "outputs_mol": outputs,
        "closure_mol": closure,
        "assumptions": assumptions,
        "scrub_na_efficiency": scrub_eff,
        "scrub_li_loss_fraction": scrub_li,
        "strip_li_fraction": strip_eff,
        "scrub_acid_mol": scrub_acid,
        "gross_strip_acid_mol": gross_strip_acid,
        "fresh_strip_acid_mol": strip_acid,
        "neutralization_naoh_mol": free_acid,
        "stripped_mol": stripped,
        "strip_na_fraction": stripped["Na"] / sum(stripped.values()),
        "product_li2co3_mol": product,
        "ge_li2co3_mol_kg": s,
        "mother_liquor_mol": mother,
        "mother_purge_mol": purge,
        "mother_nacl_mol_kg": nacl,
        "reactor_water_kg": water,
        "fresh_water_kg": net_water,
        "strip_water_recycle_fraction": 1 - net_water / (v / 10),
        "evaporation_alternative_mj_m3_latent_only": max(0, v / 10 - net_water)
        * 1000
        * 2.257,
        "mass_kg_per_l": masses,
        "prices_usd_kg": prices,
        "price_source_rows": source_rows,
        "cost_usd_m3": {k: x * 1000 for k, x in breakdown.items()},
        "cost_usd_kg_li": {
            k: x / kg_li if kg_li else None for k, x in breakdown.items()
        },
        "operating_usd_kg_li": total / kg_li if kg_li else None,
        "li_product_kg_m3": kg_li * 1000,
        "raw_li_recovery": 2 * product / original_li,
        "break_even_lce_usd_kg": total / kg_li / lce_factor if kg_li else None,
        "break_even_li_usd_kg": total / kg_li if kg_li else None,
        "price_break_evens": breaks,
        **({} if retained else rule_accounts(case, active, total, kg_li)),
    }


def rule_accounts(case, active, cost, kg_li):
    """#339 decision 1 terms per m3 raw water from per-litre amounts: q, C_r, L and the affordable price theta at
    this design; published prices of the active set are listed beside it, never filled in.
    """
    from prommis.lithium_extraction.process.costing import (
        CARBONATE_USD_KG,
        LI2CO3_PER_LI,
        register_price,
    )

    q, loss = 1000 * kg_li * LI2CO3_PER_LI, 1000 * sum(active.values())
    return {
        "active_mass_kg_per_l": active,
        "active_prices_usd_kg": {k: register_price(k) for k in active},
        "carbonate_kg_m3": q,
        "cost_outside_active_usd_m3": 1000 * cost,
        "active_loss_kg_m3": loss,
        "carbonate_price_usd_kg": CARBONATE_USD_KG[case],
        "affordable_price_usd_kg": (CARBONATE_USD_KG[case] * q - 1000 * cost) / loss,
    }
