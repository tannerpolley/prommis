"""#201 records and feed quantities only; no equilibrium or process model is built."""

import json
import math

from prommis.lithium_extraction.paths import REPO_ROOT
from prommis.lithium_extraction.process import waters as brines

SHARED = "analyses.shared.hbta_topo_configured_reactive_lle.scripts"
D2 = "analyses.prototypes.d2ehdtpa_buphen_ms2_prototype.scripts"
SAPONIFIED = "lithium_extraction.prototypes.hbta_topo_ms2_saponified"
FRESH = "lithium_extraction.v3.hbta_topo_ms2_extraction"
PACKAGE = "lithium_extraction.prototypes.produced_water_sx_prototype"
CASES = ("base", "optimistic")
STAGES = (1, 2, 3, 4, 5)
REGISTER = "data/reference/tea/extractant_price_register.csv"
SCREENING = "data/reference/tea/hbta_topo_screening_cost_assumptions.csv"
LOSS = "data/reference/tea/solvent_entrainment.csv"


def stage_adapter(name):
    from prommis.lithium_extraction.extraction import (
        d2ehdtpa_buphen_stage as d2_stage,
        fresh_organic as fresh,
        set_a,
        set_c,
    )

    return {"d2_stage": d2_stage, "fresh": fresh, "set_a": set_a, "set_c": set_c}[name]


# A solvent record (#341): its chemistry owners, stage adapter, design variables and bounds, required pretreatment,
# scrub/strip/cleanup data, dissolution loss, price rows and labels. `candidate.jobs` and `candidate.build` read these
# instead of solvent names. Values are those the owners already use; locator strings name the owner of each value.
HBTA_TOPO_PROCESS = {
    "pretreatment": "whole-water softening",  # flowsheet.washed_feed; extraction-only removes divalents at no cost
    "downstream": f"{PACKAGE}.full_process",
    "scrub_strip": "case assumptions scrub_hcl_mol_l, scrub_li_loss, scrub_na_efficiency, acid_recovery; "
    "strip fraction data/reference/extraction_models/hbta_topo_li_na/observations.csv:"
    "zhang2017_fig6_li_strip_2p5m_hcl (flowsheet.costs_and_balances)",
    "cleanup": "none: divalents leave in whole-water softening",
    "dissolution_loss": f"{PACKAGE}.flowsheet:costs_and_balances: predicted raffinate BTA plus the shared {LOSS} "
    "entrainment of HBTA, TOPO and diluent on raffinate, scrub and strip liquors",
    # #339 decision 1: extractant and synergist, counted as the purchased compound; diluent stays in C_r.
    "active_set": ("HBTA", "TOPO"),
    "prices": f"{REGISTER}: HBTA, TOPO (unpriced), sulfonated_kerosene; {SCREENING}: naoh_price, hcl_price; "
    "data/reference/tea/pretreatment_cost_assumptions.csv: soda ash, sludge (flowsheet.costs_and_balances)",
}
D2_PROCESS = {
    "stage_adapter": "d2_stage",
    # The #341 campaign keeps the fixed five-stage design; the #342 rule releases `bounds` at every rule stage count.
    "design": {
        "optimized": False,
        "stages": (5,),
        "scopes": {"extraction_only": ("base",), "full_process": CASES},
        # #188/#214 retained design starts every solve; O/A 1.25 at Mars Hill (#214).
        "fixed": {"B": 1.0, "S": 1.2, "O_A": 1.0, "Sr_decade": 0.0},
        "site_fixed": {"MarsHill": {"O_A": 1.25}},
        # #188 design range, released by #342: BuPhen M, site ratio per feed Li, O/A; organic sites M.
        "bounds": {"B": (0.25, 1.0), "S": (0.9, 1.5), "O_A": (0.5, 2.0)},
        "sites_mol_l": (0.020, 0.060),
        "li_target": 0.95,
        "sodium_limit": None,
    },
    "variants": ("fitted",),
    "pretreatment": "none",
    "nominal_acidity_input": False,  # calcium-tolerant: raw water enters extraction
    "downstream": f"{PACKAGE}.d2_full_process",
    "scrub_strip": f"{PACKAGE}.d2ehdtpa_buphen:accounts (Raiguel 2024 strip; no scrub)",
    "cleanup": f"{PACKAGE}.d2_full_process:cleanup (declared ideal Mg/Ca/Sr removal from strip liquor)",
    "dissolution_loss": f"{PACKAGE}.d2ehdtpa_buphen:accounts: octanol dissolution, BuPhen dissolution from its "
    f"partition value, and the shared {LOSS} entrainment of octanols, BuPhen, D2EHDTPA and "
    "dodecane on raffinate and strip liquor",
    "active_set": ("BuPhen", "D2EHDTPA"),
    "prices": f"{REGISTER}: BuPhen, D2EHDTPA (both unpriced), 1_octanol, 2_octanol, n_dodecane; {SCREENING}: "
    f"naoh_price, hcl_price; {PACKAGE}.d2_full_process:accounts soda ash, sludge",
}
SOLVENTS = {
    "fresh": {
        "owner": FRESH,
        "feed_owner": f"{FRESH}.live_equilibrium:FEED_MATRIX/FEED_OFFSET",
        "chemistry_owner": f"{FRESH}.live_equilibrium",
        "conservation_owner": f"{SHARED}.independent_network:_basis_matrix",
        "certificate_owner": f"{SHARED}.independent_network",
        "domain_owner": "data/processed/hbta_topo_ms2_engine_comparison/greenfield_current_v3/loaded_state_surrogate.json",
        "domain_basis": "current loaded-organic surrogate numerical interpolation; not physical calibration",
        "labels": ["#158 fresh-solvent recovery discrepancy"],
        "name": "Fresh HBTA/TOPO",
        "stage_adapter": "fresh",
        "starting_record": "results/fresh_r1.json",
        "saponified": False,
        "free_extractant": "free-carrier",
        "nominal_acidity_input": True,
        "design": {
            "optimized": True,
            "stages": STAGES,
            "scopes": {"extraction_only": CASES, "full_process": CASES},
            "bounds": {
                "organic_to_aqueous_ratio": (0.5, 1.5),
                "extractant_mol_l": (0.0075, 0.030),
                "saponification_ratio": (0.0, 0.0),
            },
            "li_target": 0.95,
            "sodium_limit": 0.02,
        },
        "variants": ("fitted",),
        **HBTA_TOPO_PROCESS,
    },
    "A": {
        "owner": SAPONIFIED,
        "feed_owner": f"{SHARED}.mechanism_candidates:inventory",
        "chemistry_owner": f"{SAPONIFIED}.stage_chemistry",
        "conservation_owner": f"{SHARED}.mechanism_candidates:MOIETIES",
        "certificate_owner": f"{SHARED}.mechanism_candidates:is_strict",
        "domain_owner": "lithium_extraction/prototypes/hbta_topo_ms2_saponified/README.md",
        "labels": ["set A cannot evaluate a state requiring hydroxide"],
        "name": "Saponified HBTA/TOPO, set A",
        "stage_adapter": "set_a",
        "starting_record": "results/set_a_r2.json",
        "saponified": True,
        "free_extractant": "hbta",
        "nominal_acidity_input": True,
        "design": {
            "optimized": True,
            "stages": STAGES,
            "scopes": {"extraction_only": CASES},
            "bounds": {
                "organic_to_aqueous_ratio": (0.5, 1.5),
                "extractant_mol_l": (0.015, 0.030),
                "saponification_ratio": (0.70, 0.95),
            },
            "li_target": None,
            "sodium_limit": None,
        },
        "variants": ("fitted",),
        **HBTA_TOPO_PROCESS,
    },
    "C": {
        "owner": SAPONIFIED,
        "feed_owner": f"{SHARED}.mechanism_candidates:inventory",
        "chemistry_owner": f"{SHARED}.mechanism_candidates:network/solve",
        "conservation_owner": "analyses.prototypes.hbta_topo_ms2_prototype.scripts.countercurrent",
        "certificate_owner": f"{SHARED}.mechanism_candidates:is_strict",
        "domain_owner": "analyses/prototypes/hbta_topo_ms2_prototype/README.md",
        "labels": [
            "support selected per stage by existing owner",
            "several fixed stage supports share Engine #452 records",
            "#158 recovery requires unsupported aqueous HBTA dissolution",
        ],
        "name": "Saponified HBTA/TOPO",
        "stage_adapter": "set_c",
        "support_rebuild": True,
        "saponified": True,
        "free_extractant": "hbta",
        "nominal_acidity_input": False,
        # Li/Na exchange fits: low = single-stage fit (lead), high = cascade fit (sensitivity).
        "variants": ("low", "high"),
        "constants": "analyses/prototypes/hbta_topo_ms2_prototype/results/set_c_summary.json",
        "design": {
            "optimized": True,
            "stages": STAGES,
            "scopes": {"extraction_only": CASES, "full_process": CASES},
            # Retained #158/#216 design: fixed solves and the held optimum every released solve starts from (#342).
            "fixed": {"organic_to_aqueous_ratio": 1.0, "saponification_ratio": 0.70},
            # Owner decision O6 (#339 decision 1): each fit's supported range, within 0.30 decade on the data that
            # vary the variable, rounded inward. `bounds` is the single-stage (lead) fit, `fit_bounds` the others.
            # O/A upper: calibration check, Zhang 2020 Fig. 7 (0.27 decade at 2.0); O/A lower: numerical, the
            # set_c.build_flowsheet accepted range (#216). Saponification: calibration, Zhang 2020 Table 2
            # measured 48.3-81.0 %; cascade fit defined at 70 % (Zhang 2021 Table 9).
            "bounds": {
                "organic_to_aqueous_ratio": (0.5, 2.0),
                "extractant_mol_l": (0.1, 0.4),
                "saponification_ratio": (0.483, 0.809),
            },
            "fit_bounds": {
                "high": {
                    "organic_to_aqueous_ratio": (0.5, 1.0),
                    "saponification_ratio": (0.483, 0.700),
                }
            },
            "li_target": 0.95,
            "sodium_limit": None,
        },
        **HBTA_TOPO_PROCESS,
    },
    "D2EHDTPA-Held": {
        "owner": "lithium_extraction.prototypes.produced_water_sx_prototype.d2ehdtpa_buphen",
        "feed_owner": f"{D2}.reaction_coordinates:aqueous_feed",
        "chemistry_owner": f"{D2}.reaction_coordinates",
        "conservation_owner": f"{D2}.reaction_coordinates:MOIETIES",
        "certificate_owner": f"{SHARED}.mechanism_candidates:is_strict",
        "domain_owner": f"{D2}.reaction_coordinates:payload",
        "labels": [
            "Held mixed-Mg discrepancy remains unresolved",
            "Na and formulation transfer not supported (F2)",
        ],
        "name": "D2EHDTPA/BuPhen, Held ion parameters",
        "cell_id": {
            "extraction_only": "D2Held_{site}",
            "full_process": "D2Held_{site}_full_process_{case}",
        },
        "ion_basis": "held_explicit_sr",
        "strip_lumped": {},
        "starting_record": "results/ms2_0908.json",
        **D2_PROCESS,  # under d2ehdtpa_buphen.ANALYSIS
    },
    "D2EHDTPA-Figiel": {
        "owner": f"{D2}.conditional_figiel_packet",
        "feed_owner": f"{D2}.conditional_figiel_packet:lumped_feed",
        "chemistry_owner": f"{D2}.conditional_figiel_packet",
        "conservation_owner": f"{D2}.reaction_coordinates:MOIETIES (Sr lumped into Ca)",
        "certificate_owner": f"{SHARED}.mechanism_candidates:is_strict",
        "domain_owner": f"{D2}.conditional_figiel_packet:packet_payload",
        "labels": [
            "provisional conditional basis (#197)",
            "binary salts only; Sr counted as Ca",
            "Na and formulation transfer not supported (F2)",
        ],
        "name": "D2EHDTPA/BuPhen, Figiel ion parameters",
        "cell_id": {
            "extraction_only": "D2Figiel_{site}",
            "full_process": "D2Figiel_{site}_full_process_{case}",
        },
        "ion_basis": "packet",
        "strip_lumped": {
            "strip_Sr_mol_L_feed": 0.0
        },  # Figiel's ion basis lumps Sr into Ca
        "starting_record": "results/conditional_figiel_packet/reference_packet_fig7_exchange.json",
        **D2_PROCESS,
    },
    "TOP-FeCl3": {"owner": None, "unavailable": "not modeled (#193)"},
}

SCOPES = {
    "extraction_only": {
        "steps": ("saponification if used", "extraction"),
        "cost_items": (
            "organic_makeup",
            "stage_operation",
            "electricity",
            "saponification_NaOH",
        ),
        "li_denominator": "Li transferred to loaded organic (kg/y)",
        "feed_basis": "raw brine after unsupported divalent/equal-Cl removal; fresh/A nominal acidity input",
        "excluded": (
            "capital",
            "labour",
            "unquantified nominal acidity-conditioning HCl and buffering",
        ),
    },
    "full_process": {
        "steps": (
            "softening if required",
            "saponification",
            "extraction",
            "scrub",
            "strip",
            "concentration",
            "carbonation",
        ),
        "cost_items": (
            "softening_NaOH",
            "softening_Na2CO3",
            "softening_HCl",
            "sludge",
            "saponification_NaOH",
            "scrub_HCl",
            "strip_HCl",
            "neutralization_NaOH",
            "carbonation_Na2CO3",
            "HBTA_loss",
            "TOPO_loss",
            "diluent_loss",
        ),
        "li_denominator": "Li in Li2CO3 product (kg/y)",
        "feed_basis": "extended softened, washed outlet",
        "excluded": ("capital", "labour", "energy"),
    },
}

# Species names are transport labels, not a new reaction or element-conservation model.
SPECIES_LABELS = {
    "water": "water",
    "lithium-cation": "Li",
    "sodium-cation": "Na",
    "hydrogen-cation": "H",
    "hydroxide-anion": "OH",
    "bta-anion": "BTA",
    "chloride-anion": "Cl",
    "potassium-cation": "K",
    "magnesium-cation": "Mg",
    "calcium-cation": "Ca",
    "strontium-cation": "Sr",
}


def bounds(solvent, end=None, rule=False):
    """Decision bounds of a solvent record for one exchange fit (`end`). Outside a #342 rule job the record's retained
    fixed design is held (the #216 campaign, app and tutorial basis)."""
    design = SOLVENTS[solvent]["design"]
    if not rule:
        return design["bounds"] | {
            k: (v, v) for k, v in design.get("fixed", {}).items()
        }
    return design["bounds"] | design.get("fit_bounds", {}).get(end, {})


def extraction_feed(
    brine,
    solvent,
    scope="extraction_only",
    *,
    process_pH=brines.PROCESS_PH,
    synthetic_softened=False,
):
    """Physical feed and signed model substitution, mol per litre of raw brine.

    Fresh and saponified HBTA share the same aqueous inventory at zero incoming organic
    loading/base/BTA; the shared owner supplies it without importing the v3 bucket.
    """
    record = SOLVENTS[solvent]
    SCOPES[scope]
    if "unavailable" in record:
        return {"unavailable": record["unavailable"]}
    softened = record["pretreatment"] == "whole-water softening"
    if scope == "full_process" and softened:
        return {
            "unavailable": "full process requires the complete washed outlet; set C uses set_c.washed_feed"
        }
    if not math.isfinite(process_pH):
        raise ValueError("process pH must be finite")
    if brine["water_kg"] is None and not synthetic_softened:
        return {
            "unavailable": "physical water inventory unavailable: SG or TDS not reported"
        }
    if synthetic_softened:
        physical = {e: brine["mol"][e] for e in ("Li", "Na", "K")}
        physical.update(water=55.508, H=10**-process_pH)
        physical["Cl"] = sum(physical[e] for e in ("Li", "Na", "K", "H"))
    else:
        physical = dict(
            brine["mol"], water=brine["water_kg"] / brines.WATER_MOLAR_MASS_KG
        )
    if any(not math.isfinite(n) or n < 0 for n in physical.values()):
        raise ValueError("physical feed must be finite and nonnegative")
    removed = {}
    if softened:
        removed = {e: physical.get(e, 0.0) for e in ("Mg", "Ca", "Sr", "Ba")}
        removed["Cl"] = sum(2 * n for n in removed.values())
        if physical["Cl"] < removed["Cl"]:
            raise ValueError(
                "chloride cannot cover the divalent charge removed at no cost"
            )
        for e, n in removed.items():
            physical[e] = physical.get(e, 0.0) - n
        from prommis.lithium_extraction.thermodynamics import mechanism_candidates as mc

        state = mc.State(
            identity="modular extraction inlet",
            temperature_k=298.15,
            organic_to_aqueous=1.0,
            li=physical["Li"],
            na=physical["Na"],
            hydrogen=10**-process_pH,
            hbta=0.0,
            topo=0.0,
        )
        aqueous, _ = mc.inventory(state)
    else:
        from prommis.lithium_extraction.thermodynamics import (
            d2ehdtpa_buphen_reactions as rc,
        )

        cations = {e: physical.get(e, 0.0) for e in rc.ION if e != "Cl"}
        if synthetic_softened:
            cations["K"] = 0.0  # #216: unchanged KCl remains in physical accounting.
        if record["ion_basis"] == "packet":
            from prommis.lithium_extraction.thermodynamics.d2ehdtpa_buphen_figiel_chloride import (
                lumped_feed,
            )

            cations = lumped_feed(cations)
        aqueous = rc.aqueous_feed(cations)
    model = {SPECIES_LABELS[e]: n for e, n in aqueous.items()}
    if synthetic_softened:
        removed = {e: brine["mol"].get(e, 0) for e in ("Mg", "Ca", "Sr", "Ba")}
        removed["Cl"] = 2 * sum(removed.values())
    conditioning = (
        {"H": physical["H"], "Cl": physical["H"]} if synthetic_softened else {}
    )
    if record["nominal_acidity_input"] and not synthetic_softened:
        # #201 decision 11 supplies the acidity inventory already used by the owner, without changing its feed.
        conditioning = {"H": model["H"], "Cl": model["H"]}
        for e, n in conditioning.items():
            physical[e] = physical.get(e, 0.0) + n
    adjustment = {
        e: model.get(e, 0.0) - physical.get(e, 0.0)
        for e in physical.keys() | model.keys()
    }
    labels = brine["labels"] + record["labels"]
    quantities = (
        ("projected Li", model["Li"], "mol/L aqueous", "li_mol_l_aqueous"),
        ("projected Na", model["Na"], "mol/L aqueous", "na_mol_l_aqueous"),
        ("process pH", process_pH, "dimensionless", "hydrogen_mol_l_aqueous"),
    )
    for quantity, value, unit, _ in quantities:
        labels.append(
            f"{quantity} = {value:.17g} {unit}: physical calibration coverage unverified for arbitrary brine; "
            f'owner {record["chemistry_owner"]}'
        )
    if "domain_basis" in record:
        bounds = json.loads((REPO_ROOT / record["domain_owner"]).read_text())[
            "input_bounds"
        ]
        for quantity, value, unit, key in quantities:
            compared = model["H"] if quantity == "process pH" else value
            low, high = bounds[key]
            position = "inside" if low <= compared <= high else "outside"
            hydrogen = (
                f" (model H = {compared:.17g} mol/L aqueous)"
                if quantity == "process pH"
                else ""
            )
            labels.append(
                f"{quantity} = {value:.17g} {unit}{hydrogen}: {position} current loaded-organic surrogate "
                f"numerical interpolation interval [{low:.17g}, {high:.17g}] mol/L aqueous; "
                f'owner {record["domain_owner"]}; not physical calibration coverage'
            )
    elif not softened:
        labels.append(
            f"process pH = {process_pH:.17g} dimensionless: H/pH not represented by "
            f'existing feed chemistry; owner {record["feed_owner"]}'
        )
    if record["nominal_acidity_input"]:
        labels += [
            "record feed uses raw-brine mass basis; R1/R2 retain their original nominal-feed conversions",
            "nominal H and equal-Cl conditioning input: existing model H amount on one nominal "
            "aqueous litre per raw litre; not measured inventory, activity-based pH or titrated acid demand",
            "nominal acidity-conditioning HCl cost and buffering unquantified and excluded; "
            "cost completeness covers only listed extraction and saponification accounts",
        ]
    if not softened:
        labels += [
            "changed model input: raw-brine source/mass basis; R6 keeps the original #188 feed and site basis"
        ]
    return {
        "basis": "mol per raw litre",
        "physical_feed": physical,
        "model_feed": model,
        "model_composition_adjustment": adjustment,
        "removed_at_no_cost": removed,
        "assumed_conditioning_input_mol_per_raw_l": conditioning,
        "synthetic_softened": synthetic_softened,
        "dropped_components": (
            ["Br", "SO4", "B", "H2S", "Fe", "Mn", "Zn", "NPOC"]
            if synthetic_softened
            else []
        ),
        "water_mass_basis": (
            "55.508 mol water per nominal litre (#216)"
            if synthetic_softened
            else "Reaktoro pitzer.dat H2O molar mass 0.018016 kg/mol"
        ),
        "process_pH": process_pH,
        "pH_basis": (
            brine["pH_basis"]
            if softened
            else "process assumption; H/pH is not represented by the existing D2EHDTPA feed owner"
        ),
        "labels": labels,
    }


def scope_costs(scope, lines, *, annual_li_kg, annual_raw_brine_m3):
    """Account already-priced USD/y lines; never present a partial sum as a total."""
    record = SCOPES[scope]
    if any(not math.isfinite(n) or n <= 0 for n in (annual_li_kg, annual_raw_brine_m3)):
        raise ValueError(
            "annual Li and raw-brine denominators must be finite and positive"
        )
    unexpected = lines.keys() - set(record["cost_items"])
    if unexpected:
        raise ValueError(f"cost items outside {scope}: {sorted(unexpected)}")
    missing, subtotal = [], 0.0
    for item in record["cost_items"]:
        line = lines.get(item, {})
        amount = line.get("usd_y")
        if amount is None or not line.get("price_source"):
            missing.append(item)
        else:
            if not math.isfinite(amount) or amount < 0:
                raise ValueError(f"{item}: annual cost must be finite and nonnegative")
            subtotal += amount
    return {
        "lines": lines,
        "li_denominator": record["li_denominator"],
        "excluded": record["excluded"],
        "missing_items": missing,
        "completeness": "partial: " + ", ".join(missing) if missing else "complete",
        "completeness_basis": "listed scope accounts only; exclusions remain unpriced",
        "priced_subtotal_usd_y": subtotal,
        "total_usd_y": None if missing else subtotal,
        "priced_subtotal_usd_kg_li": subtotal / annual_li_kg,
        "priced_subtotal_usd_m3_raw_brine": subtotal / annual_raw_brine_m3,
        "can_minimize_total": not missing,
    }
