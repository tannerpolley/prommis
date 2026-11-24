"""Helpers for the tutorial's whole process and screen: waters, softening, the extraction feed and design, downstream
connections, figures, the second water, the paper's screen results and the check of the worked example against its
stored results.

Every function that changes the flowsheet takes the model `m` as its first argument.
"""

import json
import logging
import math
import time

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Patch
from pyomo.environ import (
    Constraint,
    Expression,
    NonNegativeReals,
    Var,
    units as pyunits,
    value,
)
from pyomo.network import Arc

from lithium_tutorial_tools import ROOT, label, log_directory, number, plot_checks
from prommis.lithium_extraction.extraction import countercurrent as cc
from prommis.lithium_extraction.thermodynamics import mechanism_candidates as mc
from prommis.lithium_extraction.thermodynamics.reaction_networks import load_constants
from prommis.lithium_extraction.process import (
    waters as brines,
    costing,
    strip_and_carbonation as downstream,
    flowsheet,
    whole_process as full_process,
    optimization,
)
from prommis.lithium_extraction.extraction import set_c, stage
from prommis.lithium_extraction.process.candidate import fixed_disposition

from prommis.lithium_extraction.process.priced_designs import (
    FITS as PRICED_FITS,
    PRIMARY,
    SECONDARY,
    WATERS as PRICED_WATERS,
    load_matrix,
    selected_designs,
)

# IDAES warns that some cost flows are expressions without a zero lower bound; optimization.report checks every
# cost flow is nonnegative after the solve.
logging.getLogger("idaes.core.base.costing_base").setLevel(logging.ERROR)

STUDY = ROOT / "analyses/prototypes/modular_process"
BAND = json.loads((STUDY / "results/targets/exchange_band.json").read_text())
SCREEN = STUDY / "results/priced_margin"
WORKED_EXAMPLE = (
    "Worked example: cheapest design, 1 % organic loss per pass, HBTA at 100 USD/kg"
)
WATERS = {"MS2": "MS-2", "MarsHill": "Mars Hill"}
CASE, FIT = "base", "low"  # base-case prices; 'low' is the single-stage exchange fit
ACCEPTED = {
    site: next(
        g[FIT] for g in BAND["groups"] if g["site"] == site and g["case"] == CASE
    )
    for site in WATERS
}
LN_K = load_constants(
    ROOT / "analyses/prototypes/hbta_topo_ms2_prototype/results/set_c_summary.json"
)["ends"][FIT]["ln_k"]
LCE_PER_LI = BAND["lce_to_li_mass_ratio"]  # kg Li2CO3 per kg Li
LI_G_MOL = 6.94  # g/mol, the costing module's lithium molar mass
SAPONIFIED = 0.7  # fraction of HBTA in sodium form, fixed in the campaign


# ---------------------------------------------------------------- inputs


def waters():
    """Measured analysis and selected design of each water."""
    rows = {}
    for site, name in WATERS.items():
        raw, a = brines.raw_feed(site), ACCEPTED[site]
        rows[name] = {
            **{
                f"{ion} (mg/L)": raw["printed_mg_l"][ion]
                for ion in ("Li", "Na", "Ca", "Mg", "Cl")
            },
            "(Ca+Mg)/Li (kg/kg)": raw["ca_mg_li_mass_ratio"],
            "selected stages": a["stages"],
            "HBTA (mol/L organic)": a["extractant_mol_l"],
            "O/A": a["organic_to_aqueous_ratio"],
            "saponified fraction": a["saponification_ratio"],
        }
    return pd.DataFrame(rows).T


def soften(site):
    """Read the target-dose Reaktoro state used by the campaign."""
    return json.loads(
        (
            STUDY / f"results/softening_to_specification/softening/{site}_{CASE}.json"
        ).read_text()
    )


def softening_table(softened):
    kg = flowsheet.MASS_KG_PER_MOL
    rows = {}
    for site, soft in softened.items():
        w = soft[CASE]
        rows[WATERS[site]] = {
            "NaOH (kg/m3 raw)": 1000 * w["reagent_mol"]["NaOH"] * kg["NaOH"],
            "soda ash (kg/m3 raw)": 1000 * w["reagent_mol"]["Na2CO3"] * kg["Na2CO3"],
            "HCl, 31% (kg/m3 raw)": 1000 * w["reagent_mol"]["HCl"] * kg["HCl31"],
            "wet sludge (kg/m3 raw)": 1000 * w["wet_sludge_kg"],
            "softened water (L per L raw)": w["feed_l"],
            "Ca+Mg left (mg/L)": w["ca_mg_mg_l"],
            "Li kept in softened water (%)": 100
            * w["aqueous_mol"]["Li"]
            / w["raw_mol"]["Li"],
        }
    return pd.DataFrame(rows).T


def design(site):
    """The selected design: O/A 1, 70% saponified, the selected HBTA concentration and the single-stage fit."""
    return dict(
        organic_to_aqueous_ratio=1.0,
        extractant_mol_l=ACCEPTED[site]["extractant_mol_l"],
        saponification_ratio=SAPONIFIED,
        ln_k=LN_K,
    )


def starting_profile(site, softened):
    """Stage-by-stage countercurrent solution with the Engine alone: the flowsheet's starting point."""
    w, a = softened[site][CASE], ACCEPTED[site]
    return cc.countercurrent(
        LN_K,
        reaction_set="C",
        stage_count=a["stages"],
        organic_to_aqueous=1.0,
        extractant=a["extractant_mol_l"],
        saponification_fraction=SAPONIFIED,
        feed_li=w["aqueous_mol"]["Li"] / w["feed_l"],
        feed_na=w["aqueous_mol"]["Na"] / w["feed_l"],
        feed_acidity=cc.acidity_at_ph(w["pH"]),
    )


def stage_amounts(start):
    """Amount of each species leaving each stage in the starting profile (both phases summed per species)."""
    return [
        {s: sum(p.get(s, 0.0) for p in phases) for s in set_c.SPECIES}
        for phases in start["start"]["guesses"]
    ]


def stage_chemistries(site, start):
    """One compiled Engine record per extraction stage, anchored at that stage's starting composition.

    Set C switches water autoprotolysis on or off stage by stage; stages with the same active reactions share a record.
    """
    c, records, chemistries = ACCEPTED[site]["extractant_mol_l"], {}, []
    for row, guess in zip(start["stages"], start["start"]["guesses"], strict=True):
        anchor = cc.stage_state(
            "generic set C anchor",
            organic_to_aqueous=1.0,
            extractant=c,
            li=row["li_in_mol_l"],
            na=row["na_in_mol_l"],
            acidity=row["acidity_in_mol_l"],
            bta=row["bta_in_mol_l"],
            li_org=row["li_org_in_mol_l"],
            na_org=row["na_org_in_mol_l"],
            hbta=row["hbta_org_in_mol_l"],
        )
        active = row["water_autoprotolysis_active"]
        if active not in records:
            records[active] = set_c._chemistry(
                anchor, mc.solve(anchor, "C", LN_K, guess=guess), LN_K, 1.0, c
            )
        chemistries.append(records[active])
    return chemistries


def feed_extraction(m, site, washed):
    """Fix the softened water and the fresh organic (70% of HBTA in sodium form) entering the stages.

    Amounts are mol per litre of softened water at O/A = 1; returns them in mmol/L.
    """
    c, v = ACCEPTED[site]["extractant_mol_l"], washed["feed_l"]
    boundary = cc.stage_state(
        "generic set C boundary",
        organic_to_aqueous=1.0,
        extractant=c,
        li=washed["aqueous_mol"]["Li"] / v,
        na=washed["aqueous_mol"]["Na"] / v,
        acidity=washed.get("feed_acidity", cc.acidity_at_ph(washed["pH"])),
        na_org=SAPONIFIED * c,
    )
    aqueous, organic = mc.inventory(boundary)
    organic["diluent"] = (
        mc.model(set_c.SPECIES)
        .state(
            cc.TEMPERATURE_K,
            P=mc.PRESSURE_PA,
            phase="liquid",
            x=tuple(float(s == "diluent") for s in set_c.SPECIES),
        )
        .molar_density
        * 0.001
    )  # mol diluent per L organic
    for s in set_c.SPECIES:
        m.fs.boundary[s].fix(aqueous.get(s, 0.0) + organic.get(s, 0.0))
    m._feeds = (dict(aqueous), dict(organic))
    return (
        pd.DataFrame({"softened water": aqueous, "fresh organic": organic})
        .fillna(0.0)
        .mul(1000)
        .rename(index=label)
        .rename_axis("mmol per L of softened water")
    )


def add_extractant_inventory(m, site):
    """TOPO in the Engine record follows the extractant (O/A x c), and no stage may leave negative free HBTA or TOPO."""
    fs = m.fs
    fs.parameters["topo_total_mol"].unfix()
    fs.topo_quantity = Constraint(
        expr=fs.parameters["topo_total_mol"]
        == design(site)["organic_to_aqueous_ratio"] * fs.parameters["extractant_mol_l"]
    )

    def free_ligand_nonnegative(_, k, s):
        chemistry = m._chemistries[k - 1]
        j = chemistry.record_species.index(s)
        return (
            chemistry.feed_offset[j]
            + sum(
                a * m._incoming(k, q)
                for a, q in zip(
                    chemistry.feed_matrix[j], chemistry.control_species, strict=True
                )
            )
            >= 0
        )

    fs.affine_capacity = Constraint(
        fs.stages, ("hbta", "topo"), rule=free_ligand_nonnegative
    )


def add_design(m, site, washed):
    """The design variables (O/A, c, saponified fraction), held fixed, and the fresh organic written in terms of them."""
    m._design, m._washed, m._physical_feed = (
        design(site),
        washed,
        set_c.washed_feed(washed),
    )
    optimization.decisions(m, "C")
    return pd.Series(
        {name: value(v) for name, v in m._decisions.items()}, name="value"
    ).to_frame()


def check_extraction(m):
    """Re-solve every stage directly with the Engine and confirm it keeps the same active reactions."""
    result = set_c.report(m, stage.report(m))
    assert result[
        "strict_certificate"
    ], "a stage outlet differs from its direct Engine solution"
    assert all(
        r["certificate"]["support_matches"] for r in result["stages"]
    ), "a stage changed its active reactions"
    return result


def assumptions(site):
    """Downstream assumptions of the retained whole-process record (scrub, strip, organic loss, carbonation)."""
    return json.loads((STUDY / f"results/full_process/{site}_{CASE}.json").read_text())[
        "full_process"
    ]["assumptions"]


def downstream_alternatives(m, site, softened):
    """Every scrub-acid regime and carbonation-salt interval checked; returns the first admissible one and a summary."""
    rows = full_process.enumerate_units(
        softened[site][CASE],
        full_process.extraction(m)["loaded_organic"],
        assumptions(site),
        organic_to_aqueous_ratio=1.0,
    )
    admissible = [r for r in rows if r["feasible"]]
    candidate = admissible[0]
    regime = {"source": "source ratio", "stoichiometric": "stoichiometric demand"}
    return candidate, pd.Series(
        {
            "combinations checked": len(rows),
            "admissible combinations": len(admissible),
            "selected: scrub acid set by": regime.get(
                candidate["scrub_regime"], candidate["scrub_regime"]
            ),
            "selected: carbonation NaCl interval (mol/kg)": "{:g} to {:g}".format(
                *candidate["carbonation_interval_mol_kg"]
            ),
        },
        name="downstream alternative",
    )


def downstream_accounts(m, site, candidate):
    """Closed-form material and cost accounts of the downstream steps, used as their starting point and prices."""
    a = dict(assumptions(site), extractant_mol_l=m._design["extractant_mol_l"])
    m._full_process = {"case": CASE, "assumptions": a, "candidate": candidate}
    return a, flowsheet.costs_and_balances(
        m._washed, full_process.extraction(m), CASE, a
    )


METALS = {"Li": "lithium", "Na": "sodium"}


def add_scrub_and_strip(m, a, candidate, accounts):
    """Scrub and strip: one PrOMMiS SolventExtraction stage each, in acid exchange with the loaded organic.

    Each transfers a fixed fraction of each entering metal complex to the acid. The scrub takes the loaded organic
    leaving stage 1 and its acid is the larger of the acid supplied and the acid the scrubbed metals consume; the
    strip acid is 2.5 mol/L on a strip volume one tenth of the organic.
    """
    fs, v, ratio = m.fs, m._washed["feed_l"], m._decisions["organic_to_aqueous_ratio"]
    fractions = {
        "scrub": {"Li": a["scrub_li_loss"], "Na": a["scrub_na_efficiency"]},
        "strip": {"Li": accounts["strip_li_fraction"], "Na": 1.0},
    }
    for name, fraction in fractions.items():
        downstream.fixed_fraction_strip(
            fs,
            name,
            {e: f"{n}-carrier" for e, n in METALS.items()},
            {e: f"{n}-cation" for e, n in METALS.items()},
            dict.fromkeys(METALS, 1.0),
            dict.fromkeys(METALS, {"hbta": 1.0, "topo": 1.0}),
            fraction,
            spectators=("diluent",),
        )
    scrub, strip, loaded = fs.scrub, fs.strip, fs.stage[1].outlet
    scrub.loaded_organic = Constraint(
        fs.loaded_properties.component_list,
        rule=lambda b, s: scrub.unit.mscontactor.organic_inlet_state[0].flow_mol_comp[s]
        == v * loaded[s],
    )
    scrub.waste = Expression(
        tuple(METALS), rule=lambda b, e: b.aqueous_out[f"{METALS[e]}-cation"]
    )
    scrub.acid_mol = Var(domain=NonNegativeReals, initialize=accounts["scrub_acid_mol"])
    scrub.acid = Constraint(
        ("hydrogen-cation", "chloride-anion"),
        rule=lambda b, s: scrub.unit.mscontactor.aqueous_inlet_state[0].flow_mol_comp[s]
        == b.acid_mol,
    )
    source, demand = a["scrub_hcl_mol_l"] * ratio * v / 10, sum(scrub.waste.values())
    active, other = (
        (source, demand) if candidate["scrub_regime"] == "source" else (demand, source)
    )
    scrub.active_bound = Constraint(expr=scrub.acid_mol == active)
    scrub.other_bound = Constraint(expr=scrub.acid_mol >= other)
    strip.scrubbed_organic = Arc(
        source=scrub.unit.organic_outlet, destination=strip.unit.organic_inlet
    )
    strip.gross_acid_mol = Expression(expr=2.5 * ratio * v / 10)
    strip.acid = Constraint(
        ("hydrogen-cation", "chloride-anion"),
        rule=lambda b, s: strip.unit.mscontactor.aqueous_inlet_state[0].flow_mol_comp[s]
        == b.gross_acid_mol,
    )
    strip.metals = Expression(
        tuple(METALS), rule=lambda b, e: b.aqueous_out[f"{METALS[e]}-cation"]
    )
    strip.acid_sufficiency = Constraint(expr=strip.aqueous_out["hydrogen-cation"] >= 0)


def initialize_downstream(m, a, accounts):
    """Scrub, strip and liquor units solved in sequence from the extraction; the recycle loop from its closed form."""
    full_process._initialize_units(m.fs, accounts, a["mother_recycle"])


def priced_flows(m, accounts):
    """Each priced item: (kg per hour at 1 m3/h raw water, price), and the lithium in the Li2CO3 product (kg/h)."""
    fs, kg = m.fs, flowsheet.MASS_KG_PER_MOL
    consumed = {
        "saponification_NaOH": fs.saponification.naoh_mol,
        "scrub_HCl31": fs.scrub.acid_mol,
        "strip_HCl31": fs.strip.fresh_acid_mol,
        "neutralization_NaOH": fs.neutralization.naoh_mol,
        "carbonation_Na2CO3": fs.carbonation.carbonate_mol,
        "HBTA": fs.organic_loss.hbta_mol,
        "TOPO": fs.organic_loss.topo_mol,
        "diluent": fs.organic_loss.diluent_mol,
    }  # mol per L raw water
    reagent = {
        k: next((r for r in ("NaOH", "Na2CO3", "HCl31") if r in k), k) for k in consumed
    }
    flows = {
        k: (
            (consumed[k] * kg[reagent[k]] if k in consumed else mass)
            * costing.RAW_L_H
            * pyunits.kg,
            accounts["prices_usd_kg"][k] * costing.usd_per(pyunits.kg),
        )
        for k, mass in accounts["mass_kg_per_l"].items()
    }
    li_per_li2co3 = (
        accounts["li_product_kg_m3"] / 1000 / accounts["product_li2co3_mol"]
    )  # kg Li per mol Li2CO3
    return (
        flows,
        fs.carbonation.product_mol * li_per_li2co3 * costing.RAW_L_H * pyunits.kg,
    )


def whole_process_result(m, site, softened):
    """The repository's independent report of the solved process, and its check of every process requirement."""
    m._case = m._full_process["case"]
    result = optimization.report(m, "full_process", softened[site])
    assert (
        fixed_disposition(result, "C")["status"] == "grid"
    ), "design does not meet every process requirement"
    return result


def _decisions(model):
    return hasattr(model, "_decisions") or optimization.decisions(model, "C")


def run_water(site, softened):
    """Every step of the lead case for another water, with the repository's own flowsheet builders."""
    w, a = softened[site][CASE], ACCEPTED[site]
    model = set_c.build_flowsheet(
        w,
        stages=a["stages"],
        scenario=design(site),
        direct_cascade=starting_profile(site, softened),
    )
    optimization.decisions(model, "C")
    model, _ = set_c.solve(
        model,
        log_directory(f"{site}_extraction"),
        started=time.monotonic(),
        connect_units=_decisions,
    )
    candidate, _ = downstream_alternatives(model, site, softened)
    full_process.connect_units(model, CASE, assumptions(site), candidate)
    stage.solve(model, log_directory(f"{site}_full"))
    return model, whole_process_result(model, site, softened)


# ---------------------------------------------------------------- results


def lithium_path(result):
    """(step, stream kept, mol Li kept, (loss stream, mol Li lost)) per litre of raw water."""
    soft, p = result["softening"][CASE], result["physical_streams_mol_per_raw_l"]
    scrubbed = p["loaded organic"]["lithium-carrier"] - p["scrub waste"]["Li"]
    return [
        ("Raw water", "raw water", soft["raw_mol"]["Li"], None),
        (
            "Softening",
            "softened water",
            p["washed extraction feed"]["Li"],
            ("softening cake", soft["cake_mol"]["Li"] + soft["solid_mol"]["Li"]),
        ),
        (
            f"Extraction\n{len(result['stages'])} stages",
            "loaded organic",
            p["loaded organic"]["lithium-carrier"],
            ("raffinate", p["physical raffinate"]["Li"]),
        ),
        (
            "Scrub",
            "scrubbed organic",
            scrubbed,
            ("scrub waste", p["scrub waste"]["Li"]),
        ),
        (
            "Strip",
            "strip liquor",
            result["full_process"]["stripped_mol"]["Li"],
            ("left on organic", p["regeneration lithium discharge"]["Li"]),
        ),
        (
            "Carbonation",
            "product",
            p["Li2CO3 product"]["Li"],
            ("mother-liquor purge", p["mother liquor purge"]["Li"]),
        ),
    ]


def lithium_diagram(result, site):
    """Lithium from raw water to product, with every loss, and the balance closure (mol per L raw water)."""
    steps = lithium_path(result)
    raw = steps[0][2]
    grams = (
        lambda n: 1000 * n * LI_G_MOL
    )  # mol per L raw water -> g per m3 raw water  # noqa: E731
    fig, ax = plt.subplots(figsize=(13, 3.4))
    ax.set(xlim=(-0.6, len(steps) + 0.4), ylim=(-1.0, 1.15))
    ax.axis("off")
    for i, (name, stream, kept, lost) in enumerate(steps):
        ax.add_patch(
            FancyBboxPatch(
                (i - 0.42, 0.05),
                0.84,
                0.6,
                boxstyle="round,pad=0.02",
                fc="#e8eef6",
                ec="#33557a",
            )
        )
        ax.text(i, 0.35, name, ha="center", va="center", fontsize=10)
        ax.annotate(
            "",
            xy=(i + 0.58, 0.35),
            xytext=(i + 0.42, 0.35),
            arrowprops=dict(arrowstyle="-|>", color="#33557a"),
        )
        ax.text(
            i + 0.5,
            0.72,
            f"{stream}\n{grams(kept):.1f} g ({100 * kept / raw:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
        if lost:
            ax.annotate(
                "",
                xy=(i, -0.45),
                xytext=(i, 0.05),
                arrowprops=dict(arrowstyle="-|>", color="#a23b2a"),
            )
            ax.text(
                i,
                -0.55,
                f"{lost[0]}\n{grams(lost[1]):.2f} g ({100 * lost[1] / raw:.1f}%)",
                ha="center",
                va="top",
                fontsize=9,
                color="#a23b2a",
            )
    ax.text(
        len(steps),
        0.35,
        "Li$_2$CO$_3$\nproduct",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
    )
    ax.set_title(
        f"{WATERS[site]}, base case: lithium per m$^3$ of raw water (g Li, share of raw-water lithium)",
        fontsize=11,
    )
    plt.show()
    closure = steps[0][2] - steps[-1][2] - sum(s[3][1] for s in steps if s[3])
    print(
        f"Lithium balance residual (raw water minus product and losses): {closure:.1e} mol per L of raw water"
    )


GROUPS = {
    "Softening": {
        "softening_Na2CO3": "soda ash (softening)",
        "softening_NaOH": "sodium hydroxide (softening)",
        "softening_HCl31": "hydrochloric acid (pH reset)",
        "sludge": "wet-sludge disposal",
    },
    "Extractant and diluent replacement": {
        "HBTA": "HBTA",
        "TOPO": "TOPO",
        "diluent": "diluent",
    },
    "Extraction, scrub and strip chemicals": {
        "saponification_NaOH": "sodium hydroxide (saponification)",
        "scrub_HCl31": "hydrochloric acid (scrub)",
        "strip_HCl31": "hydrochloric acid (strip)",
        "neutralization_NaOH": "sodium hydroxide (neutralization)",
    },
    "Carbonation": {"carbonation_Na2CO3": "soda ash (carbonation)"},
}


def costs(model):
    """Each priced item from the IDAES costing block: USD per m3 raw water, per kg Li, and its price."""
    c = model.fs.costing
    assert {k for g in GROUPS.values() for k in g} == set(c.used_flows)
    li = value(c.li_kg_h)
    return pd.DataFrame(
        [
            {
                "step": step,
                "item": label,
                "USD/m3 raw water": c.aggregate_flow_costs[k].value,
                "USD/kg Li": c.aggregate_flow_costs[k].value / li,
                "price (USD/kg)": value(getattr(c, k + "_cost")),
            }
            for step, items in GROUPS.items()
            for k, label in items.items()
        ]
    )


def plot_costs(table, site, total):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = dict(zip(GROUPS, ("#4c78a8", "#e45756", "#72b7b2", "#f58518")))
    order = table.sort_values("USD/kg Li")
    ax.barh(order["item"], order["USD/kg Li"], color=[colors[s] for s in order["step"]])
    ax.legend(
        handles=[Patch(color=c, label=step) for step, c in colors.items()],
        fontsize=8,
        loc="lower right",
    )
    ax.set_xlabel("operating cost (USD per kg Li in Li$_2$CO$_3$)")
    ax.set_title(f"{WATERS[site]}, base case: {total:.0f} USD/kg Li in total")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- comparing the waters


def summary(results):
    return (
        pd.DataFrame(
            {
                WATERS[s]: {
                    "extraction stages": len(r["stages"]),
                    "HBTA (mol/L organic)": r["fixed_design"]["extractant_mol_l"],
                    "Li extraction (%)": 100 * r["li_extraction_fraction"],
                    "Li recovery to Li2CO3 (%)": 100 * r["product_recovery_fraction"],
                    "Li2CO3 product (kg per m3 raw)": r["full_process"][
                        "li_product_kg_m3"
                    ]
                    * LCE_PER_LI,
                    "operating cost (USD/kg Li)": r["cost_usd_kg_li"],
                    "operating cost (USD/m3 raw water)": r["cost_usd_m3_raw_brine"],
                    "break-even price (USD/t LCE)": 1000
                    * r["full_process"]["break_even_lce_usd_kg"],
                    "break-even / base price": r["full_process"][
                        "break_even_lce_usd_kg"
                    ]
                    / min(
                        p["lce_price_usd_kg"]
                        for p in r["full_process"]["price_break_evens"]
                    ),
                }
                for s, r in results.items()
            }
        )
        .style.format(number)
        .set_caption(WORKED_EXAMPLE)
    )


# ---------------------------------------------------------------- check against the stored worked-example results


def _compare(rows, name, actual, expected, *, rel=None, abs_=None, scale=None):
    error = abs(actual - expected)
    limit = (
        abs_
        if abs_ is not None
        else rel * abs(scale if scale is not None else expected)
    )
    rows.append(
        {
            "quantity": name,
            "reproduced": actual,
            "retained": expected,
            "difference": error,
            "limit": limit,
            "within limit": error <= limit,
        }
    )


def _compare_softening(rows, fresh, retained):
    for name in ("reagent_mol", "aqueous_mol", "cake_mol", "solid_mol"):
        assert set(fresh[name]) == set(
            retained[name]
        ), f"softening {name} species differ from the retained record"
    for name in (
        "feed_l",
        "wet_sludge_kg",
        "dry_sludge_kg",
        "pH",
        "reagent_mol",
        "aqueous_mol",
        "cake_mol",
        "solid_mol",
    ):
        pairs = (
            fresh[name].items()
            if isinstance(fresh[name], dict)
            else [(name, fresh[name])]
        )
        for key, n in pairs:
            target = (
                retained[name][key] if isinstance(fresh[name], dict) else retained[name]
            )
            tolerance = 1e-9 if key == "HCl" or name == "pH" else 1e-12
            rows.append(
                {
                    "quantity": f"softening {name}"
                    + (f" {key}" if key != name else ""),
                    "reproduced": n,
                    "retained": target,
                    "difference": abs(n - target),
                    "limit": max(tolerance, 1e-9 * abs(target)),
                    "within limit": math.isclose(
                        n, target, rel_tol=1e-9, abs_tol=tolerance
                    ),
                }
            )


KINDS = {
    "campaign": "stored design record",
    "exchange band": "stored summary table",
    "cost item": "cost item",
    "softening": "softening",
    "stage": "stage outlets",
    "physical": "process streams",
    "extractant": "design",
    "ln K": "design",
}


def check_against_paper(results, softened):
    """Every reproduced number against the worked example's stored design record and summary table; stops on any failure.

    Draws every comparison against its tolerance and returns (summary per water, every comparison).
    """
    tol, checks = BAND["reporting_relative_cost_tolerance"], []
    assert (
        set(results) == set(softened) == set(WATERS)
    ), "every water must be reproduced and checked"
    for site, r in results.items():
        a, rows = ACCEPTED[site], []
        record = json.loads((ROOT / a["source"]).read_text())
        assert (
            record["candidate"]["stages"] == len(r["stages"]) == a["stages"]
            and record["status"] == "optimal"
        )
        for key in ("cost_usd_kg_li", "cost_usd_m3_raw_brine"):
            _compare(rows, f"campaign {key}", r[key], record[key], rel=tol)
            _compare(rows, f"exchange band {key}", r[key], a[key], rel=tol)
        b = r["full_process"]["break_even_lce_usd_kg"]
        _compare(
            rows,
            "campaign break_even_lce_usd_kg",
            b,
            record["full_process"]["break_even_lce_usd_kg"],
            rel=tol,
        )
        _compare(
            rows,
            "exchange band break_even_lce_usd_kg",
            b,
            a["break_even_lce_usd_kg"],
            rel=tol,
        )
        for k, n in record["full_process"]["cost_usd_kg_li"].items():
            _compare(
                rows,
                f"cost item {k} (USD/kg Li)",
                r["full_process"]["cost_usd_kg_li"][k],
                n,
                rel=tol,
                scale=record["full_process"]["operating_usd_kg_li"],
            )
        for key in (
            "li_extraction_fraction",
            "product_recovery_fraction",
            "hbta_lost_fraction",
        ):
            _compare(rows, f"campaign {key}", r[key], record[key], abs_=1e-9)
        for key in ("li_extraction_fraction", "product_recovery_fraction"):
            _compare(rows, f"exchange band {key}", r[key], a[key], abs_=1e-9)
        for reaction, n in record["fixed_design"]["ln_k"].items():
            _compare(
                rows,
                f"ln K {reaction}",
                r["fixed_design"]["ln_k"][reaction],
                n,
                abs_=0.0,
            )
        _compare(
            rows,
            "extractant_mol_l",
            r["fixed_design"]["extractant_mol_l"],
            record["fixed_design"]["extractant_mol_l"],
            abs_=1e-9,
        )
        stage_error = max(
            abs(new["outlet_mol_per_nominal_l"][s] - old["outlet_mol_per_nominal_l"][s])
            for new, old in zip(r["stages"], record["stages"], strict=True)
            for s in old["outlet_mol_per_nominal_l"]
        )
        _compare(
            rows,
            "stage outlets, largest difference (mol/L)",
            stage_error,
            0.0,
            abs_=1e-9,
        )
        stream_error = max(
            abs(r["physical_streams_mol_per_raw_l"][k][s] - n)
            for k, stream in record["physical_streams_mol_per_raw_l"].items()
            for s, n in stream.items()
        )
        _compare(
            rows,
            "physical streams, largest difference (mol/L raw)",
            stream_error,
            0.0,
            abs_=1e-9,
        )
        _compare_softening(rows, softened[site][CASE], record["softening"][CASE])
        checks.append(pd.DataFrame(rows).assign(water=WATERS[site]))
    checks = pd.concat(checks, ignore_index=True).rename(
        columns={"within limit": "passed"}
    )
    group = checks["quantity"].str.extract(
        r"^(campaign|exchange band|cost item|softening|stage|physical|extractant|ln K)"
    )[0]
    checks["kind"] = group.map(KINDS)
    failed = checks[~checks["passed"]]
    assert failed.empty, f"reproduced numbers outside their limits:\n{failed}"
    summary = plot_checks(
        checks,
        "water",
        f"{len(checks)} comparisons with the stored worked-example results",
    )
    return summary, checks


# ---------------------------------------------------------------- The paper's screen, read from its result files


def _screen():
    """The attained selections at each price with their retained dispositions."""
    return selected_designs(*load_matrix(SCREEN))


def paper_headline():
    """Display the nominal-process quantities on a raw-water basis."""
    rows = []
    for cell in _screen():
        site = cell["water"]
        row = cell["design"]
        rows.append(
            {
                "Water": PRICED_WATERS.get(site, site),
                "Calibration": PRICED_FITS[cell["fit"]],
                "Price (USD/t LCE)": cell["price"],
                "N": int(row["N"]) if row else pd.NA,
                "Design ID": row["design_id"] if row else "",
                "Li to carbonate (%)": (
                    f"{100 * float(row['recovery']):.3f}" if row else ""
                ),
                "LCE (kg/m³)": f"{float(row['q_kg_lce_m3']):.6f}" if row else "",
                "Margin (USD/m³)": f"{float(row['margin_usd_m3']):+.5f}" if row else "",
                "Status": cell["status"],
                "Reason": cell["reason"],
            }
        )
    return pd.DataFrame(rows)


def paper_screen_summary():
    """Summarize the attained selections using the unrounded operating margins."""
    rows = list(_screen())
    primary = [cell for cell in rows if cell["price"] == PRIMARY]
    attained = [cell for cell in primary if cell["design"]]

    def margin(cell):
        return float(cell["design"]["margin_usd_m3"])

    def water(cell):
        site = cell["water"]
        return f"USGS {site}" if site.isdecimal() else PRICED_WATERS.get(site, site)

    best = max(attained, key=margin)
    negative = sum(margin(cell) < 0 for cell in attained)
    positive = [
        f"{PRICED_FITS[cell['fit']]} at {water(cell)}"
        for cell in rows
        if cell["price"] == SECONDARY and cell["design"] and margin(cell) > 0
    ]
    return (
        f"- At **{PRIMARY:,} USD/t LCE**, {negative} of {len(attained)} attained margins are negative. "
        f"The highest is **{margin(best):.2f} USD/m³** for {PRICED_FITS[best['fit']]} at {water(best)}.\n"
        f"- {len(primary) - len(attained)} water and calibration combinations have no verified target design.\n"
        f"- At **{SECONDARY:,} USD/t LCE**, positive margins occur only for {' and '.join(sorted(positive))}."
    )
