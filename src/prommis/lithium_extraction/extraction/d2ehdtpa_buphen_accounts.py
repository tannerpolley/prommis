"""#188 unsoftened MS-2 option: strict reactive stages and one-cycle reagent/loss accounts."""

import csv
import json

from prommis.lithium_extraction.thermodynamics import d2ehdtpa_buphen_reactions as rc
from prommis.lithium_extraction.extraction.cascade import sweep_stages
from prommis.lithium_extraction.process.flowsheet import _row
from prommis.lithium_extraction.paths import REPO_ROOT

ANALYSIS = REPO_ROOT / "analyses/prototypes/d2ehdtpa_buphen_ms2_prototype"
BASIS = json.loads((ANALYSIS / "inputs/stage2_basis.json").read_text())
MASSES = json.loads((ANALYSIS / "inputs/reaction_basis.json").read_text())[
    "molar_masses_kg_mol"
]
BUPHEN_ORGANIC_AQUEOUS_PARTITION = 1.4e4
MOLAR_MASS_KG_MOL = {"BuPhen": 0.29242, "D2EHDTPA": 0.35456, "dodecane": 0.17033484}
LABELS = {
    "evidence_type": "prediction",
    "holdout_support": "no holdout support",
    "activity_error_label": rc.ERROR_LABEL,
    "transfer_label": "Na and formulation transfer not supported (F2)",
    "temperature_K": rc.T,
    "pressure_Pa": rc.P,
}


def ms2_feed():
    with (
        REPO_ROOT / "data/reference/produced_water/smackover_selected_base_feed_ms2.csv"
    ).open() as stream:
        measured = {
            r["analyte"]: float(r["value"])
            for r in csv.DictReader(stream)
            if r["analyte"] in MASSES
        }
    return {
        e: measured[e] / (1e6 * MASSES[e]) for e in ("Li", "Na", "K", "Mg", "Ca", "Sr")
    }


class OutsideDomain(ValueError):
    """A design or feed outside the #188 fixed design domain."""


def site_supply(
    buphen, S, v, stage_count, *, site_basis_li_mol_l=None, design_range=None
):
    """The #188 fixed design domain and site supply, also used by the generic simulation."""
    basis = (
        BASIS["site_basis_li_mol_L"]
        if site_basis_li_mol_l is None
        else site_basis_li_mol_l
    )
    sites = S * basis / v
    if design_range is not None:
        valid = all(
            lo <= x <= hi
            for x, (lo, hi) in zip(
                (buphen, S, v, sites),
                (
                    *(design_range["bounds"][k] for k in ("B", "S", "O_A")),
                    design_range["sites_mol_l"],
                ),
                strict=True,
            )
        )
        if not valid or stage_count not in design_range["stages"]:
            raise OutsideDomain(
                f"Outside explicit design range: B={buphen}, S={S}, O/A={v}, N={stage_count}, A={sites}"
            )
        return sites
    if not (
        0.25 <= buphen <= 1
        and 0.9 <= S <= 1.5
        and 0.5 <= v <= 2
        and 0.020 <= sites <= 0.060
        and stage_count in BASIS["stage_counts"]
    ):
        raise OutsideDomain(
            f"Outside #188 domain: B={buphen}, S={S}, O/A={v}, N={stage_count}, A={sites}"
        )
    return sites


def extract(
    constants,
    buphen,
    S,
    v,
    stage_count,
    *,
    sr_decade=0,
    start=None,
    progress=None,
    resume_sweeps=0,
):
    """Available NaA sites after H2S; unavailable HA/NaHS carried outside the Engine."""
    sites = site_supply(buphen, S, v, stage_count)
    feed = ms2_feed()
    fresh = {e: 0.0 for e in rc.COMPLEX}

    def solve_stage(k, aqueous, entering, guess):
        result = rc.solve(
            aqueous,
            sites,
            buphen,
            constants,
            organic_to_aqueous=v,
            loading=entering,
            sr_decade=sr_decade,
        )
        aq, org = result["amounts_mol"]
        out_aq = {e: aq[rc.ION[e][0]] for e in feed}
        out_org = {e: org[c] / v for e, c in rc.COMPLEX.items()}
        row = {
            "stage": k + 1,
            "D": result["D"],
            "certificate": result["certificate"],
            "amounts_mol": result["amounts_mol"],
            "free_B_mol_L": org["buphen"] / v,
            **LABELS,
        }
        return out_aq, out_org, row, None

    result = sweep_stages(
        stage_count,
        feed,
        fresh,
        solve_stage,
        start=start,
        progress=progress,
        resume_sweeps=resume_sweeps,
    )
    aq, org = result["raffinate"], result["loaded_organic"]
    closure = {e: feed[e] - aq[e] - v * org.get(e, 0) for e in feed if e != "Na"}
    loaded_na = sites - sum(rc.ION[e][2] * org[e] for e in org)
    closure["Na"] = feed["Na"] + v * sites - aq["Na"] - v * loaded_na
    if max(map(abs, closure.values())) > 1e-9:
        raise RuntimeError(f"#188 cascade balance exceeds 1e-9 mol/L feed: {closure}")
    result.update(
        closure_mol=closure,
        sites_mol_L=sites,
        loaded_Na_mol_L=loaded_na,
        li_extraction_fraction=1 - aq["Li"] / feed["Li"],
        **LABELS,
    )
    return result


def entrained_l(case_assumptions, strip_outlet_l):
    """Organic litres per raw litre lost at both aqueous separators."""
    return case_assumptions["entrainment_l_per_l"] * (1 + strip_outlet_l)


def historical_strip_volume(case_assumptions, strip_li_mol):
    """Concentration proxy for the archived open-loop and #433 replay."""
    return strip_li_mol / case_assumptions["concentrate_li_mol_kg"]


def separated_losses(raffinate, strip, free_b_concentration, entrained):
    """Losses in kg/raw L from the actual separated species and nominal diluent formulation."""
    masses = MOLAR_MASS_KG_MOL
    lost = {s: (raffinate[s] if s in raffinate else 0.0) + n for s, n in strip.items()}
    losses = {
        "BuPhen_dissolution": free_b_concentration
        / BUPHEN_ORGANIC_AQUEOUS_PARTITION
        * masses["BuPhen"],
        "BuPhen_entrainment": sum(
            rc.MOIETIES["B"].get(s, 0) * n for s, n in lost.items()
        )
        * masses["BuPhen"],
        "D2EHDTPA_entrainment": sum(
            rc.MOIETIES["A"].get(s, int(s == "ha")) * n for s, n in lost.items()
        )
        * masses["D2EHDTPA"],
        "diluent": 0.75 * lost["diluent"] * masses["dodecane"],
    }
    for k in ("octanol_1", "octanol_2"):
        row = BASIS[k]
        losses[k + "_dissolution"] = row["pure_water_solubility_kg_m3"] / 1000
        losses[k + "_entrainment"] = (
            entrained * row["vol_fraction"] * row["density_kg_L"]
        )
    return losses | {
        f"octanol_{key}": sum(losses[k + "_" + key] for k in ("octanol_1", "octanol_2"))
        for key in ("dissolution", "entrainment")
    }


def losses_kg_per_l(
    entrained,
    S,
    v,
    buphen,
    h2s,
    free_b_raffinate,
    diluent_per_l_organic,
    *,
    site_basis_li_mol_l,
):
    """kg per raw litre lost: octanol dissolution (pure-water solubility, per raw litre) and entrainment; BuPhen
    dissolution (free BuPhen at the raffinate end over its partition value) and entrainment; D2EHDTPA and dodecane
    entrainment. Arguments may be numbers or Pyomo expressions of the solved design; one owner for both.
    """
    octanol = sum(
        BASIS[k]["vol_fraction"] * BASIS[k]["density_kg_L"]
        for k in ("octanol_1", "octanol_2")
    )
    return {
        "octanol_dissolution": sum(
            BASIS[k]["pure_water_solubility_kg_m3"] for k in ("octanol_1", "octanol_2")
        )
        / 1000,
        "octanol_entrainment": entrained * octanol,
        "BuPhen_dissolution": free_b_raffinate
        / BUPHEN_ORGANIC_AQUEOUS_PARTITION
        * MOLAR_MASS_KG_MOL["BuPhen"],
        "BuPhen_entrainment": entrained * buphen * MOLAR_MASS_KG_MOL["BuPhen"],
        "D2EHDTPA_entrainment": entrained
        * (S * site_basis_li_mol_l + h2s)
        / v
        * MOLAR_MASS_KG_MOL["D2EHDTPA"],
        "diluent": entrained
        * 0.75
        * diluent_per_l_organic
        * MOLAR_MASS_KG_MOL["dodecane"],
    }  # 25 vol% octanols


def accounts(
    extraction,
    S,
    v,
    buphen,
    case,
    h2s,
    *,
    case_assumptions=None,
    feed_li_mol_l=None,
    site_basis_li_mol_l=None,
    value_basis_li_mol_l=None,
    strip_volume=None,
):
    """kg/m3 losses and the priced strip intermediate (NaOH, 31 % HCl, octanols); BuPhen and D2EHDTPA are the
    unpriced active set (#339 decision 1). Independent of the connected model: amounts come from solved outlets.
    Without case_assumptions the retained #188 basis applies (per-cycle entrainment fraction of circulated organic,
    #188 octanol price, revenue fields), so retained MS-2 studies and their Engine replay still reproduce.
    """
    old = case_assumptions is None
    reg = "data/reference/tea/hbta_topo_screening_cost_assumptions.csv"
    prices = {
        k: float(_row(reg, "assumption_id", k + "_price")["base"])
        for k in ("naoh", "hcl")
    }
    octanol_price = (
        float(
            _row(
                "data/reference/tea/d2ehdtpa_buphen_price_register.csv",
                "component_id",
                "1_octanol",
            )["delivered_price_usd_kg"]
        )
        if old
        else float(
            _row(
                "data/reference/tea/extractant_price_register.csv",
                "component_id",
                "1_octanol",
            )["price_usd_kg"]
        )
    )
    feed_li = ms2_feed()["Li"] if feed_li_mol_l is None else feed_li_mol_l
    site_basis = (
        BASIS["site_basis_li_mol_L"]
        if site_basis_li_mol_l is None
        else site_basis_li_mol_l
    )
    available = S * site_basis
    loaded = {e: v * n for e, n in extraction["loaded_organic"].items()}
    loaded["Na"] = v * extraction["loaded_Na_mol_L"]
    if strip_volume is not None and len(extraction["stages"]) == 1:
        loaded = {e: n * strip_volume / v for e, n in loaded.items()}
    acid = sum(
        rc.ION[e][2] * n for e, n in loaded.items()
    )  # includes residual NaA; excludes unavailable HA.
    strip = {e: n * BASIS["strip_fractions"][case][e] for e, n in loaded.items()}
    retained = {e: loaded[e] - strip[e] for e in loaded}
    strip_closure = max(abs(loaded[e] - strip[e] - retained[e]) for e in loaded)
    strip_sites = available if strip_volume is None else available * strip_volume / v
    if strip_closure > 1e-9 or abs(acid - strip_sites) > 1e-9:
        raise RuntimeError("Strip/site mass balance exceeds 1e-9 mol/L feed")
    rec = strip["Li"] / feed_li
    li_kg = strip["Li"] * MASSES["Li"] * 1000
    prepared_l = (
        strip_volume * case_assumptions["strip_aqueous_to_organic"]
        if strip_volume is not None and "strip_aqueous_to_organic" in case_assumptions
        else None
    )
    outlet_l = (
        prepared_l
        if prepared_l is not None
        else (
            historical_strip_volume(case_assumptions, strip["Li"]) if not old else None
        )
    )
    entrained = (
        BASIS["entrainment_fraction"][case] * v
        if old
        else entrained_l(case_assumptions, outlet_l)
    )
    naoh = available + h2s
    if strip_volume is not None:
        fraction_returned = (
            1 - (entrained - case_assumptions["entrainment_l_per_l"]) / strip_volume
        )
        naoh -= fraction_returned * sum(rc.ION[e][2] * n for e, n in retained.items())
    lost = {
        k: 1000 * n
        for k, n in losses_kg_per_l(
            entrained,
            S,
            v,
            buphen,
            h2s,
            extraction["stages"][-1]["free_B_mol_L"],
            extraction["stages"][0 if strip_volume is None else -1]["amounts_mol"][1][
                "diluent"
            ]
            / v,
            site_basis_li_mol_l=site_basis,
        ).items()
    }
    if prepared_l is not None:
        raffinate_l = case_assumptions["entrainment_l_per_l"]
        raffinate = extraction["stages"][-1]["amounts_mol"][1]
        stripped = dict(extraction["stages"][0]["amounts_mol"][1])
        if len(extraction["stages"]) == 1:
            stripped = {s: n * strip_volume / v for s, n in stripped.items()}
        stripped["ha"] = sum(rc.ION[e][2] * n for e, n in strip.items())
        stripped["buphen"] += 2 * strip["Li"]
        for e, n in retained.items():
            stripped[rc.COMPLEX[e] if e != "Na" else "naa"] = n
        lost = {
            k: 1000 * n
            for k, n in separated_losses(
                {s: n * raffinate_l / v for s, n in raffinate.items()},
                {
                    s: n * (entrained - raffinate_l) / strip_volume
                    for s, n in stripped.items()
                },
                raffinate["buphen"] / v,
                entrained,
            ).items()
        }
    costs = {
        "naoh": naoh * 0.039997 * 1000 * prices["naoh"],
        "hcl31": acid * 0.036461 / 0.31 * 1000 * prices["hcl"],
        "octanol_dissolution": lost["octanol_dissolution"] * octanol_price,
        "octanol_entrainment": lost["octanol_entrainment"] * octanol_price,
    }
    subtotal = sum(costs.values())
    result = {
        "case": case,
        "h2s_protonated_mol_L_feed": h2s,
        "R_ext": extraction["li_extraction_fraction"],
        "strip_Li_fraction": BASIS["strip_fractions"][case]["Li"],
        "R_rec": rec,
        "circulated_sites_mol_L_feed": available + h2s,
        "H2S_fraction_of_available_sites": h2s / available,
        "strip_HCl_mol_L_feed": acid,
        "strip_Cl_mol_L_feed": acid,
        "strip_unconsumed_HCl_mol_L_feed": acid
        - sum(rc.ION[e][2] * n for e, n in strip.items()),
        "NaOH_mol_L_feed": naoh,
        "li_recovered_kg_m3": li_kg,
        "lce_recovered_kg_m3": strip["Li"] * 0.07389 / 2 * 1000,
        "entrained_organic_l_per_raw_l": entrained,
        "strip_na_fraction": strip["Na"] / (strip["Li"] + strip["Na"]),
        "strip_closure_mol": strip_closure,
        "BuPhen_dissolved_kg_m3": lost["BuPhen_dissolution"],
        "BuPhen_entrained_kg_m3": lost["BuPhen_entrainment"],
        "BuPhen_loss_kg_m3": lost["BuPhen_dissolution"] + lost["BuPhen_entrainment"],
        "D2EHDTPA_loss_kg_m3": lost["D2EHDTPA_entrainment"],
        "dodecane_loss_kg_m3": lost["diluent"],
        "octanol_dissolved_kg_m3": lost["octanol_dissolution"],
        "octanol_entrained_kg_m3": lost["octanol_entrainment"],
        "priced_subtotal_usd_m3": subtotal,
        "priced_subtotal_usd_kg_li": subtotal / li_kg,
        "missing_accounts": BASIS["missing_accounts"],
        "cost_expression": "C_r + theta*(BuPhen_loss + D2EHDTPA_loss); prices USD/kg, losses kg/m3 (#339 decision 1)",
        "octanol_dissolution_label": BASIS["octanol_dissolution_label"],
        "H2S_scope": "HA unavailable outside Engine; equal Na and HS leave raffinate; no sulfide activity"
        + (" or residual-loading reuse model" if strip_volume is None else ""),
        "strip_assumptions": "Na=Li strip fraction and Sr=alkaline-earth strip fraction; no independent Na or Sr strip measurement",
        **LABELS,
    }
    for e in loaded:
        result.update(
            {
                f"loaded_{e}_mol_L_feed": loaded[e],
                f"strip_{e}_mol_L_feed": strip[e],
                f"retained_{e}_mol_L_feed": retained[e],
            }
        )
        if e in ("Mg", "Ca", "Sr"):
            result[f"strip_{e}_per_Li_mol_mol"] = strip[e] / strip["Li"]
    for k, cost in costs.items():
        result[f"{k}_usd_m3"] = cost
    if not old:
        result["entrainment_l_per_l"] = case_assumptions["entrainment_l_per_l"]
        result["strip_outlet_volume_l_per_raw_l"] = outlet_l
        return result
    lce_kg = result["lce_recovered_kg_m3"]
    result.update(
        {
            "priced_subtotal_usd_kg_lce": subtotal / lce_kg,
            "cost_expression": "priced subtotal + BuPhen_loss*p_BuPhen + D2EHDTPA_loss*p_D2EHDTPA + dodecane_loss*p_dodecane; prices USD/kg, losses kg/m3",
            "break_even_label": "BuPhen-only conditional upper bound, before D2EHDTPA/dodecane prices and excluded accounts",
        }
    )
    value_scale = feed_li / (
        ms2_feed()["Li"] if value_basis_li_mol_l is None else value_basis_li_mol_l
    )
    for label, value in (
        ("low", 15.65 * value_scale),
        ("high", 18.69 * value_scale),
        ("screening_25", feed_li * 0.07389 / 2 * 1000 * 25),
    ):
        result.update(
            {
                f"revenue_{label}_usd_m3": value * rec,
                f"headroom_before_missing_{label}_usd_m3": value * rec - subtotal,
                f"BuPhen_break_even_upper_{label}_usd_kg": (value * rec - subtotal)
                / result["BuPhen_loss_kg_m3"],
            }
        )
    return result
