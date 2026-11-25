"""D2EHDTPA/BuPhen fixed simulation on #188 Held or #197 conditional Figiel chemistry."""

import json
import math
from pathlib import Path

import epcsaft
from epcsaft import equilibrium as eq
import numpy as np
from pyomo.environ import Block, Constraint, Expression, Var, value

from prommis.lithium_extraction.thermodynamics import d2ehdtpa_buphen_reactions as rc
from prommis.lithium_extraction.process import (
    waters as brines,
    mother_liquor_return,
    feeds as records,
    retained_references as references,
)
from prommis.lithium_extraction.extraction import (
    d2ehdtpa_buphen_accounts as owner,
    stage,
)
from prommis.lithium_extraction.thermodynamics import (
    d2ehdtpa_buphen_figiel_chloride as ion_basis,
)
from prommis.lithium_extraction.paths import ENGINE_ASL_LIBRARY

CONTINUATION = eq.Continuation(
    initial_step=0.25, minimum_step=1 / 64, max_steps=64, max_retries_per_step=4
)


def reference_feed(solvent="D2EHDTPA-Held"):
    """Original #188 cation conversions, preserved source fields and coherent charge closure."""
    original = ion_basis.MS2_FEED()
    raw = brines.raw_feed("MS2", reference_cations=original)
    raw["reference_conversion_basis"] = {
        "owner": "d2ehdtpa_buphen.ms2_feed",
        "molar_masses_kg_mol": {e: owner.MASSES[e] for e in original},
        "site_basis_li_mol_l": owner.BASIS["site_basis_li_mol_L"],
    }
    raw["labels"].append(
        "R6 original nominal #188 cation conversions; separate 0.024204 M site basis"
    )
    feed = records.extraction_feed(raw, solvent)
    feed["raw_brine"] = raw
    feed["labels"] = [
        label
        for label in feed["labels"]
        if not label.startswith("changed model input:")
    ]
    return feed


def build_flowsheet(
    brine,
    solvent="D2EHDTPA-Held",
    scope="extraction_only",
    stages=5,
    scenario=None,
    design_range=None,
    raffinate_entrainment_l=0.0,
    packet_id=None,
    raiguel_recipe=False,
    aqueous_return=False,
):
    """One basis per process; chemistry assembly and direct certificates remain source-owned."""
    if (
        records.SOLVENTS.get(solvent, {}).get("stage_adapter") != "d2_stage"
        or scope not in records.SCOPES
    ):
        raise ValueError("Unknown D2EHDTPA solvent or process scope")
    if not scenario or set(scenario) != {
        "B",
        "S",
        "O_A",
        "Sr_decade",
        "site_basis_li_mol_l",
        "ln_K_Li_Mg_Ca",
    }:
        raise ValueError(
            "fixed simulation requires B, S, O/A, Sr coordinate, site basis and source ln K"
        )
    b, supply, ratio, site_basis = (
        scenario[k] for k in ("B", "S", "O_A", "site_basis_li_mol_l")
    )
    sites = owner.site_supply(
        b,
        supply,
        ratio,
        stages,
        site_basis_li_mol_l=site_basis,
        design_range=design_range,
    )
    if raiguel_recipe and (
        b != owner.RAIGUEL_BUPHEN_MOL_L
        or abs(sites - owner.RAIGUEL_SITES_MOL_L) > math.ulp(owner.RAIGUEL_SITES_MOL_L)
    ):
        raise owner.OutsideDomain(
            f"Raiguel formulation requires B=0.250 M and 0.050 M sites, got {b}, {sites}"
        )
    if not all(
        math.isfinite(n)
        for n in (
            b,
            supply,
            ratio,
            sites,
            scenario["Sr_decade"],
            *scenario["ln_K_Li_Mg_Ca"],
        )
    ):
        raise ValueError("fixed design quantities must be finite")
    feed = (
        dict(brine)
        if "physical_feed" in brine
        else records.extraction_feed(brine, solvent, scope)
    )
    if "physical_feed" not in brine:
        feed["raw_brine"] = brine
    if "unavailable" in feed:
        raise ValueError(feed["unavailable"])
    basis = records.SOLVENTS[solvent]["ion_basis"]
    if packet_id is not None:
        if basis != "packet" or packet_id not in ion_basis.PACKETS:
            raise ValueError(f"Unsupported explicit packet {packet_id!r} for {solvent}")
        ion_basis.select(packet_id)
    ion_basis.use(basis)
    cations = {e: feed["model_feed"].get(e, 0) for e in rc.ION if e != "Cl"}
    constants = scenario["ln_K_Li_Mg_Ca"]
    mixture, problem, aqueous, organic, species, _, _, _ = rc.build_problem(
        cations,
        sites,
        b,
        constants,
        organic_to_aqueous=ratio,
        sr_decade=scenario["Sr_decade"],
    )
    totals = {"sites_total_mol": sites * ratio, "buphen_total_mol": b * ratio}
    fixed_species = (rc.ION["Cl"][0], "naa", "buphen") + (
        () if aqueous_return else ("water",)
    )
    controls = tuple(s for s in species if s not in fixed_species) + tuple(totals)
    charges = tuple(
        next((z for name, _, z in rc.ION.values() if name == s), 0) for s in species
    )
    inventory = problem.feed.values
    matrix, offset = [], []
    for s in species:
        row = {q: float(s == q) for q in controls}
        constant = 0.0
        if s == "water" and not aqueous_return:
            constant, row = inventory[s], dict.fromkeys(controls, 0.0)
        elif s in ("naa", "buphen"):
            moiety = rc.MOIETIES["A" if s == "naa" else "B"]
            row = {q: -moiety.get(q, 0) for q in controls}
            row["sites_total_mol" if s == "naa" else "buphen_total_mol"] = 1.0
        elif s == rc.ION["Cl"][0]:
            row = {
                q: charges[species.index(q)] if q in species else 0.0 for q in controls
            }
        offset.append(constant)
        matrix.append(tuple(row[q] for q in controls))
    anchor = tuple((inventory | totals)[s] for s in controls)
    if (
        max(
            abs(a + sum(x * y for x, y in zip(row, anchor, strict=True)) - inventory[s])
            for s, a, row in zip(species, offset, matrix, strict=True)
        )
        > 1e-12
    ):
        raise ValueError("reduced controls do not reproduce the source inventory")
    offset, matrix = tuple(offset), tuple(matrix)
    record = eq.asl_record(
        mixture,
        problem,
        CONTINUATION,
        offset,
        matrix,
        anchor,
        species.index("water"),
        0.001,
    )
    conserved, rows, omitted = [], [], {}
    for name, row in rc.MOIETIES.items():
        values = tuple(row.get(s, 0) for s in species)
        if not any(values):
            omitted[name] = "structurally absent from source-selected species support"
        elif np.linalg.matrix_rank(np.asarray([*rows, values])) > len(rows):
            rows.append(values)
            conserved.append(name)
        else:
            omitted[name] = "dependent on preceding source moiety rows on this support"

    def certify(inlet, outlet):
        v = inlet["diluent"] / (organic["diluent"] / ratio)
        aqueous_volume = inlet["water"] / aqueous["water"] if aqueous_return else 1.0
        total_sites = sum(rc.MOIETIES["A"].get(s, 0) * n for s, n in inlet.items()) / v
        total_b = sum(rc.MOIETIES["B"].get(s, 0) * n for s, n in inlet.items()) / v
        loaded = {e: inlet[c] / v for e, c in rc.COMPLEX.items() if c in inlet}
        carried = {
            e: inlet[rc.ION[e][0]] / aqueous_volume
            for e in cations
            if rc.ION[e][0] in inlet
        }
        direct = rc.solve(
            carried,
            total_sites,
            total_b,
            constants,
            organic_to_aqueous=v / aqueous_volume,
            loading=loaded,
            sr_decade=scenario["Sr_decade"],
        )
        direct_amounts = [
            {s: amount * aqueous_volume for s, amount in phase.items()}
            for phase in direct["amounts_mol"]
        ]
        difference = max(
            abs(outlet[s] - sum(p.get(s, 0) for p in direct_amounts)) for s in species
        )
        return {
            "status": "evaluated",
            **direct["certificate"],
            "direct_amounts_mol": direct_amounts,
            "D": direct["D"],
            "outlet_max_difference_mol_per_raw_l": difference,
        }

    chemistry = stage.Chemistry(
        species,
        (tuple(aqueous), tuple(organic)),
        tuple(rows),
        charges,
        controls,
        offset,
        matrix,
        record,
        Path(epcsaft.__file__).resolve().parent / ENGINE_ASL_LIBRARY,
        certify,
        parameters=tuple(totals.items()),
        temperature_k=problem.T,
        pressure_pa=problem.P,
    )
    model = stage.build_stages(
        chemistry,
        aqueous,
        organic,
        stages,
        maximum_stages=max(design_range["stages"]) if design_range else 5,
        raffinate_diluent_loss=raffinate_entrainment_l * organic["diluent"] / ratio,
        raffinate_water_basis=aqueous["water"] if aqueous_return else None,
    )
    model.fs.aqueous_volume = Expression(
        expr=model.fs.boundary["water"] / aqueous["water"] if aqueous_return else 1.0
    )
    model._aqueous_return = aqueous_return
    model._design_range = design_range
    model._raiguel_recipe = raiguel_recipe
    model._physical_feed, model._design, model._solvent = feed, dict(scenario), solvent
    model._conserved_names, model._omitted_moieties = conserved, omitted
    return model


def reference_record(solvent="D2EHDTPA-Held"):
    path = owner.ANALYSIS / records.SOLVENTS[solvent]["starting_record"]
    return path, json.loads(path.read_text())


def strip_accounts(accounts, solvent):
    """Figiel's selected ion basis lumps Sr into Ca; every other strip amount is required."""
    return accounts | records.SOLVENTS[solvent]["strip_lumped"]


def reference_profiles(model):
    """Complete phase amounts from the exact accepted fixed-design cascade, without refitting."""
    _, retained = reference_record(model._solvent)
    case = retained["case"]
    if len(model.fs.stages) != case["N"] or any(
        model._design[k] != case[k] for k in ("B", "S", "O_A", "Sr_decade")
    ):
        raise ValueError("retained profiles apply only to the original fixed design")
    if model._design["ln_K_Li_Mg_Ca"] != retained["ln_K_Li_Mg_Ca"]:
        raise ValueError("retained profiles require their original source coordinates")
    return [
        {
            s: sum(p.get(s, 0) for p in row["amounts_mol"])
            for s in model._chemistry.species
        }
        for row in retained["extraction"]["stages"]
    ]


def decisions(model, *, organic_recycle=False):
    """Release the record's #188 design range (#342): BuPhen B (M), site ratio S (sites per feed Li) and O/A set the
    organic inlet and the Engine's ligand totals; organic site concentration S*basis/(O/A) stays in its range.
    Starts fixed at the built design; `release` frees them."""
    fs, record, d = model.fs, records.SOLVENTS[model._solvent]["design"], model._design
    record = record | (model._design_range or {})
    fs.design = Block()
    b, model._decisions = fs.design, {}
    for name, bounds in record["bounds"].items():
        if name == "B" and model._raiguel_recipe:
            bounds = (owner.RAIGUEL_BUPHEN_MOL_L,) * 2
        variable = Var(bounds=bounds, initialize=d[name])
        b.add_component(name, variable)
        variable.fix(d[name])
        model._decisions[name] = variable
    B, S, ratio = (model._decisions[k] for k in ("B", "S", "O_A"))
    b.organic_volume = Expression(expr=ratio * fs.aqueous_volume)
    organic_volume = b.organic_volume
    li = rc.ION["Li"][0]
    sites = S * (fs.boundary[li] if model._aqueous_return else d["site_basis_li_mol_l"])
    diluent = model._feeds[1]["diluent"] / d["O_A"]
    links = {
        "buphen": B * organic_volume,
        "naa": sites,
        "diluent": diluent * organic_volume,
    }
    if organic_recycle:
        links = {"diluent": diluent * organic_volume}
        for s, total in (("naa", sites), ("buphen", B * organic_volume)):
            moiety = rc.MOIETIES["A" if s == "naa" else "B"]
            links[s] = total - sum(
                moiety.get(c, 0.0) * fs.boundary[c]
                for c in rc.COMPLEX.values()
                if c in fs.boundary
            )
    totals = {"sites_total_mol": sites, "buphen_total_mol": B * organic_volume}
    for s in links:
        fs.boundary[s].unfix()
    for s in totals:
        fs.parameters[s].unfix()
    b.organic_inlet = Constraint(
        tuple(links), rule=lambda _, s: fs.boundary[s] == links[s]
    )
    b.ligand_totals = Constraint(
        tuple(totals), rule=lambda _, s: fs.parameters[s] == totals[s]
    )
    if model._raiguel_recipe:
        b.site_concentration = Constraint(
            expr=sites == owner.RAIGUEL_SITES_MOL_L * organic_volume
        )
        b.site_concentration.deactivate()  # exactly constructed when S and O/A are both fixed
    else:
        low, high = record["sites_mol_l"]
        b.sites_low = Constraint(expr=sites >= low * organic_volume)
        b.sites_high = Constraint(expr=sites <= high * organic_volume)
    b.li_extraction = Expression(
        expr=1 - fs.stage[len(fs.stages)].outlet[li] / fs.boundary[li]
    )


def release(model, objective, *, minimum_li_extraction=None):
    """Free the decisions and set the #342 rule objective; the Li extraction requirement stays on extraction."""
    from prommis.lithium_extraction.process import optimization

    target = (
        records.SOLVENTS[model._solvent]["design"]["li_target"]
        if minimum_li_extraction is None
        else minimum_li_extraction
    )
    for variable in model._decisions.values():
        if variable.lb != variable.ub:
            variable.unfix()
    if model._raiguel_recipe:
        model.fs.design.site_concentration.activate()
    if getattr(model, "_d2_full_process", {}).get("optimize_mother_liquor_return"):
        model.fs.mother_liquor_return.fraction.unfix()
    if hasattr(model.fs.design, "li_target"):
        model.fs.design.del_component("li_target")
    model.fs.design.li_target = Constraint(expr=model.fs.design.li_extraction >= target)
    model.feasibility.deactivate()
    optimization.add_rule(model, objective)


def synchronize(model):
    """The solved decisions and boundaries replace the build design, so reports price the solved design."""
    if hasattr(model, "_decisions"):
        model._design.update({k: value(v) for k, v in model._decisions.items()})
    model._feeds = tuple(
        {s: value(model.fs.boundary[s]) for s in feed} for feed in model._feeds
    )


def sodium_report(aqueous, organic, outlet, entrained_na, stages):
    """Sodium leaving extraction in aqueous raffinate, entrained organic, and loaded organic."""
    ion = rc.ION["Na"][0]
    naa = outlet["naa"]
    account = {
        "brine_in": aqueous[ion],
        "saponified_organic_in": organic["naa"],
        "raffinate_out": outlet[ion],
        "loaded_organic_out": naa,
    }
    if entrained_na is None:
        return {"sodium_account_mol_per_raw_l": account}
    account["raffinate_entrained_organic_out"] = entrained_na
    if stages == 1:
        account["loaded_organic_out"] -= entrained_na
    # A one-stage split removes the same fraction of Li and NaA, so it leaves this ratio unchanged.
    metric = {
        "name": "loaded-organic Na/(Li+Na)",
        "value": naa / (naa + outlet[rc.COMPLEX["Li"]]),
    }
    return {"sodium_account_mol_per_raw_l": account, "na_metric": metric}


def report(model, result=None, *, case="base", reference_accounting=False):
    """Source moiety, physical raffinate and strip/loss evidence at the solved design."""
    synchronize(model)
    result = stage.report(model) if result is None else result
    species = model._chemistry.species
    residuals = dict(result["balance_residuals_mol_per_raw_l"])
    for row in result["stages"]:
        inlet, outlet = row["inlet_mol_per_raw_l"], row["outlet_mol_per_raw_l"]
        for name, moiety in rc.MOIETIES.items():
            residuals[f'stage{row["stage"]}/{name}'] = sum(
                moiety.get(s, 0) * (outlet[s] - inlet[s]) for s in species
            )
    final = result["outlet_mol_per_raw_l"]
    stage_count = len(result["stages"])
    entrained = {
        s: value(n)
        for s, n in getattr(model.fs, "raffinate_entrained_organic", {}).items()
    }
    for name, moiety in rc.MOIETIES.items():
        residuals[f"global/{name}"] = sum(
            moiety.get(s, 0)
            * (
                final[s]
                + (entrained.get(s, 0.0) if stage_count > 1 else 0.0)
                - sum(p.get(s, 0) for p in model._feeds)
            )
            for s in species
        )
    volume = value(model.fs.aqueous_volume)
    ratio = model._design["O_A"] * volume
    li_feed = model._feeds[0][rc.ION["Li"][0]]
    raffinate = {
        records.SPECIES_LABELS[s]: final[s] for s in model._chemistry.supports[0]
    }
    feed = mother_liquor_return.mixed_feed(model, stages_per_nominal_litre=False)
    r7 = references.check_r7(
        feed,
        raffinate,
        balance_residuals_mol_per_raw_l=residuals,
        strict_certificate=result["strict_certificate"],
    )
    extraction = {
        "loaded_organic": {
            e: final[c] / ratio for e, c in rc.COMPLEX.items() if c in final
        },
        "loaded_Na_mol_L": final["naa"] / ratio,
        "li_extraction_fraction": 1 - raffinate["Li"] / li_feed,
        "stages": [
            {
                "free_B_mol_L": row["outlet_mol_per_raw_l"]["buphen"] / ratio,
                "amounts_mol": [
                    {s: row["outlet_mol_per_raw_l"][s] for s in support}
                    for support in model._chemistry.supports
                ],
            }
            for row in result["stages"]
        ],
    }
    from prommis.lithium_extraction.process.candidate import case_assumptions

    assumptions = (
        None
        if reference_accounting
        else (getattr(model, "_d2_full_process", None) or case_assumptions(case, True))
    )
    if assumptions is not None and "strip_aqueous_to_organic" in model._design:
        assumptions = dict(
            assumptions,
            strip_aqueous_to_organic=model._design["strip_aqueous_to_organic"],
        )
    recycle = hasattr(model.fs, "organic_recycle")
    strip_volume = (
        ratio - volume * assumptions["entrainment_l_per_l"] if recycle else None
    )
    sites_li_basis = (
        li_feed if model._aqueous_return else model._design["site_basis_li_mol_l"]
    )
    accounts = owner.accounts(
        extraction,
        model._design["S"],
        ratio,
        model._design["B"],
        case,
        0.0,
        case_assumptions=assumptions,
        feed_li_mol_l=li_feed,
        site_basis_li_mol_l=sites_li_basis,
        strip_volume=strip_volume,
        aqueous_volume=volume,
    )
    accounts.update(
        accounting_basis="historical #188 extraction-plus-strip intermediate; neither user scope",
        li_denominator="kg strip-recovered Li per m3 raw brine",
        total_usd_kg_li=None,
        can_minimize_total=False,
        completeness="partial: " + accounts["missing_accounts"],
    )
    sodium = sodium_report(
        *model._feeds,
        final,
        entrained.get("naa", 0.0) if recycle else None,
        stage_count,
    )
    organic = model._feeds[1]
    sites_mol_l = (
        math.fsum(rc.MOIETIES["A"].get(s, 0.0) * n for s, n in organic.items()) / ratio
        if model._aqueous_return or model._raiguel_recipe
        else model._design["S"] * sites_li_basis / ratio
    )
    recipe = {}
    if model._raiguel_recipe:
        recipe = {
            "organic_volume_l_per_raw_l": ratio,
            "mixed_feed_l_per_raw_l": volume,
            "mixed_li_mol_l": li_feed / volume,
            "sites_mol_l_organic": sites_mol_l,
            "buphen_mol_l_organic": sum(
                rc.MOIETIES["B"].get(s, 0.0) * n for s, n in organic.items()
            )
            / ratio,
            "modifier_volume_fractions": owner.RAIGUEL_OCTANOL_FRACTIONS,
        }
    return {
        **result,
        "balance_residuals_mol_per_raw_l": residuals,
        "R7": r7,
        "feed": feed,
        "fixed_design": model._design,
        "solvent": model._solvent,
        "conserved_quantity_names": model._conserved_names,
        "omitted_moieties": model._omitted_moieties,
        "aqueous_inlet_mol_per_raw_l": model._feeds[0],
        "organic_inlet_mol_per_raw_l": model._feeds[1],
        "li_extraction_fraction": extraction["li_extraction_fraction"],
        "accounts": accounts,
        "sites_mol_l_organic": sites_mol_l,
        **({"raiguel_recipe": recipe} if recipe else {}),
        **sodium,
        "labels": model._physical_feed["labels"]
        + [label for label in owner.LABELS.values() if isinstance(label, str)]
        + [
            "fixed-design numerical verification, not physical MS-2 validation; no optimizer or full process"
        ],
    }
