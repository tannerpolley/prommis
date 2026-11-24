"""One cell of the modular produced-water process (#201, #341): a water, a solvent record, a scope, a case, a stage
count and a variant; its build, solve and disposition, solved in a fresh process with a wall limit. The caller owns the
records folder and saves the record."""

import functools
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import traceback

from prommis.lithium_extraction.thermodynamics.reaction_networks import load_constants
from prommis.lithium_extraction.paths import REPO_ROOT
from prommis.lithium_extraction.process import waters as brines, feeds as records
from prommis.lithium_extraction.process.feeds import SOLVENTS

STUDY = REPO_ROOT / "analyses/prototypes/modular_process"
MATRIX = STUDY / "results/site_solvent_matrix"
CAMPAIGN_SOLVENTS = ("fresh", "A", "C", "D2EHDTPA-Held", "D2EHDTPA-Figiel")
# Every status a record can carry. A fixed design point that misses a requirement is 'target_not_met' or
# 'no_product' (#339; before #342 both were 'infeasible'), and one whose reconstructed hydrogen is negative is
# 'outside_domain'; 'infeasible' is reserved for a bound that proves a cell infeasible over its whole design range.
DISPOSITIONS = (
    "grid",
    "optimal",
    "infeasible",
    "unresolved",
    "outside_domain",
    "no_product",
    "target_not_met",
)
FIXED_POINT_FAILURES = ("infeasible", "no_product", "target_not_met", "outside_domain")


@functools.cache
def case_assumptions(case, rule=False, *, entrainment_case=None):
    """Prices, cake moisture, wash volumes, scrub and strip fractions, acid recovery, concentrate molality and recycle
    belong to the case, not the water. Read once per case from the MS-2 R4 replay: the four retained replays agree on
    them except the starting extractant (a design start, kept with each water) and the last digit of the diluent
    price conversion. The shared sourced entrainment (#342) is added for D2EHDTPA/BuPhen; saponified HBTA/TOPO keeps
    the replay's own organic loss and HBTA/TOPO prices unless `rule` (a #342 rule job), which drops them: HBTA and
    TOPO are then the unpriced active set and lose organic by the shared entrainment."""
    from prommis.lithium_extraction.process.costing import entrainment_l_per_l

    a = json.loads((STUDY / f"results/full_process/MS2_{case}.json").read_text())[
        "full_process"
    ]["assumptions"]
    dropped = ("extractant_mol_l", "band_end") + (
        ("organic_loss", "organic_loss_provenance") if rule else ()
    )
    loss_case = case if entrainment_case is None else entrainment_case
    return {k: n for k, n in a.items() if k not in dropped} | {
        "entrainment_l_per_l": entrainment_l_per_l(loss_case),
        "entrainment_provenance": f"data/reference/tea/solvent_entrainment.csv:{loss_case} (#342); no purge",
    }


def washed_lithium(softening, case):
    washed = softening[case]
    return washed["aqueous_mol"]["Li"] / washed["feed_l"]


def assumptions(site, case, front_end=None, composition=None, rule=False):
    """One cell's inputs: the case assumptions and the water's softened state.

    A curated water keeps its retained #176/#177 replay (softening, starting design, initial downstream accounts).
    Any other sample ID reads its softening record for the case from this matrix (`run_softening --brine <ID> --case
    <case>`; each case has its own acceptance) and starts from the
    retained profile and extractant of the curated water nearest in washed-inlet lithium.
    """
    path = STUDY / f"results/full_process/{site}_{case}.json"
    if path.exists():
        result = json.loads(path.read_text())
        result["initial_accounts"] = result[
            "full_process"
        ]  # the replay's accounts start the downstream units
    else:
        softening = json.loads(
            (MATRIX / "softening" / f"{site}_{case}.json").read_text()
        )
        retained = {
            s: json.loads((STUDY / f"results/full_process/{s}_{case}.json").read_text())
            for s in brines.CURATED
        }
        lithium = washed_lithium(softening, case)
        nearest = min(
            retained,
            key=lambda s: abs(washed_lithium(retained[s]["softening"], case) - lithium),
        )
        near = retained[nearest]
        result = {
            "softening": softening,
            "fixed_design": near["fixed_design"],
            "starting_water": nearest,
            "starting_profile": {
                "organic": [
                    {
                        "li": r["li_org_out_mol_l"],
                        "na": r["na_org_out_mol_l"],
                        "hbta": r["hbta_org_out_mol_l"],
                    }
                    for r in near["initial_direct_cascade"]["stages"]
                ],
                "guesses": [
                    r["certificate"]["phase_amounts_mol"] for r in near["stages"]
                ],
            },
        }
    result["full_process"] = dict(
        result.get("full_process", {}), assumptions=case_assumptions(case, rule)
    )
    if front_end is not None:
        result["softening"] = json.loads(
            (
                STUDY / f"results/front_end/softening/{front_end}_{site}_{case}.json"
            ).read_text()
        )
    if (
        composition is not None
    ):  # a scaled water (#342 target paths): its own softening; accounts from that state
        result["softening"] = json.loads(
            (REPO_ROOT / composition["softening"]).read_text()
        )
        result.pop("initial_accounts", None)
    return result


def raw_feed(job):
    """The job's raw water: measured, or with Ca/Mg or lithium scaled (`job['composition']`, #342 target paths)."""
    c = job.get("composition") or {}
    return brines.raw_feed(
        job["site"], c.get("scale", 1.0), lithium_scale=c.get("lithium_scale", 1.0)
    )


def unavailable(site):
    """Why a water cannot enter a cell (a required analyte or its water mass is missing), else None."""
    try:
        raw = brines.raw_feed(site)
    except ValueError as error:
        if "missing required analytes" in str(error):
            return str(error)
        raise
    return (
        None
        if raw["water_kg"] is not None
        else f"{site}: water mass unavailable: SG or TDS not reported"
    )


def bound_disposition(job, reference):
    """A cell is infeasible only when a bound proves it over the whole design range (#201 campaign bounds)."""
    from prommis.lithium_extraction.process import flowsheet

    design = SOLVENTS[job["solvent"]]["design"]
    bounds, target = (
        records.bounds(job["solvent"], job.get("end"), job.get("rule", False)),
        design["li_target"],
    )
    softening = reference["softening"][job["case"]]
    inlet_li = (
        softening["aqueous_mol"]["Li"] / softening["feed_l"]
        if job["scope"] == "full_process"
        else raw_feed(job)["mol"]["Li"]
    )
    capacity = (
        0.72 * bounds["organic_to_aqueous_ratio"][1] * bounds["extractant_mol_l"][1]
    )
    if target is not None and capacity < target * inlet_li:
        return dict(
            status="infeasible",
            reason="Li target exceeds maximum loaded-organic capacity throughout decision bounds",
            capacity_mol_per_nominal_l=capacity,
            required_li_mol_per_nominal_l=target * inlet_li,
        )
    if job["scope"] == "full_process":
        a = reference["full_process"]["assumptions"]
        li_max = (
            min(inlet_li, capacity)
            * (1 - a["scrub_li_loss"])
            * flowsheet.strip_li_fraction()
        )
        salt_min = a["concentrate_li_mol_kg"] * (
            a["acid_recovery"]
            + (1 - a["acid_recovery"])
            * 0.25
            * bounds["organic_to_aqueous_ratio"][0]
            / li_max
        )
        if salt_min > 3:
            return dict(
                status="infeasible",
                reason="carbonation outside the measured solubility data throughout decision bounds",
                minimum_mother_nacl_mol_kg=salt_min,
            )
    return None


def jobs(sites=tuple(brines.CURATED), solvents=CAMPAIGN_SOLVENTS):
    """Cells of every site and solvent, from each solvent record's scopes, cases, stage counts and variants."""
    rows = []
    for solvent in solvents:
        record = SOLVENTS[solvent]
        design = record["design"]
        for site, (scope, cases) in itertools.product(sites, design["scopes"].items()):
            for case, n, end in itertools.product(
                cases, design["stages"], record["variants"]
            ):
                identity = record.get("cell_id", {}).get(
                    scope, "{solvent}_{site}_{scope}_{case}_n{stages}_{end}"
                )
                job = dict(
                    id=identity.format(
                        solvent=solvent,
                        site=site,
                        scope=scope,
                        case=case,
                        stages=n,
                        end=end,
                    ),
                    solvent=solvent,
                    site=site,
                    scope=scope,
                    case=case,
                    stages=n,
                    end=end,
                )
                if (
                    solvent == "C"
                    and scope == "full_process"
                    and site in brines.CURATED
                ):
                    job["composition"] = {
                        "softening": str(
                            (
                                STUDY
                                / "results/softening_to_specification/softening"
                                / f"{site}_{case}.json"
                            ).relative_to(REPO_ROOT)
                        )
                    }
                rows.append(job)
    return rows


def interpolated(rows, n):
    """`n` stage rows taken from a retained profile of another stage count."""
    return [rows[round(k * (len(rows) - 1) / max(n - 1, 1))] for k in range(n)]


def build(job, fixed=None, *, pinned_reference=None, fit_constants=None):
    from prommis.lithium_extraction.extraction import set_c, stage
    from prommis.lithium_extraction.extraction import countercurrent as cc

    solvent, site, scope = (job[k] for k in ("solvent", "site", "scope"))
    record = SOLVENTS[solvent]
    adapter = records.stage_adapter(record["stage_adapter"])
    bounds = records.bounds(solvent, job["end"], job.get("rule", False))
    reference = (
        pinned_reference
        if pinned_reference is not None
        else assumptions(
            site,
            job["case"],
            job.get("front_end"),
            job.get("composition"),
            job.get("rule", False),
        )
    )
    soft = reference["softening"]
    washed = soft[job["case"]]
    raw = raw_feed(job)
    if scope == "full_process":
        feed = set_c.washed_feed(washed)
        feed["raw_brine"] = raw
        feed["labels"] = [
            s for s in feed["labels"] if adapter is set_c or "#158 set C" not in s
        ]
        feed["labels"] += raw["labels"] + record["labels"]
    else:
        feed = records.extraction_feed(raw, solvent)
        feed["raw_brine"] = raw
        washed = {
            "feed_l": 1.0,
            "aqueous_mol": {
                "Li": feed["model_feed"]["Li"],
                "Na": feed["model_feed"]["Na"],
            },
            "pH": -math.log10(feed["model_feed"]["H"]),
        }
    released = (
        job["phase"] != "fixed" and fixed and fixed.get("status") in ("grid", "optimal")
    )
    if record.get("support_rebuild"):  # stage supports come from a direct cascade
        start = (fixed if released else reference)["fixed_design"]
        # A fixed solve uses the record's retained design. A released solve starts each free decision from its warm
        # record (#342 O6); a decision it holds (`job['hold']`) or cannot move keeps the record's retained value.
        retained = record["design"].get("fixed", {})
        free = {k for k, (lo, hi) in bounds.items() if lo != hi} - set(
            job.get("hold", ())
        )
        design = {
            k: (
                start[k]
                if released and (k in free or k not in retained)
                else retained.get(k, start[k])
            )
            for k in bounds
        }
        design |= job.get("design", {})
        c = design["extractant_mol_l"]
        constants = (
            fit_constants
            if fit_constants is not None
            else load_constants(REPO_ROOT / job.get("constants", record["constants"]))[
                "ends"
            ][job["end"]]["ln_k"]
        )
        design["ln_k"] = constants
        if (
            fixed and "stages" in fixed and not job.get("cold_start")
        ):  # cold start: a direct cascade at this design
            direct = direct_from_result(
                fixed, job["stages"], front_end=job.get("front_end")
            )
        else:
            start = (
                None
                if job.get("cold_start") and pinned_reference is not None
                else reference.get("starting_profile")
            )
            if start is not None:
                start = {
                    k: interpolated(rows, job["stages"]) for k, rows in start.items()
                }
            direct = cc.countercurrent(
                constants,
                reaction_set="C",
                stage_count=job["stages"],
                organic_to_aqueous=design["organic_to_aqueous_ratio"],
                extractant=c,
                saponification_fraction=design["saponification_ratio"],
                feed_li=washed["aqueous_mol"]["Li"] / washed["feed_l"],
                feed_na=washed["aqueous_mol"]["Na"] / washed["feed_l"],
                feed_acidity=cc.acidity_at_ph(washed["pH"]),
                start=start,
            )
        model = adapter.build_flowsheet(
            washed,
            stages=job["stages"],
            scenario=design,
            direct_cascade=direct,
            physical_feed=feed,
        )
        if scope == "extraction_only":
            model._physical_feed = feed
    else:
        source = json.loads((STUDY / record["starting_record"]).read_text())
        design = dict(source["fixed_design"])
        if released:
            design = dict(fixed["fixed_design"])
            if not record["saponified"]:
                design.pop("saponification_ratio", None)
        if scope == "full_process":
            v = washed["feed_l"]
            nominal = dict(
                feed, model_feed={k: n / v for k, n in feed["model_feed"].items()}
            )
            model = adapter.build_flowsheet(
                nominal, stages=job["stages"], scenario=design
            )
            model._washed, model._physical_feed = washed, feed
        else:
            model = adapter.build_flowsheet(feed, stages=job["stages"], scenario=design)
        profiles = []
        old_rows = fixed["stages"] if fixed and "stages" in fixed else source["stages"]
        for old in interpolated(old_rows, job["stages"]):
            profile = dict(
                old.get("outlet_mol_per_nominal_l", old.get("outlet_mol_per_raw_l"))
            )
            if not fixed:
                original = source["stages"][0]["inlet_mol_per_raw_l"]
                for s in model._chemistry.supports[0]:
                    if original.get(s, 0):
                        profile[s] *= model._feeds[0].get(s, 0) / original[s]
            profiles.append(profile)
        stage.initialize(model, profiles)
    model._case, model._end, model._rule = (
        job["case"],
        job["end"],
        job.get("rule", False),
    )
    return model, soft, reference


def direct_from_result(result, stage_count=None, *, front_end=None):
    from prommis.lithium_extraction.extraction import set_c

    rows, guesses = [], []
    old = result["stages"]
    n = len(old) if stage_count is None else stage_count
    profiles = None
    if front_end and n > len(old):
        # Interpolate phase-stream boundaries, preserving every stage's conserved quantities.
        aqueous = [old[0]["inlet_mol_per_nominal_l"]] + [
            r["outlet_mol_per_nominal_l"] for r in old
        ]
        organic = [r["outlet_mol_per_nominal_l"] for r in old] + [
            old[-1]["inlet_mol_per_nominal_l"]
        ]
        profiles = []
        for k in range(n):
            profile = {}
            for phase, streams, offset in zip(
                set_c.SUPPORTS, (aqueous, organic), (1, 0), strict=True
            ):
                position = (k + offset) * len(old) / n
                left = min(int(position), len(old) - 1)
                fraction = position - left
                profile.update(
                    {
                        s: (1 - fraction) * streams[left][s]
                        + fraction * streams[left + 1][s]
                        for s in phase
                    }
                )
            profiles.append(profile)
    for k in range(n):
        row = old[round(k * (len(old) - 1) / max(n - 1, 1))]
        inlet = row["inlet_mol_per_nominal_l"]
        if profiles is not None:
            inlet = {
                s: (
                    (aqueous[0] if k == 0 else profiles[k - 1])[s]
                    if s in set_c.SUPPORTS[0]
                    else (organic[-1] if k == n - 1 else profiles[k + 1])[s]
                )
                for s in set_c.SPECIES
            }
        # Per litre of organic at the record's own O/A (#342 O6; retained records are all at O/A 1).
        state = set_c.state_from_inlet(
            inlet,
            result["fixed_design"]["organic_to_aqueous_ratio"],
            result["fixed_design"]["extractant_mol_l"],
        )
        certificate = row["certificate"]
        rows.append(
            dict(
                li_in_mol_l=state.li,
                na_in_mol_l=inlet["sodium-cation"],
                acidity_in_mol_l=inlet["hydrogen-cation"]
                - inlet.get("hydroxide-anion", 0.0),
                bta_in_mol_l=state.bta,
                li_org_in_mol_l=state.li_org,
                na_org_in_mol_l=state.na_org,
                hbta_org_in_mol_l=state.hbta,
                water_autoprotolysis_active="RW" in certificate["active_reactions"],
            )
        )
        guesses.append(
            certificate["phase_amounts_mol"]
            if profiles is None
            else [{s: profiles[k][s] for s in phase} for phase in set_c.SUPPORTS]
        )
    return {"stages": rows, "start": {"guesses": guesses}}


def d2_start(job, li_mol_l):
    """The record's starting design, moved to the nearest point inside the organic-site range S x Li / (O/A) (#375):
    O/A first, then S, each within its bounds. A start already inside is unchanged; one no design reaches stays outside.
    """
    record = SOLVENTS[job["solvent"]]["design"]
    design = (
        record["fixed"]
        | record["site_fixed"].get(job["site"], {})
        | job.get("design", {})
    )
    domain = job.get("design_range") or record
    (lo, hi), bounds = domain["sites_mol_l"], domain["bounds"]
    sites = design["S"] * li_mol_l / design["O_A"]
    if lo <= sites <= hi:
        return design
    target = min(
        max(sites, lo * (1 + 1e-9)), hi * (1 - 1e-9)
    )  # just inside: the range check is inclusive
    o_a = min(max(design["S"] * li_mol_l / target, bounds["O_A"][0]), bounds["O_A"][1])
    return dict(
        design,
        O_A=o_a,
        S=min(max(target * o_a / li_mol_l, bounds["S"][0]), bounds["S"][1]),
    )


def d2_process_assumptions(job):
    if not job.get("organic_recycle") or "strip_aqueous_to_organic" not in job:
        return case_assumptions(job["case"], True)
    assumptions = case_assumptions(job["case"], True, entrainment_case="base") | {
        "strip_aqueous_to_organic": job["strip_aqueous_to_organic"]
    }
    if job.get("zero_entrainment"):
        assumptions["entrainment_l_per_l"] = 0.0
        assumptions["entrainment_provenance"] += "; zero-entrainment control (#447)"
    return assumptions


def build_d2(job, *, brine=None, start=None, retained=None, pinned_reference=None):
    """Build the fixed extraction and strip start, before any process NLP."""
    from prommis.lithium_extraction.process import (
        d2ehdtpa_buphen_whole_process as d2_full_process,
    )
    from prommis.lithium_extraction.extraction import d2ehdtpa_buphen_stage as d2, stage

    raw = brines.raw_feed(job["site"])
    old = (
        d2.reference_record(job["solvent"])[1]
        if pinned_reference is None
        else pinned_reference
    )
    reference = job.get("fixed_design_record")
    full = job["scope"] == "full_process"
    recycle = full and job.get("organic_recycle", False)
    a = d2_process_assumptions(job) if full else None
    if reference or retained is not None:
        if retained is None:
            retained = json.loads((REPO_ROOT / reference).read_text())
        if any(retained["candidate"][k] != job[k] for k in ("site", "solvent", "case")):
            raise ValueError(
                f"Fixed design and candidate conditions differ: {reference}"
            )
        fixed = {k: retained["fixed_design"][k] for k in ("B", "S", "O_A", "Sr_decade")}
        if not job.get("cold_start"):
            start = [
                r["outlet_mol_per_raw_l"]
                for r in interpolated(retained["stages"], job["stages"])
            ]
    else:
        fixed = d2_start(job, raw["mol"]["Li"])
    ratio = fixed["O_A"]
    design = dict(
        fixed, site_basis_li_mol_l=raw["mol"]["Li"], ln_K_Li_Mg_Ca=old["ln_K_Li_Mg_Ca"]
    )
    model = d2.build_flowsheet(
        raw if brine is None else brine,
        solvent=job["solvent"],
        scope=job["scope"],
        stages=job["stages"],
        scenario=design,
        design_range=job.get("design_range"),
        raffinate_entrainment_l=a["entrainment_l_per_l"] if recycle else 0.0,
    )
    model._case = job["case"]
    if start is not None:
        profiles = start
    elif pinned_reference is not None:
        profiles = [model._feeds[0] | model._feeds[1] for _ in range(job["stages"])]
    else:  # the retained cascade, scaled to this feed and design and taken at this stage count
        profiles = []
        reference_sites = d2.owner.BASIS["site_basis_li_mol_L"] * old["case"]["S"]
        total_sites = design["S"] * design["site_basis_li_mol_l"]
        reference_feed = d2.reference_feed(job["solvent"])["model_feed"]
        for oldrow in interpolated(old["extraction"]["stages"], job["stages"]):
            profile = {
                s: sum(p.get(s, 0.0) for p in oldrow["amounts_mol"])
                for s in model._chemistry.species
            }
            for s in model._chemistry.supports[0]:
                name = d2.records.SPECIES_LABELS[s]
                if reference_feed.get(name, 0.0):
                    profile[s] *= model._feeds[0][s] / reference_feed[name]
            for s in model._chemistry.supports[1]:
                if s not in ("diluent", "buphen"):
                    profile[s] *= total_sites / reference_sites
            profile["diluent"] *= ratio
            profile["buphen"] = ratio * design["B"] - 2 * profile[d2.rc.COMPLEX["Li"]]
            profiles.append(profile)
    stage.initialize(model, profiles)
    if recycle:
        d2.decisions(model, organic_recycle=True)
        d2_full_process.connect_strip(model, job["case"], a)
        stage.initialize(model, profiles)
    return model, raw, design, a


def d2_simulation(job, directory, remaining, records, brine=None, start=None):
    """Run the historical D2 fixed or released design on its declared scope."""
    from pyomo.environ import Var, value
    from prommis.lithium_extraction.process import (
        d2ehdtpa_buphen_whole_process as d2_full_process,
        optimization,
    )
    from prommis.lithium_extraction.extraction import d2ehdtpa_buphen_stage as d2, stage

    model, raw, design, a = build_d2(job, brine=brine, start=start)
    ratio = design["O_A"]
    full = job["scope"] == "full_process"
    recycle = full and job.get("organic_recycle", False)
    fixed_replay = job.get("fixed_replay", False)
    stage.solve(model, directory, wall_seconds=remaining(), fixed_replay=fixed_replay)
    result = d2.report(
        model,
        stage.report(model, certify=False) if fixed_replay else None,
        case=job["case"],
    )
    result["sites_mol_l_organic"] = d2.owner.site_supply(
        design["B"],
        design["S"],
        ratio,
        job["stages"],
        site_basis_li_mol_l=raw["mol"]["Li"],
        design_range=job.get("design_range"),
    )
    result["strip_li_recovery_fraction"] = result["accounts"]["R_rec"]
    result["product_recovery_fraction"] = None
    result["cost_usd_kg_li"] = result["accounts"]["priced_subtotal_usd_kg_li"]
    result["cost_usd_m3_raw_brine"] = result["accounts"]["priced_subtotal_usd_m3"]
    result["cost_basis"] = "kg strip-recovered Li per m3 raw brine"
    result["calibration_coverage"] = [
        {
            "quantity": name,
            "value": result["feed"]["model_feed"][name],
            "unit": "mol/L aqueous",
            "basis": "nominal raw aqueous model feed",
            "comparison": "physical calibration coverage unverified",
            "source_owner": SOLVENTS[job["solvent"]]["feed_owner"],
        }
        for name in ("Li", "Na")
    ] + [
        {
            "quantity": "process pH",
            "value": None,
            "unit": "dimensionless",
            "comparison": "H/pH not represented by existing D2 feed chemistry",
            "source_owner": SOLVENTS[job["solvent"]]["feed_owner"],
        }
    ]
    if not recycle:
        outlet = result["outlet_mol_per_raw_l"]
        result["na_metric"] = {
            "name": "loaded-organic Na/(Li+Na)",
            "value": outlet["naa"] / (outlet["naa"] + outlet[d2.rc.COMPLEX["Li"]]),
        }
    result["outlet_li_loading_fractions"] = []
    if not full:
        return result
    stopping_point = "construction"
    certified = not fixed_replay
    try:
        try:
            initial = d2_full_process.accounts(
                d2.strip_accounts(result["accounts"], job["solvent"]),
                a,
                raw["mol"]["Li"],
            )
        except ValueError as error:
            if "no Li2CO3 product" not in str(error):
                raise
            if not certified:
                result.update(d2.report(model, case=job["case"]))
            return result | {
                "downstream_admissible": False,
                "reason": f"no carbonate product: {error}",
            }
        interval = d2_full_process.carbonation_interval(initial["mother_nacl_mol_kg"])
        if interval is None:
            if not certified:
                result.update(d2.report(model, case=job["case"]))
            return result | {
                "downstream_admissible": False,
                "reason": "no carbonate product: outside the measured solubility data",
            }
        extraction_strip_solve = result["solve"]
        if not recycle:
            d2.decisions(model)
        d2_full_process.connect_units(model, initial, a, interval)
        stopping_point = "solve"
        stage.solve(
            model,
            directory / "full_process",
            wall_seconds=remaining(),
            fixed_replay=fixed_replay,
        )
        starting = optimization.rule_values(model) | {
            "design": {k: value(v) for k, v in model._decisions.items()},
            "carbonation_interval_mol_kg": list(interval),
        }
        if job["phase"] == "selected":
            stopping_point = "optimization"
            d2.release(
                model,
                job["objective"],
                minimum_li_extraction=job.get("minimum_li_extraction"),
            )
            stage.solve(model, directory / "released", wall_seconds=remaining())
            moves = []
            while (
                True
            ):  # measured intervals are enumerated outside the solver, never extrapolated
                salt, (lo, hi) = (
                    value(model.fs.carbonation.nacl_mol_kg),
                    model._carbonation_interval,
                )
                knots = d2_full_process.carbonation_interval
                neighbour = (
                    knots(hi + 1e-12)
                    if abs(salt - hi) <= 1e-9
                    else (
                        knots(lo - 1e-12) if abs(salt - lo) <= 1e-9 and lo > 0 else None
                    )
                )
                if neighbour is None or neighbour == [lo, hi]:
                    break
                kept = (
                    [(v, v.value) for v in model.component_data_objects(Var)],
                    value(model.rule),
                    model._solve,
                    model._solved_values,
                    [lo, hi],
                )
                d2_full_process.set_interval(model, neighbour)
                try:
                    stage.solve(
                        model,
                        directory / f"interval_{neighbour[0]:g}_{neighbour[1]:g}",
                        wall_seconds=remaining(),
                    )
                    improved = value(model.rule) < kept[1] - 1e-9 * abs(kept[1])
                except RuntimeError as error:
                    improved, error_text = False, str(error)
                else:
                    error_text = None
                moves.append(
                    dict(interval=neighbour, improved=improved, error=error_text)
                )
                if not improved:
                    for v, x in kept[0]:
                        v.set_value(x, skip_validation=True)
                    model._solve, model._solved_values = kept[2], kept[3]
                    d2_full_process.set_interval(model, kept[4])
                    break
            result["carbonation_interval_moves"] = moves
        stopping_point = "reconciliation"
        downstream_result = result | d2.report(model, case=job["case"])
        result = downstream_result
        certified = True
        reported = d2_full_process.report(model, downstream_result)
        solved = {k: value(v) for k, v in model._decisions.items()}
        return (
            reported
            | ({"extraction_strip_solve": extraction_strip_solve} if recycle else {})
            | {
                "starting_design_rule": starting,
                "rule": optimization.rule_values(model)
                | {"objective": getattr(model, "_objective", None)},
                # organic site concentration S x feed Li / (O/A) at the solved design (the #188 0.020-0.060 M range)
                "sites_mol_l_organic": solved["S"] * raw["mol"]["Li"] / solved["O_A"],
            }
        )
    except Exception as error:
        if not certified:
            result.update(d2.report(model, case=job["case"]))
        return result | {
            "reason": f"{type(error).__name__}: {error}",
            "stopping_point": stopping_point,
            "exception_traceback": traceback.format_exc(),
        }


def fixed_disposition(result, solvent):
    """An uncertified state cannot prove a fixed design physically infeasible."""
    from prommis.lithium_extraction.process import optimization as opt

    if not result["strict_certificate"]:
        reasons = ["owner strict certificate does not hold"]
        reasons += [
            r["certificate"]["reason"]
            for r in result["stages"]
            if r["certificate"].get("reason")
        ]
        return dict(status="unresolved", reason="; ".join(dict.fromkeys(reasons)))
    # #339: 'infeasible' is reserved for a bound that proves a cell infeasible over its whole design range
    # (bound_disposition). One fixed design point that has no product, leaves the composition mapping or misses a
    # requirement proves nothing about the range, so it gets its own label.
    if result.get("downstream_admissible") is False:
        return dict(
            status="no_product",
            reason="no carbonate product: fixed design: carbonation outside the measured solubility data or acid bounds",
        )
    failures = opt.target_failures(result, solvent)
    numerical = any(
        "certificate" in f or "balance" in f or "charge" in f or "inconsistent" in f
        for f in failures
    )
    numerical |= any(
        not math.isfinite(n) for n in result["R7"]["physical_raffinate"].values()
    )
    if (
        SOLVENTS[solvent].get("support_rebuild")
        and not numerical
        and result["R7"]["physical_raffinate"]["H"] < 0
    ):
        return dict(
            status="outside_domain",
            reason="; ".join(
                [
                    "Fixed design is infeasible under the declared nominal-acidity composition mapping: reconstructed hydrogen is negative. "
                    "Buffered-brine behavior and additional acidity conditioning were not evaluated.",
                    *(f for f in failures if "physical raffinate" not in f),
                ]
            ),
        )
    return dict(
        status=(
            "grid" if not failures else "unresolved" if numerical else "target_not_met"
        ),
        reason="fixed design grid result" if not failures else "; ".join(failures),
    )


def objective_value(result, objective):
    """The value an objective maximizes (sign-adjusted), from a record's rule values."""
    kind = objective["kind"]
    if kind not in ("affordable", "headroom", "uniform", "priced_margin"):
        raise ValueError(f"unknown objective {kind}")
    rule = result["rule"]
    if kind == "affordable":
        return rule["affordable_price_usd_kg"]
    if kind == "headroom":
        return rule["headroom_usd_m3"]
    if kind == "priced_margin":
        from prommis.lithium_extraction.process.costing import ligand_cost

        q, cost = rule["carbonate_kg_m3"], rule["cost_outside_active_usd_m3"]
        return (
            objective["carbonate_price_usd_t_lce"] * q / 1000
            - cost
            - ligand_cost(
                rule["active_loss_by_component_kg_m3"],
                objective["ligand_prices_usd_kg"],
            )
        )
    return (
        -(
            rule["cost_outside_active_usd_m3"]
            + objective["theta"] * rule["active_loss_kg_m3"]
        )
        / rule["carbonate_kg_m3"]
    )


def worse(result, fixed, objective):
    """A local optimum below its own accepted starting design, beyond the 1e-6 relative comparison limit."""
    if objective is None:
        return result["cost_usd_kg_li"] > fixed["cost_usd_kg_li"] * (1 + 1e-6)
    optimum, start = objective_value(result, objective), objective_value(
        fixed, objective
    )
    return optimum < start - 1e-6 * abs(start)


def starting_record(job, records):
    """(parent fixed record, record the solve starts from, its label or None) of a non-D2 candidate. A job's
    `warm_start` (repository path) names the record to start from, as a #342 check starts from a solved optimum.
    """
    if job.get("warm_start"):
        parent = (
            None
            if job["phase"] == "fixed"
            else json.loads(
                (records / "fixed" / (job["parent_id"] + ".json")).read_text()
            )
        )
        return (
            parent,
            json.loads((REPO_ROOT / job["warm_start"]).read_text()),
            job["warm_start"],
        )
    fixed = (
        None
        if job["phase"] == "fixed"
        else json.loads((records / "fixed" / (job["parent_id"] + ".json")).read_text())
    )
    if fixed is None and SOLVENTS[job["solvent"]].get("support_rebuild"):
        neighbours = [
            r
            for f in (records / "fixed").glob("*.json")
            if (r := json.loads(f.read_text())).get("stages")
            and all(
                r["candidate"][k] == job[k]
                for k in ("solvent", "site", "scope", "case", "end")
            )
            and (not job.get("front_end") or r["candidate"]["stages"] < job["stages"])
        ]
        fixed = min(
            neighbours,
            key=lambda r: abs(r["candidate"]["stages"] - job["stages"]),
            default=None,
        )
    initial, source = fixed, None
    if job.get("front_end"):
        source = (
            "fixed/" + fixed["candidate"]["id"] + ".json" if fixed else "direct cascade"
        )
    if job["phase"] != "fixed":
        paths = [records / "selected" / (job["id"] + ".json")]
        paths += sorted((records / "selected").glob("*.json"))
        paths += sorted((records / "feasibility").glob("*.json"))
        warm = [
            (p, r)
            for p in paths
            if p.exists()
            and (r := json.loads(p.read_text())).get("status") in ("optimal", "grid")
            and (not job.get("front_end") or r["candidate"]["stages"] < job["stages"])
            and all(
                r["candidate"][k] == job[k]
                for k in ("solvent", "site", "scope", "case", "end")
            )
        ]
        # Keep a feasible same-stage fixed profile before interpolating another stage count.
        if fixed["status"] == "grid":
            source = "fixed/" + job["parent_id"] + ".json"
            warm = [
                (p, r) for p, r in warm if r["candidate"]["stages"] == job["stages"]
            ]
        if warm:
            p, initial = min(
                warm,
                key=lambda item: (
                    abs(item[1]["candidate"]["stages"] - job["stages"]),
                    item[1]["cost_usd_kg_li"],
                ),
            )
            source = str(p.relative_to(records))
    return fixed, initial, source


def source_identity():
    """sha256 of every loaded module under this package's parent, found from the module's own file."""
    package = __spec__.name.rsplit(".", 2)[0] + "."
    return {
        name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
        for name, m in sorted(sys.modules.items())
        if name.startswith(package) and getattr(m, "__file__", None)
    }


def solve_candidate(job, records, scratch, *, wall_seconds=290):
    """Solve one campaign candidate and return its record without saving it.

    `records` is the campaign records folder read for parent, neighbour and warm-start records; solver files go to
    `scratch`.
    """
    started = time.monotonic()
    directory = Path(scratch)
    directory.mkdir(parents=True, exist_ok=True)
    from pyomo.common.tempfiles import TempfileManager
    from prommis.lithium_extraction.process import (
        flowsheet,
        whole_process as full_process,
        optimization as opt,
    )
    from prommis.lithium_extraction.extraction import set_c, stage

    TempfileManager.tempdir = str(directory)
    result = dict(
        type="prototype",
        candidate=job,
        status="unresolved",
        engine=flowsheet.installed_engine(),
        claim="model prediction; no physical validation",
        candidate_wall_limit_s=wall_seconds,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    if job.get("design_range"):
        result["claim"] = (
            "extrapolation sensitivity (#373); local solution; no physical validation"
        )

    def remaining():
        left = wall_seconds - (time.monotonic() - started)
        if left <= 0:
            raise TimeoutError(f"{wall_seconds} s candidate wall budget exhausted")
        return left

    record = SOLVENTS[job["solvent"]]
    rebuild = record.get(
        "support_rebuild"
    )  # set C selects stage supports; a change rebuilds the model
    try:
        if (reason := unavailable(job["site"])) is not None:
            result.update(status="unavailable", reason=reason)
        elif (
            record["stage_adapter"] == "d2_stage"
        ):  # the adapter's own cell: start, connect and release in one process
            from prommis.lithium_extraction.process import optimization as opt

            result.update(d2_simulation(job, directory / "solver", remaining, records))
            extraction_failed = not result["strict_certificate"] or bool(
                result["R7"]["failures"]
            )
            minimum = (
                0.0
                if job.get("fixed_design_record") and job["phase"] == "fixed"
                else job.get("minimum_li_extraction")
            )
            failures = (
                []
                if extraction_failed
                or "exception_traceback" in result
                or "rule" not in result
                else opt.target_failures(result, job["solvent"], minimum)
            )
            selected = job["phase"] == "selected" and "rule" in result
            if selected and worse(
                result, {"rule": result["starting_design_rule"]}, job["objective"]
            ):
                failures.append(
                    "optimum worse than accepted fixed design at identical conditions"
                )
            result["status"] = (
                "unresolved"
                if extraction_failed or "exception_traceback" in result or failures
                else (
                    "no_product"
                    if result.get("downstream_admissible") is False
                    else "optimal" if selected else "grid"
                )
            )
            if extraction_failed:
                result["reason"] = "uncertified extraction; " + "; ".join(
                    result["R7"]["failures"]
                )
            elif failures:
                result["reason"] = "; ".join(failures)
            result.setdefault(
                "reason",
                (
                    "IPOPT optimum; strict certificates, balances, account recomputation and KKT "
                    "limits pass"
                    if selected
                    else "design solved at fixed decisions"
                ),
            )
        else:
            fixed, initial, source = starting_record(job, records)
            if source is not None:
                result["initialization_source"] = source
            model, soft, reference = build(job, initial)
            opt.decisions(model, job["solvent"])
            if job["phase"] == "fixed":
                if rebuild:
                    # A support rebuild returns a new model; its decisions and costing join the solve.
                    model, extracted = set_c.solve(
                        model,
                        directory / "extraction",
                        started=started,
                        wall_seconds=wall_seconds,
                        connect_units=lambda m: hasattr(m, "_decisions")
                        or opt.decisions(m, job["solvent"]),
                    )
                else:
                    stage.solve(
                        model, directory / "extraction", wall_seconds=remaining()
                    )
                if job["scope"] == "full_process":
                    if rebuild:
                        result.update(extracted)
                    else:
                        result.update(
                            opt.nominal_to_raw_litre(
                                stage.report(model), model._washed["feed_l"]
                            ),
                            fixed_design=dict(model._design),
                        )
                    extracted = full_process.extraction(model)
                    enumeration = full_process.enumerate_units(
                        model._washed,
                        extracted["loaded_organic"],
                        reference["full_process"]["assumptions"],
                        organic_to_aqueous_ratio=model._design[
                            "organic_to_aqueous_ratio"
                        ],
                    )
                    result["unit_enumeration"] = enumeration
                    candidate = next((r for r in enumeration if r["feasible"]), None)
                    if candidate is None:
                        result["downstream_admissible"] = False
                    else:
                        full_process.connect_units(
                            model,
                            job["case"],
                            reference["full_process"]["assumptions"],
                            candidate,
                        )
                        stage.solve(model, directory / "full", wall_seconds=remaining())
                        result.update(opt.report(model, job["scope"], soft))
                else:
                    result.update(opt.report(model, job["scope"]))
                result.update(fixed_disposition(result, job["solvent"]))
            else:
                candidate = job.get("unit")
                if job["scope"] == "full_process" and candidate is None:
                    enumeration = full_process.enumerate_units(
                        model._washed,
                        full_process.extraction(model)["loaded_organic"],
                        reference["full_process"]["assumptions"],
                        organic_to_aqueous_ratio=model._design[
                            "organic_to_aqueous_ratio"
                        ],
                    )
                    candidate = next(
                        (r for r in enumeration if r["feasible"]),
                        reference.get("unit_enumeration_selected"),
                    )
                    if candidate is None:
                        raise RuntimeError(
                            "no admissible downstream interval/regime at the starting design"
                        )
                provisional = (
                    job["phase"] in ("selected", "feasibility")
                    and job["scope"] == "full_process"
                )
                selections = []

                def connect(m):
                    m._case, m._end, m._rule = (
                        job["case"],
                        job["end"],
                        job.get("rule", False),
                    )
                    if not hasattr(m, "_decisions"):
                        opt.decisions(m, job["solvent"])
                    if job["scope"] == "full_process" and not hasattr(
                        m, "_full_process"
                    ):
                        full_process.connect_units(
                            m,
                            job["case"],
                            reference["full_process"]["assumptions"],
                            candidate,
                            initial_accounts=(
                                None
                                if job.get("front_end") == "sorbent"
                                else (initial or {}).get(
                                    "full_process", reference.get("initial_accounts")
                                )
                            ),
                        )
                    if not hasattr(m, "minimum_cost") and not hasattr(m, "rule"):
                        opt.add_optimization(
                            m,
                            job["scope"],
                            objective=job.get("objective"),
                            minimum_li_extraction=job.get("minimum_li_extraction"),
                            hold=job.get("hold", ()),
                        )
                        if job["phase"] == "feasibility":
                            m.minimum_cost.deactivate()
                            m.feasibility.activate()
                    if provisional:
                        m.fs.carbonation.interval.deactivate()
                        m.fs.scrub.other_bound.deactivate()

                while True:
                    if rebuild:
                        model, _ = set_c.solve(
                            model,
                            directory / "released",
                            started=started,
                            wall_seconds=wall_seconds,
                            connect_units=connect,
                        )
                    else:
                        connect(model)
                        stage.solve(
                            model, directory / "released", wall_seconds=remaining()
                        )
                    opt.synchronize(model)
                    if not provisional:
                        break
                    enumeration = full_process.enumerate_units(
                        model._washed,
                        full_process.extraction(model)["loaded_organic"],
                        reference["full_process"]["assumptions"],
                        organic_to_aqueous_ratio=model._design[
                            "organic_to_aqueous_ratio"
                        ],
                        model=model,
                    )
                    selected = next((r for r in enumeration if r["feasible"]), None)
                    if selected is None:
                        raise RuntimeError(
                            "converged extraction has no admissible downstream interval/regime"
                        )
                    selections.append(selected)
                    consistent = all(
                        selected[k] == candidate[k]
                        for k in ("scrub_regime", "carbonation_interval_mol_kg")
                    )
                    candidate = selected
                    model._full_process["candidate"] = candidate
                    full_process.select_units(model, candidate)
                    if consistent:
                        break
                    remaining()
                result["unit_selections"] = selections
                result.update(opt.report(model, job["scope"], soft))
                failures = opt.target_failures(
                    result, job["solvent"], job.get("minimum_li_extraction")
                )
                if (
                    job["phase"] != "feasibility"
                    and fixed["status"] == "grid"
                    and worse(result, fixed, job.get("objective"))
                ):
                    failures.append(
                        "optimum worse than accepted fixed design at identical conditions"
                    )
                result.update(
                    status="optimal" if not failures else "unresolved",
                    reason="; ".join(failures)
                    or "IPOPT optimum; strict certificates, balances and version 1.0 KKT limits pass",
                )
                if job["phase"] == "feasibility" and not failures:
                    result.update(
                        status="grid",
                        reason="feasibility-only solve; all original bounds and targets retained; cost not minimized",
                    )
                result["fixed_design_cost_usd_kg_li"] = (
                    fixed.get("cost_usd_kg_li") if fixed["status"] == "grid" else None
                )
    except Exception as error:
        from prommis.lithium_extraction.extraction.d2ehdtpa_buphen_accounts import (
            OutsideDomain,
        )

        result.update(
            status=(
                "outside_domain" if isinstance(error, OutsideDomain) else "unresolved"
            ),
            reason=f"{type(error).__name__}: {error}",
            stopping_point=job["phase"],
        )
        result["exception_traceback"] = traceback.format_exc()
        if "model" in locals():
            from pyomo.environ import Constraint, value

            result["last_stage_amounts"] = [
                {s: value(n) for s, n in model.fs.stage[k].outlet.items()}
                for k in model.fs.stages
            ]
            result["last_decision_values"] = {
                k: value(v) for k, v in model._decisions.items()
            }
            result["last_li_extraction_fraction"] = value(model.fs.design.li_extraction)
            if hasattr(model.fs, "carbonation"):
                result["last_carbonation_nacl_mol_kg"] = value(
                    model.fs.carbonation.nacl_mol_kg
                )
            residuals = {}
            for c in model.component_data_objects(Constraint, active=True):
                # Engine responses are evaluated only in IPOPT's fresh process, not accumulated here.
                if ".equilibrium[" in c.name:
                    continue
                body = value(c.body)
                residuals[c.name] = dict(
                    lower=None if c.lower is None else body - value(c.lower),
                    upper=None if c.upper is None else value(c.upper) - body,
                )
            result["last_algebraic_margins"] = residuals
            logs = sorted(directory.rglob("ipopt.log"), key=lambda p: p.stat().st_mtime)
            if logs:
                logfile = logs[-1]
                result["solver_log"] = str(
                    logfile.relative_to(REPO_ROOT)
                    if logfile.is_relative_to(REPO_ROOT)
                    else logfile
                )
                result["solver_log_sha256"] = hashlib.sha256(
                    logfile.read_bytes()
                ).hexdigest()
                result["solver_exit"] = (
                    logfile.read_text().rsplit("EXIT:", 1)[-1].strip()
                )
    finally:
        result["source_sha256"] = source_identity()  # the modules this solve imported
        result["elapsed_wall_s"] = time.monotonic() - started
        if result["elapsed_wall_s"] > wall_seconds:
            result.update(
                status="unresolved",
                reason=f"{wall_seconds} s candidate wall limit, including initialization and checks",
            )
    print(
        job["id"],
        result["status"],
        result["elapsed_wall_s"],
        result.get("reason", ""),
        flush=True,
    )
    return result


def solve_in_process(job, records, wall_seconds=290):
    """`solve_candidate` in a fresh Python process, so each candidate keeps its own Engine state, as in the campaign.

    At the wall limit the process and every process it started are killed; a failed or killed candidate is returned
    as unresolved with the error and the end of its log.
    """
    started = time.monotonic()
    environment = dict(
        os.environ, PYTHONPATH=os.pathsep.join(sys.path)
    )  # imports resolve as in this process
    with tempfile.TemporaryDirectory(
        prefix="li_candidate_", ignore_cleanup_errors=True
    ) as folder:
        output, log = Path(folder) / "result.json", Path(folder) / "worker.log"
        command = [
            sys.executable,
            "-m",
            __spec__.name,
            json.dumps(job),
            str(records),
            str(output),
            json.dumps(wall_seconds),
        ]
        try:
            with log.open("w") as stream:
                process = subprocess.Popen(
                    command,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    code = process.wait(
                        timeout=max(0, wall_seconds - (time.monotonic() - started))
                    )
                except subprocess.TimeoutExpired:
                    if hasattr(os, "killpg"):  # the worker and every process it started
                        os.killpg(process.pid, signal.SIGKILL)
                    else:  # Windows
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                            capture_output=True,
                        )
                    process.wait()
                    raise
            if code:
                raise subprocess.CalledProcessError(code, command[:3])
            return json.loads(output.read_text())
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            OSError,
            ValueError,
        ) as error:
            return {
                "candidate": job,
                "status": "unresolved",
                "elapsed_wall_s": time.monotonic() - started,
                "reason": f"{error}\n{log.read_text(errors='replace')[-2000:]}",
            }


if __name__ == "__main__":
    job, folder, output, wall_seconds = sys.argv[
        1:
    ]  # 'records' names the solvent-record module here
    result = solve_candidate(
        json.loads(job),
        Path(folder),
        Path(output).parent / "scratch",
        wall_seconds=json.loads(wall_seconds),
    )
    Path(output).write_text(json.dumps(result, allow_nan=False))
