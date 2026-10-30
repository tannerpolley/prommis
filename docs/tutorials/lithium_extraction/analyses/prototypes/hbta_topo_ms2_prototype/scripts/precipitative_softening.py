"""Stage 3 of #166: NaOH then Na2CO3 precipitative softening at equilibrium (Reaktoro 2.13, PHREEQC ``pitzer.dat``).

Plan: the #166 stage 3 comment (fixed before any model run). Two equilibrium steps with the solids removed between
them: (1) NaOH, (2) Na2CO3; then, for MS-2 only, HCl back to the feed pH 5.77 with the step-2 solids removed.

- Database: embedded ``pitzer.dat`` plus the repository Li2CO3 extension (``custom_pitzer_contents``); the
  Strontianite and Witherite blocks are copied at run time from Reaktoro's embedded ``phreeqc.dat``.
- Solids allowed: ``PHASES``; Nesquehonite only as a sensitivity.
- Reagents are added as ions without water (no dilution). Amounts are per L of feed brine; the water per L is
  1 kg for the literature feeds (densities not printed; the design-review probe's basis) and SG - TDS = 0.905 kg
  for MS-2.
- Removal = 1 - dissolved after / feed as printed (feed ions after the charge closure).
- Closure: feed + reagents = final water + every removed solid, per element, and aqueous charge, each <= 1e-9 mol;
  the script stops if any state misses it.

Run in the conda environment ``lithium-extraction-reaktoro`` with ``PYTHONPATH=.``. Writes ``results/softening_*``.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import reaktoro as rkt
from scipy.optimize import brentq

from lithium_extraction.common.economics.hbta_topo import load_screening_cost_assumptions
from lithium_extraction.common.pretreatment import NA2CO3_USD_KG, softening
from lithium_extraction.common.precipitation_payload import custom_pitzer_contents
from lithium_extraction.common.support.paths import REPO_ROOT
from lithium_extraction.prototypes.produced_water_sx_prototype.brines import MOLAR_MASS, feed

ANALYSIS = Path(__file__).resolve().parents[1]
RESULTS = ANALYSIS / "results"
SOFTENING = REPO_ROOT / "data/reference/softening"
MS2_RECORD = REPO_ROOT / "data/reference/produced_water/smackover_selected_base_feed_ms2.csv"
PRETREATMENT_REGISTER = REPO_ROOT / "data/reference/tea/pretreatment_cost_assumptions.csv"

PHASES = ("Brucite", "Calcite", "Portlandite", "Strontianite", "Witherite", "Celestite", "Barite", "Gypsum",
          "Anhydrite", "Halite", "Pirssonite", "Gaylussite", "Li2CO3")
ELEMENTS = "H O C Na K Mg Ca Sr Ba Li Cl Br S B"
ION = {"Na": "Na+", "K": "K+", "Mg": "Mg+2", "Ca": "Ca+2", "Sr": "Sr+2", "Ba": "Ba+2", "Li": "Li+", "Cl": "Cl-",
       "Br": "Br-", "SO4": "SO4-2", "B": "B(OH)3", "HCO3": "HCO3-"}
REAGENT = {"NaOH": {"Na+": 1, "OH-": 1}, "Na2CO3": {"Na+": 2, "CO3-2": 1}, "HCl": {"H+": 1, "Cl-": 1},
           "Ca(OH)2": {"Ca+2": 1, "OH-": 2}}
REAGENT_ELEMENTS = {"NaOH": {"Na": 1, "O": 1, "H": 1}, "Na2CO3": {"Na": 2, "C": 1, "O": 3},
                    "HCl": {"H": 1, "Cl": 1}, "Ca(OH)2": {"Ca": 1, "O": 2, "H": 2}}
CLOSURE_MOL = 1e-9
MS2_PH = 5.77  # USGS row, measured at PHT 29.5 degC
MS2_LI_VALUE_USD_M3 = (15.7, 18.7)  # #163 research record, at 100 % recovery (analyses/prototypes/hbta_topo_ms2_saponified)
HCL_SOLUTION_FRACTION = 0.31
LITERATURE_WATER_KG = 1.0  # per L of brine; densities are not printed


def _database() -> rkt.PhreeqcDatabase:
    lines = custom_pitzer_contents(REPO_ROOT).splitlines()
    source = rkt.PhreeqcDatabase.contents("phreeqc.dat").splitlines()
    added = []
    for name in ("Strontianite", "Witherite"):
        start = next(i for i, line in enumerate(source) if line.strip() == name)
        end = start + 1
        while source[end].startswith(("\t", " ")):
            end += 1
        added += source[start:end]
    at = next(i for i, line in enumerate(lines) if line.strip() == "PHASES")
    return rkt.PhreeqcDatabase.fromContents("\n".join(lines[:at + 1] + added + lines[at + 1:]) + "\n")


DB = _database()
_SYSTEMS: dict[tuple, tuple] = {}


def _system(phases: tuple[str, ...]):
    if phases not in _SYSTEMS:
        aqueous = rkt.AqueousPhase(rkt.speciate(ELEMENTS))
        aqueous.set(rkt.ActivityModelPitzer())
        system = rkt.ChemicalSystem(DB, aqueous, *[rkt.MineralPhase(n) for n in phases])
        _SYSTEMS[phases] = (system, rkt.EquilibriumSolver(system))
    return _SYSTEMS[phases]


def _state(system, temperature_c: float, water_kg: float, mol: dict[str, float]) -> rkt.ChemicalState:
    state = rkt.ChemicalState(system)
    state.temperature(temperature_c, "celsius")
    state.pressure(1.0, "atm")
    state.set("H2O", water_kg, "kg")
    for key, amount in mol.items():
        state.set(ION.get(key, key), amount, "mol")
    return state


def _solve(solver, state) -> None:
    result = solver.solve(state)
    if not result.succeeded():
        raise RuntimeError("Reaktoro equilibrium did not converge")


def _add(state, reagent: str, amount: float) -> None:
    for species, nu in REAGENT[reagent].items():
        state.add(species, nu * amount, "mol")


def _elements(system, vector) -> dict[str, float]:
    return {e.symbol(): float(v) for e, v in zip(system.elements(), vector)}


def _dosed(system, solver, state, reagent, amount) -> rkt.ChemicalState:
    trial = rkt.ChemicalState(state)
    _add(trial, reagent, amount)
    _solve(solver, trial)
    return trial


def _to_ph(system, solver, state, reagent, target, upper) -> tuple[float, rkt.ChemicalState]:
    """Reagent amount (mol) that brings the equilibrium to ``target`` pH; bracket [0, upper]."""
    def miss(amount):
        return float(rkt.AqueousProps(_dosed(system, solver, state, reagent, amount)).pH()) - target
    amount = brentq(miss, 0.0, upper, xtol=1e-13, rtol=1e-12)
    return amount, _dosed(system, solver, state, reagent, amount)


def _record(system, state, phases) -> dict:
    props = rkt.ChemicalProps(state)
    aq = rkt.AqueousProps(state)
    phase = props.phaseProps("AqueousPhase")
    solids = {n: float(state.speciesAmount(n)) for n in phases if float(state.speciesAmount(n)) > 1e-12}
    return {"pH": float(aq.pH()),
            "aq": _elements(system, props.elementAmountsInPhase("AqueousPhase")),
            "solids_mol": solids,
            "solids_kg": sum(n * DB.species(k).molarMass() for k, n in solids.items()),  # per L of brine
            "liquor_kg": float(phase.mass()), "liquor_l": float(phase.volume()) * 1000.0,
            "charge_mol": float(props.chargeInPhase("AqueousPhase")),
            "li2co3_si": float(aq.saturationIndex("Li2CO3"))}


def _strip(state, phases) -> None:
    for name in phases:
        state.setSpeciesAmount(name, 0.0, "mol")


def soften(mol, water_kg, temperature_c, *, naoh=None, naoh_ph=None, lime=None, na2co3=0.0, neutralize_ph=None,
           phases=PHASES) -> dict:
    """Feed -> step 1 (NaOH, fixed amount or to ``naoh_ph``; or lime) -> solids out -> step 2 (Na2CO3) -> solids
    out -> optional HCl to ``neutralize_ph``. Returns every step record, reagent amounts and closure."""
    system, solver = _system(tuple(phases))
    state = _state(system, temperature_c, water_kg, mol)
    feed_b = _elements(system, rkt.ChemicalProps(state).elementAmounts())
    added = {"NaOH": 0.0, "Na2CO3": na2co3, "HCl": 0.0, "Ca(OH)2": lime or 0.0}
    out = {"T_C": temperature_c, "steps": {}}
    removed = {e: 0.0 for e in feed_b}
    if naoh_ph is not None or naoh is not None or lime is not None:
        if naoh_ph is not None:
            naoh, state = _to_ph(system, solver, state, "NaOH", naoh_ph, upper=2.0)
        elif naoh is not None:
            state = _dosed(system, solver, state, "NaOH", naoh)
        else:
            state = _dosed(system, solver, state, "Ca(OH)2", lime)
        added["NaOH"] = naoh or 0.0
        out["steps"]["step1"] = _record(system, state, phases)
        for e, v in _elements(system, rkt.ChemicalProps(state).elementAmounts()).items():
            removed[e] += v - out["steps"]["step1"]["aq"][e]
        _strip(state, phases)
    state = _dosed(system, solver, state, "Na2CO3", na2co3)
    out["steps"]["step2"] = _record(system, state, phases)
    for e, v in _elements(system, rkt.ChemicalProps(state).elementAmounts()).items():
        removed[e] += v - out["steps"]["step2"]["aq"][e]
    last = out["steps"]["step2"]
    if neutralize_ph is not None:
        _strip(state, phases)
        upper = 2.0 * last["aq"]["C"] + 1.0
        added["HCl"], state = _to_ph(system, solver, state, "HCl", neutralize_ph, upper=upper)
        out["steps"]["neutralized"] = last = _record(system, state, phases)
        for e, v in _elements(system, rkt.ChemicalProps(state).elementAmounts()).items():
            removed[e] += v - last["aq"][e]
    reagent_b = {e: sum(n * REAGENT_ELEMENTS[r].get(e, 0) for r, n in added.items()) for e in feed_b}
    closure = {e: last["aq"][e] + removed[e] - feed_b[e] - reagent_b[e] for e in feed_b}
    out |= {"reagent_mol": added,
            "element_closure_mol": max(abs(v) for v in closure.values()),
            "charge_closure_mol": max(abs(s["charge_mol"]) for s in out["steps"].values())}
    if max(out["element_closure_mol"], out["charge_closure_mol"]) > CLOSURE_MOL:
        raise RuntimeError(f"closure above {CLOSURE_MOL} mol: {closure}")
    return out


# ------------------------------------------------------------------ literature feeds


def _rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def shafer_feeds(close_with="Cl"):
    """Shafer-Peltier 2020 Table 2: ions as printed, alkalinity as HCO3-, charge closed on Cl (or Na)."""
    feeds = {}
    for county in ("Douglas County (DC)", "Reno County (RC)"):
        rows = {r["quantity"]: r["value"] for r in _rows(SOFTENING / "shafer2020_chemical_precipitation.csv")
                if r["sample"] == county and r["stage"] == "feed"}
        mg_l = {k: float(rows[k]) for k in ("Na", "Ca", "Sr", "Mg", "Ba", "Cl", "Br", "HCO3", "SO4")}
        feeds[county[-3:-1]] = {"mol": feed(mg_l, close_with), "water_kg": LITERATURE_WATER_KG,
                                "sampling_T_C": float(rows["Temperature"]), "printed_mg_l": mg_l}
    return feeds


def khajouei_feeds():
    """Khajouei 2021 Table 1; alkalinity (mg/L as CaCO3) as HCO3-; Fe, Mn, Al omitted (no allowed solid)."""
    feeds = {}
    rows = _rows(SOFTENING / "khajouei2021_softening.csv")
    for sample in dict.fromkeys(r["sample"] for r in rows):
        q = {r["quantity"]: r["value"] for r in rows if r["sample"] == sample and r["stage"] == "feed"}
        mg_l = {k: float(q[k]) for k in ("Ca", "Mg", "Na", "Ba", "Sr", "Cl", "SO4")}
        mg_l["HCO3"] = float(q["Alkalinity"]) / 50.0436 * MOLAR_MASS["HCO3"]  # meq/L -> mg/L HCO3-
        feeds[sample.split()[0]] = {"mol": feed(mg_l), "water_kg": LITERATURE_WATER_KG, "printed_mg_l": mg_l}
    return feeds


def lee_feed():
    """Lee 2022 Table 1 synthesized fluid, midpoints of the printed ranges; metal chlorides (Cl closes charge);
    Fe and SiO2 omitted."""
    rows = [r for r in _rows(SOFTENING / "lee2022_chemical_precipitation.csv")
            if r["sample"] == "synthesized geothermal fluid" and r["stage"] == "feed"]
    mg_l = {r["quantity"]: (float(r["value"]) + float(r["value_max"])) / 2 for r in rows
            if r["quantity"] in ("Ca", "Li", "Na", "Sr", "Mg", "Ba", "K")}
    mg_l["Cl"] = 0.0
    return {"mol": feed(mg_l), "water_kg": LITERATURE_WATER_KG}


def _removal(result, mol, ion):
    return 100.0 * (1.0 - result["steps"]["step2"]["aq"][ion] / mol[ion])


def falsifiers() -> tuple[list[dict], list[dict], dict]:
    rows, curves, cases = [], [], {}
    shafer = _rows(SOFTENING / "digitization/shafer2020_figure3/points.csv")
    variants = [("base", {}), ("pH 11.5", {"ph": 11.5}), ("pH 12", {"ph": 12.0}),
                ("sampling temperature", {"T": "sampling"}), ("Na closure", {"close": "Na"}),
                ("nesquehonite", {"phases": PHASES + ("Nesquehonite",)})]
    for variant, v in variants:
        feeds = shafer_feeds(v.get("close", "Cl"))
        for p in shafer:
            county = p["brine"][-3:-1]
            f = feeds[county]
            dose = float(p["na2co3_added_mol_per_l_brine"])
            ion = p["ion"]
            key = (variant, county, dose)
            if key not in cases:
                t = f["sampling_T_C"] if v.get("T") == "sampling" else 25.0
                cases[key] = soften(f["mol"], f["water_kg"], t, naoh_ph=v.get("ph", 11.0), na2co3=dose,
                                    phases=v.get("phases", PHASES))
            r = cases[key]
            model = _removal(r, f["mol"], ion)
            measured = float(p["removal_percent"])
            test, criterion = _shafer_test(county, dose, ion)
            ok = None if test is None else (abs(model - measured) <= 10.0 if test in ("F1", "F2")
                                            else model >= measured - 10.0)
            rows.append({"variant": variant, "test": test or "reported", "criterion": criterion,
                         "source": "Shafer-Peltier 2020 Fig. 3", "sample": county, "ion": ion,
                         "na2co3_mol_l": dose, "naoh_mol_l": r["reagent_mol"]["NaOH"], "measured_removal_pct": measured,
                         "model_removal_pct": model, "model_minus_measured": model - measured, "pass": ok,
                         "step1_pH": r["steps"]["step1"]["pH"], "final_pH": r["steps"]["step2"]["pH"],
                         "T_C": r["T_C"], "element_closure_mol": r["element_closure_mol"],
                         "charge_closure_mol": r["charge_closure_mol"],
                         "solids_step1": json.dumps(r["steps"]["step1"]["solids_mol"]),
                         "solids_step2": json.dumps(r["steps"]["step2"]["solids_mol"])})
    khajouei = _rows(SOFTENING / "digitization/khajouei2021_figure4/points.csv")
    for variant, phases in (("base", PHASES), ("nesquehonite", PHASES + ("Nesquehonite",))):
        feeds = khajouei_feeds()
        for p in khajouei:
            f = feeds[p["brine"].split()[0]]
            dose = float(p["na2co3_mol_per_l_pw_derived"])
            key = (variant, p["brine"], dose)
            if key not in cases:
                cases[key] = soften(f["mol"], f["water_kg"], 25.0, na2co3=dose, phases=phases)
            r = cases[key]
            model = _removal(r, f["mol"], p["ion"])
            measured = float(p["removal_percent"])
            scored = p["ion"] == "Ca"
            rows.append({"variant": variant, "test": "F3" if scored else "reported",
                         "criterion": "model >= measured - 10" if scored else "",
                         "source": "Khajouei 2021 Fig. 4", "sample": p["brine"].split()[0], "ion": p["ion"],
                         "na2co3_mol_l": dose, "naoh_mol_l": 0.0, "measured_removal_pct": measured,
                         "model_removal_pct": model, "model_minus_measured": model - measured,
                         "pass": (model >= measured - 10.0) if scored else None,
                         "step1_pH": None, "final_pH": r["steps"]["step2"]["pH"], "T_C": 25.0,
                         "element_closure_mol": r["element_closure_mol"],
                         "charge_closure_mol": r["charge_closure_mol"], "solids_step1": "",
                         "solids_step2": json.dumps(r["steps"]["step2"]["solids_mol"])})
    # F4: Lee 2022, Ca(OH)2 1.0 x Mg, then Na2CO3 1.2 x (Ca + Sr + Ba) after step 1 (Ca "newly analyzed").
    lee = lee_feed()
    step1 = soften(lee["mol"], lee["water_kg"], 25.0, lime=lee["mol"]["Mg"], na2co3=0.0)
    after1 = step1["steps"]["step1"]["aq"]
    dose = 1.2 * (after1["Ca"] + after1["Sr"] + after1["Ba"])
    r = soften(lee["mol"], lee["water_kg"], 25.0, lime=lee["mol"]["Mg"], na2co3=dose)
    for ion in ("Ca", "Sr", "Ba", "Mg", "Li"):
        model = 100.0 * (1.0 - r["steps"]["step2"]["aq"][ion] / after1[ion])
        rows.append({"variant": "base", "test": "F4" if ion == "Ca" else "reported",
                     "criterion": "model > 95" if ion == "Ca" else "", "source": "Lee 2022 Section 3.1",
                     "sample": "synthesized", "ion": ion, "na2co3_mol_l": dose, "naoh_mol_l": 0.0,
                     "measured_removal_pct": {"Ca": 95.0, "Sr": 95.0, "Ba": 98.0}.get(ion),
                     "model_removal_pct": model, "model_minus_measured": None,
                     "pass": model > 95.0 if ion == "Ca" else None, "step1_pH": r["steps"]["step1"]["pH"],
                     "final_pH": r["steps"]["step2"]["pH"], "T_C": 25.0,
                     "element_closure_mol": r["element_closure_mol"], "charge_closure_mol": r["charge_closure_mol"],
                     "solids_step1": json.dumps(r["steps"]["step1"]["solids_mol"]),
                     "solids_step2": json.dumps(r["steps"]["step2"]["solids_mol"])})
    # Model lines for the figure: base representation, dose sweep at pH 11.
    feeds = shafer_feeds()
    for county, top in (("RC", 0.30), ("DC", 0.06)):
        f = feeds[county]
        for i in range(31):
            dose = top * i / 30
            r = soften(f["mol"], f["water_kg"], 25.0, naoh_ph=11.0, na2co3=dose)
            curves.append({"sample": county, "na2co3_mol_l": dose, "final_pH": r["steps"]["step2"]["pH"]}
                          | {f"{ion}_removal_pct": _removal(r, f["mol"], ion) for ion in ("Ca", "Sr", "Mg", "Ba")})
    return rows, curves, {"n_states": len(cases) + 2 + len(curves)}


def _shafer_test(county, dose, ion):
    if county == "RC" and 0.0 < dose < 0.2 and ion in ("Ca", "Sr"):
        return "F1", "|model - measured| <= 10"
    if county == "DC" and dose == 0.0 and ion in ("Ca", "Ba", "Sr"):
        return "F2", "|model - measured| <= 10"
    if (county == "RC" and dose > 0.2 and ion in ("Ca", "Sr", "Mg")) or (county == "DC" and dose > 0.0):
        return "F6", "model >= measured - 10"
    return None, ""


# ------------------------------------------------------------------ MS-2


def _register(name: str) -> tuple[float, float, float]:
    row = next(r for r in _rows(PRETREATMENT_REGISTER) if r["parameter"] == name)
    return float(row["low_value"]), float(row["base_value"]), float(row["high_value"])


def ms2_feed():
    """USGS MS-2 row (selected copy): major ions, B as B(OH)3, DIC 18.61 mg/L taken as mg C/L (raw row); Cl closes
    the +0.84 % imbalance (acetate, Si, H2S and trace metals omitted: not in pitzer.dat or trace)."""
    record = {r["analyte"]: r["value"] for r in _rows(MS2_RECORD)}
    with (REPO_ROOT / "data/reference/smackover_sar_li/input/usgs_southAR/southAR_brines_2022.txt").open() as stream:
        raw = next(r for r in csv.DictReader(stream) if r["SITEID"] == "MS-2")
    mg_l = {k: float(record[k]) for k in ("Na", "K", "Mg", "Ca", "Sr", "Ba", "Li", "Cl", "Br", "SO4", "B")}
    return {"mol": feed(mg_l, co2_mol=float(raw["DIC"]) / 1000.0 / MOLAR_MASS["C"]), "printed_mg_l": mg_l,
            "water_kg": float(record["specific_gravity"]) - float(record["TDS"]) / 1e6,
            "T_C": float(record["TEMP"])}


def _ms2_row(r, f, variant, naoh_x, na2co3_x, prices, moisture) -> dict:
    s1, s2, fin = r["steps"]["step1"], r["steps"]["step2"], r["steps"]["neutralized"]
    kg = {"NaOH": r["reagent_mol"]["NaOH"] * 39.997, "Na2CO3": r["reagent_mol"]["Na2CO3"] * 105.988,
          "HCl31": r["reagent_mol"]["HCl"] * 36.461 / HCL_SOLUTION_FRACTION}  # kg per m3
    cost = {"NaOH": kg["NaOH"] * prices.naoh_price, "Na2CO3": kg["Na2CO3"] * NA2CO3_USD_KG,
            "HCl": kg["HCl31"] * prices.hcl_31pct_solution_price}
    residual = {e: s2["aq"][e] * MOLAR_MASS[e] * 1000.0 for e in ("Ca", "Mg", "Sr", "Ba", "Li", "Na")}
    held = _cake(r, moisture)
    solids = {}
    for step, s in (("step1", s1), ("step2", s2)):
        for k, n in s["solids_mol"].items():
            solids[f"{step}_{k}_mol_l"] = n
    return {"variant": variant, "T_C": r["T_C"], "naoh_x_2Mg": naoh_x, "na2co3_x_CaSrBa": na2co3_x,
            "naoh_mol_l": r["reagent_mol"]["NaOH"], "na2co3_mol_l": r["reagent_mol"]["Na2CO3"],
            "hcl_mol_l": r["reagent_mol"]["HCl"], "step1_pH": s1["pH"], "step2_pH": s2["pH"],
            **{f"residual_{e}_mg_l": v for e, v in residual.items()},
            "residual_CaMg_mg_l": residual["Ca"] + residual["Mg"],
            "portlandite_mol_l": s1["solids_mol"].get("Portlandite", 0.0) + s2["solids_mol"].get("Portlandite", 0.0),
            "halite_mol_l": s1["solids_mol"].get("Halite", 0.0) + s2["solids_mol"].get("Halite", 0.0),
            "solids_step1_kg_m3": 1000.0 * s1["solids_kg"], "solids_step2_kg_m3": 1000.0 * s2["solids_kg"],
            "sludge_disposal_usd_m3_at_base": 1000.0 * (s1["solids_kg"] + s2["solids_kg"]) / (1.0 - moisture)
            * _register("sludge_disposal_usd_kg")[1],
            "li2co3_si_step2": s2["li2co3_si"], "li_in_solids_mol_l": _li_solids(r),
            "naoh_kg_m3": kg["NaOH"], "na2co3_kg_m3": kg["Na2CO3"], "hcl31_kg_m3": kg["HCl31"],
            "cost_naoh_usd_m3": cost["NaOH"], "cost_na2co3_usd_m3": cost["Na2CO3"], "cost_hcl_usd_m3": cost["HCl"],
            "reagent_cost_usd_m3": sum(cost.values()),
            "li_loss_unwashed_pct_at_base_moisture": 100.0 * held["li_mol"] / f["mol"]["Li"],
            "softened_liquor_l_per_l": fin["liquor_l"], "softened_Na_mol_l_liquor": fin["aq"]["Na"] / fin["liquor_l"],
            "element_closure_mol": r["element_closure_mol"], "charge_closure_mol": r["charge_closure_mol"], **solids}


def _li_solids(r) -> float:
    return sum(2.0 * s["solids_mol"].get("Li2CO3", 0.0) for s in r["steps"].values())


def _cake(r, moisture) -> dict:
    """Liquid held by each separated cake (wet-mass fraction ``moisture``) at that step's liquor composition.
    Step 2 is solved on the full step-1 liquor, so its held liquid is overstated by the step-1 fraction held."""
    held = {"li_mol": 0.0, "na_mol": 0.0, "liquid_kg": 0.0, "liquid_l": 0.0}
    for step in ("step1", "step2"):
        s = r["steps"][step]
        liquid_kg = s["solids_kg"] * moisture / (1.0 - moisture)
        held["li_mol"] += liquid_kg * s["aq"]["Li"] / s["liquor_kg"]
        held["na_mol"] += liquid_kg * s["aq"]["Na"] / s["liquor_kg"]
        held["liquid_kg"] += liquid_kg
        held["liquid_l"] += liquid_kg * s["liquor_l"] / s["liquor_kg"]
        held[f"{step}_liquid_fraction"] = liquid_kg / s["liquor_kg"]
    return held


def cake_table(r, f) -> list[dict]:
    """Li loss by cake moisture and one displacement wash of 0, 1, 2 cake-liquid volumes. The wash removes a
    fraction exp(-n) (fully mixed limit, reported) or min(n, 1) (plug-flow limit) of the held solutes."""
    rows = []
    fin = r["steps"]["neutralized"]
    for moisture in _register("sludge_moisture_fraction"):
        held = _cake(r, moisture)
        for n in (0, 1, 2):
            for limit, left in (("fully mixed", math.exp(-n)), ("plug flow", max(0.0, 1.0 - n))):
                for route in ("to extraction feed", "to waste"):
                    lost = held["li_mol"] * (left if route == "to extraction feed" else 1.0)
                    na = fin["aq"]["Na"] - held["na_mol"] * (left if route == "to extraction feed" else 1.0)
                    li = f["mol"]["Li"] - lost
                    volume = fin["liquor_l"] - held["liquid_l"] + (n * held["liquid_l"]
                                                                    if route == "to extraction feed" else 0.0)
                    rows.append({"T_C": r["T_C"], "sludge_moisture_fraction": moisture, "wash_volumes": n,
                                 "wash_limit": limit, "wash_route": route,
                                 "cake_liquid_l_per_l": held["liquid_l"],
                                 "step1_liquor_fraction_held": held["step1_liquid_fraction"],
                                 "li_loss_pct": 100.0 * lost / f["mol"]["Li"],
                                 "feed_volume_l_per_l_brine": volume,
                                 "feed_li_mg_l": li * MOLAR_MASS["Li"] * 1000.0 / volume,
                                 "feed_na_mol_l": na / volume})
    return rows


def ms2(prices) -> tuple[list[dict], list[dict], list[dict], dict]:
    f = ms2_feed()
    mg, carbonate = f["mol"]["Mg"], f["mol"]["Ca"] + f["mol"]["Sr"] + f["mol"]["Ba"]
    grid = sorted({(x / 10, 1.2) for x in range(8, 15)} | {(1.2, x / 10) for x in range(8, 15)})
    moisture = _register("sludge_moisture_fraction")[1]
    rows, cakes, feeds, crossings = [], [], [], {}
    nesq = PHASES + ("Nesquehonite",)
    no_na_ca = tuple(p for p in PHASES if p not in ("Pirssonite", "Gaylussite"))

    def run(temperature, naoh_x, na2co3_x, phases=PHASES):
        return soften(f["mol"], f["water_kg"], temperature, naoh=naoh_x * 2 * mg, na2co3=na2co3_x * carbonate,
                      neutralize_ph=MS2_PH, phases=phases)

    for temperature in (25.0, f["T_C"]):
        system, solver = _system(PHASES)
        raw = _state(system, temperature, f["water_kg"], f["mol"])
        _solve(solver, raw)
        feeds.append({"T_C": temperature} | _record(system, raw, PHASES))
        for naoh_x, na2co3_x in grid:
            r = run(temperature, naoh_x, na2co3_x)
            rows.append(_ms2_row(r, f, "base", naoh_x, na2co3_x, prices, moisture))
            rows.append(_ms2_row(run(temperature, naoh_x, na2co3_x, nesq), f, "nesquehonite", naoh_x, na2co3_x,
                                 prices, moisture))
            if 1.4 in (naoh_x, na2co3_x):
                rows.append(_ms2_row(run(temperature, naoh_x, na2co3_x, no_na_ca), f,
                                     "no pirssonite or gaylussite", naoh_x, na2co3_x, prices, moisture))
            if (naoh_x, na2co3_x) == (1.2, 1.2):
                cakes += cake_table(r, f)
        for line in ("naoh", "na2co3"):
            def excess(x):
                args = (x, 1.2) if line == "naoh" else (1.2, x)
                s2 = run(temperature, *args)["steps"]["step2"]["aq"]
                return (s2["Ca"] * MOLAR_MASS["Ca"] + s2["Mg"] * MOLAR_MASS["Mg"]) * 1000.0 - 10.0
            ends = (excess(0.8), excess(1.4))
            crossings[f"{temperature:g}C_{line}"] = (
                {"multiplier": brentq(excess, 0.8, 1.4, xtol=1e-6)} if ends[0] * ends[1] < 0
                else {"multiplier": None, "ca_mg_minus_10_at_0.8_and_1.4": ends})
    return rows, cakes, feeds, {"crossings_CaMg_10mg_l": crossings, "feed_mol_l": f["mol"],
                                "water_kg_per_l": f["water_kg"]}


def _write(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (RESULTS / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    prices = load_screening_cost_assumptions()
    start = time.perf_counter()
    rows, curves, info = falsifiers()
    falsifier_seconds = time.perf_counter() - start
    _write("softening_falsifiers.csv", rows)
    _write("softening_model_lines.csv", curves)
    start = time.perf_counter()
    ms2_rows, cakes, feeds, ms2_info = ms2(prices)
    ms2_seconds = time.perf_counter() - start
    _write("softening_ms2.csv", ms2_rows)
    _write("softening_cake.csv", cakes)
    scored = [r for r in rows if r["variant"] == "base" and r["pass"] is not None]
    summary = {
        "scores": {t: {"passed": sum(r["pass"] for r in scored if r["test"] == t),
                       "points": sum(r["test"] == t for r in scored)} for t in ("F1", "F2", "F3", "F4", "F6")},
        "max_element_closure_mol": max(r["element_closure_mol"] for r in rows + ms2_rows),
        "max_charge_closure_mol": max(r["charge_closure_mol"] for r in rows + ms2_rows),
        "ms2_feed_equilibrium": feeds, **ms2_info,
        "issue164_softening": softening().__dict__,
        "ms2_li_value_usd_m3": MS2_LI_VALUE_USD_M3,
        "runtime_s": {"falsifiers": falsifier_seconds, "ms2": ms2_seconds, "falsifier_states": info["n_states"]},
        "database": {"base": "pitzer.dat (Reaktoro 2.13 embedded) + reaktoro_li2co3_pitzer_extension.dat",
                     "added_from_phreeqc.dat": ["Strontianite", "Witherite"], "reaktoro": rkt.__version__},
    }
    (RESULTS / "softening_summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(json.dumps({k: summary[k] for k in ("scores", "max_element_closure_mol", "max_charge_closure_mol",
                                              "crossings_CaMg_10mg_l", "runtime_s")}, indent=1, default=str))


if __name__ == "__main__":
    main()
