"""CONDITIONAL #188 rerun on one named Engine chloride packet (Lithium #197, #213).

The packet supplies the whole aqueous basis: Figiel-Yu-Held 2025 water, Li/Na/K/Mg/Ca/Cl ions, Eq. 11
permittivity and corrected Born Eq. 7. The #188 organic prior, reaction topology, conventions and strict
certificate are unchanged. The packets have no Sr2+, so MS-2 Sr is lumped into Ca on equal moles and charge
(#188 already sets ln K_Sr = ln K_Ca); a Held 2008 control with the same lumping isolates the packet effect.
Retained #188 values are read from Lithium commit RETAINED; only OUT, the packet's own folder, is written.

The packet basis sets reaction_coordinates.HELD_BASIS = False, so its Held-only check (Born zero, constant
permittivity 78.09) applies to the Held control only.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import subprocess
import time
from copy import deepcopy
from functools import lru_cache

import numpy as np

from analyses.shared.hbta_topo_configured_reactive_lle.scripts.independent_network import PRESSURE_RELATIVE_LIMIT  # noqa: E402
from lithium_extraction.prototypes.produced_water_sx_prototype import d2ehdtpa_buphen as fs  # noqa: E402
from . import aqueous_activities as aqa, calibrate_and_score as cs, reaction_coordinates as rc  # noqa: E402

RETAINED = "a2096c5707165e65158ed0dff0bfe2ff8f0ee932"
# Engine packet id: (Engine commit holding it, data/packet-locks.json fingerprint, results folder under
# ../results whose packet/ is a byte-for-byte copy, assembled document id, limits). The first is the default.
PACKETS = {
    "figiel-2025-chloride-activity-refit/2": (
        "9c70c7ae16775c6ebcba69f740a31ee4008ce1e9", "d79865152abaf527b4602c13b20e3e2c46db8eb0b3f2b294b138b8bdde65a865",
        "conditional_figiel_packet", "lithium-188-conditional-figiel-chloride-refit-2",
        "provisional packet (owner-relaxed target; delivered review and owner acceptance pending); binary MgCl2/CaCl2 "
        "only, 298.15 K, 1 bar, 0.001-4.5 mol/kg; not qualified for mixed brines; 73/168 rows above 0.02 decade; "
        "Mg/Ca-water and Mg/Ca-chloride k_ij at bounds (not identified); no Sr2+ parameters (Sr lumped into Ca)"),
    "figiel-2025-chloride-mixture-refit/1": (
        "21e761c318c2ac5b555a245365c844d38a93e4cb", "ad1194a1dab24e5cd748e8992500e91b664715be59892cdd82ac5888a7b85004",
        "conditional_qualified_packet", "lithium-213-conditional-figiel-chloride-mixture-refit-1",
        "qualified by the Engine owner on 2026-10-16 for aqueous NaCl-MgCl2-CaCl2 brines at 298.15 K, 1 bar; "
        "fails one of 1,467 source rounding realizations (r1408, Dinane-Mounir I <= 3 water-activity RMS 1.7 % over); "
        "K+ and Li+ not refit, so Li- and K-bearing brines are outside the qualification; held-out data are two-salt "
        "mixtures at I 0.45-8.03 mol/kg, so Na-Mg-Ca brines are a composition extrapolation; numerical fit, not "
        "physical validation; no Sr2+ parameters (Sr lumped into Ca)"),
}
LABEL = {"held": f"Held 2008 control, Sr lumped into Ca where present; {rc.ERROR_LABEL}"}
CASE = {"B": 1.0, "S": 1.2, "O_A": 1.0, "N": 5, "Sr_decade": 0}
RETAINED_CASCADE = {"fig7_exchange": "ms2_0908.json", "table1": "ms2_1948.json"}
HELD, MS2_FEED, COMPLEX = rc.payload, fs.ms2_feed, rc.COMPLEX
HELD_LABELS = rc.ERROR_LABEL, cs.LABELS, fs.LABELS


def retained(name):
    # RETAINED predates the 2026-10-08 bucket move, so the file sits at its earlier path there.
    return subprocess.check_output(["git", "show", f"{RETAINED}:analyses/d2ehdtpa_buphen_ms2_prototype/results/{name}"],
                                   cwd=aqa.ANALYSIS, text=True)


def fingerprint(root):
    """Engine data/tools/validate_packets.py packet fingerprint."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big") + relative + path.stat().st_size.to_bytes(8, "big") + path.read_bytes())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def packet_payload():
    """Packet water, ions, pairs and model families; #188 organic records and zero organic pairs unchanged."""
    if fingerprint(PACKET) != FINGERPRINT:
        raise RuntimeError("Packet copy differs from the Engine lock fingerprint")
    packet, held = json.loads((PACKET / "parameters/parameters.json").read_text()), HELD()
    aqueous = {"water", *(aqa.ION[e][0] for e in ("Li", "Na", "K", "Mg", "Ca", "Cl"))}
    held_aqueous = {"water", *(name for name, _, _ in aqa.ION.values())}
    document = deepcopy(packet)
    document["components"] = [c for c in packet["components"] if c["component_id"] in aqueous]
    document["components"] += [deepcopy(c) for c in held["components"] if c["component_id"] not in held_aqueous]
    prior = {c["component_id"]: c for c in rc.ligand_payload()["components"]}
    for record in document["components"][len(aqueous):]:  # Born needs the #158 solvation factor that Held omitted.
        name = record["component_id"]
        factor = deepcopy(next(c for c in prior["diluent" if name == "diluent" else "hbta"]["coefficients"]
                               if c["family"] == "solvation_factor"))
        factor["identity"] = f"component/{name}/solvation_factor"
        record["coefficients"].append(factor)
    names = {c["component_id"] for c in document["components"]}
    document["pairs"] = [p for p in packet["pairs"] if {p["component_id_a"], p["component_id_b"]} <= aqueous]
    document["pairs"] += [deepcopy(p) for p in held["pairs"] if {p["component_id_a"], p["component_id_b"]} <= names
                          and not {p["component_id_a"], p["component_id_b"]} <= held_aqueous]
    document["correlations"] = [c for c in packet["correlations"] if c["component_id"] in aqueous]
    document["topology"]["sites"] = [s for s in packet["topology"]["sites"] if s["component_id"] in aqueous]
    document["topology"]["edges"] = [e for e in packet["topology"]["edges"]
                                     if {e["endpoint_a"]["component_id"], e["endpoint_b"]["component_id"]} <= aqueous]
    used = json.dumps(document["components"] + document["pairs"])
    for key, field in (("sources", "source_id"), ("domains", "domain_id")):
        present = {r[field] for r in document[key]}
        document[key] += [r for r in held[key] if r[field] not in present and f'"{r[field]}"' in used]
    document["document_id"] = DOCUMENT_ID
    return document


def select(packet_id):
    """Point the packet basis, its labels and OUT at one named Engine packet."""
    global PACKET_ID, PACKET_COMMIT, FINGERPRINT, OUT, PACKET, DOCUMENT_ID, LIMITS
    PACKET_ID, (PACKET_COMMIT, FINGERPRINT, folder, DOCUMENT_ID, LIMITS) = packet_id, PACKETS[packet_id]
    OUT = aqa.RESULTS / folder
    PACKET = OUT / "packet"
    LABEL["packet"] = f"CONDITIONAL on {PACKET_ID}: {LIMITS}"
    packet_payload.cache_clear()


select(next(iter(PACKETS)))


def lumped_feed(feed=None):
    """MS-2 Sr2+ counted as Ca2+ (equal moles and charge); the packet has no Sr2+ parameters."""
    feed = MS2_FEED() if feed is None else feed.copy()
    feed["Ca"] += feed.pop("Sr")
    return feed


def use(basis):
    """Select the packet, Sr-lumped Held control, or original explicit-Sr Held basis."""
    explicit_sr = basis == "held_explicit_sr"
    rc.payload = packet_payload if basis == "packet" else HELD
    rc.HELD_BASIS = basis != "packet"
    rc.model.cache_clear()
    rc.ERROR_LABEL = HELD_LABELS[0] if explicit_sr else LABEL[basis]
    rc.COMPLEX = COMPLEX if explicit_sr else {e: c for e, c in COMPLEX.items() if e != "Sr"}
    fs.ms2_feed = MS2_FEED if explicit_sr else lumped_feed
    cs.LABELS = HELD_LABELS[1] if explicit_sr else {
        **HELD_LABELS[1], "activity_error_label": LABEL[basis], "aqueous_basis": basis}
    fs.LABELS = HELD_LABELS[2] if explicit_sr else {
        **HELD_LABELS[2], "activity_error_label": LABEL[basis], "aqueous_basis": basis,
                 "sr_treatment": "MS-2 Sr lumped into Ca"}


def log10_activities(names, amounts):
    """Molal infinite-dilution activities with the #188 offsets of rc.model."""
    mixture, offsets, _ = rc.model(names)
    total = math.fsum(amounts.values())
    x = [amounts.get(s, 0.)/total for s in names]
    state = mixture.state(rc.T, P=rc.P, x=x, phase="liquid")
    if abs(state.pressure/rc.P - 1) > PRESSURE_RELATIVE_LIMIT:
        raise RuntimeError(f"Pressure residual {state.pressure/rc.P - 1}")
    return {e: (math.log(x[j]*state.molar_density) + state.residual_chemical_potential_over_rt[j] - offsets[s])/math.log(10)
            for e, (s, _, _) in aqa.ION.items() if amounts.get(s, 0) > 0 for j in [names.index(s)]}


def f0():
    states = {r["state"]: r for r in csv.DictReader((aqa.INPUTS / "f0_states.csv").open())}
    pitzer = {(r["target_state"], r["ratio"]): r for r in csv.DictReader(io.StringIO(retained("f0.csv")))}
    rows = []
    for basis, lump in (("held", False), ("held", True), ("packet", True)):
        use(basis)
        names = ("water", *(aqa.ION[e][0] for e in aqa.ION if not (lump and e == "Sr")), "diluent")
        values = {}
        for name, row in states.items():
            mol = {e: float(row[e])/1000 for e in aqa.ION if e != "Cl"}
            if lump:
                mol["Ca"] += mol.pop("Sr")
            mol["Cl"] = math.fsum(n*aqa.ION[e][2] for e, n in mol.items())
            values[name] = aqa.ratios(log10_activities(names, {aqa.ION[e][0]: n for e, n in mol.items()}
                                                       | {"water": float(row["water_mol"])}))
        for target in ("table1", "ms2"):
            for ratio in ("R_Mg", "R_Ca", "R_Li"):
                engine, p = values[target][ratio] - values["fig7"][ratio], pitzer[(target, ratio)]
                difference = engine - float(p["pitzer_delta_decade"])
                rows.append({"aqueous_basis": basis, "sr_treatment": "lumped into Ca" if lump else "explicit Sr2+",
                             "target_state": target, "ratio": ratio, "engine_delta_decade": engine,
                             "pitzer_delta_decade": float(p["pitzer_delta_decade"]), "disagreement_decade": difference,
                             "tolerance_decade": aqa.F0_LIMIT, "passed": abs(difference) <= aqa.F0_LIMIT,
                             "retained_held_disagreement_decade": float(p["disagreement_decade"]),
                             "evidence_type": "comparison between numerical models (retained Pitzer has explicit Sr2+)",
                             "label": LABEL[basis]})
    aqa.write_csv("f0.csv", rows)
    mixture = rc.model(names)[0]  # packet basis, from the last loop pass
    solvent = [float(s == "water") for s in names]
    binary = []
    for r in csv.DictReader(io.StringIO(retained("pure_salt_f0_molalities.csv"))):
        cation = {"NaCl": "Na", "LiCl": "Li", "MgCl2": "Mg", "CaCl2": "Ca"}[r["salt"]]
        reference = mixture.reference(rc.T, rc.P, solvent_fractions=solvent, ions=(aqa.ION[cation][0], "chloride-anion"),
                                      stoichiometry=(1, aqa.ION[cation][2]))
        model = reference.activity(float(r["molality_mol_kg"])).mean_ionic_log_molality_scale/math.log(10)
        binary.append({"state": r["source_state"], "salt": r["salt"], "molality_mol_kg": float(r["molality_mol_kg"]),
                       "gamma_evaluated_interpolated": float(r["gamma_observed"]), "packet_gamma": 10**model,
                       "packet_error_decade": model - math.log10(float(r["gamma_observed"])),
                       "retained_held_error_decade": float(r["engine_gamma_error_decade"]),
                       "retained_pitzer_error_decade": float(r["pitzer_gamma_error_decade"]),
                       "source_locator": r["source_locator"],
                       "evidence_type": "comparison with experiment: evaluated binary data, interpolated in log gamma",
                       "label": LABEL["packet"]})
    aqa.write_csv("f0_binary_gamma.csv", binary)


def fit(calibration, start):
    use("packet")
    reused = OUT / f"{calibration}_retained_coordinates.json"
    if start == 0 and not reused.exists():
        q = json.loads(retained(f"{calibration}_start0.json"))["ln_K_Li_Mg_Ca"]
        reused.write_text(json.dumps({"ln_K_Li_Mg_Ca": q, "residual_decade": cs.residual(np.array(q), calibration).tolist(),
                                      "meaning": "retained Held-basis coordinates evaluated on the packet basis",
                                      **cs.LABELS}, indent=2) + "\n")
    cs.fit(calibration, start)


def score():
    use("packet")
    for calibration in ("fig7_exchange", "table1"):
        q = [json.loads((OUT / f"{calibration}_start{i}.json").read_text())["ln_K_Li_Mg_Ca"] for i in range(3)]
        if np.max(np.ptp(q, axis=0)) > 1e-6:
            raise RuntimeError(f"{calibration} start spread exceeds 1e-6 ln K")
    cs.score()


def cascade(basis, calibration):
    """Strict MS-2 reference cascade; warm-started from the retained Held profile, resumable after timeout."""
    use(basis)
    path, checkpoint = OUT / f"reference_{basis}_{calibration}.json", OUT / f"reference_{basis}_{calibration}_progress.json"
    if path.exists():
        return
    fits = (OUT / f"{calibration}_start0.json").read_text() if basis == "packet" else retained(f"{calibration}_start0.json")
    q = json.loads(fits)["ln_K_Li_Mg_Ca"]
    prior = json.loads(checkpoint.read_text()) if checkpoint.exists() else None
    if prior is None:
        held = json.loads(retained(RETAINED_CASCADE[calibration]))
        if held["case"] != {"calibration": calibration, **CASE}:
            raise RuntimeError(f"Retained cascade is not the reference case: {held['case']}")
        organic = [row["amounts_mol"][1] for row in held["extraction"]["stages"]]
        prior = {"completed_sweeps": 0, "elapsed_seconds": 0., "start": {
            "organic": [{"Li": o["lib2a"], "Mg": o["mga2"], "Ca": o["caa2"] + o["sra2"]} for o in organic],
            "guesses": [None]*CASE["N"]}}
    started = time.perf_counter()

    def progress(sweep, change, state):
        record = {"completed_sweeps": sweep, "change_mol_L": change, "start": state,
                  "elapsed_seconds": prior["elapsed_seconds"] + time.perf_counter() - started}
        temporary = checkpoint.with_suffix(".tmp")
        temporary.write_text(json.dumps(record) + "\n")
        temporary.replace(checkpoint)
    result = fs.extract(q, CASE["B"], CASE["S"], CASE["O_A"], CASE["N"], start=prior["start"], progress=progress,
                        resume_sweeps=prior["completed_sweeps"])
    result.pop("start")
    accounts = [fs.accounts(result, CASE["S"], CASE["O_A"], CASE["B"], case, h2s)
                for case in ("base", "optimistic") for h2s in (0., fs.BASIS["h2s_protonation_mol_L"])]
    engine = aqa.installed_engine()
    path.write_text(json.dumps({"case": {"calibration": calibration, **CASE}, "ln_K_Li_Mg_Ca": q, "extraction": result,
                                "accounts": accounts, "wall_seconds_all_invocations":
                                prior["elapsed_seconds"] + time.perf_counter() - started,
                                "warm_start": "retained Held reference-case organic profile, Sr loading moved to Ca",
                                "engine_commit": engine["source_commit"], "engine_sha256": engine["sha256"],
                                **fs.LABELS}, indent=2) + "\n")
    checkpoint.unlink()


def summary():
    """Side-by-side retained and conditional values; reads files only."""
    def table(text):
        return list(csv.DictReader(io.StringIO(text)))
    f0_rows, f2_rows = table((OUT / "f0.csv").read_text()), table((OUT / "f2.csv").read_text())
    groups = {}
    for r in f0_rows:
        groups.setdefault(f"{r['aqueous_basis']}, Sr {r['sr_treatment']}", []).append(r)
    f0_result = {key: {"max_abs_disagreement_decade": max(abs(float(r["disagreement_decade"])) for r in rows),
                       "rows_passed": sum(r["passed"] == "True" for r in rows), "rows": len(rows)} for key, rows in groups.items()}
    f2 = {}
    for name, rows in (("packet", f2_rows), ("retained_held", table(retained("f2.csv")))):
        f2[name] = {"points_passed": sum(r["passed"] == "True" for r in rows), "points": len(rows),
                    **{f"Fig{s['figure']} {s['series']}": f"{sum(r['passed'] == 'True' for r in rows if r['figure'] == s['figure'] and r['series'] == s['series'])}"
                       f"/{sum(r['figure'] == s['figure'] and r['series'] == s['series'] for r in rows)}" for s in rows}}
    calibrations = {}
    for calibration in ("fig7_exchange", "table1"):
        starts = [json.loads((OUT / f"{calibration}_start{i}.json").read_text()) for i in range(3)]
        calibrations[calibration] = {
            "coordinates": "refit on the packet basis from the three #188 starts (6,5,4), (7,7,6), (8,9,8)",
            "packet_ln_K_Li_Mg_Ca": starts[0]["ln_K_Li_Mg_Ca"],
            "retained_held_ln_K_Li_Mg_Ca": json.loads(retained(f"{calibration}_start0.json"))["ln_K_Li_Mg_Ca"],
            "start_spread_ln_K": float(np.max(np.ptp([s["ln_K_Li_Mg_Ca"] for s in starts], axis=0))),
            **{k: starts[0][k] for k in ("residual_decade", "singular_values", "rank", "condition_number")},
            "retained_coordinates_on_packet": json.loads((OUT / f"{calibration}_retained_coordinates.json").read_text())}
    reference = [{"aqueous_basis": "held (retained, explicit Sr2+, wheel b5566590)", "calibration": r["calibration"],
                  "R_ext": float(r["R_ext"]), "R_rec_strip_base": float(r["R_rec"]),
                  "priced_subtotal_usd_m3": float(r["priced_subtotal_usd_m3"]),
                  "priced_subtotal_usd_kg_li": float(r["priced_subtotal_usd_kg_li"]), "sweeps": "", "wall_seconds": ""}
                 for r in table(retained("ms2_costs.csv")) if r["case"] == "base" and float(r["h2s_protonated_mol_L_feed"]) == 0
                 and all(float(r[k]) == v for k, v in CASE.items())]
    generated_by = set()  # the Engine that solved the retained references, not the current runtime pin
    for basis in ("held", "packet"):
        for calibration in ("fig7_exchange", "table1"):
            record = json.loads((OUT / f"reference_{basis}_{calibration}.json").read_text())
            generated_by.add((record["engine_commit"], record["engine_sha256"]))
            a = record["accounts"][0]
            if (a["case"], a["h2s_protonated_mol_L_feed"]) != ("base", 0.):
                raise RuntimeError("First account is not the base, no-H2S case")
            reference.append({"aqueous_basis": f"{basis} (Sr lumped into Ca, wheel {record['engine_commit'][:8]})", "calibration": calibration,
                              "R_ext": a["R_ext"], "R_rec_strip_base": a["R_rec"],
                              "priced_subtotal_usd_m3": a["priced_subtotal_usd_m3"],
                              "priced_subtotal_usd_kg_li": a["priced_subtotal_usd_kg_li"],
                              "sweeps": record["extraction"]["sweeps"], "wall_seconds": record["wall_seconds_all_invocations"]})
    for row in reference:
        row.update(evidence_type="prediction; no holdout support", **CASE, case="base strip, no H2S protonation")
    if len(generated_by) != 1:
        raise RuntimeError(f"Reference records name different Engines: {sorted(generated_by)}")
    (commit, sha256), = generated_by
    aqa.write_csv("reference_case.csv", reference)
    calibration = json.loads((OUT / "calibration_summary.json").read_text())
    document = {
        "label": LABEL["packet"], "limits": LIMITS.split("; "),
        "engine": {"commit": commit, "sha256": sha256},
        "packet": {"id": PACKET_ID, "fingerprint": FINGERPRINT, "merged_commit": PACKET_COMMIT, "local_copy": "packet/"},
        "retained_commit": RETAINED,
        "basis": ("Packet water, Li, Na, K, Mg, Ca, Cl, their pairs, Eq. 11 permittivity (C from packet) and corrected Born "
                  "replace the whole Held 2008 aqueous basis; #188 organic prior, topology, reference conventions and strict "
                  "certificate unchanged. Packet qualified at 1 bar; #188 states use 101325 Pa."),
        "sr_treatment": "Packet has no Sr2+: MS-2 Sr lumped into Ca (equal moles and charge); Held control repeats the lumping.",
        "F0": {"tolerance_decade": aqa.F0_LIMIT, "groups": f0_result,
               "evidence_type": "comparison between numerical models; retained Pitzer comparator has explicit Sr2+"},
        "calibrations": calibrations,
        "F1": {"packet": table((OUT / "f1.csv").read_text()), "retained_held": table(retained("f1.csv")),
               "packet_passed": calibration["F1_passed"]},
        "F2": f2, "reference_case": reference,
        "wall_times": [json.loads(line) for line in (OUT / "wall_times.jsonl").read_text().splitlines()]}
    (OUT / "summary.json").write_text(json.dumps(document, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("f0", "fit", "score", "cascade", "summary"))
    parser.add_argument("--calibration", choices=("fig7_exchange", "table1"), default="fig7_exchange")
    parser.add_argument("--start", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--basis", choices=("packet", "held"), default="packet")
    parser.add_argument("--packet", choices=PACKETS, default=PACKET_ID)
    args = parser.parse_args()
    select(args.packet)
    aqa.installed_engine()
    cs.RESULTS = aqa.RESULTS = OUT  # every reused writer goes to OUT; retained #188 results are never written
    started = time.perf_counter()
    {"f0": f0, "score": score, "summary": summary, "fit": lambda: fit(args.calibration, args.start),
     "cascade": lambda: cascade(args.basis, args.calibration)}[args.step]()
    with (OUT / "wall_times.jsonl").open("a") as stream:
        stream.write(json.dumps({**vars(args), "wall_seconds": time.perf_counter() - started}) + "\n")


if __name__ == "__main__":
    main()
