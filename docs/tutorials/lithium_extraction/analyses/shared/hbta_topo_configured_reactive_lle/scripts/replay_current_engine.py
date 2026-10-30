"""Bounded physical H1/dodecane replay; no fitting, surrogate training or IDAES."""

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .apparent_application import SMACKOVER, ZHANG_2017
from .generate import _sha256, _verify_wheel
from .independent_network import INITIALIZATION_SEEDS, parameter_document, solve_condition

ROOT = Path(__file__).resolve().parents[4]
RESULTS = Path(__file__).resolve().parents[1] / "results"
TRAINING = ROOT / "analyses/v3/hbta_topo_ms2_idaes_optimization/results/lhs_training_matrix.csv"
BLIND = TRAINING.with_name("whole_trajectory_verification.csv")
FIT = ROOT / "analyses/shared/hbta_topo_calibrated_surrogate/results/application_parameter_fits.json"
BLIND_IDS = tuple(
    f"blind_blind_{side}_ratio_{count}_{stage}" for side in ("low", "high") for count, stage in ((1, 1), (2, 1), (2, 2))
)


def _condition(row):
    carrier = float(row["carrier_mol_l_organic"])
    return replace(
        SMACKOVER,
        identity=row["sample_id"],
        role=row.get("role", "retained_blind_surrogate_comparison"),
        aqueous_volume_l=1.0,
        organic_volume_l=float(row["organic_to_aqueous_ratio"]),
        li_mol_l_aqueous=float(row["li_mol_l_aqueous"]),
        na_mol_l_aqueous=float(row["na_mol_l_aqueous"]),
        hydrogen_mol_l_aqueous=float(row["hydrogen_mol_l_aqueous"]),
        hbta_mol_l_organic=carrier,
        topo_mol_l_organic=carrier,
        temperature_k=float(row["temperature_k"]),
        initial_ph=-math.log10(float(row["hydrogen_mol_l_aqueous"])),
        li_mol_l_organic=carrier * float(row["incoming_li_loading_fraction"]),
        na_mol_l_organic=carrier * float(row["incoming_na_loading_fraction"]),
        saponification_fraction=0.0,
    )


def conditions():
    with TRAINING.open(newline="") as stream:
        row = next(
            r for r in csv.DictReader(stream) if r["sample_id"] == "training_fresh_oa_1_carrier_0.015_remaining_1"
        )
    return (
        ZHANG_2017,
        replace(SMACKOVER, identity="smackover_fresh_one_stage", saponification_fraction=0.0),
        _condition(row),
    )


def blind_rows():
    with BLIND.open(newline="") as stream:
        rows = {row["sample_id"]: row for row in csv.DictReader(stream) if row["sample_id"] in BLIND_IDS}
    return [rows[name] for name in BLIND_IDS]


def fit_coordinates():
    fit = next(row for row in json.loads(FIT.read_text())["fits"] if row["diluent_id"] == "dodecane")
    return float(fit["ln_k_exchange"]), float(fit["ln_k_na_uptake"])


def _run_child(arguments, unit, remaining):
    limit = min(60.0, remaining)
    started = time.monotonic()
    try:
        run = subprocess.run(
            [sys.executable, "-m", __spec__.name, *sys.argv[1:], *arguments],
            capture_output=True,
            text=True,
            timeout=limit,
        )
        if run.returncode:
            result = {
                "status": "non_evaluable",
                "failure": {"code": "replay_process_failed", "diagnostic": run.stderr, "stdout": run.stdout},
            }
        else:
            result = json.loads(
                next(
                    line.removeprefix("REPLAY_JSON=")
                    for line in run.stdout.splitlines()
                    if line.startswith("REPLAY_JSON=")
                )
            )
            if len(result.get("attempts", [])) == 1:
                del result["attempts"]  # the sole attempt is already the result record
    except subprocess.TimeoutExpired:
        result = {"status": "timeout", "failure": {"code": "external_timeout"}}
    result["execution"] = {
        "unit": unit,
        "elapsed_s": time.monotonic() - started,
        "limit_s": limit,
        "remaining_budget_s": max(0.0, remaining - (time.monotonic() - started)),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-wheel", required=True, type=Path)
    parser.add_argument("--engine-sha256", required=True)
    parser.add_argument("--engine-commit", required=True)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--condition", type=int, choices=range(3))
    choice.add_argument("--blind", type=int, choices=range(6))
    parser.add_argument("--seed", type=int, choices=range(4))
    args = parser.parse_args()
    artifact = _verify_wheel(args.engine_wheel, args.engine_sha256, args.engine_commit)
    if args.condition is not None or args.blind is not None:
        if args.condition is not None and args.seed is None:
            parser.error("an isolated direct condition requires a seed")
        condition = conditions()[args.condition] if args.condition is not None else _condition(blind_rows()[args.blind])
        seeds = (INITIALIZATION_SEEDS[args.seed],) if args.condition is not None else INITIALIZATION_SEEDS
        result = solve_condition(condition, "t1_mbta_topo", "dodecane", *fit_coordinates(), initialization_seeds=seeds)
        print("REPLAY_JSON=" + json.dumps(result, allow_nan=False))
        return
    started = time.monotonic()
    records = [{"inputs": asdict(c), "trials": []} for c in conditions()]
    jobs = [(i, j) for i in range(3) for j in range(4)]
    unattempted = []
    for position, (i, j) in enumerate(jobs):
        remaining = 1080 - (time.monotonic() - started)
        if remaining <= 0:
            unattempted.extend(f"{conditions()[a].identity}/seed{b}" for a, b in jobs[position:])
            break
        records[i]["trials"].append(
            _run_child(["--condition", str(i), "--seed", str(j)], f"{conditions()[i].identity}/seed{j}", remaining)
        )
    comparisons, errors = [], []
    direct_passed = all(
        len(row["trials"]) == 4 and all(t["status"] == "evaluated" for t in row["trials"]) for row in records
    )
    if direct_passed:
        for i, row in enumerate(blind_rows()):
            remaining = 1080 - (time.monotonic() - started)
            if remaining <= 0:
                unattempted.extend(BLIND_IDS[i:])
                break
            condition = _condition(row)
            result = _run_child(["--blind", str(i)], condition.identity, remaining)
            comparison = {"inputs": asdict(condition), "result": result}
            if result["status"] == "evaluated":
                for species in ("li", "na"):
                    direct = result[f"{species}_org_mol"] / result["carrier_total_org_mol"]
                    predicted = float(row[f"{species}_surrogate_loading_fraction"])
                    error = abs(direct - predicted)
                    comparison[species] = {"direct": direct, "retained_surrogate": predicted, "absolute_error": error}
                    errors.append(error)
            comparisons.append(comparison)
    else:
        unattempted.extend(BLIND_IDS)
    files = [
        Path(__file__),
        Path(__file__).with_name("independent_network.py"),
        Path(__file__).with_name("apparent_application.py"),
        FIT,
        TRAINING,
        BLIND,
        ROOT / "data/processed/hbta_topo_reactive_lle_surrogate/loaded_state_surrogate.json",
    ]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *map(str, files)], cwd=ROOT).returncode != 0
    payload = json.dumps(parameter_document("t1_mbta_topo", "dodecane"), sort_keys=True, separators=(",", ":"))
    output = {
        "engine": artifact,
        "application": {
            "revision": revision,
            "source_dirty": dirty,
            "files_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in files},
            "physical_payload_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        },
        "fit_coordinates": fit_coordinates(),
        "conditions": records,
        "blind_comparisons": comparisons,
        "unattempted_units": unattempted,
        "direct_passed": direct_passed,
        "surrogate_translation": {
            "qualified": bool(len(errors) == 12 and max(errors) <= 0.08 and np.quantile(errors, 0.95) <= 0.03),
            "p95": float(np.quantile(errors, 0.95)) if errors else None,
            "maximum": max(errors) if errors else None,
            "p95_limit": 0.03,
            "maximum_limit": 0.08,
        },
        "limits": [
            "Fixed H1/dodecane inputs; physical MS-2 transfer accuracy remains unvalidated.",
            "Comparison with retained surrogate outputs only; no IDAES stages, refit or retraining.",
            "Unattempted units are explicitly listed; the external aggregate cap is 1080 seconds.",
        ],
        "wall_time_s": time.monotonic() - started,
    }
    path = RESULTS / "current_engine_replay.json"
    path.write_text(json.dumps(output, separators=(",", ":"), allow_nan=False) + "\n")
    print(path)
    if not direct_passed or not output["surrogate_translation"]["qualified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
