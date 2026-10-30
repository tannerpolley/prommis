"""Conditional #188 reaction calibration, identifiability and F1/F2; no MS-2 states.

One resumable fit start per command; a failed strict state raises, never becomes a residual.
"""

import argparse
import csv
import json
import math

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .aqueous_activities import INPUTS, RESULTS, ANALYSIS, installed_engine, write_csv
from .reaction_coordinates import solve, payload, ERROR_LABEL, EVIDENCE, T, P

REFERENCE = ANALYSIS.parents[2] / "data/reference/extraction_models/d2ehdtpa_buphen_raiguel_2024"
BASIS = json.loads((INPUTS / "reaction_basis.json").read_text())
STARTS = ((6, 5, 4), (7, 7, 6), (8, 9, 8))  # span the concentration-based source estimates.
GROUPS = (2.8606e-3, 1.2281e-3)  # #188 Fig7 0.75 M exchange groups; dimensionless.
LABELS = {"evidence_type": EVIDENCE, "activity_error_label": ERROR_LABEL, "temperature_K": T, "pressure_Pa": P,
          "reference_states": "aqueous molal infinite dilution; organic 1 mol/L infinite dilution in dodecane; D/alpha dimensionless on declared input litres"}


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


TABLE = {r["element"]: r for r in read(REFERENCE / "table1.csv")}


def state(name, q):
    s = BASIS["states"][name]
    return solve(s["feed_mol_L"], s["sites_mol_L"], s["buphen_mol_L"], q)


def residual(q, alternative):
    table = state("table1", q)
    values = [table["D"]["Li"]]
    if alternative == "fig7_exchange":
        fig = state("fig7", q)
        values += [fig["D"][e]/fig["D"]["Li"]**2 for e in ("Mg", "Ca")]
        targets = [float(TABLE["Li"]["D"]), *GROUPS]
    else:
        values += [table["alpha"][e] for e in ("Mg", "Ca")]
        targets = [float(TABLE[e]["D" if e == "Li" else "alpha_prose"]) for e in ("Li", "Mg", "Ca")]
    return np.log10(np.array(values)/targets)


def fit(alternative, start):
    path = RESULTS / f"{alternative}_start{start}.json"
    if path.exists():
        print(f"Retained fit: {path}", flush=True)
        return
    progress = RESULTS / f"{alternative}_start{start}_progress.jsonl"
    initial = json.loads(progress.read_text().splitlines()[-1])["q"] if progress.exists() else STARTS[start]
    def evaluate(q):
        r = residual(q, alternative)
        record = {"q": q.tolist(), "residual_decade": r.tolist(), **LABELS}
        with progress.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        return r
    result = least_squares(evaluate, initial, diff_step=1e-4, xtol=1e-10, ftol=1e-12, gtol=1e-12, max_nfev=200)
    h = 1e-4
    jacobian = np.column_stack([(residual(result.x + np.eye(3)[j]*h, alternative) - residual(result.x - np.eye(3)[j]*h, alternative))/(2*h) for j in range(3)])
    singular = np.linalg.svd(jacobian, compute_uv=False)
    record = {"alternative": alternative, "initial_ln_K": list(STARTS[start]), "ln_K_Li_Mg_Ca": result.x.tolist(),
              "ln_K_Sr": float(result.x[2]), "residual_decade": result.fun.tolist(), "jacobian_decade_per_ln_K": jacobian.tolist(),
              "singular_values": singular.tolist(), "rank": int(np.linalg.matrix_rank(jacobian)),
              "condition_number": float(singular[0]/singular[-1]), "solver_message": result.message, "nfev": result.nfev, **LABELS}
    accepted = record["rank"] == 3 and record["condition_number"] <= 1e6 and max(abs(result.fun)) <= 1e-6
    (path if accepted else path.with_suffix(".rejected.json")).write_text(json.dumps(record, indent=2) + "\n")
    if not accepted:
        raise RuntimeError(f"#188 identifiability requirements fail; record kept in {path.with_suffix('.rejected.json')}")


def score():
    fits = [json.loads((RESULTS / f"fig7_exchange_start{i}.json").read_text()) for i in range(3)]
    for i, record in enumerate(fits):
        (RESULTS / f"fig7_exchange_start{i}.json").write_text(json.dumps({**record, **LABELS}, indent=2) + "\n")
    q = np.array(fits[0]["ln_K_Li_Mg_Ca"])
    spread = float(np.max(np.ptp([r["ln_K_Li_Mg_Ca"] for r in fits], axis=0)))
    assert spread <= 1e-6, spread
    table, fig = state("table1", q), state("fig7", q)
    f1 = []
    for e, null in (("Mg", 690), ("Ca", 1608)):
        target, model = float(TABLE[e]["alpha_prose"]), table["alpha"][e]
        error, null_error = math.log10(model/target), math.log10(null/target)
        f1.append({"series": f"Li/{e}", "alpha_model": model, "alpha_prose": target,
                   "alpha_table": float(TABLE[e]["alpha_table"]), "error_decade": error, "passed": abs(error) <= .30,
                   "table_error_decade": math.log10(model/float(TABLE[e]["alpha_table"])), "null_alpha": null,
                   "null_error_decade": null_error, "activities_improve_absolute_error": abs(error) < abs(null_error),
                   "issue_null_error_decade": .06383 if e == "Mg" else -.15354,
                   "table_locator": TABLE[e]["table_locator"], "prose_locator": TABLE[e]["prose_locator"], **LABELS})
    write_csv("f1.csv", f1)
    f2, certificates = [], [{"state": name, **r["certificate"], **LABELS} for name, r in (("table1", table), ("fig7_endpoint", fig))]
    for figure in (7, 9):
        for row in read(REFERENCE / f"figure{figure}.csv"):
            if row["fitted_endpoint"] == "True":
                continue
            s = BASIS["states"][f"fig{figure}"]
            feed, b = s["feed_mol_L"].copy(), s["buphen_mol_L"]
            x = float(row["x"])
            if figure == 7:
                b = x
            else:
                feed["Na"] = (0 if int(row["point"]) == 0 else x)/(1000*(BASIS["molar_masses_kg_mol"]["Na"] + BASIS["molar_masses_kg_mol"]["Cl"]))
            result = solve(feed, s["sites_mol_L"], b, q)
            e = row["series"].split("/")[1]
            model, observed = result["alpha"][e], float(row["alpha"])
            error = math.log10(model/observed)
            f2.append({"figure": figure, "series": row["series"], "point": row["point"], "source_x": x,
                       "x_unit": row["x_unit"], "alpha_model": model, "alpha_observed": observed,
                       "alpha_abs_bound": row["alpha_abs_bound"], "alpha_lower": row["alpha_lower"], "alpha_upper": row["alpha_upper"],
                       "error_decade": error, "passed": abs(error) <= .30, "Na_eq_mol_kg_water": result["Na_mol_kg_water"],
                       "log10_Na_eq": math.log10(result["Na_mol_kg_water"]), "slope_included": figure == 9 and int(row["point"]) > 0,
                       "locator": row["locator"], "condition_label": "2.5/20 vol% octanols; calibration 5/20; modifiers implicit", **LABELS})
            certificates.append({"state": f"fig{figure}_{row['series']}_{row['point']}", **result["certificate"], **LABELS})
    write_csv("f2.csv", f2)
    write_csv("reactive_certificates.csv", certificates)
    slopes = []
    for series in ("Li/Mg", "Li/Ca"):
        rows = [r for r in f2 if r["series"] == series and r["slope_included"]]
        x = np.array([r["log10_Na_eq"] for r in rows])
        model, observed = [float(np.polyfit(x, np.log10([r[k] for r in rows]), 1)[0]) for k in ("alpha_model", "alpha_observed")]
        slopes.append({"series": series, "model_slope": model, "observed_slope": observed, "difference": model - observed,
                       "basis": "d log10(alpha) / d log10(equilibrium aqueous Na molality); dimensionless", **LABELS})
    write_csv("f2_na_slopes.csv", slopes)
    summary = {"max_start_spread_ln_K": spread, "fig7_endpoint": fig, "table1": table, "F1_passed": all(r["passed"] for r in f1),
               "F2_passed": all(r["passed"] for r in f2), "F2_points_passed": sum(r["passed"] for r in f2), "F2_points": len(f2), **LABELS}
    summary["table1_alpha_Li_Na"] = table["D"]["Li"]*table["amounts_mol"][0]["sodium-cation"]/table["amounts_mol"][1]["naa"]
    summary["table1_model_NaA_mol_L"] = table["amounts_mol"][1]["naa"]
    summary["table1_strip_Na_diagnostic_mol_L"] = float(TABLE["Na"]["strip_mg_L"])/(1e6*BASIS["molar_masses_kg_mol"]["Na"])
    summary["fig7_Li_extraction_percent"] = 100*fig["D"]["Li"]/(1 + fig["D"]["Li"])
    if not summary["F1_passed"]:
        summary["holdout_support"] = "none; F1 failed; Fig7-exchange and Table1 calibrations are alternatives"
        paths = [RESULTS / f"table1_start{i}.json" for i in range(3)]
        if all(p.exists() for p in paths):
            fallback = [json.loads(p.read_text()) for p in paths]
            spread = float(np.max(np.ptp([r["ln_K_Li_Mg_Ca"] for r in fallback], axis=0)))
            assert spread <= 1e-6, spread
            result = state("table1", fallback[0]["ln_K_Li_Mg_Ca"])
            record = {"fits": fallback, "max_start_spread_ln_K": spread, "table1": result, "holdout_support": "none; all Table1 targets fitted", **LABELS}
            (RESULTS / "fallback_summary.json").write_text(json.dumps(record, indent=2) + "\n")
        else:
            summary["fallback_pending"] = "Run --alternative table1 --start 0, 1, 2; original F1/F2 failures remain."
    (RESULTS / "calibration_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


M_NACL = BASIS["molar_masses_kg_mol"]["Na"] + BASIS["molar_masses_kg_mol"]["Cl"]  # kg/mol
FIG8_CONDITION = "2.5/20 vol% octanols, calibration 5/20, implicit; base not printed, sites Na-saponified; calibrated sites 0.032, 0.050 mol/L"
FIG10_CONDITION = "2.5/20 vol% octanols, calibration 5/20, implicit; LiOH-saponified organic as fully Li-loaded sites; Na trace cancels"


def conditions(row, na_trace=0.):
    """Recorded Raiguel feed (mol/L; NaCl from g/L), D2EHDTPA sites and BuPhen (mol/L organic)."""
    return ({"Li": float(row["feed_LiCl_mol_L"]), "Na": float(row["feed_NaCl_g_L"])/(1000*M_NACL) + na_trace,
             "Mg": float(row["feed_MgCl2_mol_L"]), "Ca": float(row["feed_CaCl2_mol_L"])},
            float(row["d2ehdtpa_mol_L_organic"]), float(row["buphen_mol_L_organic"]))


def score_holdouts(ln_k, out_dir, labels):
    """#210 Raiguel Fig. 8 and 25 °C Fig. 10 holdouts on frozen ln K; writes only out_dir/holdout_scores.csv."""
    s = BASIS["states"]["fig9"]
    for row in (r for r in read(out_dir / "f2.csv") if r["figure"] == "9"):  # retained F2 through the same calls first
        feed = {**s["feed_mol_L"], "Na": (0 if row["point"] == "0" else float(row["source_x"]))/(1000*M_NACL)}
        model = solve(feed, s["sites_mol_L"], s["buphen_mol_L"], ln_k)["alpha"][row["series"][3:]]
        if abs(model/float(row["alpha_model"]) - 1) > 1e-9:
            raise RuntimeError(f"F2 reproduction fails at {row['series']} point {row['point']}: {model}")
    engine, challenge = installed_engine(), next(h["reason"] for h in read(REFERENCE / "holdouts.csv") if h["role"] == "challenge")
    common = {"comparison": "holdout comparison under declared representation assumptions", "ln_K_Li_Mg_Ca": json.dumps(list(ln_k)),
              "engine_version": engine["version"], "engine_commit": engine["source_commit"], "engine_sha256": engine["sha256"], **labels}

    def entry(row, reason=""):
        return {"figure": row["figure"], "series": row["series"], "point": row["point"], "source_temperature_K": row["temperature_K"],
                "role": "holdout" if row["temperature_K"] == "298.15" else "challenge", "scored": not reason,
                "reason": reason, "source_x": row["x"], "x_unit": row["x_unit"], "locator": row["locator"]}

    def scored(row, result, condition, **extra):
        model, observed = result["alpha"][row["series"][3:]], float(row["alpha"])
        error = math.log10(model/observed)
        return {**entry(row), "site_mol_L": row["d2ehdtpa_mol_L_organic"], "alpha_model": model, "alpha_observed": observed,
                "alpha_abs_bound": row["alpha_abs_bound"], "error_decade": error, "passed": abs(error) <= .30, **extra,
                **result["certificate"], "condition_label": condition, **common}
    rows, solved, fig9 = [], {}, read(REFERENCE / "figure9.csv")
    for row in read(REFERENCE / "figure8.csv"):  # recorded D2EHDTPA = feed LiCl; one solve serves both series
        feed, sites, b = conditions(row)
        result = solved[sites] = solved.get(sites) or solve(feed, sites, b, ln_k)
        o = result["amounts_mol"][1]  # mol per litre organic at O/A 1
        extra = {"below_ms2_design_site_range": sites < .020, "Li_site_fraction": o["lib2a"]/sites,
                 "metal_site_fraction": (o["lib2a"] + 2*o["mga2"] + 2*o["caa2"])/sites}
        if sites == .05:  # Fig. 9 at 50 g/L NaCl has the same recorded conditions; alpha interpolated linearly in g/L.
            nine = [r for r in fig9 if r["series"] == row["series"]]
            a9 = float(np.interp(50, [float(r["x"]) for r in nine], [float(r["alpha"]) for r in nine]))
            extra |= {"fig9_alpha_50_g_L_NaCl": a9, "source_scatter_fig8_vs_fig9_decade": math.log10(float(row["alpha"])/a9)}
        rows.append(scored(row, result, FIG8_CONDITION, **extra))
    for figure in (10, 11):
        for row in read(REFERENCE / f"figure{figure}.csv"):
            if row["temperature_K"] != "298.15" or figure == 11:
                rows.append({**entry(row, challenge if row["temperature_K"] != "298.15" else "authors' ln K transform of Fig. 10"), **common})
                continue
            r6, r7 = (solve(*conditions(row, eps), ln_k, loading={"Li": float(row["d2ehdtpa_mol_L_organic"])}) for eps in (1e-6, 1e-7))
            a7 = r7["alpha"][row["series"][3:]]
            check = math.log10(r6["alpha"][row["series"][3:]]/a7)
            rows.append(scored(row, r6, FIG10_CONDITION, alpha_model_Na_1e_7=a7, Na_trace_falsifier_decade=check,
                               Na_trace_falsifier_passed=abs(check) <= 1e-3))
    pd.DataFrame(rows).to_csv(out_dir / "holdout_scores.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alternative", choices=("fig7_exchange", "table1"), default="fig7_exchange")
    parser.add_argument("--start", type=int, choices=(0, 1, 2))
    parser.add_argument("--holdouts", action="store_true", help="#210 holdout scores, Held then provisional Figiel")
    args = parser.parse_args()
    installed_engine()
    if args.holdouts:  # never rewrites inputs/ or retained results; labels and folders passed, not module globals
        from . import conditional_figiel_packet as cf
        score_holdouts(json.loads((RESULTS / "fig7_exchange_start0.json").read_text())["ln_K_Li_Mg_Ca"], RESULTS, LABELS)
        cf.select("figiel-2025-chloride-activity-refit/2")
        cf.use("packet")
        score_holdouts(json.loads((cf.OUT / "fig7_exchange_start0.json").read_text())["ln_K_Li_Mg_Ca"], cf.OUT, cf.cs.LABELS)
        return
    (INPUTS / "reactive_parameters.json").write_text(json.dumps(payload(), indent=2) + "\n")
    fit(args.alternative, args.start) if args.start is not None else score()


if __name__ == "__main__":
    main()
