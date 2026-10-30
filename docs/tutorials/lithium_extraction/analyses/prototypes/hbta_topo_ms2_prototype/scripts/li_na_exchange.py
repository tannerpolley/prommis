"""Set A Li/Na exchange refit on saponified Zhang et al. 2020 data and its holdouts (Lithium #166, stage 1).

The plan, split, tolerance and falsifier were fixed before fitting (issue #166, recalibration plan comment).

- Fitted: ln K_ex = ln K(R1) - ln K(R2) only, on log10 D_Li of Zhang 2020 Fig. 6 (9) and Table 2 (6). ln K(R2) is
  held at the #158 set A value in ``mechanism_candidates.json``.
- Independent holdouts, each passing if its worst |log10 pred - log10 obs| <= 0.30: Shan 2025 Fig. 3 (printed
  D_Li) and the Zhang 2021 Table 9 Li step (log10 raffinate Li, 3-stage countercurrent, R 70 %).
- Same-paper holdout, reported against 0.30: Zhang 2020 Fig. 7, D_Li from E only.
- Reported: Zhang 2020 Fig. 2 equilibrium pH against the model (Table 2 states), Shan's slope of log D against
  log Na, sensitivities (feed pH +/- 1, Shan 60 mg/L Na, Zhang 2021 at R 66.54 %), and MS-2 one stage at
  0.1-0.4 M with ln K(R2) +/- 2.

Sulfate media (Zhang 2020, 2021) are represented with the same Li and Na charge-closed by chloride; kerosene by
dodecane; 298.15 K. The saponified organic enters as NaA.T. Every state is a strict Engine state (mc.solve).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import time

from scipy.optimize import least_squares

from analyses.shared.hbta_topo_configured_reactive_lle.scripts import mechanism_candidates as mc
from analyses.shared.hbta_topo_configured_reactive_lle.scripts.apparent_application import shan_condition
from analyses.prototypes.hbta_topo_ms2_prototype.scripts.countercurrent import acidity_at_ph, countercurrent, stage_state
from lithium_extraction.prototypes.produced_water_sx_prototype.flowsheet import installed_engine

ROOT = Path(__file__).resolve().parents[4]
RESULTS = Path(__file__).resolve().parents[1] / "results"
DATA = ROOT / "data/reference/extraction_models/hbta_topo_li_na"
LN_K_R2 = json.loads((mc.RESULTS / "mechanism_candidates.json").read_text())["summary"]["A"]["ln_k_298"]["R2"]
LI_G_MOL, NA_G_MOL = 6.941, 22.99  # tables/conditions.csv molar conversions
TOLERANCE_DECADE = 0.30
E_LIMIT_PERCENT = 99.5  # D from E is not scored at or above this extraction
FEED_PH = 8.52  # Zhang 2020 Fig. 2 / Table 3 feed; not reported for Figs. 6, 7 or Zhang 2021
R_NOMINAL_70 = 0.6654  # Zhang 2020 Table 2 measured saponification at nominal 70 %
FIT_STARTS = (0.0, 2.5, 5.0, 7.5, 10.0)
FIT_BOUNDS = (-5.0, 15.0)
START_AGREEMENT = 1.0e-6
EXTRACTANT_MS2 = (0.1, 0.2, 0.3, 0.4)


@dataclass(frozen=True)
class Contact:
    """One single batch contact, per litre of aqueous; ``saponification`` is NaA.T / extractant before contact."""

    dataset: str
    role: str
    identity: str
    organic_to_aqueous: float
    extractant: float
    saponification: float
    li: float
    na: float
    feed_ph: float
    observed_d: float | None
    observed_basis: str
    sensitivity: str = ""

    def state(self) -> mc.State:
        return stage_state(self.identity, organic_to_aqueous=self.organic_to_aqueous, extractant=self.extractant,
                           li=self.li, na=self.na, acidity=acidity_at_ph(self.feed_ph),
                           na_org=self.saponification * self.extractant)


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def contacts() -> list[Contact]:
    """Fit and single-contact holdout rows at their base conditions."""
    rows = []
    for r in _csv(DATA / "digitization/zhang2020_figure6/points.csv"):  # O/A 1, Na 1.370 M, R 70 % nominal
        li0, li_org = float(r["li_initial_aq_mol_l"]), float(r["li_org_mol_l"])
        rows.append(Contact("zhang2020_figure6", "fit", r["point_id"], 1.0, 0.4, R_NOMINAL_70, li0, 1.370, FEED_PH,
                            li_org / (li0 - li_org), "organic Li / (initial - organic Li)"))
    for r in _csv(DATA / "tables/zhangLithiumRecoveryEffluent2020_table2.csv"):  # Table 1 feed, O/A 1 inferred
        li_org = float(r["li_organic_mol_l"])
        rows.append(Contact("zhang2020_table2", "fit", f"zhang2020_table2_rs_{r['saponification_degree_cal_percent']}",
                            1.0, 0.4, float(r["na_saponified_organic_mol_l"]) / 0.4, 0.267, 1.36, FEED_PH,
                            li_org / (0.267 - li_org), "organic Li / (0.267 - organic Li)"))
    for r in _csv(DATA / "digitization/zhang2020_figure7/points.csv"):  # E only: the Li-organic series is aqueous Li
        oa, e = float(r["organic_to_aqueous_numeric"]), float(r["li_extraction_percent"])
        if e >= E_LIMIT_PERCENT:
            raise ValueError(f"{r['point_id']}: E {e} % is above the {E_LIMIT_PERCENT} % scoring limit")
        rows.append(Contact("zhang2020_figure7", "holdout_same_paper", r["point_id"], oa, 0.4, R_NOMINAL_70, 0.267,
                            1.370, FEED_PH, e / (100.0 - e) / oa, "E / (100 - E) / (O/A)"))
    for r in _csv(DATA / "digitization/shan2025_figure3/points.csv"):
        source = shan_condition(float(r["na_mg_l"]), float(r["li_extraction_percent"]), "holdout")
        rows.append(Contact("shan2025_figure3", "holdout_independent", r["point_id"], source.organic_volume_l / source.aqueous_volume_l,
                            source.hbta_mol_l_organic, source.saponification_fraction, source.li_mol_l_aqueous,
                            source.na_mol_l_aqueous, source.initial_ph, float(r["li_distribution_ratio"]),
                            "printed D_Li"))
    return rows


def sensitivity_contacts(base: list[Contact]) -> list[Contact]:
    """Feed pH +/- 1 for Zhang 2020 Figs. 6 and 7 and Shan; Shan's 0 Na control at the methods' 60 mg/L."""
    rows = []
    for c in base:
        if c.dataset in ("zhang2020_figure6", "zhang2020_figure7", "shan2025_figure3"):
            for shift in (-1.0, 1.0):
                rows.append(replace(c, identity=f"{c.identity}_ph_{c.feed_ph + shift:.2f}", feed_ph=c.feed_ph + shift,
                                    sensitivity=f"feed pH {shift:+.0f}"))
        if c.identity == "shan_fig3_na_0":
            rows.append(replace(c, identity="shan_fig3_na_60", na=0.060 / 22.98976928, sensitivity="Na 60 mg/L"))
    return rows


def ln_k(ln_k_ex: float, ln_k_r2: float = LN_K_R2) -> dict[str, float]:
    return {"R1": ln_k_ex + ln_k_r2, "R2": ln_k_r2}


_GUESS: dict[str, dict] = {}  # last strict composition per contact: a warm start only


def solve(contact: Contact, constants: dict[str, float]) -> dict:
    result = mc.solve(contact.state(), "A", constants, guess=_GUESS.get(contact.identity))
    _GUESS[contact.identity] = result["phase_amounts_mol"]
    return result


def record(contact: Contact, result: dict, ln_k_ex: float, ln_k_r2: float = LN_K_R2) -> dict:
    residual = None if contact.observed_d is None else math.log10(result["li_distribution"] / contact.observed_d)
    return {
        "dataset": contact.dataset, "role": contact.role, "identity": contact.identity,
        "sensitivity": contact.sensitivity, "organic_to_aqueous": contact.organic_to_aqueous,
        "extractant_mol_l": contact.extractant, "saponification_fraction": contact.saponification,
        "li_feed_mol_l": contact.li, "na_feed_mol_l": contact.na, "feed_ph": contact.feed_ph,
        "ln_k_ex": ln_k_ex, "ln_k_r2": ln_k_r2, "observed_d_li": contact.observed_d,
        "observed_basis": contact.observed_basis, "model_d_li": result["li_distribution"],
        "log10_model_over_observed_d_li": residual, "model_d_na": result["na_distribution"],
        "model_li_extraction_percent": result["li_extraction_percent"], "model_equilibrium_ph": result["ph"],
        "active_reactions": " ".join(result["active_reactions"]), "status": result["status"],
        **{k: result[k] for k in ("balance_inf_norm", "charge_inf_norm", "absence_inf_norm",
                                  "reaction_affinity_inf_norm", "kkt_stationarity_inf_norm",
                                  "reduced_hessian_min_eigenvalue", "pressure_relative_inf_norm")},
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fit(args) -> None:
    """Bounded least squares on log10 D_Li over the 15 fit contacts from each start; a failed state stops the fit."""
    engine = installed_engine()
    fit_rows = [c for c in contacts() if c.role == "fit"]
    started, solves = time.perf_counter(), 0

    def residual(q):
        nonlocal solves
        solves += len(fit_rows)
        return [math.log10(solve(c, ln_k(q[0]))["li_distribution"] / c.observed_d) for c in fit_rows]

    starts = []
    for x0 in FIT_STARTS:
        t0 = time.perf_counter()
        s = least_squares(residual, [x0], bounds=FIT_BOUNDS, diff_step=1e-4, xtol=1e-10, ftol=1e-12, gtol=1e-12,
                          max_nfev=200)
        starts.append({"start": x0, "ln_k_ex": float(s.x[0]), "cost": float(s.cost), "nfev": int(s.nfev),
                       "status": int(s.status), "message": s.message, "wall_time_seconds": time.perf_counter() - t0})
        print(f"start {x0}: ln K_ex {s.x[0]:.10f} cost {s.cost:.8g} nfev {s.nfev} status {s.status}", flush=True)
    best = min(starts, key=lambda s: s["cost"])
    spread = max(abs(s["ln_k_ex"] - best["ln_k_ex"]) for s in starts)
    document = {
        "engine": engine, "issue": "tannerpolley/Lithium_Extraction#166",
        "plan": "issue #166 recalibration plan comment (stage 1), fixed before fitting",
        "ln_k_ex": best["ln_k_ex"], "ln_k_r2_fixed": LN_K_R2, "ln_k_r1": best["ln_k_ex"] + LN_K_R2,
        "k_ex": math.exp(best["ln_k_ex"]), "log10_k_ex": best["ln_k_ex"] / math.log(10.0),
        "bounds": FIT_BOUNDS, "starts": starts, "start_spread_ln_k_ex": spread,
        "starts_agree": spread <= START_AGREEMENT, "fit_points": len(fit_rows), "strict_solves": solves,
        "wall_time_seconds": time.perf_counter() - started,
    }
    (RESULTS / "fit.json").write_text(json.dumps(document, indent=1, allow_nan=False) + "\n")
    if not document["starts_agree"]:
        raise SystemExit(f"starts disagree on ln K_ex by {spread:g} (> {START_AGREEMENT}): goes to diagnosis")


def _slope(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / sum((a - mx) ** 2 for a in xs)


def zhang2021_cascade(constants: dict[str, float], label: str, fraction: float, ph: float) -> tuple[dict, list[dict]]:
    """Zhang 2021 Table 9 Li step: 3-stage countercurrent, O/A 1, 0.2 M + 0.2 M, 907 mg/L Li, 5.21e4 mg/L Na."""
    feed_li, feed_na, observed_mg_l = 907.0 / LI_G_MOL / 1000.0, 5.21e4 / NA_G_MOL / 1000.0, 5.16
    t0 = time.perf_counter()
    out = countercurrent(constants, reaction_set="A", stage_count=3, organic_to_aqueous=1.0, extractant=0.2,
                         saponification_fraction=fraction, feed_li=feed_li, feed_na=feed_na,
                         feed_acidity=acidity_at_ph(ph))
    raffinate_mg_l = out["raffinate"]["li"] * LI_G_MOL * 1000.0
    row = {
        "case": label, "saponification_fraction": fraction, "feed_ph": ph, "raffinate_li_mg_l": raffinate_mg_l,
        "observed_raffinate_li_mg_l": observed_mg_l,
        "log10_model_over_observed_raffinate_li": math.log10(raffinate_mg_l / observed_mg_l),
        "raffinate_na_mg_l": out["raffinate"]["na"] * NA_G_MOL * 1000.0, "observed_raffinate_na_mg_l": 5.52e4,
        "li_extraction_percent": 100.0 * out["li_extraction_fraction"], "sweeps": out["sweeps"],
        "closure_mol": out["closure_mol"], "stage_ph": [s["stage_ph"] for s in out["stages"]],
        "stages_with_water_autoprotolysis": sum(s["water_autoprotolysis_active"] for s in out["stages"]),
        "strict_stage_states": len(out["stages"]), "wall_time_seconds": time.perf_counter() - t0,
    }
    return row, [{"case": label, **s} for s in out["stages"]]


def ms2_one_stage(kex: float, ln_k_r2: float, tag: str, sensitivity: str) -> list[dict]:
    """MS-2 one stage: measured Li and Na (divalents omitted, chloride charge-closed), feed pH 5.77, O/A 1, R 0.70."""
    rows = []
    for c in EXTRACTANT_MS2:
        contact = Contact("ms2", "prediction", f"ms2_c_{c}_ln_k_r2_{tag}", 1.0, c, 0.70, mc.MS2_LI, mc.MS2_NA,
                          mc.MS2_PH, None, "", sensitivity)
        rows.append(record(contact, solve(contact, ln_k(kex, ln_k_r2)), kex, ln_k_r2))
    return rows


def predict(args) -> None:
    engine = installed_engine()
    started = time.perf_counter()
    fitted = json.loads((RESULTS / "fit.json").read_text())
    if not fitted["starts_agree"]:
        raise SystemExit("the fit's starts disagree: no predictions until diagnosed")
    kex = fitted["ln_k_ex"]
    base = contacts()
    rows = [record(c, solve(c, ln_k(kex)), kex) for c in base + sensitivity_contacts(base)]
    # Check of the plan's identifiability statement: fit-contact D_Li at ln K(R2) +/- 2 with ln K_ex held.
    for shift in (-2.0, 2.0):
        rows += [record(replace(c, sensitivity=f"ln K(R2) {shift:+.0f}"), solve(c, ln_k(kex, LN_K_R2 + shift)), kex,
                        LN_K_R2 + shift) for c in base if c.role == "fit"]
    _write_csv(RESULTS / "contacts.csv", rows)
    basic = [r for r in rows if not r["sensitivity"]]

    def worst(dataset):
        return max(abs(r["log10_model_over_observed_d_li"]) for r in basic if r["dataset"] == dataset)

    cascades, stage_rows = [], []
    for label, fraction, ph in (("scored_r_0.70", 0.70, FEED_PH), ("r_0.6654", R_NOMINAL_70, FEED_PH),
                                ("r_0.70_feed_ph_7.52", 0.70, FEED_PH - 1), ("r_0.70_feed_ph_9.52", 0.70, FEED_PH + 1)):
        summary_row, stages = zhang2021_cascade(ln_k(kex), label, fraction, ph)
        cascades.append(summary_row)
        stage_rows += stages
    _write_csv(RESULTS / "zhang2021_cascade_stages.csv", stage_rows)

    ms2 = []
    for shift in (-2.0, 0.0, 2.0):
        ms2 += ms2_one_stage(kex, LN_K_R2 + shift, f"{shift:+.0f}", f"ln K(R2) {shift:+.0f}" if shift else "")
    _write_csv(RESULTS / "ms2_one_stage.csv", ms2)

    shan = sorted((r for r in basic if r["dataset"] == "shan2025_figure3" and r["na_feed_mol_l"] > 0),
                  key=lambda r: r["na_feed_mol_l"])
    log_na = [math.log10(r["na_feed_mol_l"]) for r in shan]
    table2 = {int(r["identity"].rsplit("_", 1)[1]): r for r in basic if r["dataset"] == "zhang2020_table2"}
    figure2 = [{"saponification_degree_cal_percent": int(p["saponification_degree_cal_percent"]),
                "observed_ph": float(p["ph_extraction_raffinate"]),
                "model_ph": table2[int(p["saponification_degree_cal_percent"])]["model_equilibrium_ph"]}
               for p in _csv(DATA / "digitization/zhang2020_figure2/points.csv")]
    for p in figure2:
        p["model_minus_observed"] = p["model_ph"] - p["observed_ph"]
    fit_rows = [r for r in basic if r["role"] == "fit"]
    zhang2021 = cascades[0]
    holdouts = {
        "shan2025_figure3": {"worst_abs_log10_d": worst("shan2025_figure3"), "points": sum(r["dataset"] == "shan2025_figure3" for r in basic)},
        "zhang2021_table9_li_step": {"worst_abs_log10_d": abs(zhang2021["log10_model_over_observed_raffinate_li"]),
                                     "points": 1},
    }
    for h in holdouts.values():
        h["passes"] = h["worst_abs_log10_d"] <= TOLERANCE_DECADE
    identifiability = [r for r in rows if r["sensitivity"].startswith("ln K(R2)")]
    base_d = {r["identity"]: r["model_d_li"] for r in basic}
    sensitivity = {}
    for r in rows:
        if r["sensitivity"].startswith("feed pH"):
            key = f"{r['dataset']} {r['sensitivity']}"
            change = abs(math.log10(r["model_d_li"] / base_d[r["identity"].rsplit("_ph_", 1)[0]]))
            sensitivity[key] = max(sensitivity.get(key, 0.0), change)
    shan60 = next(r for r in rows if r["identity"] == "shan_fig3_na_60")
    by_id = {r["identity"]: r for r in fit_rows}
    summary = {
        "engine": engine, "issue": "tannerpolley/Lithium_Extraction#166",
        "ln_k_ex": kex, "ln_k_r2_fixed": LN_K_R2, "tolerance_decade": TOLERANCE_DECADE,
        "fit": {"points": len(fit_rows), "residuals_log10": {r["identity"]: r["log10_model_over_observed_d_li"]
                                                             for r in fit_rows},
                "rms_log10": math.sqrt(sum(r["log10_model_over_observed_d_li"] ** 2 for r in fit_rows) / len(fit_rows)),
                "worst_abs_log10": max(abs(r["log10_model_over_observed_d_li"]) for r in fit_rows),
                "identifiability_max_abs_log10_d_change_ln_k_r2_pm2": max(
                    abs(math.log10(r["model_d_li"] / by_id[r["identity"]]["model_d_li"])) for r in identifiability)},
        "holdouts_independent": holdouts,
        "recalibration_passes": all(h["passes"] for h in holdouts.values()),
        "zhang2020_figure7_same_paper": {"worst_abs_log10_d": worst("zhang2020_figure7"),
                                         "within_0.30": worst("zhang2020_figure7") <= TOLERANCE_DECADE},
        "zhang2021_cascades": cascades,
        "shan_slope_log10_d_per_log10_na": {
            "points": "Na 2000-10900 mg/L (0 Na has no logarithm)",
            "measured": _slope(log_na, [math.log10(r["observed_d_li"]) for r in shan]),
            "model": _slope(log_na, [math.log10(r["model_d_li"]) for r in shan])},
        "zhang2020_figure2_ph": figure2,
        "feed_ph_sensitivity_worst_abs_change_log10_d": sensitivity,
        "shan_na_60_mg_l": {k: shan60[k] for k in ("model_d_li", "log10_model_over_observed_d_li")},
        "strict_states": {"single_contacts": len(rows) + len(ms2),
                          "cascade_stages": sum(c["strict_stage_states"] for c in cascades),
                          "all_evaluated": all(r["status"] == "evaluated" for r in rows + ms2)},
        "reported_not_modeled": {
            "zhang2022_table3": "synergist is TRPO, not TOPO",
            "xing2024_table3": "unsaponified, no pH reported",
            "zhang2020_figure3": "saponification and O/A unstated",
        },
        "wall_time_seconds": time.perf_counter() - started,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n")


def ph_consistent(args) -> None:
    """Diagnostic, added after review: ln K(R2) fitted to Zhang 2020 Fig. 2 pH with ln K_ex held, and every score
    repeated there. The pre-registered result (ln K(R2) held at the #158 value) is not changed.

    Rule: bounded least squares of (model pH - Fig. 2 extraction-raffinate pH) over the six Table 2 contacts (the
    same experiments as Fig. 2, matched by calculated saponification degree), from four starts.
    """
    engine = installed_engine()
    started = time.perf_counter()
    kex = json.loads((RESULTS / "fit.json").read_text())["ln_k_ex"]
    base = contacts()
    fig2_ph = {f"zhang2020_table2_rs_{p['saponification_degree_cal_percent']}": float(p["ph_extraction_raffinate"])
               for p in _csv(DATA / "digitization/zhang2020_figure2/points.csv")}
    table2 = [c for c in base if c.dataset == "zhang2020_table2"]

    def residual(q):
        return [solve(c, ln_k(kex, q[0]))["ph"] - fig2_ph[c.identity] for c in table2]

    starts = []
    for x0 in (LN_K_R2, -12.0, -18.0, -24.0):
        fit_ = least_squares(residual, [x0], bounds=(-40.0, 0.0), diff_step=1e-4, xtol=1e-10, ftol=1e-12,
                             gtol=1e-12, max_nfev=200)
        starts.append({"start": x0, "ln_k_r2": float(fit_.x[0]), "cost": float(fit_.cost), "nfev": int(fit_.nfev),
                       "status": int(fit_.status)})
        print(f"start {x0}: ln K(R2) {fit_.x[0]:.10f} cost {fit_.cost:.8g} nfev {fit_.nfev}", flush=True)
    k2 = min(starts, key=lambda s: s["cost"])["ln_k_r2"]
    spread = max(abs(s["ln_k_r2"] - k2) for s in starts)
    held = {r["identity"]: r for r in _csv(RESULTS / "contacts.csv") if not r["sensitivity"]}
    rows = [record(c, solve(c, ln_k(kex, k2)), kex, k2) for c in base]
    ms2 = ms2_one_stage(kex, k2, "ph_consistent", "pH-consistent ln K(R2)")
    _write_csv(RESULTS / "ph_consistent_contacts.csv", rows + ms2)
    cascades, stage_rows = [], []
    for label, fraction in (("scored_r_0.70", 0.70), ("r_0.6654", R_NOMINAL_70)):
        summary_row, stages = zhang2021_cascade(ln_k(kex, k2), label, fraction, FEED_PH)
        cascades.append(summary_row)
        stage_rows += stages
    _write_csv(RESULTS / "ph_consistent_zhang2021_cascade_stages.csv", stage_rows)

    def worst(dataset):
        return max(abs(r["log10_model_over_observed_d_li"]) for r in rows if r["dataset"] == dataset)

    fit_rows = [r for r in rows if r["role"] == "fit"]
    shan = sorted((r for r in rows if r["dataset"] == "shan2025_figure3" and r["na_feed_mol_l"] > 0),
                  key=lambda r: r["na_feed_mol_l"])
    log_na = [math.log10(r["na_feed_mol_l"]) for r in shan]
    holdouts = {
        "shan2025_figure3": worst("shan2025_figure3"),
        "zhang2021_table9_li_step": abs(cascades[0]["log10_model_over_observed_raffinate_li"]),
    }
    held_summary = json.loads((RESULTS / "summary.json").read_text())
    document = {
        "engine": engine, "issue": "tannerpolley/Lithium_Extraction#166",
        "purpose": "diagnostic added after independent review; the pre-registered scores in summary.json stand",
        "rule": "bounded least squares of model pH - Zhang 2020 Fig. 2 extraction-raffinate pH over the six Table 2 "
                "contacts, ln K_ex held at fit.json",
        "ln_k_ex": kex, "ln_k_r2_ph_consistent": k2, "ln_k_r1": kex + k2, "ln_k_r2_held": LN_K_R2,
        "starts": starts, "start_spread_ln_k_r2": spread,
        "figure2_ph": [{"identity": c.identity, "observed_ph": fig2_ph[c.identity],
                        "model_ph": next(r["model_equilibrium_ph"] for r in rows if r["identity"] == c.identity)}
                       for c in table2],
        "fit": {"rms_log10": math.sqrt(sum(r["log10_model_over_observed_d_li"] ** 2 for r in fit_rows) / len(fit_rows)),
                "worst_abs_log10": max(abs(r["log10_model_over_observed_d_li"]) for r in fit_rows),
                "max_abs_residual_change_from_held": max(
                    abs(r["log10_model_over_observed_d_li"] - float(held[r["identity"]]["log10_model_over_observed_d_li"]))
                    for r in fit_rows)},
        "holdouts_independent": {k: {"worst_abs_log10": v, "passes": v <= TOLERANCE_DECADE,
                                     "verdict_changed": (v <= TOLERANCE_DECADE)
                                     != held_summary["holdouts_independent"][k]["passes"]}
                                 for k, v in holdouts.items()},
        "zhang2020_figure7_worst_abs_log10": worst("zhang2020_figure7"),
        "shan_slope_log10_d_per_log10_na_model": _slope(log_na, [math.log10(r["model_d_li"]) for r in shan]),
        "zhang2021_cascades": cascades,
        "ms2_one_stage": [{k: r[k] for k in ("extractant_mol_l", "model_li_extraction_percent", "model_d_li",
                                             "model_equilibrium_ph")} for r in ms2],
        "strict_states": {"single_contacts": len(rows) + len(ms2),
                          "cascade_stages": sum(c["strict_stage_states"] for c in cascades),
                          "all_evaluated": all(r["status"] == "evaluated" for r in rows + ms2)},
        "wall_time_seconds": time.perf_counter() - started,
    }
    (RESULTS / "ph_consistent.json").write_text(json.dumps(document, indent=1, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("fit", "predict", "ph-consistent"))
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    {"fit": fit, "predict": predict, "ph-consistent": ph_consistent}[args.step](args)


if __name__ == "__main__":
    main()
