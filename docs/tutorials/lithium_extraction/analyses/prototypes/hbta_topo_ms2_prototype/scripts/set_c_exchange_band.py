"""pH-consistent set C with a Li/Na exchange band (Lithium #166, stage 2).

The constants' roles, falsifiers, delta bracket and expected outcome were fixed before fitting (issue #166, stage 2
plan comment).

- Set C (#158): R1 Li+ + HA + T = LiA.T + H+, R2 Na+ + HA + T = NaA.T + H+, R5 HA(org) = H+ + BTA-(aq).
- ln K(R2) = #158 value + delta and ln K(R5) = #158 value + delta, so ln K5 - ln K2 stays at #158 set C (-4.495);
  ln K(R1) = ln K(R2) + ln K_ex.
- delta: bounded least squares of (model - Zhang 2020 Fig. 2 extraction-raffinate pH) over the six Table 2 contacts,
  ln K_ex held.
- Low end of ln K_ex: least squares on log10 D_Li over the 15 Zhang 2020 contacts (as stage 1), delta held. delta and
  ln K_ex are refit alternately until both change by < 1e-6 (at most 10 rounds).
- High end of ln K_ex: the root at which the 3-stage Zhang 2021 Table 9 Li step leaves 5.16 mg/L Li at R 70 %, with
  delta refit alternately by the same rule. R 66.54 % is a sensitivity cascade at the high-end constants, not a refit
  (owner decision, 2026-09-29).
- Each one-parameter fit and each root uses at least three starts, which must agree to 1e-6.
- delta bracket: delta +/- 0.5 ln 10 with ln K_ex held; applied to the Shan scores and every MS-2 prediction.

Scored: Zhang 2020 Fig. 2 pH within +/- 0.5 at the low end; Shan 2025 Fig. 3 worst |log10 D_Li| <= 0.30 at both
ends, each at its fitted delta. Reported: Shan slope and delta bracket, the ln K_ex sub-range where Shan passes,
Zhang 2020 Fig. 7, the Cha 2025 HBTA loss, and MS-2 one-stage and 3-stage predictions for the divalents-omitted and
softened feeds. Representations are those of stage 1 (``li_na_exchange.py``): chloride for sulfate, dodecane for
kerosene, 298.15 K. Every state is a strict Engine state (mc.solve).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import time

from scipy.optimize import brentq, least_squares, root_scalar

from analyses.shared.hbta_topo_configured_reactive_lle.scripts import mechanism_candidates as mc
from analyses.shared.hbta_topo_configured_reactive_lle.scripts.apparent_application import HBTA_MOLAR_MASS
from analyses.prototypes.hbta_topo_ms2_prototype.scripts import li_na_exchange as lx
from analyses.prototypes.hbta_topo_ms2_prototype.scripts.countercurrent import acidity_at_ph, countercurrent
from lithium_extraction.common.pretreatment import softening

RESULTS = lx.RESULTS
SET_C_158 = json.loads((mc.RESULTS / "mechanism_candidates.json").read_text())["summary"]["C"]["ln_k_298"]
DELTA_BRACKET = 0.5 * math.log(10.0)  # +/- 0.5 pH unit
AGREE = 1.0e-6
MAX_ROUNDS = 10
DELTA_STARTS, DELTA_BOUNDS = (0.0, -6.0, -12.0, -18.0), (-30.0, 5.0)
KEX_STARTS = (2.5, 5.0, 7.5)
ROOT_OFFSETS = (-0.1, 0.0, 0.1)  # root starts about the previous round's ln K_ex (round 1: ROOT_FIRST_CENTER)
ROOT_FIRST_CENTER = 5.88695630  # R 70 % root at the low-end delta, from a first run stopped for CPU sharing
PROGRESS = RESULTS / "set_c_progress.json"  # resumable work between lock-held units; removed when complete
MORE = 3  # exit status: the step has more units to run
PH_TOLERANCE = 0.5
ZHANG2021_RAFFINATE_LI_MG_L = 5.16
# Cha et al. 2025 section 2.2 and 3.2: 0.4 M + 0.4 M, 80 % saponified, O/A 1, Li 1.85 g/L, Na 31.2 g/L, one contact;
# 626 mg/L HBTA in the water (0.72 %). Its feed is Zhang 2020's (Cha ref. 53); no feed pH is given, so Fig. 2's.
CHA = lx.Contact("cha2025", "context", "cha2025_r_0.80", 1.0, 0.4, 0.80, 1.85 / lx.LI_G_MOL, 31.2 / lx.NA_G_MOL,
                 lx.FEED_PH, None, "")
CHA_HBTA_LOSS_MG_L, CHA_HBTA_LOSS_PERCENT = 626.0, 0.72
MS2_SAPONIFICATION = 0.70
# #210 step 5 holdouts (Round 2 plan, P2): representation assumptions on every scored row, and reasons not scored.
ASSUMED = {"seo2025_figure2": "preprint v1; R inferred in 10 % steps; TOPO 0.4 M = HBTA; sulfate as chloride; D80 as "
                              "dodecane; 298.15 K; 10 min contact as equilibrium",
           "ni2024_figure10": "0.1 + 0.1 M; kerosene as dodecane; 298.15 K (Ni: 20 C); chloride closes charge (1.35 "
                              "eq/L, printed 2.094); K, Rb omitted; stripped before re-saponification; 5 min contact"}
NI_FEED_PH = (7.0, 10.68)  # unbuffered alkaline raffinate: above 7 (Ni Section 1) to the buffered optimum's initial pH
NOT_SCORED = {"niSustainableEfficientRecovery2024_section3_1_6": "no pH-imposed countercurrent driver exists",
              "buffer": "NH3-NH4Cl buffer not representable: it sets the initial pH (10.68), not the equilibrium pH "
                        "(8.02); imposing an equilibrium pH would erase the saponification axis"}

_GUESS: dict[str, dict] = {}  # last strict composition per contact: a warm start only
_CASCADE_START: dict[str, dict] = {}  # last converged cascade per feed: a warm start only
_SOLVES = [0]


def ln_k(ln_k_ex: float, delta: float) -> dict[str, float]:
    ln_k2 = SET_C_158["R2"] + delta
    return {"R1": ln_k2 + ln_k_ex, "R2": ln_k2, "R5": SET_C_158["R5"] + delta}


def solve(contact: lx.Contact, constants: dict[str, float]) -> dict:
    _SOLVES[0] += 1
    result = mc.solve(contact.state(), "C", constants, guess=_GUESS.get(contact.identity))
    _GUESS[contact.identity] = result["phase_amounts_mol"]
    return result


def _one_parameter(residual, starts, bounds, name: str) -> dict:
    """Bounded least squares from each start; the lowest cost is kept, and the starts must agree to 1e-6."""
    runs = []
    for x0 in starts:
        s = least_squares(residual, [x0], bounds=bounds, diff_step=1e-4, xtol=1e-10, ftol=1e-12, gtol=1e-12,
                          max_nfev=200)
        runs.append({"start": x0, name: float(s.x[0]), "cost": float(s.cost), "nfev": int(s.nfev),
                     "status": int(s.status)})
        print(f"  {name} start {x0}: {s.x[0]:.10f} cost {s.cost:.8g} nfev {s.nfev}", flush=True)
    best = min(runs, key=lambda r: r["cost"])
    spread = max(abs(r[name] - best[name]) for r in runs)
    return {"value": best[name], "cost": best["cost"], "starts": runs, "spread": spread, "agree": spread <= AGREE}


def _table2_ph():
    table2 = [c for c in lx.contacts() if c.dataset == "zhang2020_table2"]
    fig2 = {f"zhang2020_table2_rs_{p['saponification_degree_cal_percent']}": float(p["ph_extraction_raffinate"])
            for p in lx._csv(lx.DATA / "digitization/zhang2020_figure2/points.csv")}
    return table2, fig2


def fit_delta(kex: float) -> dict:
    table2, fig2 = _table2_ph()
    return _one_parameter(lambda q: [solve(c, ln_k(kex, q[0]))["ph"] - fig2[c.identity] for c in table2],
                          DELTA_STARTS, DELTA_BOUNDS, "delta")


def fit_kex(delta: float) -> dict:
    fit_rows = [c for c in lx.contacts() if c.role == "fit"]
    return _one_parameter(
        lambda q: [math.log10(solve(c, ln_k(q[0], delta))["li_distribution"] / c.observed_d) for c in fit_rows],
        KEX_STARTS, lx.FIT_BOUNDS, "ln_k_ex")


def zhang2021(constants: dict[str, float], fraction: float) -> dict:
    """Zhang 2021 Table 9 Li step: 3-stage countercurrent, O/A 1, 0.2 M + 0.2 M, feed pH 8.52."""
    key = f"zhang2021_{fraction}"
    out = countercurrent(constants, reaction_set="C", stage_count=3, organic_to_aqueous=1.0, extractant=0.2,
                         saponification_fraction=fraction, feed_li=907.0 / lx.LI_G_MOL / 1000.0,
                         feed_na=5.21e4 / lx.NA_G_MOL / 1000.0, feed_acidity=acidity_at_ph(lx.FEED_PH),
                         start=_CASCADE_START.get(key))
    _CASCADE_START[key] = out["start"]
    _SOLVES[0] += 3 * out["sweeps"]
    return out


def root_start(delta: float, fraction: float, x0: float) -> dict:
    """One secant start for the ln K_ex at which the Zhang 2021 cascade leaves 5.16 mg/L Li."""
    def f(kex):
        return math.log(zhang2021(ln_k(kex, delta), fraction)["raffinate"]["li"] * lx.LI_G_MOL * 1000.0
                        / ZHANG2021_RAFFINATE_LI_MG_L)
    r = root_scalar(f, method="secant", x0=x0, x1=x0 + 0.05, xtol=1e-9, maxiter=50)
    print(f"  root start {x0}: {r.root:.10f} converged {r.converged} calls {r.function_calls}", flush=True)
    return {"start": x0, "ln_k_ex": float(r.root), "converged": bool(r.converged),
            "function_calls": int(r.function_calls)}


def _progress() -> dict:
    progress = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {}
    _CASCADE_START.update(progress.get("cascade_start", {}))
    return progress


def _save(progress: dict) -> None:
    progress["cascade_start"] = _CASCADE_START
    PROGRESS.write_text(json.dumps(progress, allow_nan=False) + "\n")


def alternation_unit(state: dict, kex0: float, units: int, kex_unit) -> None:
    """Run the next unit of the delta / ln K_ex alternation held in ``state`` (resumable across processes).

    A round is a delta fit (pH rule) at the previous round's ln K_ex, then ``units`` ln K_ex units, each
    ``kex_unit(round, index)``; their records are combined into the round's ln K_ex (starts agreeing to 1e-6). The
    alternation stops when delta and ln K_ex both change by < 1e-6 (at most 10 rounds).
    """
    rounds = state.setdefault("rounds", [])
    if not rounds or "ln_k_ex_fit" in rounds[-1]:
        kex_in = rounds[-1]["ln_k_ex_fit"]["value"] if rounds else kex0
        rounds.append({"round": len(rounds) + 1, "ln_k_ex_in": kex_in, "kex_runs": []})
    r = rounds[-1]
    if "delta_fit" not in r:
        r["delta_fit"] = fit_delta(r["ln_k_ex_in"])
        if not r["delta_fit"]["agree"]:
            state["status"] = "delta starts disagree"
        return
    if len(r["kex_runs"]) < units:
        r["kex_runs"].append(kex_unit(r, len(r["kex_runs"])))
        if units > 1:
            return
    runs = r.pop("kex_runs")
    if units == 1:
        k = runs[0]
    else:
        spread = max(abs(x["ln_k_ex"] - runs[0]["ln_k_ex"]) for x in runs)
        k = {"value": runs[0]["ln_k_ex"], "starts": runs, "spread": spread,
             "agree": spread <= AGREE and all(x["converged"] for x in runs)}
    r["ln_k_ex_fit"] = k
    if not k["agree"]:
        state["status"] = "ln K_ex starts disagree"
        return
    delta = r["delta_fit"]["value"]
    previous = rounds[-2]["delta_fit"]["value"] if len(rounds) > 1 else None
    change = (None if previous is None else abs(delta - previous), abs(k["value"] - r["ln_k_ex_in"]))
    r.update(delta_change=change[0], ln_k_ex_change=change[1])
    print(f"round {r['round']}: delta {delta:.10f} ln K_ex {k['value']:.10f} changes {change}", flush=True)
    if change[0] is not None and max(change) < AGREE:
        state.update(status="converged", ln_k_ex=k["value"], delta=delta, ln_k=ln_k(k["value"], delta))
    elif len(rounds) == MAX_ROUNDS:
        state["status"] = f"no convergence in {MAX_ROUNDS} rounds"


def fit(args) -> None:
    """One unit per run (one lock-held job): a delta fit, a ln K_ex fit (low end) or one root start (high end, R 70 %).
    Exits with MORE while the part has units left; the part is written to set_c_fit.json when it ends."""
    engine = lx.installed_engine()
    started = time.perf_counter()
    path = RESULTS / "set_c_fit.json"
    document = json.loads(path.read_text()) if path.exists() else {
        "issue": "tannerpolley/Lithium_Extraction#166", "plan": "issue #166 stage 2 plan comment, fixed before fitting",
        "set_c_158_ln_k": SET_C_158, "ln_k5_minus_ln_k2": SET_C_158["R5"] - SET_C_158["R2"],
        "delta_bracket": DELTA_BRACKET, "starts": {"delta": DELTA_STARTS, "delta_bounds": DELTA_BOUNDS,
                                                   "ln_k_ex": KEX_STARTS, "ln_k_ex_bounds": lx.FIT_BOUNDS,
                                                   "root_offsets": ROOT_OFFSETS,
                                                   "root_first_center": ROOT_FIRST_CENTER}, "high": {}}
    progress = _progress()
    state = progress.setdefault("fit", {}).setdefault(args.part, {"status": "running", "strict_solves": 0,
                                                                   "wall_time_seconds": 0.0})
    if args.part == "low":
        kex0 = json.loads((RESULTS / "fit.json").read_text())["ln_k_ex"]  # stage 1 set A refit: first round's ln K_ex
        alternation_unit(state, kex0, 1, lambda r, i: fit_kex(r["delta_fit"]["value"]))
    else:
        if document.get("low", {}).get("status") != "converged":
            raise SystemExit("the low end has not converged")
        def unit(r, i):
            center = r["ln_k_ex_in"] if r["round"] > 1 else ROOT_FIRST_CENTER
            return root_start(r["delta_fit"]["value"], 0.70, center + ROOT_OFFSETS[i])

        state["saponification_fraction"] = 0.70
        alternation_unit(state, document["low"]["ln_k_ex"], len(ROOT_OFFSETS), unit)
    state["strict_solves"] += _SOLVES[0]
    state["wall_time_seconds"] += time.perf_counter() - started
    if state["status"] == "running":
        _save(progress)
        raise SystemExit(MORE)
    document["engine"] = engine
    if args.part == "low":
        document["low"] = state
    else:
        document["high"][args.part.removeprefix("high_")] = state
    path.write_text(json.dumps(document, indent=1, allow_nan=False) + "\n")
    del progress["fit"][args.part]
    _save(progress)
    if state["status"] != "converged":
        raise SystemExit(f"{args.part}: {state['status']}: goes to diagnosis")


def _row(contact: lx.Contact, end: str, delta_case: str, kex: float, delta: float) -> dict:
    """One single contact; a contact with no strict state is recorded as such (it is not retried)."""
    head = {"end": end, "delta_case": delta_case, "delta": delta}
    try:
        result = solve(contact, ln_k(kex, delta))
    except RuntimeError as error:
        return {**head, "dataset": contact.dataset, "identity": contact.identity, "ln_k_ex": kex,
                "status": "no strict state", "error": str(error)[:500]}
    aq, org = result["phase_amounts_mol"]
    na_org = org.get("sodium-carrier", 0.0) / contact.organic_to_aqueous
    return {**head, **lx.record(contact, result, kex, ln_k(kex, delta)["R2"]),
            "ln_k_r5": ln_k(kex, delta)["R5"], "aqueous_bta_mol_l": aq.get("bta-anion", 0.0),
            "hbta_lost_mg_l": aq.get("bta-anion", 0.0) * HBTA_MOLAR_MASS * 1.0e6,
            "hbta_lost_percent": 100.0 * result["hbta_lost_to_aqueous_fraction"],
            "na_loaded_organic_mol_l": na_org,
            "na_loaded_over_feed_na": na_org * contact.organic_to_aqueous / contact.na if contact.na else None}


def _cascade(label: str, warm: str, constants: dict, stage_rows: list, **conditions) -> dict:
    """One 3-stage set C cascade, warm-started from the last converged cascade on the same ``warm`` key."""
    t0 = time.perf_counter()
    try:
        out = countercurrent(constants, reaction_set="C", stage_count=3, organic_to_aqueous=1.0,
                             start=_CASCADE_START.get(warm), **conditions)
    except RuntimeError as error:
        return {"case": label, "status": "failed", "error": str(error)[:500]}
    _CASCADE_START[warm] = out["start"]
    _SOLVES[0] += 3 * out["sweeps"]
    stage_rows += [{"case": label, **s} for s in out["stages"]]
    loaded = out["loaded_organic"]
    return {"case": label, "status": "evaluated", "raffinate_li_mol_l": out["raffinate"]["li"],
            "li_extraction_percent": 100.0 * out["li_extraction_fraction"],
            "raffinate_na_mol_l": out["raffinate"]["na"], "na_loaded_organic_mol_l": loaded["na"],
            "na_loaded_over_feed_na": loaded["na"] / conditions["feed_na"],
            "raffinate_bta_mol_l": out["raffinate"]["bta"],
            "hbta_lost_mg_l": out["raffinate"]["bta"] * HBTA_MOLAR_MASS * 1.0e6,
            "hbta_lost_percent": 100.0 * out["hbta_lost_fraction"], "loaded_organic_hbta_mol_l": loaded["hbta"],
            "stage_ph": [s["stage_ph"] for s in out["stages"]], "sweeps": out["sweeps"],
            "closure_mol": out["closure_mol"], "strict_stage_states": len(out["stages"]),
            "wall_time_seconds": time.perf_counter() - t0}


def _bracket(delta: float) -> tuple:
    return ("fitted", delta), ("delta_minus", delta - DELTA_BRACKET), ("delta_plus", delta + DELTA_BRACKET)


def _slope(rows):
    shan = sorted((r for r in rows if r["na_feed_mol_l"] > 0), key=lambda r: r["na_feed_mol_l"])
    return lx._slope([math.log10(r["na_feed_mol_l"]) for r in shan], [math.log10(r["model_d_li"]) for r in shan])


def _shan_worst(rows):
    if any(r["status"] != "evaluated" for r in rows):
        return None
    return max(abs(r["log10_model_over_observed_d_li"]) for r in rows)


def shan_subrange(low: float, high: float, grid_points: int = 6) -> dict:
    """ln K_ex in [low, high] where Shan's worst miss is <= 0.30, delta refit by the pH rule at each ln K_ex."""
    shan = [c for c in lx.contacts() if c.dataset == "shan2025_figure3"]

    def worst(kex):
        d = fit_delta(kex)
        rows = [_row(c, "interior", "fitted", kex, d["value"]) for c in shan]
        value = _shan_worst(rows)
        print(f"  Shan at ln K_ex {kex:.6f}, delta {d['value']:.6f}: worst {value}", flush=True)
        return value, d

    grid = [low + (high - low) * i / (grid_points - 1) for i in range(grid_points)]
    points = []
    for kex in grid:
        value, d = worst(kex)
        points.append({"ln_k_ex": kex, "delta": d["value"], "delta_starts_agree": d["agree"], "shan_worst": value})
    crossings = []
    for a, b in zip(points, points[1:]):
        if None not in (a["shan_worst"], b["shan_worst"]) and \
                (a["shan_worst"] - lx.TOLERANCE_DECADE) * (b["shan_worst"] - lx.TOLERANCE_DECADE) < 0:
            crossings.append(brentq(lambda k: worst(k)[0] - lx.TOLERANCE_DECADE, a["ln_k_ex"], b["ln_k_ex"],
                                    xtol=1e-5))
    return {"rule": "delta refit by the pH rule at each ln K_ex; grid then brentq (xtol 1e-5) on each sign change",
            "grid": points, "crossings_ln_k_ex": crossings,
            "passes_at_grid": [p["ln_k_ex"] for p in points
                               if p["shan_worst"] is not None and p["shan_worst"] <= lx.TOLERANCE_DECADE]}


def _end_summary(kex: float, delta: float, rows: list[dict]) -> dict:
    table2, fig2 = _table2_ph()
    at = {r["identity"]: r for r in rows if r["delta_case"] == "fitted"}
    fit_rows = [r for r in at.values() if r.get("role") == "fit"]
    ph = [{"identity": c.identity, "observed_ph": fig2[c.identity], "model_ph": at[c.identity]["model_equilibrium_ph"],
           "model_minus_observed": at[c.identity]["model_equilibrium_ph"] - fig2[c.identity]} for c in table2]
    shan = {case: [r for r in rows if r["dataset"] == "shan2025_figure3" and r["delta_case"] == case]
            for case, _ in _bracket(delta)}
    summary = {
        "ln_k_ex": kex, "delta": delta, "ln_k": ln_k(kex, delta), "zhang2020_figure2_ph": ph,
        "ph_max_abs_residual": max(abs(p["model_minus_observed"]) for p in ph),
        "ph_within_0.5": all(abs(p["model_minus_observed"]) <= PH_TOLERANCE for p in ph),
        "zhang2020_fit_rms_log10": math.sqrt(sum(r["log10_model_over_observed_d_li"] ** 2 for r in fit_rows)
                                             / len(fit_rows)),
        "zhang2020_fit_worst_abs_log10": max(abs(r["log10_model_over_observed_d_li"]) for r in fit_rows),
        "shan": {case: {"delta": v[0]["delta"], "worst_abs_log10_d": _shan_worst(v),
                        "slope_log10_d_per_log10_na": _slope(v) if _shan_worst(v) is not None else None,
                        "residuals": {r["identity"]: r.get("log10_model_over_observed_d_li") for r in v},
                        "model_ph": [r.get("model_equilibrium_ph") for r in v],
                        "hbta_lost_percent": [r.get("hbta_lost_percent") for r in v]}
                 for case, v in shan.items()},
        "zhang2020_figure7_worst_abs_log10": max(abs(r["log10_model_over_observed_d_li"]) for r in at.values()
                                                 if r["dataset"] == "zhang2020_figure7"),
        "cha2025": [{k: r.get(k) for k in ("delta_case", "delta", "hbta_lost_mg_l", "hbta_lost_percent",
                                            "model_li_extraction_percent", "model_equilibrium_ph", "status")}
                    for r in rows if r["dataset"] == "cha2025"],
    }
    worst = summary["shan"]["fitted"]["worst_abs_log10_d"]
    summary["shan"]["fitted"]["passes"] = worst is not None and worst <= lx.TOLERANCE_DECADE
    return summary


def _contacts_unit(end: str, kex: float, delta: float) -> dict:
    """Single contacts at one end (all data at its fitted delta; Shan and Cha across the bracket) and the Zhang 2021
    cascades at R 70 % and 66.54 %."""
    base, rows, stage_rows, cascades = lx.contacts(), [], [], []
    for case, d in _bracket(delta):
        chosen = base if case == "fitted" else [c for c in base if c.dataset == "shan2025_figure3"]
        rows += [_row(c, end, case, kex, d) for c in chosen + [CHA]]
    for fraction, label in ((0.70, "zhang2021_r_0.70"), (lx.R_NOMINAL_70, "zhang2021_r_0.6654")):
        row = _cascade(f"{end}_{label}", label, ln_k(kex, delta), stage_rows, extractant=0.2,
                       saponification_fraction=fraction, feed_li=907.0 / lx.LI_G_MOL / 1000.0,
                       feed_na=5.21e4 / lx.NA_G_MOL / 1000.0, feed_acidity=acidity_at_ph(lx.FEED_PH))
        if row["status"] == "evaluated":
            row["raffinate_li_mg_l"] = row["raffinate_li_mol_l"] * lx.LI_G_MOL * 1000.0
            row["log10_model_over_observed_raffinate_li"] = math.log10(row["raffinate_li_mg_l"]
                                                                       / ZHANG2021_RAFFINATE_LI_MG_L)
        cascades.append({"end": end, "delta_case": "fitted", "feed": label, **row})
    return {"rows": rows, "cascades": cascades, "stage_rows": stage_rows}


def _ms2_feeds() -> dict:
    softened = softening()
    return {"ms2_divalents_omitted": (mc.MS2_LI, mc.MS2_NA),
            "ms2_softened": (softened.li_feed_mol_l, softened.na_feed_mol_l)}


def _ms2_unit(end: str, case: str, kex: float, d: float) -> dict:
    """MS-2 one stage and three countercurrent stages at 0.1-0.4 M, O/A 1, R 70 %, feed pH 5.77, both feeds."""
    ms2, stage_rows = [], []
    for feed, (li, na) in _ms2_feeds().items():
        for c in lx.EXTRACTANT_MS2:
            head = {"end": end, "delta_case": case, "delta": d, "ln_k_ex": kex, "feed": feed, "feed_li_mol_l": li,
                    "feed_na_mol_l": na, "extractant_mol_l": c}
            contact = lx.Contact(feed, "prediction", f"{feed}_{c}_{end}_{case}", 1.0, c, MS2_SAPONIFICATION, li, na,
                                 mc.MS2_PH, None, "")
            one = _row(contact, end, case, kex, d)
            ms2.append({**head, "stages": 1, **{k: one.get(k) for k in (
                "identity", "status", "error", "model_li_extraction_percent", "model_d_li", "model_d_na",
                "na_loaded_organic_mol_l", "na_loaded_over_feed_na", "hbta_lost_mg_l", "hbta_lost_percent",
                "model_equilibrium_ph", "balance_inf_norm", "reaction_affinity_inf_norm",
                "kkt_stationarity_inf_norm")}})
            three = _cascade(f"{feed}_{c}_{end}_{case}", f"{feed}_{c}", ln_k(kex, d), stage_rows, extractant=c,
                             saponification_fraction=MS2_SAPONIFICATION, feed_li=li, feed_na=na,
                             feed_acidity=acidity_at_ph(mc.MS2_PH))
            ms2.append({**head, "stages": 3, "identity": three["case"],
                        **{("model_li_extraction_percent" if k == "li_extraction_percent" else k): v
                           for k, v in three.items() if k != "case"}})
            print(f"MS-2 {feed} {c} {end} {case}: 1 stage {one.get('model_li_extraction_percent')} "
                  f"3 stages {three.get('li_extraction_percent')}", flush=True)
    return {"ms2": ms2, "stage_rows": stage_rows}


def predict(args) -> None:
    """One unit per run (one lock-held job), in order: contacts and Zhang 2021 at each end, the Shan sub-range, MS-2
    at each end and delta case; then the retained CSVs and summary. Exits with MORE while units remain."""
    engine = lx.installed_engine()
    started = time.perf_counter()
    fitted = json.loads((RESULTS / "set_c_fit.json").read_text())
    ends = {"low": (fitted["low"]["ln_k_ex"], fitted["low"]["delta"]),
            "high": (fitted["high"]["r_0.70"]["ln_k_ex"], fitted["high"]["r_0.70"]["delta"])}
    units = {f"contacts_{end}": (lambda end=end: _contacts_unit(end, *ends[end])) for end in ends}
    units["shan_subrange"] = lambda: shan_subrange(ends["low"][0], ends["high"][0])
    for end, (kex, delta) in ends.items():
        for case, d in _bracket(delta):
            units[f"ms2_{end}_{case}"] = lambda end=end, case=case, kex=kex, d=d: _ms2_unit(end, case, kex, d)
    progress = _progress()
    done = progress.setdefault("predict", {})
    pending = [name for name in units if name not in done]
    if pending:
        print(f"predict unit {pending[0]}", flush=True)
        done[pending[0]] = {**units[pending[0]](), "strict_solves": _SOLVES[0],
                            "wall_time_seconds": time.perf_counter() - started}
        _save(progress)
        raise SystemExit(MORE)

    rows = [r for end in ends for r in done[f"contacts_{end}"]["rows"]]
    cascades = [c for end in ends for c in done[f"contacts_{end}"]["cascades"]]
    ms2 = [r for name in units if name.startswith("ms2_") for r in done[name]["ms2"]]
    stage_rows = [s for name in units for s in done[name].get("stage_rows", [])]
    lx._write_csv(RESULTS / "set_c_contacts.csv", rows)
    lx._write_csv(RESULTS / "set_c_ms2_predictions.csv", ms2)
    lx._write_csv(RESULTS / "set_c_cascade_stages.csv", stage_rows)
    summary_ends = {end: _end_summary(kex, delta, [r for r in rows if r["end"] == end])
                    for end, (kex, delta) in ends.items()}
    three = [r for r in ms2 if r["stages"] == 3]
    evaluated_cascades = [c for c in cascades + three if c["status"] == "evaluated"]
    singles = rows + [r for r in ms2 if r["stages"] == 1]
    feeds = _ms2_feeds()
    summary = {
        "engine": engine, "issue": "tannerpolley/Lithium_Extraction#166",
        "plan": "issue #166 stage 2 plan comment, fixed before fitting", "tolerance_decade": lx.TOLERANCE_DECADE,
        "ph_tolerance": PH_TOLERANCE, "delta_bracket": DELTA_BRACKET, "ends": summary_ends,
        "shan_passes_both_ends": all(e["shan"]["fitted"]["passes"] for e in summary_ends.values()),
        "shan_subrange": {k: v for k, v in done["shan_subrange"].items() if k != "stage_rows"},
        "zhang2021_cascades": cascades,
        "ms2_divalents_omitted_feed": {"li_mol_l": feeds["ms2_divalents_omitted"][0],
                                       "na_mol_l": feeds["ms2_divalents_omitted"][1], "ph": mc.MS2_PH},
        "ms2_softened_feed": {"li_mol_l": feeds["ms2_softened"][0], "na_mol_l": feeds["ms2_softened"][1],
                              "source": "lithium_extraction/common/pretreatment.py"},
        "cha2025_observed": {"hbta_lost_mg_l": CHA_HBTA_LOSS_MG_L, "hbta_lost_percent": CHA_HBTA_LOSS_PERCENT},
        "strict_states": {
            "single_contacts": len(singles),
            "single_contacts_evaluated": sum(r["status"] == "evaluated" for r in singles),
            "cascades": len(cascades) + len(three), "cascades_evaluated": len(evaluated_cascades),
            "cascade_stages": sum(c["strict_stage_states"] for c in evaluated_cascades),
            "failed": [{"identity": r.get("identity") or r.get("case"), "error": r.get("error")}
                       for r in singles + cascades + three if r["status"] != "evaluated"],
            "max_single_norms": {k: max(r[k] for r in singles if r["status"] == "evaluated" and r.get(k) is not None)
                                 for k in ("balance_inf_norm", "reaction_affinity_inf_norm",
                                           "kkt_stationarity_inf_norm")},
            "max_abs_cascade_closure_mol": {x: max(abs(c["closure_mol"][x]) for c in evaluated_cascades)
                                            for x in ("li", "na", "h", "bta")},
            "cascade_sweeps": [min(c["sweeps"] for c in evaluated_cascades),
                               max(c["sweeps"] for c in evaluated_cascades)]},
        "units": {name: {"strict_solves": u["strict_solves"], "wall_time_seconds": u["wall_time_seconds"]}
                  for name, u in done.items()},
    }
    (RESULTS / "set_c_summary.json").write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n")
    del progress["predict"]
    if set(progress) <= {"cascade_start", "fit"} and not progress.get("fit"):
        PROGRESS.unlink()
    else:
        _save(progress)


def _observed_d(point: dict) -> float:
    e = float(point["li_extraction_percent"])
    return e / (100.0 - e)  # O/A 1


def _scored(row: dict, observed_ph: float | None = None) -> dict:
    """|log10 D_model/D_obs| <= 0.30; equilibrium pH within +/- 0.5 where an observed pH is given."""
    if row["status"] == "evaluated":
        row["d_pass"] = abs(row["log10_model_over_observed_d_li"]) <= lx.TOLERANCE_DECADE
        if observed_ph is not None:
            row.update(observed_ph=observed_ph, ph_error=row["model_equilibrium_ph"] - observed_ph)
            row["ph_pass"] = abs(row["ph_error"]) <= PH_TOLERANCE
    return row


def _direct(contact: lx.Contact, state: mc.State, route: str, end: str, case: str, kex: float, delta: float,
            target_ph: float | None = None) -> dict:
    """A state not built by Contact.state(): an imposed equilibrium pH (mc.solve_at_ph) or saponification above 1.
    A state that cannot be built (ValueError) or has no strict solution (RuntimeError) is recorded with its reason."""
    head = {"end": end, "delta_case": case, "delta": delta, "route": route, "dataset": contact.dataset,
            "role": contact.role, "identity": contact.identity, "imposed_ph": target_ph}
    try:
        result = (mc.solve(state, "C", ln_k(kex, delta)) if target_ph is None
                  else mc.solve_at_ph(state, "C", ln_k(kex, delta), target_ph))
    except (RuntimeError, ValueError) as error:
        return {**head, "status": type(error).__name__, "error": str(error)[:500]}
    return {**head, **lx.record(contact, result, kex, ln_k(kex, delta)["R2"]),
            "feed_ph": contact.feed_ph if target_ph is None else None,  # an imposed pH replaces the feed's acid/base
            "net_base_mol": result.get("net_base_mol"), "ph_unreachable": result.get("ph_unreachable")}


def holdout_scores(args) -> None:
    """#210 step 5 (Round 2 plan, P2): the frozen set C band ends on Seo 2025 Fig. 2 and Ni 2024 Fig. 10(A).

    Nothing is fitted and no retained file is rewritten. Shan 2025 Fig. 3 is first reproduced from
    set_c_contacts.csv through _row at both ends and every delta case, within 1e-9 relative. Every holdout runs at
    each end and across the delta bracket.
    - Seo, two routes on equal footing: (i) forward, inferred saponification at the Cha feed pH 8.52, D_Li and pH
      scored; (ii) each digitized equilibrium pH imposed with mc.solve_at_ph, D_Li scored. Zhang 2020 Table 2 on
      route (ii) is a route-sensitivity row, reported only.
    - Ni, forward route at feed pH 7 and 10.68 (unbuffered band): 40-100 % primary; at 110 and 120 % a
      sensitivity, the organic fully saponified (na_org = c) and (R - 1) c entering as free NaOH, which adds Na+.
    Every holdouts.csv row not scored is listed with its reason.
    """
    engine = lx.installed_engine()
    ends = json.loads((RESULTS / "set_c_summary.json").read_text())["ends"]
    retained = {(r["end"], r["delta_case"], r["identity"]): r for r in lx._csv(RESULTS / "set_c_contacts.csv")}
    for end, e in ends.items():
        _GUESS.clear()  # predict() ran each end in its own process
        for case, d in _bracket(e["delta"]):
            for c in (c for c in lx.contacts() if c.dataset == "shan2025_figure3"):
                row, old = _row(c, end, case, e["ln_k_ex"], d), retained[(end, case, c.identity)]
                for k in ("model_d_li", "model_equilibrium_ph"):
                    if row.get(k) is None or abs(row[k] / float(old[k]) - 1.0) > 1e-9:
                        raise SystemExit(f"Shan reproduction fails at {end} {case} {c.identity} {k}: {row.get(k)} "
                                         f"against {old[k]}; nothing is scored")
    print("Shan Fig. 3 reproduced at both ends within 1e-9 relative", flush=True)
    table2, fig2 = _table2_ph()
    seo = [(float(p["equilibrium_ph"]), lx.Contact(
        "seo2025_figure2", "holdout", p["point_id"], 1.0, 0.4, float(p["saponification_degree_percent_inferred"]) / 100,
        1.85 / lx.LI_G_MOL, 31.2 / lx.NA_G_MOL, lx.FEED_PH, _observed_d(p), "E / (100 - E) at O/A 1"))
        for p in lx._csv(lx.DATA / "digitization/seo2025_figure2/points.csv")]
    feed = {r["component"]: float(r["content_g_l"])
            for r in lx._csv(lx.DATA / "tables/niSustainableEfficientRecovery2024_table1.csv")}
    ni = [lx.Contact("ni2024_figure10", "primary" if float(p["saponification_degree_percent"]) <= 100 else "sensitivity",
                     f"{p['point_id']}_feed_ph_{ph}", 1.0, 0.1, float(p["saponification_degree_percent"]) / 100,
                     feed["Li+"] / lx.LI_G_MOL, feed["Na+"] / lx.NA_G_MOL, ph, _observed_d(p), "E / (100 - E) at O/A 1")
          for p in lx._csv(lx.DATA / "digitization/ni2024_figure10/points.csv") for ph in NI_FEED_PH]
    rows = []
    for end, e in ends.items():
        kex = e["ln_k_ex"]
        for case, d in _bracket(e["delta"]):
            for ph, c in seo:
                rows.append(_scored({**_row(c, end, case, kex, d), "route": "forward"}, ph))
                rows.append(_scored({**_direct(c, replace(c.state(), na=c.na), "imposed_ph", end, case, kex, d, ph),
                                     "imposed_ph_outside_calibration": not min(fig2.values()) <= ph <= max(fig2.values())}))
            for c in ni:
                if c.saponification <= 1.0:
                    rows.append(_scored({**_row(c, end, case, kex, d), "route": "forward"}))
                else:  # the organic fully saponified; (R - 1) c enters as free NaOH, adding Na+ and OH-
                    s = replace(c, saponification=1.0).state()
                    s = replace(s, base=s.base + (c.saponification - 1.0) * c.extractant)
                    rows.append(_scored(_direct(c, s, "forward_free_naoh", end, case, kex, d)))
            print(f"{end} {case}: Seo and Ni done", flush=True)
        for c in table2:  # route sensitivity at calibration conditions, fitted delta: reported, not scored
            row = _direct(c, replace(c.state(), na=c.na), "imposed_ph_route_check", end, "fitted", kex, e["delta"],
                          fig2[c.identity])
            forward = float(retained[(end, "fitted", c.identity)]["model_d_li"])  # retained forward-route value
            rows.append({**row, "log10_imposed_over_forward_d_li":
                         math.log10(row["model_d_li"] / forward) if "model_d_li" in row else None})
    for h in lx._csv(lx.DATA / "holdouts.csv"):
        if not (h["role"] == "holdout" and h["campaign"] in ASSUMED):
            rows.append({"route": "not_scored", "dataset": h["campaign"], "role": h["role"], "identity": h["locator"],
                         "status": "not scored", "error": NOT_SCORED.get(h["campaign"], h["reason"])})
    rows.append({"route": "not_scored", "dataset": "ni2024_figure10", "role": "holdout", "status": "not scored",
                 "identity": "Ni et al. 2024, Figure 10(A), NH3-NH4Cl buffered", "error": NOT_SCORED["buffer"]})
    lx._write_csv(RESULTS / "set_c_holdout_scores.csv", [
        {**r, "assumptions": ASSUMED.get(r["dataset"], "") if "delta_case" in r else "",
         "engine_version": engine["version"], "engine_sha256": engine["sha256"]} for r in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("fit", "predict", "holdouts"))
    parser.add_argument("--part", choices=("low", "high_r_0.70"), help="fit step only")
    args = parser.parse_args()
    {"fit": fit, "predict": predict, "holdouts": holdout_scores}[args.step](args)


if __name__ == "__main__":
    main()
