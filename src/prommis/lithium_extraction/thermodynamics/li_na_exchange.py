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

from prommis.lithium_extraction.thermodynamics.reaction_networks import (
    AQUEOUS_PACKET,
    load_constants,
)
from prommis.lithium_extraction.thermodynamics import mechanism_candidates as mc
from prommis.lithium_extraction.thermodynamics.apparent_reactive_lle import (
    shan_condition,
)
from prommis.lithium_extraction.extraction.countercurrent import (
    acidity_at_ph,
    stage_state,
)
from prommis.lithium_extraction.process.flowsheet import installed_engine

from prommis.lithium_extraction.paths import REPO_ROOT

ROOT = REPO_ROOT
RESULTS = REPO_ROOT / "analyses/prototypes/hbta_topo_ms2_prototype/results"
DATA = ROOT / "data/reference/extraction_models/hbta_topo_li_na"
LN_K_R2 = load_constants((mc.RESULTS / "mechanism_candidates.json"))["summary"]["A"][
    "ln_k_298"
]["R2"]
LI_G_MOL, NA_G_MOL = 6.941, 22.99  # tables/conditions.csv molar conversions
TOLERANCE_DECADE = 0.30
E_LIMIT_PERCENT = 99.5  # D from E is not scored at or above this extraction
FEED_PH = (
    8.52  # Zhang 2020 Fig. 2 / Table 3 feed; not reported for Figs. 6, 7 or Zhang 2021
)
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
        return stage_state(
            self.identity,
            organic_to_aqueous=self.organic_to_aqueous,
            extractant=self.extractant,
            li=self.li,
            na=self.na,
            acidity=acidity_at_ph(self.feed_ph),
            na_org=self.saponification * self.extractant,
        )


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def contacts() -> list[Contact]:
    """Fit and single-contact holdout rows at their base conditions."""
    rows = []
    for r in _csv(
        DATA / "digitization/zhang2020_figure6/points.csv"
    ):  # O/A 1, Na 1.370 M, R 70 % nominal
        li0, li_org = float(r["li_initial_aq_mol_l"]), float(r["li_org_mol_l"])
        rows.append(
            Contact(
                "zhang2020_figure6",
                "fit",
                r["point_id"],
                1.0,
                0.4,
                R_NOMINAL_70,
                li0,
                1.370,
                FEED_PH,
                li_org / (li0 - li_org),
                "organic Li / (initial - organic Li)",
            )
        )
    for r in _csv(
        DATA / "tables/zhangLithiumRecoveryEffluent2020_table2.csv"
    ):  # Table 1 feed, O/A 1 inferred
        li_org = float(r["li_organic_mol_l"])
        rows.append(
            Contact(
                "zhang2020_table2",
                "fit",
                f"zhang2020_table2_rs_{r['saponification_degree_cal_percent']}",
                1.0,
                0.4,
                float(r["na_saponified_organic_mol_l"]) / 0.4,
                0.267,
                1.36,
                FEED_PH,
                li_org / (0.267 - li_org),
                "organic Li / (0.267 - organic Li)",
            )
        )
    for r in _csv(
        DATA / "digitization/zhang2020_figure7/points.csv"
    ):  # E only: the Li-organic series is aqueous Li
        oa, e = float(r["organic_to_aqueous_numeric"]), float(
            r["li_extraction_percent"]
        )
        if e >= E_LIMIT_PERCENT:
            raise ValueError(
                f"{r['point_id']}: E {e} % is above the {E_LIMIT_PERCENT} % scoring limit"
            )
        rows.append(
            Contact(
                "zhang2020_figure7",
                "holdout_same_paper",
                r["point_id"],
                oa,
                0.4,
                R_NOMINAL_70,
                0.267,
                1.370,
                FEED_PH,
                e / (100.0 - e) / oa,
                "E / (100 - E) / (O/A)",
            )
        )
    for r in _csv(DATA / "digitization/shan2025_figure3/points.csv"):
        source = shan_condition(
            float(r["na_mg_l"]), float(r["li_extraction_percent"]), "holdout"
        )
        rows.append(
            Contact(
                "shan2025_figure3",
                "holdout_independent",
                r["point_id"],
                source.organic_volume_l / source.aqueous_volume_l,
                source.hbta_mol_l_organic,
                source.saponification_fraction,
                source.li_mol_l_aqueous,
                source.na_mol_l_aqueous,
                source.initial_ph,
                float(r["li_distribution_ratio"]),
                "printed D_Li",
            )
        )
    return rows


def sensitivity_contacts(base: list[Contact]) -> list[Contact]:
    """Feed pH +/- 1 for Zhang 2020 Figs. 6 and 7 and Shan; Shan's 0 Na control at the methods' 60 mg/L."""
    rows = []
    for c in base:
        if c.dataset in ("zhang2020_figure6", "zhang2020_figure7", "shan2025_figure3"):
            for shift in (-1.0, 1.0):
                rows.append(
                    replace(
                        c,
                        identity=f"{c.identity}_ph_{c.feed_ph + shift:.2f}",
                        feed_ph=c.feed_ph + shift,
                        sensitivity=f"feed pH {shift:+.0f}",
                    )
                )
        if c.identity == "shan_fig3_na_0":
            rows.append(
                replace(
                    c,
                    identity="shan_fig3_na_60",
                    na=0.060 / 22.98976928,
                    sensitivity="Na 60 mg/L",
                )
            )
    return rows


def ln_k(ln_k_ex: float, ln_k_r2: float = LN_K_R2) -> dict[str, float]:
    return {"R1": ln_k_ex + ln_k_r2, "R2": ln_k_r2}


_GUESS: dict[str, dict] = {}  # last strict composition per contact: a warm start only


def solve(contact: Contact, constants: dict[str, float]) -> dict:
    result = mc.solve(
        contact.state(), "A", constants, guess=_GUESS.get(contact.identity)
    )
    _GUESS[contact.identity] = result["phase_amounts_mol"]
    return result


def record(
    contact: Contact, result: dict, ln_k_ex: float, ln_k_r2: float = LN_K_R2
) -> dict:
    residual = (
        None
        if contact.observed_d is None
        else math.log10(result["li_distribution"] / contact.observed_d)
    )
    return {
        "dataset": contact.dataset,
        "role": contact.role,
        "identity": contact.identity,
        "sensitivity": contact.sensitivity,
        "organic_to_aqueous": contact.organic_to_aqueous,
        "extractant_mol_l": contact.extractant,
        "saponification_fraction": contact.saponification,
        "li_feed_mol_l": contact.li,
        "na_feed_mol_l": contact.na,
        "feed_ph": contact.feed_ph,
        "ln_k_ex": ln_k_ex,
        "ln_k_r2": ln_k_r2,
        "observed_d_li": contact.observed_d,
        "observed_basis": contact.observed_basis,
        "model_d_li": result["li_distribution"],
        "log10_model_over_observed_d_li": residual,
        "model_d_na": result["na_distribution"],
        "model_li_extraction_percent": result["li_extraction_percent"],
        "model_equilibrium_ph": result["ph"],
        "active_reactions": " ".join(result["active_reactions"]),
        "status": result["status"],
        **{
            k: result[k]
            for k in (
                "balance_inf_norm",
                "charge_inf_norm",
                "absence_inf_norm",
                "reaction_affinity_inf_norm",
                "kkt_stationarity_inf_norm",
                "reduced_hessian_min_eigenvalue",
                "pressure_relative_inf_norm",
            )
        },
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
        return [
            math.log10(solve(c, ln_k(q[0]))["li_distribution"] / c.observed_d)
            for c in fit_rows
        ]

    starts = []
    for x0 in FIT_STARTS:
        t0 = time.perf_counter()
        s = least_squares(
            residual,
            [x0],
            bounds=FIT_BOUNDS,
            diff_step=1e-4,
            xtol=1e-10,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=200,
        )
        starts.append(
            {
                "start": x0,
                "ln_k_ex": float(s.x[0]),
                "cost": float(s.cost),
                "nfev": int(s.nfev),
                "status": int(s.status),
                "message": s.message,
                "wall_time_seconds": time.perf_counter() - t0,
            }
        )
        print(
            f"start {x0}: ln K_ex {s.x[0]:.10f} cost {s.cost:.8g} nfev {s.nfev} status {s.status}",
            flush=True,
        )
    best = min(starts, key=lambda s: s["cost"])
    spread = max(abs(s["ln_k_ex"] - best["ln_k_ex"]) for s in starts)
    document = {
        "aqueous_basis": AQUEOUS_PACKET,
        "engine": engine,
        "issue": "tannerpolley/Lithium_Extraction#166",
        "plan": "issue #166 recalibration plan comment (stage 1), fixed before fitting",
        "ln_k_ex": best["ln_k_ex"],
        "ln_k_r2_fixed": LN_K_R2,
        "ln_k_r1": best["ln_k_ex"] + LN_K_R2,
        "k_ex": math.exp(best["ln_k_ex"]),
        "log10_k_ex": best["ln_k_ex"] / math.log(10.0),
        "bounds": FIT_BOUNDS,
        "starts": starts,
        "start_spread_ln_k_ex": spread,
        "starts_agree": spread <= START_AGREEMENT,
        "fit_points": len(fit_rows),
        "strict_solves": solves,
        "wall_time_seconds": time.perf_counter() - started,
    }
    (RESULTS / "fit.json").write_text(
        json.dumps(document, indent=1, allow_nan=False) + "\n"
    )
    if not document["starts_agree"]:
        raise SystemExit(
            f"starts disagree on ln K_ex by {spread:g} (> {START_AGREEMENT}): goes to diagnosis"
        )


def predict(args) -> None:
    fitted = load_constants(RESULTS / "fit.json")
    if not fitted["starts_agree"]:
        raise SystemExit("the fit's starts disagree: no predictions until diagnosed")
    kex = fitted["ln_k_ex"]
    base = contacts()
    rows = [
        record(c, solve(c, ln_k(kex)), kex) for c in base + sensitivity_contacts(base)
    ]
    # Check of the plan's identifiability statement: fit-contact D_Li at ln K(R2) +/- 2 with ln K_ex held.
    for shift in (-2.0, 2.0):
        rows += [
            record(
                replace(c, sensitivity=f"ln K(R2) {shift:+.0f}"),
                solve(c, ln_k(kex, LN_K_R2 + shift)),
                kex,
                LN_K_R2 + shift,
            )
            for c in base
            if c.role == "fit"
        ]
    _write_csv(RESULTS / "contacts.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("fit", "predict"))
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    {"fit": fit, "predict": predict}[args.step](args)


if __name__ == "__main__":
    main()
