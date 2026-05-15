from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from pyomo.environ import value


def locate_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        src_dir = candidate / "src" / "prommis"
        if src_dir.exists():
            return candidate
    raise RuntimeError("Could not locate repo root containing src/prommis.")


REPO_ROOT = locate_repo_root()
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prommis.solvent_extraction.pcsaft_integration.sx_li_na_cl_pcsaft_poc import (  # noqa: E402
    _ensure_pcsaft_pse_import,
    build_li_na_cl_pcsaft_poc_model,
    solve_li_na_cl_pcsaft_poc,
    summarize_results,
)


OUTPUT_DIR = Path(__file__).resolve().parent
ASSET_DIR = OUTPUT_DIR / "assets"
ASSET_DIR.mkdir(parents=True, exist_ok=True)

_, run_lle_stage, _, _, StageInputs = _ensure_pcsaft_pse_import()

palette = {
    "deep_green": "#0B6E4F",
    "burnt_orange": "#C84C09",
    "slate": "#2F4858",
    "sand": "#EAD8B1",
    "mist": "#F4F1EA",
}

plt.rcParams["figure.figsize"] = (8, 4.5)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["figure.dpi"] = 140

nominal_inputs = {
    "li_mg_L": 60.0,
    "na_mg_L": 10900.0,
    "o_to_a_ratio": 1.0,
    "top_mol_L": 0.015,
}

mw_li = 6.94e-3
mw_na = 22.98976928e-3
mw_cl = 35.45e-3


def neutral_cl_mg_L(li_mg_L: float, na_mg_L: float) -> float:
    li_mol_L = li_mg_L * 1e-6 / mw_li
    na_mol_L = na_mg_L * 1e-6 / mw_na
    return (li_mol_L + na_mol_L) * mw_cl * 1e6


def render_workflow() -> None:
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")

    boxes = [
        (0.4, 0.7, 2.3, 1.5, palette["slate"], "1. Direct PC-SAFT\nDetailed stage LLE"),
        (3.3, 0.7, 2.3, 1.5, palette["burnt_orange"], "2. pcsaft-pse\nBridge and surrogate outputs"),
        (6.2, 0.7, 2.3, 1.5, palette["deep_green"], "3. PrOMMiS\nMSContactor-based stage model"),
        (9.1, 0.7, 2.3, 1.5, "#7A5C3E", "4. Process KPIs\nExtraction and outlet streams"),
    ]

    for x, y, w, h, color, label in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.12",
            linewidth=1.5,
            edgecolor=color,
            facecolor=palette["mist"],
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=10,
            color=color,
            fontweight="bold",
        )

    for start, end, label in [
        (2.7, 3.3, "calibration"),
        (5.6, 6.2, "surrogate signal"),
        (8.5, 9.1, "process prediction"),
    ]:
        ax.annotate(
            "",
            xy=(end, 1.45),
            xytext=(start, 1.45),
            arrowprops=dict(arrowstyle="-|>", lw=2, color="#555555"),
        )
        ax.text(
            (start + end) / 2,
            1.72,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#555555",
        )

    ax.set_title(
        "PC-SAFT to pcsaft-pse to PrOMMiS Integration Architecture",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "workflow_overview.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


def main() -> None:
    truth = run_lle_stage(StageInputs(**nominal_inputs))

    model, calibration = build_li_na_cl_pcsaft_poc_model()
    stage = model.fs.solex.mscontactor.elements.first()
    default_li_constraint = model.fs.solex.mass_transfer_constraint[0, stage, "aqueous", "organic", "Li"]
    replacement_li_constraint = model.fs.li_mass_transfer_constraint[0]

    solved_model, _, results = solve_li_na_cl_pcsaft_poc(tee=False)
    summary = summarize_results(solved_model)
    solved_stage = solved_model.fs.solex.mscontactor.elements.first()

    aq_in = solved_model.fs.solex.mscontactor.aqueous_inlet_state[0]
    aq_out = solved_model.fs.solex.mscontactor.aqueous[0, solved_stage]
    og_in = solved_model.fs.solex.mscontactor.organic_inlet_state[0]
    og_out = solved_model.fs.solex.mscontactor.organic[0, solved_stage]

    def state_rows(label: str, state_block, components: list[str]) -> list[dict[str, float | str]]:
        return [
            {
                "Stream": label,
                "Component": comp,
                "Concentration (mg/L)": value(state_block.conc_mass_comp[comp]),
            }
            for comp in components
        ]

    stream_table = pd.DataFrame(
        state_rows("Aqueous In", aq_in, ["Li", "Na", "Cl"])
        + state_rows("Aqueous Out", aq_out, ["Li", "Na", "Cl"])
        + state_rows("Organic In", og_in, ["TOP", "Hexane", "Li", "Na", "Cl"])
        + state_rows("Organic Out", og_out, ["TOP", "Hexane", "Li", "Na", "Cl"])
    )

    focus = stream_table[stream_table["Component"].isin(["Li", "Na"])].copy()
    pivot = (
        focus.pivot(index="Stream", columns="Component", values="Concentration (mg/L)")
        .reindex(["Aqueous In", "Aqueous Out", "Organic In", "Organic Out"])
    )
    kpi = pd.DataFrame(
        {
            "Metric": ["Li extraction (%)", "Li distribution ratio", "Li partition proxy"],
            "Value": [summary["eta_li_pct"], summary["d_li"], summary["k_li"]],
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    pivot[["Li", "Na"]].plot(
        kind="bar",
        ax=axes[0],
        color=[palette["deep_green"], palette["burnt_orange"]],
    )
    axes[0].set_title("Species Concentration by Stream")
    axes[0].set_ylabel("Concentration (mg/L)")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend(frameon=False)

    axes[1].barh(
        kpi["Metric"],
        kpi["Value"],
        color=[palette["deep_green"], palette["slate"], palette["burnt_orange"]],
    )
    axes[1].set_title("Key Stage Performance Indicators")
    axes[1].set_xlabel("Value")
    for idx, val in enumerate(kpi["Value"]):
        axes[1].text(val, idx, f"  {val:,.4f}", va="center", ha="left")

    fig.tight_layout()
    fig.savefig(ASSET_DIR / "stage_outcome.png", bbox_inches="tight", dpi=180)
    plt.close(fig)

    comparison = pd.DataFrame(
        {
            "Quantity": [
                "Li extraction efficiency",
                "Li distribution ratio",
                "Li partition coefficient",
            ],
            "Direct PC-SAFT stage": [
                truth.eta_li_pct,
                truth.D_li,
                truth.kappa_li,
            ],
            "Integrated PrOMMiS model": [
                summary["eta_li_pct"],
                summary["d_li"],
                summary["k_li"],
            ],
        }
    )
    comparison["Absolute difference"] = (
        comparison["Integrated PrOMMiS model"] - comparison["Direct PC-SAFT stage"]
    ).abs()

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    y = range(len(comparison))
    ax.barh(
        [i + 0.18 for i in y],
        comparison["Direct PC-SAFT stage"],
        height=0.34,
        color=palette["slate"],
        label="Direct PC-SAFT",
    )
    ax.barh(
        [i - 0.18 for i in y],
        comparison["Integrated PrOMMiS model"],
        height=0.34,
        color=palette["deep_green"],
        label="Integrated PrOMMiS",
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(comparison["Quantity"])
    ax.set_title("Direct Chemistry and Process Model Agreement")
    ax.set_xlabel("Value")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "agreement.png", bbox_inches="tight", dpi=180)
    plt.close(fig)

    ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    sensitivity_records: list[dict[str, float]] = []
    for oa in ratios:
        out = run_lle_stage(
            StageInputs(
                li_mg_L=nominal_inputs["li_mg_L"],
                na_mg_L=nominal_inputs["na_mg_L"],
                o_to_a_ratio=float(oa),
                top_mol_L=nominal_inputs["top_mol_L"],
            )
        )
        sensitivity_records.append(
            {
                "O/A ratio": oa,
                "Li extraction (%)": out.eta_li_pct,
                "Na extraction (%)": out.eta_na_pct,
                "D_Li": out.D_li,
            }
        )

    sensitivity = pd.DataFrame(sensitivity_records)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(
        sensitivity["O/A ratio"],
        sensitivity["Li extraction (%)"],
        marker="o",
        linewidth=2.0,
        color=palette["deep_green"],
    )
    axes[0].set_title("Lithium Extraction vs O/A Ratio")
    axes[0].set_xlabel("Organic-to-aqueous ratio")
    axes[0].set_ylabel("Li extraction (%)")

    axes[1].plot(
        sensitivity["O/A ratio"],
        sensitivity["D_Li"],
        marker="o",
        linewidth=2.0,
        color=palette["burnt_orange"],
    )
    axes[1].set_title("Lithium Distribution Ratio vs O/A Ratio")
    axes[1].set_xlabel("Organic-to-aqueous ratio")
    axes[1].set_ylabel("D_Li")

    fig.suptitle("Direct PC-SAFT Stage Sensitivity", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "sensitivity.png", bbox_inches="tight", dpi=180)
    plt.close(fig)

    render_workflow()

    feed_basis = {
        **nominal_inputs,
        "cl_mg_L": neutral_cl_mg_L(nominal_inputs["li_mg_L"], nominal_inputs["na_mg_L"]),
    }
    payload = {
        "feed_basis": feed_basis,
        "truth": {
            "eta_li_pct": truth.eta_li_pct,
            "eta_na_pct": truth.eta_na_pct,
            "D_li": truth.D_li,
            "kappa_li": truth.kappa_li,
            "kappa_na": truth.kappa_na,
            "kappa_cl": truth.kappa_cl,
            "li_na_selectivity": truth.li_na_selectivity,
        },
        "calibration": calibration,
        "constraint_status": {
            "default_li_mass_transfer_active": bool(default_li_constraint.active),
            "replacement_li_mass_transfer_active": bool(replacement_li_constraint.active),
            "k_li_from_eta_active": bool(model.fs.k_li_from_eta_constraint.active),
            "replacement_li_expression": str(replacement_li_constraint.expr),
        },
        "summary": summary,
        "comparison": comparison.to_dict(orient="records"),
        "sensitivity": sensitivity_records,
        "solver_termination": str(results.solver.termination_condition),
        "stream_focus": pivot.reset_index().to_dict(orient="records"),
    }

    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    stream_table.to_csv(OUTPUT_DIR / "stream_table.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "comparison.csv", index=False)
    sensitivity.to_csv(OUTPUT_DIR / "sensitivity.csv", index=False)

    print(f"Wrote assets to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
