#####################################################################################################
# "PrOMMiS" was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# ("PrOMMiS") initiative, and is copyright (c) 2023-2024 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""IDAES/Pyomo costing layer for the HBTA/TOPO produced-water case study.

The Lithium_Extraction repository owns the source-regressed ePC-SAFT/HBTA-TOPO
transfer artifacts. This module is the downstream PrOMMiS-side costing consumer:
it reads the Lithium handoff table, builds an IDAES flowsheet block with formal
Pyomo expressions for the economic terms, and exports the scenario results.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pyomo.environ as pyo
from idaes.core import FlowsheetBlock

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_LITHIUM_REPO = Path(os.environ.get("LITHIUM_EXTRACTION_REPO", r"C:\Users\Tanner\Documents\git\Lithium_Extraction"))
DEFAULT_INPUT_CSV = (
    DEFAULT_LITHIUM_REPO
    / "data"
    / "reference"
    / "produced_water"
    / "hbta_topo_idaes_costing_input.csv"
)
OUT_CSV = THIS_DIR / "hbta_topo_idaes_costing_results.csv"
OUT_MD = THIS_DIR / "hbta_topo_idaes_costing_results.md"


@dataclass(frozen=True)
class CostingScenario:
    scenario: str
    feed_flow_m3_day: float
    li_feed_mg_L: float
    tds_mg_L: float
    li_recovery_pct: float
    li2co3_product_t_year: float
    gross_revenue_usd_year: float
    pretreatment_opex_usd_year: float
    extraction_opex_usd_year: float
    concentration_precip_opex_usd_year: float
    solvent_makeup_opex_usd_year: float
    installed_contactor_capex_usd: float
    annualized_capex_usd_year: float


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def load_costing_scenarios(path: str | Path = DEFAULT_INPUT_CSV) -> list[CostingScenario]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        CostingScenario(
            scenario=row["scenario"],
            feed_flow_m3_day=_float(row, "feed_flow_m3_day"),
            li_feed_mg_L=_float(row, "Li_feed_mg_L"),
            tds_mg_L=_float(row, "TDS_mg_L"),
            li_recovery_pct=_float(row, "Li_recovery_pct_used_for_costing"),
            li2co3_product_t_year=_float(row, "Li2CO3_product_t_year"),
            gross_revenue_usd_year=_float(row, "gross_revenue_usd_year"),
            pretreatment_opex_usd_year=_float(row, "pretreatment_opex_usd_year"),
            extraction_opex_usd_year=_float(row, "extraction_opex_usd_year"),
            concentration_precip_opex_usd_year=_float(row, "concentration_precip_opex_usd_year"),
            solvent_makeup_opex_usd_year=_float(row, "solvent_makeup_opex_usd_year"),
            installed_contactor_capex_usd=_float(row, "installed_contactor_capex_usd"),
            annualized_capex_usd_year=_float(row, "annualized_capex_usd_year"),
        )
        for row in rows
    ]


def build_idaes_costing_model(scenarios: Iterable[CostingScenario]) -> pyo.ConcreteModel:
    scenario_list = list(scenarios)
    if not scenario_list:
        raise ValueError("At least one costing scenario is required.")

    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.costing = pyo.Block()
    m.fs.costing.scenario = pyo.Set(initialize=[row.scenario for row in scenario_list], ordered=True)

    by_name = {row.scenario: row for row in scenario_list}
    c = m.fs.costing
    c.feed_flow_m3_day = pyo.Param(c.scenario, initialize={k: v.feed_flow_m3_day for k, v in by_name.items()})
    c.li_feed_mg_L = pyo.Param(c.scenario, initialize={k: v.li_feed_mg_L for k, v in by_name.items()})
    c.tds_mg_L = pyo.Param(c.scenario, initialize={k: v.tds_mg_L for k, v in by_name.items()})
    c.li_recovery_pct = pyo.Param(c.scenario, initialize={k: v.li_recovery_pct for k, v in by_name.items()})
    c.li2co3_product_kg_year = pyo.Param(
        c.scenario,
        initialize={k: v.li2co3_product_t_year * 1000.0 for k, v in by_name.items()},
    )
    c.gross_revenue_usd_year = pyo.Param(
        c.scenario,
        initialize={k: v.gross_revenue_usd_year for k, v in by_name.items()},
    )
    c.pretreatment_opex_usd_year = pyo.Param(
        c.scenario,
        initialize={k: v.pretreatment_opex_usd_year for k, v in by_name.items()},
    )
    c.extraction_opex_usd_year = pyo.Param(
        c.scenario,
        initialize={k: v.extraction_opex_usd_year for k, v in by_name.items()},
    )
    c.concentration_precip_opex_usd_year = pyo.Param(
        c.scenario,
        initialize={k: v.concentration_precip_opex_usd_year for k, v in by_name.items()},
    )
    c.solvent_makeup_opex_usd_year = pyo.Param(
        c.scenario,
        initialize={k: v.solvent_makeup_opex_usd_year for k, v in by_name.items()},
    )
    c.installed_contactor_capex_usd = pyo.Param(
        c.scenario,
        initialize={k: v.installed_contactor_capex_usd for k, v in by_name.items()},
    )
    c.annualized_capex_usd_year = pyo.Param(
        c.scenario,
        initialize={k: v.annualized_capex_usd_year for k, v in by_name.items()},
    )

    c.total_opex_usd_year = pyo.Expression(
        c.scenario,
        rule=lambda b, s: b.pretreatment_opex_usd_year[s]
        + b.extraction_opex_usd_year[s]
        + b.concentration_precip_opex_usd_year[s]
        + b.solvent_makeup_opex_usd_year[s],
    )
    c.total_annualized_cost_usd_year = pyo.Expression(
        c.scenario,
        rule=lambda b, s: b.total_opex_usd_year[s] + b.annualized_capex_usd_year[s],
    )
    c.net_before_tax_usd_year = pyo.Expression(
        c.scenario,
        rule=lambda b, s: b.gross_revenue_usd_year[s] - b.total_annualized_cost_usd_year[s],
    )
    c.breakeven_li2co3_price_usd_kg = pyo.Expression(
        c.scenario,
        rule=lambda b, s: b.total_annualized_cost_usd_year[s] / b.li2co3_product_kg_year[s],
    )
    c.revenue_price_usd_kg = pyo.Expression(
        c.scenario,
        rule=lambda b, s: b.gross_revenue_usd_year[s] / b.li2co3_product_kg_year[s],
    )
    c.margin_usd_kg = pyo.Expression(
        c.scenario,
        rule=lambda b, s: b.revenue_price_usd_kg[s] - b.breakeven_li2co3_price_usd_kg[s],
    )
    return m


def scenario_result_rows(model: pyo.ConcreteModel) -> list[dict[str, object]]:
    c = model.fs.costing
    rows: list[dict[str, object]] = []
    for scenario in c.scenario:
        rows.append(
            {
                "scenario": str(scenario),
                "framework": "idaes_pyomo_flowsheet_block",
                "feed_flow_m3_day": pyo.value(c.feed_flow_m3_day[scenario]),
                "Li_feed_mg_L": pyo.value(c.li_feed_mg_L[scenario]),
                "TDS_mg_L": pyo.value(c.tds_mg_L[scenario]),
                "Li_recovery_pct": pyo.value(c.li_recovery_pct[scenario]),
                "Li2CO3_product_kg_year": pyo.value(c.li2co3_product_kg_year[scenario]),
                "gross_revenue_usd_year": pyo.value(c.gross_revenue_usd_year[scenario]),
                "total_opex_usd_year": pyo.value(c.total_opex_usd_year[scenario]),
                "annualized_capex_usd_year": pyo.value(c.annualized_capex_usd_year[scenario]),
                "installed_contactor_capex_usd": pyo.value(c.installed_contactor_capex_usd[scenario]),
                "total_annualized_cost_usd_year": pyo.value(c.total_annualized_cost_usd_year[scenario]),
                "net_before_tax_usd_year": pyo.value(c.net_before_tax_usd_year[scenario]),
                "revenue_price_usd_kg": pyo.value(c.revenue_price_usd_kg[scenario]),
                "breakeven_li2co3_price_usd_kg": pyo.value(c.breakeven_li2co3_price_usd_kg[scenario]),
                "margin_usd_kg": pyo.value(c.margin_usd_kg[scenario]),
                "status": "formal_idaes_costing_expression_from_lithium_handoff",
            }
        )
    return rows


def export_costing_results(
    rows: list[dict[str, object]],
    *,
    out_csv: str | Path = OUT_CSV,
    out_md: str | Path = OUT_MD,
) -> tuple[Path, Path]:
    csv_path = Path(out_csv)
    md_path = Path(out_md)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# HBTA/TOPO IDAES Costing Results",
        "",
        "This artifact is generated inside the PrOMMiS environment from the Lithium_Extraction HBTA/TOPO costing handoff. It is a formal IDAES/Pyomo flowsheet costing expression layer, not a vendor-grade TEA.",
        "",
        "| Scenario | Flow (m3/day) | Li2CO3 (kg/year) | Revenue price ($/kg) | Breakeven ($/kg) | Margin ($/kg) | Net before tax ($/year) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {float(row['feed_flow_m3_day']):.0f} | "
            f"{float(row['Li2CO3_product_kg_year']):.2f} | {float(row['revenue_price_usd_kg']):.2f} | "
            f"{float(row['breakeven_li2co3_price_usd_kg']):.2f} | {float(row['margin_usd_kg']):.2f} | "
            f"{float(row['net_before_tax_usd_year']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Lithium recovery is supplied by the source-regressed Li/Na HBTA/TOPO model and capped to the Shan/Gando 97.17% three-stage anchor before costing.",
            "- Equipment and operating costs are scenario assumptions from the Lithium_Extraction handoff, represented as IDAES/Pyomo expressions here.",
            "- This module is the required downstream costing path; vendor quotes, solvent-loss experiments, and detailed pretreatment sizing remain future TEA inputs.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def run_idaes_costing(input_csv: str | Path = DEFAULT_INPUT_CSV) -> tuple[pyo.ConcreteModel, list[dict[str, object]]]:
    scenarios = load_costing_scenarios(input_csv)
    model = build_idaes_costing_model(scenarios)
    return model, scenario_result_rows(model)


def main() -> None:
    _, rows = run_idaes_costing()
    csv_path, md_path = export_costing_results(rows)
    print(f"Saved IDAES costing CSV: {csv_path}")
    print(f"Saved IDAES costing report: {md_path}")


if __name__ == "__main__":
    main()
