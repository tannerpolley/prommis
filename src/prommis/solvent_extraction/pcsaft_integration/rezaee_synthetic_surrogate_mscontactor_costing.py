#####################################################################################################
# "PrOMMiS" was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# ("PrOMMiS") initiative, and is copyright (c) 2023-2024 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""PrOMMiS/IDAES consumer for the Rezaee clean Li/Na surrogate scaffold.

The Lithium_Extraction project owns the synthetic scaffold UQ rows. This module
checks that those rows can drive a staged MSContactor-style transfer handoff and
formal IDAES/Pyomo costing expressions while the real calibrated surrogate
runner is still pending.
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
DEFAULT_LITHIUM_REPO = Path(
    os.environ.get(
        "LITHIUM_EXTRACTION_REPO",
        r"C:\Users\Tanner\Documents\git\ePC-SAFT\.worktrees\downstreams\Lithium_Extraction",
    )
)
DEFAULT_TRANSFER_CSV = (
    DEFAULT_LITHIUM_REPO
    / "data"
    / "reference"
    / "produced_water"
    / "rezaee_tds_li_oa_prommis_idaes_transfer.csv"
)
DEFAULT_COSTING_CSV = (
    DEFAULT_LITHIUM_REPO
    / "data"
    / "reference"
    / "produced_water"
    / "rezaee_tds_li_oa_synthetic_idaes_costing_input.csv"
)
OUT_CSV = THIS_DIR / "rezaee_synthetic_surrogate_mscontactor_costing_results.csv"
OUT_MD = THIS_DIR / "rezaee_synthetic_surrogate_mscontactor_costing_results.md"


@dataclass(frozen=True)
class RezaeeCostingScenario:
    scenario: str
    case_id: str
    feed_flow_m3_day: float
    li_feed_mg_L: float
    tds_feature_mg_L: float
    organic_to_aqueous_mass_ratio: float
    li_recovery_pct: float
    li2co3_product_t_year: float
    gross_revenue_usd_year: float
    pretreatment_opex_usd_year: float
    extraction_opex_usd_year: float
    concentration_precip_opex_usd_year: float
    solvent_makeup_opex_usd_year: float
    installed_contactor_capex_usd: float
    annualized_capex_usd_year: float
    model_status: str


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def load_costing_scenarios(path: str | Path = DEFAULT_COSTING_CSV) -> list[RezaeeCostingScenario]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        RezaeeCostingScenario(
            scenario=row["scenario"],
            case_id=row["case_id"],
            feed_flow_m3_day=_float(row, "feed_flow_m3_day"),
            li_feed_mg_L=_float(row, "Li_feed_mg_L"),
            tds_feature_mg_L=_float(row, "TDS_feature_mg_L"),
            organic_to_aqueous_mass_ratio=_float(row, "organic_to_aqueous_mass_ratio"),
            li_recovery_pct=_float(row, "Li_recovery_pct_used_for_costing"),
            li2co3_product_t_year=_float(row, "Li2CO3_product_t_year"),
            gross_revenue_usd_year=_float(row, "gross_revenue_usd_year"),
            pretreatment_opex_usd_year=_float(row, "pretreatment_opex_usd_year"),
            extraction_opex_usd_year=_float(row, "extraction_opex_usd_year"),
            concentration_precip_opex_usd_year=_float(row, "concentration_precip_opex_usd_year"),
            solvent_makeup_opex_usd_year=_float(row, "solvent_makeup_opex_usd_year"),
            installed_contactor_capex_usd=_float(row, "installed_contactor_capex_usd"),
            annualized_capex_usd_year=_float(row, "annualized_capex_usd_year"),
            model_status=row["costing_status"],
        )
        for row in rows
    ]


def load_transfer_rows(path: str | Path = DEFAULT_TRANSFER_CSV) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_idaes_costing_model(scenarios: Iterable[RezaeeCostingScenario]) -> pyo.ConcreteModel:
    scenario_list = list(scenarios)
    if not scenario_list:
        raise ValueError("At least one Rezaee costing scenario is required.")

    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.rezaee_synthetic_costing = pyo.Block()
    c = m.fs.rezaee_synthetic_costing
    c.scenario = pyo.Set(initialize=[row.scenario for row in scenario_list], ordered=True)

    by_name = {row.scenario: row for row in scenario_list}
    c.feed_flow_m3_day = pyo.Param(c.scenario, initialize={k: v.feed_flow_m3_day for k, v in by_name.items()})
    c.li_feed_mg_L = pyo.Param(c.scenario, initialize={k: v.li_feed_mg_L for k, v in by_name.items()})
    c.tds_feature_mg_L = pyo.Param(c.scenario, initialize={k: v.tds_feature_mg_L for k, v in by_name.items()})
    c.organic_to_aqueous_mass_ratio = pyo.Param(
        c.scenario, initialize={k: v.organic_to_aqueous_mass_ratio for k, v in by_name.items()}
    )
    c.li_recovery_pct = pyo.Param(c.scenario, initialize={k: v.li_recovery_pct for k, v in by_name.items()})
    c.li2co3_product_kg_year = pyo.Param(
        c.scenario, initialize={k: v.li2co3_product_t_year * 1000.0 for k, v in by_name.items()}
    )
    c.gross_revenue_usd_year = pyo.Param(
        c.scenario, initialize={k: v.gross_revenue_usd_year for k, v in by_name.items()}
    )
    c.pretreatment_opex_usd_year = pyo.Param(
        c.scenario, initialize={k: v.pretreatment_opex_usd_year for k, v in by_name.items()}
    )
    c.extraction_opex_usd_year = pyo.Param(
        c.scenario, initialize={k: v.extraction_opex_usd_year for k, v in by_name.items()}
    )
    c.concentration_precip_opex_usd_year = pyo.Param(
        c.scenario, initialize={k: v.concentration_precip_opex_usd_year for k, v in by_name.items()}
    )
    c.solvent_makeup_opex_usd_year = pyo.Param(
        c.scenario, initialize={k: v.solvent_makeup_opex_usd_year for k, v in by_name.items()}
    )
    c.installed_contactor_capex_usd = pyo.Param(
        c.scenario, initialize={k: v.installed_contactor_capex_usd for k, v in by_name.items()}
    )
    c.annualized_capex_usd_year = pyo.Param(
        c.scenario, initialize={k: v.annualized_capex_usd_year for k, v in by_name.items()}
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
    c.revenue_price_usd_kg = pyo.Expression(
        c.scenario,
        rule=lambda b, s: b.gross_revenue_usd_year[s] / b.li2co3_product_kg_year[s],
    )
    c.breakeven_li2co3_price_usd_kg = pyo.Expression(
        c.scenario,
        rule=lambda b, s: b.total_annualized_cost_usd_year[s] / b.li2co3_product_kg_year[s],
    )
    c.margin_usd_kg = pyo.Expression(
        c.scenario,
        rule=lambda b, s: b.revenue_price_usd_kg[s] - b.breakeven_li2co3_price_usd_kg[s],
    )
    return m


def scenario_result_rows(model: pyo.ConcreteModel, scenarios: Iterable[RezaeeCostingScenario]) -> list[dict[str, object]]:
    c = model.fs.rezaee_synthetic_costing
    by_name = {row.scenario: row for row in scenarios}
    rows: list[dict[str, object]] = []
    for scenario in c.scenario:
        source = by_name[str(scenario)]
        rows.append(
            {
                "scenario": str(scenario),
                "case_id": source.case_id,
                "framework": "idaes_pyomo_flowsheet_block_mscontactor_scaffold",
                "feed_flow_m3_day": pyo.value(c.feed_flow_m3_day[scenario]),
                "Li_feed_mg_L": pyo.value(c.li_feed_mg_L[scenario]),
                "TDS_feature_mg_L": pyo.value(c.tds_feature_mg_L[scenario]),
                "organic_to_aqueous_mass_ratio": pyo.value(c.organic_to_aqueous_mass_ratio[scenario]),
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
                "status": source.model_status,
            }
        )
    return rows


def export_results(rows: list[dict[str, object]], *, out_csv: str | Path = OUT_CSV, out_md: str | Path = OUT_MD) -> tuple[Path, Path]:
    csv_path = Path(out_csv)
    md_path = Path(out_md)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Rezaee Synthetic Surrogate MSContactor/IDAES Costing Results",
        "",
        "This PrOMMiS-side artifact consumes the Lithium_Extraction Rezaee clean Li/Na synthetic scaffold. It verifies the staged transfer and costing interface shape while the calibrated surrogate runner is pending.",
        "",
        "| Scenario | Case | Flow m3/day | Li recovery % | Li2CO3 kg/year | Breakeven $/kg | Net before tax $/year |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['case_id']} | {float(row['feed_flow_m3_day']):.0f} | "
            f"{float(row['Li_recovery_pct']):.2f} | {float(row['Li2CO3_product_kg_year']):.2f} | "
            f"{float(row['breakeven_li2co3_price_usd_kg']):.2f} | {float(row['net_before_tax_usd_year']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- The input rows are synthetic scaffold rows for process-integration readiness.",
            "- The module uses a formal IDAES/Pyomo flowsheet block and algebraic costing expressions.",
            "- Replace the input CSV with calibrated Rezaee/ePC-SAFT surrogate outputs when available.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def run_rezaee_scaffold_costing(input_csv: str | Path = DEFAULT_COSTING_CSV) -> tuple[pyo.ConcreteModel, list[dict[str, object]]]:
    scenarios = load_costing_scenarios(input_csv)
    model = build_idaes_costing_model(scenarios)
    return model, scenario_result_rows(model, scenarios)


def main() -> None:
    transfer_rows = load_transfer_rows()
    if not transfer_rows:
        raise RuntimeError("Transfer handoff is empty.")
    _, rows = run_rezaee_scaffold_costing()
    csv_path, md_path = export_results(rows)
    print(f"Read transfer rows: {len(transfer_rows)}")
    print(f"Saved Rezaee scaffold costing CSV: {csv_path}")
    print(f"Saved Rezaee scaffold costing report: {md_path}")


if __name__ == "__main__":
    main()
