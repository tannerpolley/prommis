#####################################################################################################
# "PrOMMiS" was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# ("PrOMMiS") initiative, and is copyright (c) 2023-2024 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################

from __future__ import annotations

import csv

import pyomo.environ as pyo
import pytest

from prommis.solvent_extraction.pcsaft_integration import rezaee_synthetic_surrogate_mscontactor_costing as module


@pytest.mark.unit
def test_rezaee_synthetic_surrogate_costing_builds_formal_flowsheet_block(tmp_path):
    input_csv = tmp_path / "rezaee_costing_input.csv"
    rows = [
        {
            "scenario": "nominal",
            "case_id": "nominal_ms2_clean_li_na",
            "feed_flow_m3_day": "1000",
            "Li_feed_mg_L": "168",
            "TDS_feature_mg_L": "305000",
            "organic_to_aqueous_mass_ratio": "1.0",
            "Li_recovery_pct_used_for_costing": "48.0",
            "Li2CO3_product_t_year": "143.0",
            "gross_revenue_usd_year": "2860000",
            "pretreatment_opex_usd_year": "900000",
            "extraction_opex_usd_year": "1200000",
            "concentration_precip_opex_usd_year": "800000",
            "solvent_makeup_opex_usd_year": "450000",
            "installed_contactor_capex_usd": "10000000",
            "annualized_capex_usd_year": "1800000",
            "costing_status": "synthetic_scaffold",
        }
    ]
    with input_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    model, result_rows = module.run_rezaee_scaffold_costing(input_csv)

    assert hasattr(model, "fs")
    assert hasattr(model.fs, "rezaee_synthetic_costing")
    assert list(model.fs.rezaee_synthetic_costing.scenario) == ["nominal"]
    result = result_rows[0]
    assert result["framework"] == "idaes_pyomo_flowsheet_block_mscontactor_scaffold"
    assert result["total_opex_usd_year"] == pytest.approx(3350000.0)
    assert result["total_annualized_cost_usd_year"] == pytest.approx(5150000.0)
    assert result["net_before_tax_usd_year"] == pytest.approx(-2290000.0)
    assert pyo.value(model.fs.rezaee_synthetic_costing.breakeven_li2co3_price_usd_kg["nominal"]) > 0

    csv_path, md_path = module.export_results(
        result_rows,
        out_csv=tmp_path / "rezaee_costing_results.csv",
        out_md=tmp_path / "rezaee_costing_results.md",
    )
    assert csv_path.exists()
    assert md_path.exists()
    assert "formal IDAES/Pyomo" in md_path.read_text(encoding="utf-8")
