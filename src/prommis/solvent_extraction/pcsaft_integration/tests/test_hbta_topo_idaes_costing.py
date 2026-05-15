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

from prommis.solvent_extraction.pcsaft_integration import hbta_topo_idaes_costing as module


@pytest.mark.unit
def test_hbta_topo_idaes_costing_builds_formal_flowsheet_block(tmp_path):
    input_csv = tmp_path / "costing_input.csv"
    rows = [
        {
            "scenario": "base",
            "feed_flow_m3_day": "1000",
            "Li_feed_mg_L": "168",
            "TDS_mg_L": "305000",
            "Li_recovery_pct_used_for_costing": "97.17",
            "Li2CO3_product_t_year": "289.6825319308357",
            "gross_revenue_usd_year": "5793650.638616714",
            "pretreatment_opex_usd_year": "1666666.6666666665",
            "extraction_opex_usd_year": "1000000",
            "concentration_precip_opex_usd_year": "2000000",
            "solvent_makeup_opex_usd_year": "666666.6666666666",
            "installed_contactor_capex_usd": "15000000",
            "annualized_capex_usd_year": "3000000",
        }
    ]
    with input_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    model, result_rows = module.run_idaes_costing(input_csv)

    assert hasattr(model, "fs")
    assert hasattr(model.fs, "costing")
    assert list(model.fs.costing.scenario) == ["base"]
    result = result_rows[0]
    assert result["framework"] == "idaes_pyomo_flowsheet_block"
    assert result["total_opex_usd_year"] == pytest.approx(5333333.333333333)
    assert result["total_annualized_cost_usd_year"] == pytest.approx(8333333.333333333)
    assert result["net_before_tax_usd_year"] == pytest.approx(-2539682.6947166193)
    assert pyo.value(model.fs.costing.breakeven_li2co3_price_usd_kg["base"]) > 0

    csv_path, md_path = module.export_costing_results(
        result_rows,
        out_csv=tmp_path / "idaes_costing.csv",
        out_md=tmp_path / "idaes_costing.md",
    )
    assert csv_path.exists()
    assert md_path.exists()
    assert "formal IDAES/Pyomo" in md_path.read_text(encoding="utf-8")
