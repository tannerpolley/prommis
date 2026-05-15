#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2024 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################

from __future__ import annotations

import csv

import pytest

from prommis.solvent_extraction.pcsaft_integration import hbta_topo_produced_water_case_study as module


def _mock_stage_runner(feed_mg_L, stage_idx: int):
    solvent = module._solvent_stage_feed_mg_L(top_mol_L=module.TOP_MOL_L, hexane_mol_L=module.HEXANE_MOL_L)
    return module._run_stage_fallback(
        feed_mg_L,
        stage=stage_idx,
        top_mg_L=solvent["TOP"],
        hexane_mg_L=solvent["Hexane"],
        org_flow_L_h=module.AQ_FLOW_L_H * module.OA_RATIO,
        stage_index_scale=1.0,
    )


@pytest.mark.unit
def test_hbta_topo_artifact_build_with_injected_stage_runner_roundtrips_exports(tmp_path):
    artifact = module.build_case_study_artifact(n_stages=3, stage_runner=_mock_stage_runner)

    assert artifact["bridge_status"]["mode"] == "injectable_stage_runner"
    assert len(artifact["stages"]) == 3
    assert artifact["pretreatment"]["status"] == "implemented"

    stages = artifact["stages"]
    final_li = stages[-1].aqueous_out_mg_L["Li+"]
    initial_li = artifact["pretreatment"]["outlet_mg_L"]["Li+"]
    assert final_li < initial_li

    cumulative_row = next(row for row in artifact["rows"] if row.section == "cumulative" and row.item == "Cumulative Li extraction")
    reported = float(cumulative_row.bridge_proxy.strip("%"))
    expected = 100.0 * (1.0 - final_li / max(initial_li, 1e-30))
    assert reported == pytest.approx(expected, abs=5e-8)

    for unit in (
        "pretreatment",
        "extraction_contacting",
        "stripping",
        "solvent_makeup_loss",
        "concentration",
        "li2co3_precipitation",
    ):
        assert any(row.section == "costing" and row.item == unit for row in artifact["rows"])

    csv_path, md_path = module.export_hbta_topo_case_study_artifacts(
        artifact,
        out_csv=tmp_path / "hbta_case_study.csv",
        out_md=tmp_path / "hbta_case_study.md",
    )
    assert csv_path.exists()
    assert md_path.exists()

    table = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    assert table[0]["section"] == "feed"
    assert any(row["section"] == "mscontactor" and row["item"].startswith("Stage 1") for row in table)

    md_lines = md_path.read_text(encoding="utf-8").splitlines()
    assert any("Stage-wise MSContactor Material Balance" in line for line in md_lines)
    assert any("Costing Hooks" in line for line in md_lines)


@pytest.mark.unit
def test_hbta_topo_stage_fallback_outputs_closed_form_profiles():
    solvent = module._solvent_stage_feed_mg_L(top_mol_L=module.TOP_MOL_L, hexane_mol_L=module.HEXANE_MOL_L)
    feed = {
        "Li+": 60.0,
        "K+": 560.0,
        "Na+": 10900.0,
        "Mg+2": 820.0,
        "Ca+2": 8600.0,
        "Ba+2": 560.0,
        "Cl-": 47220.0,
    }
    balance = module._run_stage_fallback(
        feed,
        stage=1,
        top_mg_L=solvent["TOP"],
        hexane_mg_L=solvent["Hexane"],
        org_flow_L_h=module.AQ_FLOW_L_H * module.OA_RATIO,
    )

    assert balance.source_status == "deterministic_fallback"
    for species in ("Li+", "Na+", "K+", "Mg+2", "Ca+2", "Ba+2", "Cl-", "TOP", "Hexane"):
        assert species in balance.feed_aq_mg_L
        assert species in balance.aqueous_out_mg_L
        assert species in balance.organic_out_mg_L
        assert species in balance.material_balance_gap
        assert balance.material_balance_gap[species] > 0.0
