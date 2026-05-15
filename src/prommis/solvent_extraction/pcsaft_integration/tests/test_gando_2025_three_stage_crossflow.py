#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2024 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################

import pytest

from prommis.solvent_extraction.pcsaft_integration.gando_2025_three_stage_crossflow import (
    export_gando_2025_artifacts,
    run_gando_2025_three_stage_crossflow,
    solve_gando_2025_selective_stage,
)


@pytest.mark.unit
def test_gando_stage_reference_matches_expected_nominal_values():
    ref = solve_gando_2025_selective_stage(li_mg_L=60.0, na_mg_L=10900.0)
    assert ref.li_extraction_pct == pytest.approx(52.00472287377231, rel=1e-10)
    assert ref.na_extraction_pct == pytest.approx(0.906746516991823, rel=1e-10)
    assert ref.li_distribution_ratio == pytest.approx(1.0835383393453435, rel=1e-10)


@pytest.mark.component
def test_gando_three_stage_crossflow_matches_expected_cumulative_profile(tmp_path):
    result = run_gando_2025_three_stage_crossflow()

    assert result.li_cumulative_pct == pytest.approx([52.004722873772316, 84.84990814348146, 97.80245363319239], rel=1e-10)
    assert result.na_cumulative_pct == pytest.approx([0.9067465169918187, 2.0979329787334855, 3.5827126965000677], rel=1e-10)
    assert result.paper_li_cumulative_pct == pytest.approx([54.95, 85.6, 97.17], rel=1e-12)

    csv_path, md_path, png_path = export_gando_2025_artifacts(
        result,
        out_csv=tmp_path / 'gando_2025_stage3_comparison.csv',
        out_md=tmp_path / 'gando_2025_stage3_comparison.md',
        out_png=tmp_path / 'gando_2025_stage3_efficiency_plot.png',
    )
    assert csv_path.exists()
    assert md_path.exists()
    assert png_path.exists()
