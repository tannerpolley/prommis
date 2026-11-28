"""Reference case: the set C MS-2 whole-process fixed design (base prices, accepted stage count, low exchange fit)
reproduces its retained campaign record."""

import json

import pytest


@pytest.mark.component
def test_ms2_fixed_design_reproduces_retained_record(tmp_path, monkeypatch):
    # The conda Engine's threaded numerical libraries can exhaust IPOPT's CPU budget at iteration zero (#425).
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    pytest.importorskip("epcsaft")
    from prommis.lithium_extraction.process.candidate import STUDY, solve_candidate

    records = STUDY / "results/campaign"
    retained = json.loads(
        (records / "fixed/C_MS2_full_process_base_n5_low.json").read_text()
    )
    result = solve_candidate(
        dict(retained["candidate"], phase="fixed"), records, tmp_path
    )
    # The campaign compares costs at 1e-6 relative and recovery at 1e-9 absolute.
    assert result["status"] == retained["status"]
    assert result["cost_usd_kg_li"] == pytest.approx(
        retained["cost_usd_kg_li"], rel=1e-6, abs=0
    )
    assert result["product_recovery_fraction"] == pytest.approx(
        retained["product_recovery_fraction"], rel=0, abs=1e-9
    )
