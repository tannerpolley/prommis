"""Reference case: the set C MS-2 whole-process fixed design (base prices, accepted stage count, low exchange fit)
reproduces its retained campaign record."""

import json

import pytest


@pytest.mark.component
def test_ms2_fixed_design_reproduces_retained_record(tmp_path):
    pytest.importorskip("epcsaft")
    from prommis.lithium_extraction.process.candidate import (
        STUDY,
        jobs,
        solve_candidate,
    )

    band = json.loads((STUDY / "results/targets/exchange_band.json").read_text())
    stages = next(
        g["low"]["stages"]
        for g in band["groups"]
        if g["site"] == "MS2" and g["case"] == "base"
    )
    job = next(j for j in jobs() if j["id"] == f"C_MS2_full_process_base_n{stages}_low")
    records = STUDY / "results/campaign"
    retained = json.loads((records / "fixed" / (job["id"] + ".json")).read_text())
    result = solve_candidate(dict(job, phase="fixed"), records, tmp_path)
    # The produced-water notebook's limits: relative cost tolerance of the exchange band, 1e-9 on recovery.
    assert result["status"] == retained["status"]
    assert result["cost_usd_kg_li"] == pytest.approx(
        retained["cost_usd_kg_li"], rel=band["reporting_relative_cost_tolerance"], abs=0
    )
    assert result["product_recovery_fraction"] == pytest.approx(
        retained["product_recovery_fraction"], rel=0, abs=1e-9
    )
