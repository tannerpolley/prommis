#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
from importlib import reload

from pyomo.environ import check_optimal_termination, value
from pyomo.opt import SolverStatus

from idaes.core.util.model_statistics import degrees_of_freedom

import pytest

pytest.importorskip("idaes_fi")
# pylint: disable-next=wrong-import-position
from prommis.examples import solvent_extraction_runner


@pytest.mark.component
def test_solvent_extraction_runner():
    # Each run must annotate a fresh model, regardless of earlier module use.
    runner = reload(solvent_extraction_runner).FS
    runner.run_steps(save_report=False)

    assert runner.results.solver.status == SolverStatus.ok
    assert check_optimal_termination(runner.results)
    assert degrees_of_freedom(runner.model) == 0

    annotations = runner.annotated_vars
    assert {key for key, ann in annotations.items() if ann["is_input"]} == {
        "aqueous_feed_flow",
        "feed_nd",
        "feed_fe",
        "barren_organic_flow",
        "makeup_organic_flow",
        "raffinate_recycle",
    }
    expected_outputs = {
        "ree_recovery": 11.9715359629,
        "ree_purity": 1.69813038915,
        "mixer_settler_cost": 0.068032113156,
    }
    assert {key for key, ann in annotations.items() if ann["is_output"]} == set(
        expected_outputs
    )
    for key, expected in expected_outputs.items():
        assert value(annotations[key]["var"]) == pytest.approx(expected, rel=1e-3)

    baseline = {key: value(annotations[key]["var"]) for key in expected_outputs}
    recycle = runner.model.fs.raffinate_splitter.split_fraction[0, "recycle"]
    original_recycle = value(recycle)
    for fraction in (0.5, original_recycle):
        recycle.fix(fraction)
        runner.run_steps(first="solve_initial", save_report=False)
        assert runner.results.solver.status == SolverStatus.ok
        assert check_optimal_termination(runner.results)
        assert degrees_of_freedom(runner.model) == 0

    annotations = runner.annotated_vars
    for key, expected in baseline.items():
        assert value(annotations[key]["var"]) == pytest.approx(expected, rel=1e-6)
