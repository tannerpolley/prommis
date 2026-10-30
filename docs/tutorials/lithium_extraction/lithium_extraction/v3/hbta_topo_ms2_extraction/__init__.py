"""Canonical HBTA/TOPO Smackover MS-2 extraction-section model."""

from .loaded_state_surrogate import (
    LoadedOrganicStateSurrogate,
    attach_loaded_state_surrogate,
    input_domain_violations,
    load_loaded_state_contract,
)
from .multistage import (
    MultistageCase,
    build_multistage_model,
    case_flow,
    maximum_balance_error_mol_h,
    nominal_ms2_case,
    solve_square_model,
    stage_rows,
)
from .multistage_optimization import (
    OptimizationResult,
    add_minimum_cost_optimization,
    add_pareto_optimization,
    optimization_result,
    reference_anchored_organic_cost_usd_l,
    solve_optimization,
)
from .reactive_lle_surrogate import (
    attach_reactive_lle_surrogate,
    evaluate_reactive_lle_surrogate,
    load_reactive_lle_contract,
    load_reactive_lle_surrogate,
)
from .state_surrogate import (
    ReactiveLleStateSurrogate,
    attach_state_surrogate,
    equilibrium_statistics,
    load_state_surrogate_contract,
)

__all__ = [
    "LoadedOrganicStateSurrogate",
    "MultistageCase",
    "OptimizationResult",
    "ReactiveLleStateSurrogate",
    "add_minimum_cost_optimization",
    "add_pareto_optimization",
    "attach_loaded_state_surrogate",
    "attach_reactive_lle_surrogate",
    "attach_state_surrogate",
    "build_multistage_model",
    "case_flow",
    "equilibrium_statistics",
    "evaluate_reactive_lle_surrogate",
    "input_domain_violations",
    "load_loaded_state_contract",
    "load_reactive_lle_contract",
    "load_reactive_lle_surrogate",
    "load_state_surrogate_contract",
    "maximum_balance_error_mol_h",
    "nominal_ms2_case",
    "optimization_result",
    "reference_anchored_organic_cost_usd_l",
    "solve_optimization",
    "solve_square_model",
    "stage_rows",
]
