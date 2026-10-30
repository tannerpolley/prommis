"""Attach the installed ePC-SAFT reactive response to the MS-2 stage model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import epcsaft
from epcsaft import equilibrium as eq
from pyomo.environ import Constraint, Expression, ExternalFunction, Var, value
from pyomo.environ import units as pyunits

from analyses.shared.hbta_topo_configured_reactive_lle.scripts import independent_network
from analyses.shared.hbta_topo_configured_reactive_lle.scripts.replay_current_engine import conditions, fit_coordinates
from lithium_extraction.common.support.paths import ENGINE_ASL_LIBRARY


ASL_FUNCTION = "epcsaft_equilibrium_phase_component_amount"
CONTROL_LABELS = ("diluent", "Li", "Na", "H", "freeExtractant", "NaComplex", "LiComplex")
CONTINUATION = eq.Continuation(initial_step=0.25, minimum_step=0.25, max_steps=4, max_retries_per_step=1)
SPECIES = independent_network.SPECIES
FEED_OFFSET = (55.508, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
FEED_MATRIX = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
)


@dataclass(frozen=True, slots=True)
class Reference:
    condition: object
    model: object
    problem: object
    anchor: tuple[float, ...]
    record: str
    library: Path
    dodecane_mol_l_organic: float
    # The Engine problem's species map. Another reaction network (e.g. the saponified flowsheet's separate HBTA and
    # TOPO) passes its own; the seven stage controls and their order are shared.
    species: tuple[str, ...] = SPECIES
    feed_offset: tuple[float, ...] = FEED_OFFSET
    feed_matrix: tuple[tuple[float, ...], ...] = FEED_MATRIX
    outputs: tuple[int, int] = (8, 7)  # component indices of the Li and Na complexes
    # (problem, compiled, result) -> (diagnostics, status) strict certificate; None is the lumped-carrier network's.
    certify: Callable | None = None
    continuation: object = CONTINUATION  # the record's continuation from its anchor; the ladders reuse it


def _raw_problem(condition, topology, model, reactions):
    seeded = independent_network._problem(condition, topology, model, reactions, independent_network.INITIALIZATION_SEEDS[0])
    initial = independent_network._initial_amounts(condition, topology, model, independent_network.INITIALIZATION_SEEDS[0])
    free, sodium, lithium = independent_network._declared_carrier_amounts(condition, topology)
    li = condition.li_mol_l_aqueous * condition.aqueous_volume_l
    na = condition.na_mol_l_aqueous * condition.aqueous_volume_l
    hydrogen = condition.hydrogen_mol_l_aqueous * condition.aqueous_volume_l
    raw_feed = dict(zip(SPECIES, (
        55.508 * condition.aqueous_volume_l,
        initial[1][SPECIES.index("diluent")], li, na, hydrogen, li + na + hydrogen,
        free, sodium, lithium,
    ), strict=True))
    problem = eq.Problem(
        phases=seeded.phases, T=seeded.T, P=seeded.P, feed=eq.Amounts(raw_feed),
        reactions=seeded.reactions, row_tolerance=1.0e-10,
    )
    dodecane = initial[1][SPECIES.index("diluent")] / condition.organic_volume_l
    return problem, raw_feed, dodecane


@lru_cache(maxsize=1)
def build_reference() -> Reference:
    condition = conditions()[2]
    ln_k_exchange, ln_k_na_uptake = fit_coordinates()
    model = independent_network.build_model("t1_mbta_topo", "dodecane")[1]
    topology = independent_network.TOPOLOGIES["t1_mbta_topo"]
    reactions = independent_network._reactions(condition, ln_k_exchange, ln_k_na_uptake)
    problem, feed, dodecane = _raw_problem(condition, topology, model, reactions)
    anchor = tuple(float(feed[name]) for name in (
        "diluent", "lithium-cation", "sodium-cation", "hydrogen-cation",
        "free-carrier", "sodium-carrier", "lithium-carrier",
    ))
    library = Path(epcsaft.__file__).resolve().parent / ENGINE_ASL_LIBRARY
    record = eq.asl_record(model, problem, CONTINUATION, FEED_OFFSET, FEED_MATRIX, anchor, 0, 0.001)
    return Reference(condition, model, problem, anchor, record, library, dodecane)


def _stage_controls(block, reference: Reference):
    inventory = block.organic_to_aqueous_ratio * block.carrier_mol_l_organic * pyunits.mol
    return (
        block.organic_to_aqueous_ratio * reference.dodecane_mol_l_organic * pyunits.mol,
        block.li_mol_l_aqueous * pyunits.mol,
        block.na_mol_l_aqueous * pyunits.mol,
        block.hydrogen_mol_l_aqueous * pyunits.mol,
        inventory * block.incoming_free_loading_fraction,
        inventory * block.incoming_na_loading_fraction,
        inventory * block.incoming_li_loading_fraction,
    ), inventory


def attach_live_equilibrium(model, reference: Reference | None = None):
    """Replace initialized surrogate closures with the fixed-T/P ASL response."""

    reference = build_reference() if reference is None else reference
    if not reference.library.is_file():
        raise FileNotFoundError(f"installed ASL library is unavailable: {reference.library}")
    fs = model.fs
    if not fs.temperature.fixed or value(fs.temperature) != reference.problem.T:
        raise ValueError("live ASL attachment requires fixed temperature equal to the reference problem")
    fs.epcsaft_live_response = ExternalFunction(
        library=str(reference.library), function=ASL_FUNCTION, units=pyunits.mol,
        arg_units=[pyunits.mol] * 7 + [None, None, None],
    )
    feed = fs.extractor.organic_inlet_state[0].conc_mol_comp
    fresh_feed = all(feed[m].fixed and value(feed[m]) == 0.0 for m in ("Li", "Na"))
    for stage in fs.extractor.elements:
        block = fs.stage_equilibrium[stage]
        if not hasattr(block, "loaded_state_surrogate"):
            raise ValueError(f"stage {stage} has no initialized loaded-state surrogate")
        block.del_component("loaded_state_surrogate")
        # Live amounts obey physical capacity, not the surrogate's training cutoff.
        block.incoming_li_loading_fraction.setub(1.0)
        block.incoming_na_loading_fraction.setub(1.0)
        # Nor its 1e-4 mol/L aqueous Li training floor, which four- and five-stage cascades cross.
        block.li_mol_l_aqueous.setlb(0.0)
        block.incoming_free_loading_fraction = Var(
            initialize=1.0 - value(block.incoming_li_loading_fraction) - value(block.incoming_na_loading_fraction),
            bounds=(0.0, 1.0),
        )
        block.incoming_loading_domain.set_value(
            block.incoming_free_loading_fraction
            + block.incoming_li_loading_fraction
            + block.incoming_na_loading_fraction == 1.0
        )
        controls, inventory = _stage_controls(block, reference)
        block.incoming_extractant_inventory = Expression(expr=inventory)
        block.live_li_equilibrium = Constraint(
            expr=block.equilibrium_li_loading_fraction == fs.epcsaft_live_response(*controls, reference.record, "1", str(reference.outputs[0]))
            / block.incoming_extractant_inventory
        )
        block.live_na_equilibrium = Constraint(
            expr=block.equilibrium_na_loading_fraction == fs.epcsaft_live_response(*controls, reference.record, "1", str(reference.outputs[1]))
            / block.incoming_extractant_inventory
        )
        if fresh_feed and stage == fs.extractor.elements.last():
            # Loadings of a metal-free organic feed are zero by construction. Fixing them removes bounds held
            # active by equalities (violating LICQ), whose unbounded multipliers stall Ipopt near the optimum.
            for name, fraction in (("li", 0.0), ("na", 0.0), ("free", 1.0)):
                getattr(block, f"incoming_{name}_loading_fraction").fix(fraction)
            block.incoming_li_link.deactivate()
            block.incoming_na_link.deactivate()
            block.incoming_loading_domain.deactivate()
    model._hbta_topo_live_reference = reference
    return model
