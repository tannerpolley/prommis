"""Lithium #158 reaction sets on the MS-2 stage: separate HBTA (HA) and TOPO (T), 1:1:1 complexes.

Set A (R1 Li+ + HA + T = LiA.T + H+, R2 Na+ + HA + T = NaA.T + H+) is the working model; set C adds
R5 HA(org) = H+ + BTA-(aq). The ln K are the #158 fits (PR #160), read from the retained result; nothing is refit.
The mixture, reference-state offsets, reactions and strict certificate are those of
``analyses/shared/hbta_topo_configured_reactive_lle/scripts/mechanism_candidates.py``.

A saponified organic is NaA.T: NaOH + HA + T = NaA.T + H2O has the same element inventory as NaOH dosed to the
contact. The live stage response uses set A's species and the seven controls of the fresh-organic live route
(diluent, Li, Na, H, free extractant, NaA.T, LiA.T), with free HA = free T.
"""

from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path

import epcsaft
from epcsaft import equilibrium as eq
from pyomo.environ import ExternalFunction

from analyses.shared.hbta_topo_configured_reactive_lle.scripts import mechanism_candidates as mc
from lithium_extraction.v3.hbta_topo_ms2_extraction.live_equilibrium import ASL_FUNCTION, Reference
from lithium_extraction.common.support.paths import ENGINE_ASL_LIBRARY

TEMPERATURE_K = 298.15
LIVE_SPECIES = (
    "water", "lithium-cation", "sodium-cation", "hydrogen-cation", "chloride-anion",
    "diluent", "hbta", "topo", "lithium-carrier", "sodium-carrier",
)
LIVE_OUTPUTS = (8, 9)  # lithium-carrier, sodium-carrier
# The fresh-organic route's continuation (four fixed 0.25 steps) stalls at s = 0.5 on combined Li/Na/H moves between
# saponified stages (e.g. the second stage of a two-stage cascade), which single-axis moves pass. Halving the step
# down to 1/64 reaches the direct strict state (agreement 1e-18 mol at that point).
CONTINUATION = eq.Continuation(initial_step=0.25, minimum_step=1.0 / 64, max_steps=64, max_retries_per_step=4)
FEED_OFFSET = (55.508,) + (0.0,) * 9
# Controls: diluent, Li, Na, H, free extractant (HA = T), NaA.T, LiA.T (mol per litre of aqueous); Cl closes charge.
FEED_MATRIX = (
    (0, 0, 0, 0, 0, 0, 0),
    (0, 1, 0, 0, 0, 0, 0),
    (0, 0, 1, 0, 0, 0, 0),
    (0, 0, 0, 1, 0, 0, 0),
    (0, 1, 1, 1, 0, 0, 0),
    (1, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 1, 0, 0),
    (0, 0, 0, 0, 1, 0, 0),
    (0, 0, 0, 0, 0, 0, 1),
    (0, 0, 0, 0, 0, 1, 0),
)


@lru_cache(maxsize=None)
def fitted_ln_k(reaction_set: str) -> dict[str, float]:
    document = json.loads((mc.RESULTS / "mechanism_candidates.json").read_text(encoding="utf-8"))
    return dict(document["summary"][reaction_set]["ln_k_298"])


@lru_cache(maxsize=1)
def dodecane_mol_l() -> float:
    """Diluent moles per litre of organic: pure n-dodecane at 298.15 K, the mechanism_candidates convention."""
    mixture = mc.model(LIVE_SPECIES)
    x = tuple(float(s == "diluent") for s in LIVE_SPECIES)
    return mixture.state(TEMPERATURE_K, P=mc.PRESSURE_PA, x=x, phase="liquid").molar_density * 1.0e-3


def stage_state(identity: str, *, organic_to_aqueous: float, li: float, na: float, hydrogen: float, carrier: float,
                incoming_li: float = 0.0, incoming_na: float = 0.0, hbta: float | None = None, topo: float | None = None,
                bta: float = 0.0) -> mc.State:
    """One stage per litre of aqueous. Loadings are fractions of ``carrier`` (mol/L organic each of HBTA, TOPO);
    ``hbta``/``topo`` override the organic totals where set C has dissolved HBTA upstream."""
    return mc.State(identity, TEMPERATURE_K, organic_to_aqueous, li, na, carrier if hbta is None else hbta,
                    carrier if topo is None else topo, hydrogen=hydrogen, li_org=carrier * incoming_li,
                    na_org=carrier * incoming_na, bta=bta)


def solve_stage(state: mc.State, reaction_set: str = "A", guess=None) -> dict:
    """Strict direct Engine state (mechanism_candidates.solve) or RuntimeError."""
    return mc.solve(state, reaction_set, fitted_ln_k(reaction_set), guess=guess)


def _problem(state: mc.State, amounts) -> tuple[object, eq.Problem, list, dict[str, float]]:
    """The set A Engine problem at ``state``, composition guess from a strict state's phase amounts."""
    active = [(n, mc.REACTIONS[n]) for n in mc.SETS["A"]]
    mixture = mc.model(LIVE_SPECIES)
    offset = mc.offsets(LIVE_SPECIES, TEMPERATURE_K)
    ln_k = fitted_ln_k("A")
    ln_k_engine = {n: mc.ln_k_at(n, ln_k[n], TEMPERATURE_K) + math.fsum(c * offset[s] for s, c in nu.items())
                   for n, nu in active}
    aqueous, organic = mc.inventory(state)
    organic = {"diluent": dodecane_mol_l() * state.organic_to_aqueous, **organic}
    feed = {s: aqueous.get(s, 0.0) + organic.get(s, 0.0) for s in LIVE_SPECIES}
    phases = []
    for name, phase_amounts, members in (("aqueous", amounts[0], mc.AQUEOUS), ("organic", amounts[1], mc.ORGANIC)):
        support = tuple(s for s in LIVE_SPECIES if s in members)
        total = math.fsum(phase_amounts.get(s, 0.0) for s in support)
        phases.append(eq.Phase(name, kind="liquid", support=support, amount=eq.Free(),
                               composition_guess=[max(phase_amounts.get(s, 0.0), 1e-30) / total for s in support]))
    problem = eq.Problem(T=TEMPERATURE_K, P=mc.PRESSURE_PA, feed=eq.Amounts(feed), phases=phases, row_tolerance=1e-10,
                         reactions=[eq.Reaction(nu, equilibrium_constant=math.exp(ln_k_engine[n])) for n, nu in active])
    return mixture, problem, active, ln_k_engine


def build_live_reference(anchor: mc.State) -> Reference:
    """Set A live stage response anchored at a strict direct state (the ASL continuation starts there)."""
    direct = solve_stage(anchor)
    if direct["active_reactions"] != list(mc.SETS["A"]):
        raise RuntimeError(f"anchor needs {direct['active_reactions']}; the live species map carries no OH-")
    mixture, problem, active, ln_k_engine = _problem(anchor, direct["phase_amounts_mol"])
    v = anchor.organic_to_aqueous
    controls = (dodecane_mol_l() * v, anchor.li, anchor.na, anchor.hydrogen,
                v * (anchor.hbta - anchor.li_org - anchor.na_org), v * anchor.na_org, v * anchor.li_org)
    feed = problem.feed.values
    for species, row, offset in zip(LIVE_SPECIES, FEED_MATRIX, FEED_OFFSET, strict=True):
        mapped = offset + math.fsum(m * q for m, q in zip(row, controls, strict=True))
        if abs(mapped - feed.get(species, 0.0)) > 1e-12 * max(1.0, abs(mapped)):
            raise RuntimeError(f"feed map disagrees for {species}: {mapped} against {feed.get(species)}")

    def certify(problem_, compiled, result):
        _, _, check = mc._certificate(anchor, mixture, LIVE_SPECIES, problem_, compiled, result, active, ln_k_engine)
        return check, "evaluated" if mc.is_strict(result, check, active) else "not_strict"

    record = eq.asl_record(mixture, problem, CONTINUATION, FEED_OFFSET, FEED_MATRIX, controls, 0, 0.001)
    library = Path(epcsaft.__file__).resolve().parent / ENGINE_ASL_LIBRARY
    return Reference(anchor, mixture, problem, controls, record, library, dodecane_mol_l(),
                     species=LIVE_SPECIES, feed_offset=FEED_OFFSET, feed_matrix=FEED_MATRIX,
                     outputs=LIVE_OUTPUTS, certify=certify, continuation=CONTINUATION)


def live_stage_response(reference: Reference):
    """(Li, Na) equilibrium loading of one stage from the live ASL record, for initialization."""
    external = ExternalFunction(library=str(reference.library), function=ASL_FUNCTION)

    def response(inputs: dict[str, float]) -> tuple[float, float]:
        inventory = inputs["organic_to_aqueous_ratio"] * inputs["carrier_mol_l_organic"]
        li_in, na_in = inputs["incoming_li_loading_fraction"], inputs["incoming_na_loading_fraction"]
        controls = [
            inputs["organic_to_aqueous_ratio"] * reference.dodecane_mol_l_organic,
            inputs["li_mol_l_aqueous"], inputs["na_mol_l_aqueous"], inputs["hydrogen_mol_l_aqueous"],
            inventory * (1.0 - li_in - na_in), inventory * na_in, inventory * li_in,
        ]
        return tuple(external.evaluate([*controls, reference.record, "1", str(k)]) / inventory
                     for k in reference.outputs)

    return response


def anchor_state(feed_li: float, feed_na: float, feed_h: float, *, organic_to_aqueous: float = 1.0,
                 carrier: float = 0.03, saponification_ratio: float = 0.9) -> mc.State:
    """The live reference anchor: one stage at the extraction feed with a saponified organic."""
    return stage_state("live_anchor", organic_to_aqueous=organic_to_aqueous, li=feed_li, na=feed_na,
                       hydrogen=feed_h, carrier=carrier, incoming_na=saponification_ratio)
