"""Candidate HBTA/TOPO Li/Na extraction reaction sets against Zhang (2017) (Lithium #158).

The design, fit/holdout split and acceptance criteria were fixed in issue #158 before any fitting.

HBTA (HA) and TOPO (T) are separate organic species; the complexes are 1:1:1.

- R1 Li+ + HA + T = LiA.T + H+
- R2 Na+ + HA + T = NaA.T + H+ (R1 - R2 is Li/Na exchange on the saponified organic)
- R3 Li+ + Cl- + T = LiCl.T (neutral TOPO solvation, no proton)
- R5 HA(org) = H+ + BTA-(aq) (extractant loss to the aqueous phase)

The sets are A = R1, R2; B = A + R3; C = A + R5; D = A + R3 + R5.

Constants are ln K on conventional reference states:
- aqueous solutes: molal infinite dilution in water;
- water: pure water;
- organic solutes: 1 mol/L at infinite dilution in dodecane.

The Engine-basis constant is ln K + sum(nu_i offset_i), with each offset taken from a trace state (mole fraction
1e-20) of that species in its own solvent. ln K(T) = ln K(298.15 K) - dH/R (1/T - 1/298.15). dH is -5.73 kJ/mol on R1
and R3 (Zhang et al. 2020, J. Hazard. Mater. 398:122840, Section 3.11, HBTA/TOPO/kerosene) and 0 on R2 and R5 (no
source). None of the dH values is fitted.

Organic species share the retained ligand-derived prior (sigma 3.70 A, eps/k 360 K, permittivity 4). Their segment
numbers are the retained topologies' increments: HA 7.39, T 13.21, MA.T 20.60, LiCl.T 13.21. BTA-(aq) and OH-(aq)
take chloride's ion parameters and their own masses; this is provisional, since no source gives them.

NaOH saponifies the extractant first (NaOH + HA + T = NaA.T + H2O, the same element inventory). NaOH beyond the
extractant's capacity stays free and brings OH- with water autoprotolysis, RW H2O = H+ + OH-, at the Harned and
Robinson (1940) molal Kw (pKw 14.167 at 293.15 K). Fig. 1 at initial pH 13.13 holds more NaOH (0.09 M) than HBTA
(0.05 M).

States at a printed equilibrium pH are solved at the net NaOH (base > 0) or HCl (base < 0) that gives that pH on the
Engine's H+ molal-activity scale. Each retained state passes the strict certificate, with unchanged tolerances.

epcsaft.regression.fit exposes only equation-of-state families, so the ln K are fitted by bounded least squares over
Engine-solved strict states, from several starts.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache
import json
import math
from pathlib import Path
import time

import epcsaft
from epcsaft import equilibrium
from scipy.optimize import least_squares

from .apparent_application import HBTA_MOLAR_MASS, LI_MOLAR_MASS, NA_MOLAR_MASS, PRESSURE_PA, TOPO_MOLAR_MASS
from .generate import _verify_wheel
from .independent_network import H_MOLAR_MASS, PRESSURE_RELATIVE_LIMIT, TOPOLOGIES, _pair, _rename_component
from .independent_network import parameter_payload as carrier_payload

ROOT = Path(__file__).resolve().parents[4]
RESULTS = Path(__file__).resolve().parents[1] / "results"
DIGITIZED = ROOT / "data/reference/extraction_models/hbta_topo_li_capacity/digitization"
LI_G_MOL, NA_G_MOL, CL_MOLAR_MASS = 6.941, 22.98977, 0.035453
AQUEOUS = ("water", "lithium-cation", "sodium-cation", "hydrogen-cation", "chloride-anion", "hydroxide-anion", "bta-anion")
ORGANIC = ("diluent", "hbta", "topo", "lithium-carrier", "sodium-carrier", "lithium-chloride-topo")
SPECIES = AQUEOUS + ORGANIC
CHARGE = {"lithium-cation": 1, "sodium-cation": 1, "hydrogen-cation": 1, "chloride-anion": -1, "hydroxide-anion": -1,
          "bta-anion": -1}
MOIETIES = {  # conserved rows: every reaction below leaves each one unchanged
    "O": {"water": 1, "hydroxide-anion": 1},
    "diluent": {"diluent": 1},
    "Li": {"lithium-cation": 1, "lithium-carrier": 1, "lithium-chloride-topo": 1},
    "Na": {"sodium-cation": 1, "sodium-carrier": 1},
    "Cl": {"chloride-anion": 1, "lithium-chloride-topo": 1},
    "BTA": {"hbta": 1, "lithium-carrier": 1, "sodium-carrier": 1, "bta-anion": 1},
    "TOPO": {"topo": 1, "lithium-carrier": 1, "sodium-carrier": 1, "lithium-chloride-topo": 1},
    "H": {"hydrogen-cation": 1, "hbta": 1, "hydroxide-anion": -1},
}
REACTIONS = {
    "RW": {"water": -1, "hydrogen-cation": 1, "hydroxide-anion": 1},
    "R1": {"lithium-cation": -1, "hbta": -1, "topo": -1, "lithium-carrier": 1, "hydrogen-cation": 1},
    "R2": {"sodium-cation": -1, "hbta": -1, "topo": -1, "sodium-carrier": 1, "hydrogen-cation": 1},
    "R3": {"lithium-cation": -1, "chloride-anion": -1, "topo": -1, "lithium-chloride-topo": 1},
    "R5": {"hbta": -1, "hydrogen-cation": 1, "bta-anion": 1},
}
SETS = {"A": ("R1", "R2"), "B": ("R1", "R2", "R3"), "C": ("R1", "R2", "R5"), "D": ("R1", "R2", "R3", "R5")}  # + RW
ENTHALPY_J_MOL = {"R1": -5730.0, "R3": -5730.0}  # Zhang et al. (2020) Section 3.11; R2, R5: 0 (no source)
REFERENCE_T = 298.15
GAS_CONSTANT = 8.314462618
SEGMENTS = {"hbta": 7.39, "topo": 13.21, "lithium-chloride-topo": 13.21}  # retained t3 - t1 and t2 - t1 increments
SEEDS = (0.05, 0.3, 1.0e-3, 0.9)
PH_TOLERANCE = 1.0e-7


@dataclass(frozen=True)
class State:
    """One contact per litre of aqueous phase. base > 0 is NaOH entering as NaA.T; base < 0 is HCl."""

    identity: str
    temperature_k: float
    organic_to_aqueous: float
    li: float  # aqueous feed, mol/L
    na: float
    hbta: float  # organic, mol/L of organic
    topo: float
    base: float = 0.0  # mol per L of aqueous
    hydrogen: float = 1.0e-10  # aqueous H+ before base, mol/L
    li_org: float = 0.0  # loaded organic Li and Na chelates, mol/L of organic
    na_org: float = 0.0
    bta: float = 0.0  # aqueous BTA- entering, mol/L (a set C cascade stage downstream of another)


def payload() -> dict:
    """The retained carrier payload with HA, T, LiCl.T and BTA- added on the priors named in the module docstring."""
    document = carrier_payload(TOPOLOGIES["t1_mbta_topo"], "dodecane")
    by_id = {row["component_id"]: row for row in document["components"]}
    prior = {"source_id": "hbta-topo-independent-network", "domain_id": "hbta-topo-independent-network",
             "locator": "Lithium #158 separate-extractant prior: retained ligand-derived priors, topology increments"}
    template = by_id.pop("free-carrier")
    masses = {"hbta": HBTA_MOLAR_MASS, "topo": TOPO_MOLAR_MASS,
              "lithium-chloride-topo": TOPO_MOLAR_MASS + LI_MOLAR_MASS + CL_MOLAR_MASS}
    for name, mass in masses.items():
        record = deepcopy(template)
        _rename_component(record, "free-carrier", name)
        record["fixed"]["molar_mass"] = {"value": {"magnitude": mass, "unit": "kilogram / mole"}, "provenance": prior}
        next(c for c in record["coefficients"] if c["family"] == "segment_count")["value"]["magnitude"] = SEGMENTS[name]
        by_id[name] = record
    for name in ("lithium-carrier", "sodium-carrier"):  # MA.T keeps its retained record; mass without the proton
        metal = LI_MOLAR_MASS if name == "lithium-carrier" else NA_MOLAR_MASS
        by_id[name]["fixed"]["molar_mass"]["value"]["magnitude"] = HBTA_MOLAR_MASS - H_MOLAR_MASS + TOPO_MOLAR_MASS + metal
    anion = deepcopy(by_id["chloride-anion"])
    _rename_component(anion, "chloride-anion", "bta-anion")
    anion["fixed"]["molar_mass"] = {"value": {"magnitude": HBTA_MOLAR_MASS - H_MOLAR_MASS, "unit": "kilogram / mole"},
                                    "provenance": {**prior, "locator": "Lithium #158 provisional: chloride ion parameters"}}
    by_id["bta-anion"] = anion
    hydroxide = deepcopy(by_id["chloride-anion"])
    _rename_component(hydroxide, "chloride-anion", "hydroxide-anion")
    hydroxide["fixed"]["molar_mass"] = {"value": {"magnitude": 0.0170073, "unit": "kilogram / mole"},
                                        "provenance": {**prior, "locator": "Lithium #158 provisional: chloride ion parameters"}}
    by_id["hydroxide-anion"] = hydroxide
    document["components"] = [by_id[name] for name in SPECIES]
    pairs = {}
    for row in document["pairs"]:
        names = [("hbta" if n == "free-carrier" else n) for n in (row["component_id_a"], row["component_id_b"])]
        row = _pair(*names, row["coefficients"][0]["value"]["magnitude"], **{
            k: row["coefficients"][0]["provenance"][k] for k in ("source_id", "locator")})
        pairs[frozenset(names)] = row
    for key, row in list(pairs.items()):  # BTA- and OH- copy chloride's pairs
        if "chloride-anion" in key and len(key) == 2:
            other = next(iter(key - {"chloride-anion"}))
            for anion in ("bta-anion", "hydroxide-anion"):
                pairs.setdefault(frozenset((anion, other)), _pair(
                    anion, other, row["coefficients"][0]["value"]["magnitude"],
                    source_id="hbta-topo-independent-network", locator="Lithium #158 provisional: chloride pair"))
    for i, a in enumerate(SPECIES):
        for b in SPECIES[i + 1:]:
            if CHARGE.get(a, 0) * CHARGE.get(b, 0) <= 0:  # same-sign ion pairs are inactive and must be omitted
                pairs.setdefault(frozenset((a, b)), _pair(a, b, 0.0, source_id="hbta-topo-independent-network",
                                                          locator="zero cross-interaction prior"))
    document["pairs"] = list(pairs.values())
    return document


@lru_cache(maxsize=None)
def model(components: tuple[str, ...]) -> epcsaft.Mixture:
    return epcsaft.Mixture(epcsaft.Parameters.from_mapping(payload(), components=components))


@lru_cache(maxsize=None)
def offsets(components: tuple[str, ...], temperature_k: float) -> dict[str, float]:
    """ln(c_ref) + mu_res(infinite dilution) per species: ln a_i = ln c_i + mu_res_i - offset_i."""
    mixture, ids = model(components), components
    result = {}
    for solvent, members, ln_reference in (("water", AQUEOUS, None), ("diluent", ORGANIC, math.log(1000.0))):
        s = ids.index(solvent)
        pure = mixture.state(temperature_k, P=PRESSURE_PA, x=tuple(float(k == s) for k in range(len(ids))), phase="liquid")
        if solvent == "water":
            result["water"] = math.log(pure.molar_density) + pure.residual_chemical_potential_over_rt[s]
            ln_reference = math.log(pure.molar_density * 0.01801528)  # c/m at infinite dilution, kg/m3: molal basis
        for name in members:
            if name == solvent or name not in ids:
                continue
            i = ids.index(name)
            x = [0.0] * len(ids)
            x[s], x[i] = 1.0 - 1.0e-20, 1.0e-20
            trace = mixture.state(temperature_k, P=PRESSURE_PA, x=tuple(x), phase="liquid")
            result[name] = ln_reference + trace.residual_chemical_potential_over_rt[i]
    return result


def ln_k_at(reaction: str, ln_k_298: float, temperature_k: float) -> float:
    if reaction == "RW":  # Harned and Robinson (1940) molal Kw: pKw 14.167 at 293.15 K, 13.995 at 298.15 K
        t = temperature_k
        return math.log(10.0) * (-4470.99 / t + 6.0875 - 0.01706 * t)
    return ln_k_298 - ENTHALPY_J_MOL.get(reaction, 0.0) / GAS_CONSTANT * (1.0 / temperature_k - 1.0 / REFERENCE_T)


def inventory(state: State) -> tuple[dict[str, float], dict[str, float]]:
    """Aqueous and organic feed amounts (mol) for one litre of aqueous; chloride closes the aqueous charge."""
    v = state.organic_to_aqueous
    base, acid = max(state.base, 0.0), max(-state.base, 0.0)
    li_c, na_c = state.li_org * v, state.na_org * v
    # NaOH saponifies the extractant first (NaOH + HA + T -> NaA.T + H2O); only NaOH beyond its capacity stays free.
    saponified = min(base, max(min(state.hbta, state.topo) * v - li_c - na_c, 0.0))
    na_c += saponified
    free = base - saponified
    aqueous = {"water": 55.508, "lithium-cation": state.li, "sodium-cation": state.na + free,
               "hydrogen-cation": state.hydrogen + acid, "hydroxide-anion": free, "bta-anion": state.bta}
    aqueous["chloride-anion"] = state.li + state.na + state.hydrogen + acid - state.bta
    organic = {"hbta": state.hbta * v - li_c - na_c, "topo": state.topo * v - li_c - na_c,
               "lithium-carrier": li_c, "sodium-carrier": na_c}
    if min(organic["hbta"], organic["topo"]) < -1.0e-15 or (li_c + na_c > 0 and min(state.hbta, state.topo) == 0.0):
        raise ValueError(f"{state.identity}: organic metal inventory exceeds the extractant")
    organic = {k: max(value, 0.0) for k, value in organic.items()}
    return aqueous, organic


def network(state: State, reaction_set: str, feed: dict[str, float],
            with_water: bool = False) -> tuple[tuple[str, ...], list[tuple[str, dict]]]:
    """Present species and active reactions: closure of the positive feed under the set's reactions.

    Water autoprotolysis (RW) and OH- enter with free NaOH or when solve() finds the contact above pH 7. In acidic
    contacts OH- is below 1e-7 mol/L; carrying it there leaves unbuffered states stalled above the row tolerance.
    """
    water = with_water or feed.get("hydroxide-anion", 0.0) > 0.0
    reactions = [(name, REACTIONS[name]) for name in (("RW",) if water else ()) + SETS[reaction_set]]
    present = {name for name, amount in feed.items() if amount > 0.0}
    changed = True
    while changed:
        changed = False
        for _, nu in reactions:
            for sign in (1, -1):
                if all(s in present for s, c in nu.items() if c * sign < 0):
                    new = {s for s, c in nu.items() if c * sign > 0} - present
                    present |= new
                    changed |= bool(new)
    active = [(n, nu) for n, nu in reactions if all(s in present for s in nu)]
    return tuple(s for s in SPECIES if s in present), active


def _seeded(feed_phases, active, fraction, species=SPECIES, aqueous=AQUEOUS):
    """Positive starting amounts: a small extent of each active reaction in whichever direction is feasible."""
    phases = [dict(p) for p in feed_phases]
    where = {s: 0 if s in aqueous else 1 for s in species}
    for _ in range(3):
        for _, nu in active:
            for sign in (1, -1):
                reactants = [s for s, c in nu.items() if c * sign < 0]
                if all(phases[where[s]].get(s, 0.0) > 0.0 for s in reactants):
                    extent = fraction * min(phases[where[s]][s] / abs(nu[s]) for s in reactants)
                    if "water" in nu:  # water autoprotolysis: a trace extent, not a fraction of 55 mol of water
                        extent = min(extent, 1.0e-7)
                    for s, c in nu.items():
                        phases[where[s]][s] = phases[where[s]].get(s, 0.0) + sign * c * extent
                    break
    return phases


def _certificate(state, mixture, components, problem, compiled, result, active, ln_k_engine,
                 aqueous=AQUEOUS, moieties=MOIETIES, charge=CHARGE):
    count = len(components)
    amounts, potentials, states = [], {}, []
    for p in range(2):
        x = tuple(result.mole_fractions[p * count:(p + 1) * count])
        phase_state = mixture.state(state.temperature_k, rho=result.molar_densities[p], x=x)
        states.append(phase_state)
        amounts.append({name: result.amounts[p] * x[i] for i, name in enumerate(components)})
        for i, name in enumerate(components):
            if (name in aqueous) == (p == 0):
                potentials[name] = math.log(x[i] * phase_state.molar_density) + phase_state.residual_chemical_potential_over_rt[i]
    feed = problem.feed.values
    point = list(result.coordinates[:compiled.primal_size])
    gradient, jacobian = compiled.objective_gradient(point), compiled.jacobian(point)
    rows = [i for i, row in enumerate(compiled.rows) if str(row.kind).split(".")[-1] not in ("Chemical", "Pressure")]
    kkt = max(abs(gradient[j] + math.fsum(mu * jacobian[i * compiled.primal_size + j]
                                          for mu, i in zip(result.constraint_multipliers, rows, strict=True)))
              for j in range(compiled.primal_size))
    return amounts, potentials, {
        "balance_inf_norm": max(abs(math.fsum(w * (amounts[0][s] + amounts[1][s] - feed.get(s, 0.0))
                                              for s, w in row.items() if s in components)) for row in moieties.values()),
        "charge_inf_norm": max(abs(math.fsum(charge.get(s, 0) * a for s, a in phase.items())) for phase in amounts),
        "absence_inf_norm": max(abs(amounts[p][s]) for p in range(2) for s in components if (s in aqueous) != (p == 0)),
        "reaction_affinity_inf_norm": max(abs(math.fsum(c * potentials[s] for s, c in nu.items()) - ln_k_engine[n])
                                          for n, nu in active),
        "kkt_stationarity_inf_norm": kkt,
        "pressure_relative_inf_norm": max(abs(s.pressure / PRESSURE_PA - 1.0) for s in states),
        "reduced_hessian_min_eigenvalue": result.reduced_hessian_minimum,
        "direction_rank": count - compiled.rank,
    }


def is_strict(result, check: dict, active) -> bool:
    """The strict-state criteria on an Engine result and its _certificate.

    Curvature is the Engine's (EqID reduced_hessian_stability, ALG-EQUILIBRIUM): a result has a state only when the
    equilibrated reduced Hessian exceeds its rounding band. The raw minimum is a diagnostic whose sign is noise on
    trace states (#220)."""
    return (result.requested_tolerance_met and result.topology_event.name == "None_"
            and result.jacobian_rank == result.unknown_count and check["direction_rank"] == len(active)
            and check["balance_inf_norm"] <= 1e-9 and check["charge_inf_norm"] <= 1e-9
            and check["absence_inf_norm"] <= 1e-14 and check["reaction_affinity_inf_norm"] <= 1e-6
            and check["kkt_stationarity_inf_norm"] <= 1e-7
            and check["pressure_relative_inf_norm"] <= PRESSURE_RELATIVE_LIMIT)


def solve(state: State, reaction_set: str, ln_k: dict[str, float], guess=None) -> dict:
    """_solve_once without OH- first. If that fails, or the contact is above pH 7, where OH- is not negligible,
    the retained state carries water autoprotolysis (warm-started from the first state when there is one)."""
    try:
        result = _solve_once(state, reaction_set, ln_k, guess)
    except RuntimeError:
        return _solve_once(state, reaction_set, ln_k, None, with_water=True)
    if "RW" in result["active_reactions"] or result["ph"] is None or result["ph"] <= 7.0:
        return result
    aqueous, organic = result["phase_amounts_mol"]
    start = [dict(aqueous, **{"hydroxide-anion": 10.0 ** (result["ph"] - 14.0)}), dict(organic)]
    try:
        return _solve_once(state, reaction_set, ln_k, start, with_water=True)
    except RuntimeError:
        return _solve_once(state, reaction_set, ln_k, None, with_water=True)


def _solve_once(state: State, reaction_set: str, ln_k: dict[str, float], guess=None, with_water: bool = False) -> dict:
    """Strict Engine state or a RuntimeError; the first strict start is kept.

    guess: the phase amounts of a neighbouring strict state, tried first as the composition guess. Weakly
    buffered states near pH 5-9 with trace H+, OH- and chelate stall above the 1e-10 row tolerance from the
    reaction-extent starts; a neighbouring state's composition starts them on the right scale.
    """
    aqueous, organic = inventory(state)
    components, active = network(state, reaction_set, {**aqueous, **organic, "diluent": 1.0}, with_water)
    mixture = model(components)
    offset = offsets(components, state.temperature_k)
    ln_k_engine = {}
    for name, nu in active:
        ln_k_engine[name] = ln_k_at(name, ln_k.get(name, 0.0), state.temperature_k) + math.fsum(
            c * offset[s] for s, c in nu.items())
    diluent = mixture.state(state.temperature_k, P=PRESSURE_PA, x=tuple(float(s == "diluent") for s in components),
                            phase="liquid").molar_density * state.organic_to_aqueous * 1.0e-3
    organic = {"diluent": diluent, **organic}
    amounts, potentials, check, _, _, _, fraction, elapsed = solve_declared(
        state, mixture, components, active, offset, ln_k_engine, (aqueous, organic), guess)
    aq, org = amounts
    v = state.organic_to_aqueous
    li_org = org.get("lithium-carrier", 0.0) + org.get("lithium-chloride-topo", 0.0)
    ph = None
    if "hydrogen-cation" in components:
        ph = -(potentials["hydrogen-cation"] - offset["hydrogen-cation"]) / math.log(10.0)
    return {
        "identity": state.identity, "reaction_set": reaction_set, "status": "evaluated",
        "active_reactions": [n for n, _ in active], "seed": fraction,
        "wall_time_seconds": elapsed,
        "li_distribution": (li_org / v) / aq["lithium-cation"],
        "na_distribution": (org.get("sodium-carrier", 0.0) / v) / aq["sodium-cation"] if aq.get("sodium-cation") else None,
        "li_extraction_percent": 100.0 * li_org / (li_org + aq["lithium-cation"]),
        "neutral_li_fraction": org.get("lithium-chloride-topo", 0.0) / li_org if li_org > 0 else 0.0,
        "aqueous_bta_mol": aq.get("bta-anion", 0.0),
        "hbta_lost_to_aqueous_fraction": aq.get("bta-anion", 0.0) / (state.hbta * v) if state.hbta > 0 else 0.0,
        "ph": ph, "phase_amounts_mol": amounts, **check,
    }


def solve_declared(state, mixture, components, active, offset, ln_k_engine, unseeded, guess=None,
                   aqueous_members=AQUEOUS, moieties=MOIETIES, charge=CHARGE, row_tolerance=1e-10):
    """The common strict configured-equilibrium procedure; chemistry stays with the caller."""
    failures = []
    feed = {s: unseeded[0].get(s, 0.0) + unseeded[1].get(s, 0.0) for s in components}
    for fraction in (("guess",) if guess else ()) + SEEDS:
        started = time.perf_counter()
        aq, org = guess if fraction == "guess" else _seeded(unseeded, active, fraction, components, aqueous_members)
        phases = []
        for name, amounts, members in (("aqueous", aq, aqueous_members), ("organic", org, tuple(s for s in components if s not in aqueous_members))):
            support = tuple(s for s in components if s in members)
            total = math.fsum(amounts.get(s, 0.0) for s in support)
            phases.append(equilibrium.Phase(name, kind="liquid", support=support, amount=equilibrium.Free(),
                                            composition_guess=[max(amounts.get(s, 0.0), 1e-30) / total for s in support]))
        problem = equilibrium.Problem(
            T=state.temperature_k, P=PRESSURE_PA, feed=equilibrium.Amounts(feed), phases=phases, row_tolerance=row_tolerance,
            reactions=[equilibrium.Reaction(nu, equilibrium_constant=math.exp(ln_k_engine[n])) for n, nu in active])
        try:
            compiled = equilibrium.compile_problem(mixture, problem)
            result = equilibrium.solve_equilibrium(mixture, problem)
        except Exception as error:  # noqa: BLE001 - a failed start is recorded and the next start tried
            failures.append(f"{fraction}: {type(error).__name__}: {error}")
            continue
        if not result.success:
            failures.append(f"{fraction}: {result.solver_status} {result.message}")
            continue
        try:
            amounts, potentials, check = _certificate(state, mixture, components, problem, compiled, result, active, ln_k_engine,
                                                          aqueous_members, moieties, charge)
        except epcsaft.ParameterError as error:
            if not str(error).startswith('non-electroneutral composition: charge sum '):
                raise
            failures.append(f"{fraction}: {type(error).__name__}: {error}")
            continue
        if not is_strict(result, check, active):
            failures.append(f"{fraction}: not strict {check}")
            continue
        return amounts, potentials, check, compiled, result, problem, fraction, time.perf_counter() - started
    raise RuntimeError(f"{state.identity} (declared reactions): no strict start: {failures}")


_WARM: dict[tuple[State, str, float], float] = {}


def solve_at_ph(state: State, reaction_set: str, ln_k: dict[str, float], target_ph: float) -> dict:
    """Strict state at the net NaOH or HCl (u: base = 1e-7 sinh u, up to 1 mol/L) whose equilibrium pH equals target_ph.

    Marches from a strict start toward the target, stepping over non-strict points (an unbuffered contact near
    pH 5.5-7.5 with trace H+, OH- and chelate can stall above the row tolerance), then bisects between the
    nearest strict points on either side, each solve starting from its strict neighbour's composition. If no
    strict state lies between them (a pH jump, e.g. at the equivalence point of a carrier-limited contact),
    the nearer strict state is kept and its pH flagged.
    """
    low, high = -math.asinh(1.0 / 1.0e-7), math.asinh(1.0 / 1.0e-7)
    memo = {}

    def evaluate(u):
        if u not in memo:
            strict = [v for v, r in memo.items() if r is not None]
            guess = memo[min(strict, key=lambda v: abs(v - u))]["phase_amounts_mol"] if strict else None
            try:
                memo[u] = solve(replace(state, base=1.0e-7 * math.sinh(u)), reaction_set, ln_k, guess=guess)
            except RuntimeError:
                memo[u] = None
        return memo[u]

    def done(u, unreachable):
        result = memo[u]
        _WARM[key] = u
        return {**result, "net_base_mol": 1.0e-7 * math.sinh(u),
                "ph_unreachable": unreachable or abs(result["ph"] - target_ph) > 1e-4}

    key = (replace(state, identity=""), reaction_set, target_ph)  # the physical state, shared by duplicate rows
    warm = key in _WARM
    starts = tuple(min(max(u, low), high) for u in ((_WARM[key],) if warm else ()) + (0.0, 7.6, -7.6, 10.0, -10.0, 12.6, -12.6, 5.0, -5.0))
    a = next((u for u in starts if evaluate(u) is not None), None)
    if a is None:
        raise RuntimeError(f"{state.identity} ({reaction_set}): no strict start for the pH search")
    fa = memo[a]["ph"] - target_ph
    direction, step, u = (1.0 if fa < 0 else -1.0), (0.05 if warm else 0.5), a
    while True:
        u = min(max(u + direction * step, low), high)
        if evaluate(u) is not None:
            if (memo[u]["ph"] - target_ph < 0) != (fa < 0):
                b = u
                break
            a, fa = u, memo[u]["ph"] - target_ph
        if u in (low, high):
            return done(a, True)
        step = min(step * 1.5, 1.0)  # scan, never leap over a strict region between non-strict points
    stalled, side = 0, None
    while abs(b - a) > 1e-10 and abs(memo[a]["ph"] - target_ph) > 1e-9:
        fb = memo[b]["ph"] - target_ph
        secant = min(max(fa / (fa - fb), 0.02), 0.98) if stalled < 2 else 0.5  # regula falsi, bisect when one end sticks
        middle = next((m for f in (secant, 0.5, 0.25, 0.75) if evaluate(m := a + f * (b - a)) is not None), None)
        if middle is None:
            break
        if (memo[middle]["ph"] - target_ph < 0) == (fa < 0):
            a, fa, stalled = middle, memo[middle]["ph"] - target_ph, (stalled + 1 if side == "b" else 0)
            side = "b"
        else:
            b, stalled = middle, (stalled + 1 if side == "a" else 0)
            side = "a"
    nearest = min((a, b), key=lambda v: abs(memo[v]["ph"] - target_ph))
    return done(nearest, False)


# ----- Zhang (2017) conditions -----
FIG1_LI = 0.138 / LI_G_MOL
FIGURE1 = State("figure1", 293.15, 1.0, FIG1_LI, 1.0e-6, 0.05, 0.10)
BRINE = State("brine_oa_1.5", 298.15, 1.5, 1.581 / LI_G_MOL, 2.263, 0.3, 0.3)
SECTION_3_6_PH, SECTION_3_4_PH = 8.0, 9.5
TABLE4 = ((0.0, 0.26, 82.8, 7.55), (0.5, 0.14, 90.7, 7.82), (0.6, 0.13, 91.4, 8.02), (0.7, 0.12, 92.1, 8.24),
          (0.8, 0.13, 91.4, 8.48), (0.9, 0.14, 90.7, 8.77), (1.0, 0.14, 90.7, 8.89))
TABLE3 = (  # scrub solution HCl mol/L, equilibrium pH, Li org g/L, Li scrubbed %, Na org g/L, Na scrubbed % (p. 171)
    (0.0, 8.66, 1.50, 0.66, 0.79, 2.47), (0.01, 8.50, 1.50, 0.66, 0.75, 7.41), (0.015, 8.42, 1.48, 1.99, 0.72, 11.11),
    (0.04, 8.20, 1.47, 2.65, 0.68, 16.05), (0.1, 7.54, 1.44, 4.64, 0.42, 48.15), (0.3, 4.90, 1.42, 5.96, 0.23, 71.60),
    (0.5, 4.68, 1.39, 7.95, 0.02, 98.67))
LOADED_LI, LOADED_NA = 1.51 / LI_G_MOL, 0.81 / NA_G_MOL  # Section 3.6 loaded organic, g/L -> mol/L
FIGURE5_OA = (10.0, 5.0, 3.0, 2.0, 1.5, 1.0, 2.0 / 3.0, 0.5)
MS2_LI, MS2_NA, MS2_PH = 0.168 / LI_G_MOL, 64.1 / NA_G_MOL, 5.77


def _digitized(figure, series):
    with (DIGITIZED / figure / "points.csv").open(newline="") as stream:
        return [(float(r["x"]), float(r["y"])) for r in csv.DictReader(stream) if r["series"] == series]


def figure1():
    return list(zip(_digitized("zhang2017_figure1", "extraction_ratio"),
                    _digitized("zhang2017_figure1", "equilibrium_ph"), strict=True))


def observations():
    """(role, case, state, target pH or None, observed D_Li, observed D_Na or None, x) for every Zhang row."""
    rows = []
    f1 = figure1()
    for k, ((initial, e), (_, ph)) in enumerate(f1):
        role = "fit" if k in (0, len(f1) - 1) else "holdout_gating"
        rows.append((role, "figure1_ph", replace(FIGURE1, identity=f"figure1_initial_{initial:.2f}"), ph, e / (100 - e), None, ph))
    rows.append(("fit", "figure5_oa_1.5", BRINE, SECTION_3_6_PH, 0.96144 / 0.13877, 0.016, 1.5))
    rows.append(("fit", "figure2_topo_alone", replace(FIGURE1, identity="figure2_hbta_0.0", hbta=0.0, topo=0.15),
                 None, 0.08045, None, 0.0))
    for regeneration, raffinate, e, ph in TABLE4:
        feed = raffinate / (1 - e / 100) / LI_G_MOL
        rows.append(("holdout_gating", "table4_ph", replace(BRINE, identity=f"table4_r{regeneration:.1f}", li=feed), ph,
                     e / (100 - e) / 1.5, None, ph))
    for (temperature, e), _ in zip(_digitized("zhang2017_figure4", "extraction_ratio"), _digitized("zhang2017_figure4", "d_li")):
        rows.append(("holdout_gating", "figure4_temperature", replace(FIGURE1, identity=f"figure4_{temperature:.0f}k",
                     temperature_k=round(temperature, 2)), SECTION_3_4_PH, e / (100 - e), None, round(temperature, 2)))
    points = dict(_digitized("zhang2017_figure5", "equilibrium"))
    for oa, (aqueous, organic) in zip(FIGURE5_OA, sorted(points.items())):
        if oa != 1.5:
            # Same brine at every O/A: its alkalinity is the base the O/A 1.5 state needs at pH 8.0, solved forward.
            rows.append(("holdout_reported", "figure5_isotherm", replace(BRINE, identity=f"figure5_oa_{oa:.3f}", organic_to_aqueous=oa),
                         None, organic / aqueous, None, oa))
    for fraction, d in _digitized("zhang2017_figure2", "d_li_mixture"):
        if 0.05 < fraction < 0.95:
            rows.append(("holdout_reported", "figure2_ratio", replace(FIGURE1, identity=f"figure2_hbta_{fraction:.1f}",
                         hbta=0.15 * fraction, topo=0.15 * (1 - fraction)), SECTION_3_4_PH, d, None, round(fraction, 1)))
    for series, varied in (("log_hbta", "hbta"), ("log_topo", "topo")):
        for log_c, log_d in _digitized("zhang2017_figure3", series):
            rows.append(("holdout_reported", f"figure3_{series}", replace(FIGURE1, identity=f"figure3_{series}_{log_c:.2f}",
                         **{"hbta": 0.3, "topo": 0.3, varied: 10**log_c}), SECTION_3_4_PH, 10**log_d, None, log_c))
    for hcl, ph, li_org, li_scrub, na_org, na_scrub in TABLE3:
        state = State(f"table3_hcl_{hcl:g}", 298.15, 10.0, 0.0, 0.0, 0.3, 0.3, hydrogen=max(hcl, 1e-7),
                      li_org=LOADED_LI, na_org=LOADED_NA)
        rows.append(("holdout_reported", "table3_scrub_forward", state, None,
                     li_org / (1.51 * li_scrub / 100 * 10.0), (na_org / (0.81 * na_scrub / 100 * 10.0)), ph))
    return rows


def solve_forward(state: State, reaction_set: str, ln_k: dict[str, float]) -> dict:
    """solve(), and if no start is strict, continuation in base from a strict neighbouring contact.

    Contacts with trace H+ and OH- (fresh organic on an unbuffered brine, a fully saponified organic) can stall just
    above the row tolerance from every reaction-extent start; the neighbour's composition starts them on scale.
    """
    try:
        return solve(state, reaction_set, ln_k)
    except RuntimeError as error:
        failure = error
    for delta in (1e-6, -1e-6, 1e-5, -1e-5, 1e-4, -1e-4):
        try:
            guess = solve(replace(state, base=state.base + delta), reaction_set, ln_k)["phase_amounts_mol"]
        except (RuntimeError, ValueError):
            continue
        for fraction in (0.5, 0.1, 0.0):
            try:
                result = solve(replace(state, base=state.base + fraction * delta), reaction_set, ln_k, guess=guess)
            except RuntimeError:
                break
            guess = result["phase_amounts_mol"]
        else:
            return result
    raise failure


def evaluate(row, reaction_set, ln_k):
    role, case, state, target, observed, observed_na, x = row
    if case == "figure2_topo_alone" and "R3" not in SETS[reaction_set]:
        return None  # no route without HBTA: D = 0 is not a finite log residual
    if case == "figure5_isotherm":
        alkalinity = solve_at_ph(BRINE, reaction_set, ln_k, SECTION_3_6_PH)["net_base_mol"]
        state = replace(state, base=alkalinity)
    return solve_forward(state, reaction_set, ln_k) if target is None else solve_at_ph(state, reaction_set, ln_k, target)


def _residuals(rows, reaction_set, ln_k):
    out = []
    for row in rows:
        result = evaluate(row, reaction_set, ln_k)
        if result is None:
            continue
        miss = abs(result["ph"] - row[3]) if result.get("ph_unreachable") else 0.0
        out.append(math.log10(result["li_distribution"] / row[4]) + math.copysign(miss, math.log10(result["li_distribution"] / row[4])))
        if row[5] is not None:
            ratio = math.log10(result["na_distribution"] / row[5])
            out.append(ratio + math.copysign(miss, ratio))
    return out


STARTS = {  # ln K starting points (R1, R2[, R3][, R5]) spanning the ideal-solution triage minima
    "A": ((-3.2, -6.9), (-7.0, -11.5), (0.0, -4.6)),
    "B": ((-3.2, -6.9, 3.5), (-15.9, -18.9, 3.6), (-4.6, -8.1, 2.3)),
    "C": ((-3.2, -8.4, -12.1), (-3.2, -6.9, -18.4), (-4.6, -9.2, -15.0)),
    "D": ((-3.2, -8.4, 2.3, -12.1), (-4.6, -8.1, 2.3, -15.0), (-15.9, -18.9, 3.6, -20.7)),
}


def fit(reaction_set, rows, log):
    names = SETS[reaction_set]
    best = None
    for start in STARTS[reaction_set]:
        def residual(q):
            try:
                return _residuals(rows, reaction_set, dict(zip(names, q)))
            except RuntimeError as error:
                print(f"  {reaction_set} residual failure at {q}: {str(error)[:200]}", file=log, flush=True)
                return [10.0] * (len(rows) + 1 - ("R3" not in names))
        started = time.perf_counter()
        solution = least_squares(residual, start, diff_step=1e-4, xtol=1e-10, ftol=1e-12, gtol=1e-12, max_nfev=200)
        print(f"{reaction_set} start {start}: ln K {list(solution.x)} cost {solution.cost:.6g} "
              f"({solution.nfev} evaluations, {time.perf_counter() - started:.0f} s)", file=log, flush=True)
        if best is None or solution.cost < best.cost:
            best = solution
    return dict(zip(names, best.x)), list(best.fun), [list(map(float, s)) for s in STARTS[reaction_set]], float(best.cost)


def _slope(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / sum((a - mx) ** 2 for a in xs)


def ms2_rows(reaction_set, ln_k):
    rows = []
    for oa, c in ((1.0, 0.015), (1.5, 0.030)):
        base = State(f"ms2_oa_{oa}_{c}", 298.15, oa, MS2_LI, MS2_NA, c, c, hydrogen=10**-MS2_PH)
        cases = [("fresh", replace(base, identity=base.identity + "_fresh"), None),
                 ("saponified_0.7", replace(base, identity=base.identity + "_r0.7", base=0.7 * c * oa), None),
                 ("saponified_1.0", replace(base, identity=base.identity + "_r1.0", base=c * oa), None),
                 ("ph_held_5.77", replace(base, identity=base.identity + "_ph"), MS2_PH)]
        for organic, state, target in cases:
            try:
                result = (solve_forward(state, reaction_set, ln_k) if target is None
                          else solve_at_ph(state, reaction_set, ln_k, target))
            except RuntimeError as error:  # retained as a failed state, never reconstructed
                rows.append({"reaction_set": reaction_set, "organic": organic, "organic_to_aqueous": oa,
                             "extractant_mol_l": c, "status": "no_strict_state", "failure": str(error)[:300]})
                continue
            rows.append({"reaction_set": reaction_set, "organic": organic, "organic_to_aqueous": oa, "extractant_mol_l": c,
                         "li_extraction_percent": result["li_extraction_percent"], "li_distribution": result["li_distribution"],
                         "na_distribution": result["na_distribution"], "equilibrium_ph": result["ph"],
                         "ph_unreachable": result.get("ph_unreachable", False),
                         "net_base_mol_per_l_aqueous": result.get("net_base_mol", state.base),
                         "neutral_li_fraction": result["neutral_li_fraction"],
                         "hbta_lost_to_aqueous_fraction": result["hbta_lost_to_aqueous_fraction"], "status": result["status"],
                         "reaction_affinity_inf_norm": result["reaction_affinity_inf_norm"],
                         "kkt_stationarity_inf_norm": result["kkt_stationarity_inf_norm"],
                         "balance_inf_norm": result["balance_inf_norm"]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-wheel", required=True, type=Path)
    parser.add_argument("--engine-sha256", required=True)
    parser.add_argument("--engine-commit", required=True)
    parser.add_argument("--sets", default="ABCD")
    parser.add_argument("--log", type=Path, default=Path("/dev/stdout"))
    args = parser.parse_args()
    engine = _verify_wheel(args.engine_wheel, args.engine_sha256, args.engine_commit)
    rows = observations()
    fit_rows = [r for r in rows if r[0] == "fit"]
    table, ms2, summary = [], [], {}
    with args.log.open("a") as log:
        for reaction_set in args.sets:
            ln_k, residual, starts, cost = fit(reaction_set, fit_rows, log)
            print(f"{reaction_set}: fitted ln K {ln_k}", file=log, flush=True)
            for row in rows:
                role, case, state, target, observed, observed_na, x = row
                try:
                    result = evaluate(row, reaction_set, ln_k)
                except RuntimeError as error:  # retained as a failed holdout state, never reconstructed
                    table.append({"reaction_set": reaction_set, "role": role, "case": case, "identity": state.identity,
                                  "x": x, "observed_d_li": observed, "status": "no_strict_state",
                                  "failure": str(error)[:300]})
                    print(f"  {reaction_set} {state.identity}: FAILED {str(error)[:200]}", file=log, flush=True)
                    continue
                if result is None:
                    table.append({"reaction_set": reaction_set, "role": role, "case": case, "identity": state.identity,
                                  "x": x, "observed_d_li": observed, "model_d_li": 0.0, "status": "not_representable"})
                    continue
                table.append({
                    "reaction_set": reaction_set, "role": role, "case": case, "identity": state.identity, "x": x,
                    "temperature_k": state.temperature_k, "organic_to_aqueous": state.organic_to_aqueous,
                    "hbta_mol_l": state.hbta, "topo_mol_l": state.topo, "target_ph": target,
                    "model_ph": result["ph"], "ph_unreachable": result.get("ph_unreachable", False),
                    "net_base_mol": result.get("net_base_mol", state.base),
                    "observed_d_li": observed, "model_d_li": result["li_distribution"],
                    "log10_model_over_observed_d_li": math.log10(result["li_distribution"] / observed),
                    "observed_d_na": observed_na, "model_d_na": result["na_distribution"],
                    "model_li_extraction_percent": result["li_extraction_percent"],
                    "neutral_li_fraction": result["neutral_li_fraction"], "aqueous_bta_mol": result["aqueous_bta_mol"],
                    "hbta_lost_to_aqueous_fraction": result["hbta_lost_to_aqueous_fraction"],
                    "status": result["status"], "balance_inf_norm": result["balance_inf_norm"],
                    "charge_inf_norm": result["charge_inf_norm"],
                    "reaction_affinity_inf_norm": result["reaction_affinity_inf_norm"],
                    "kkt_stationarity_inf_norm": result["kkt_stationarity_inf_norm"],
                    "reduced_hessian_min_eigenvalue": result["reduced_hessian_min_eigenvalue"],
                })
                print(f"  {reaction_set} {state.identity}: D {result['li_distribution']:.4g} vs {observed:.4g}", file=log, flush=True)
            ms2 += ms2_rows(reaction_set, ln_k)
            mine = [r for r in table if r["reaction_set"] == reaction_set and r["status"] == "evaluated"]
            gating = [r for r in mine if r["case"] in ("figure1_ph", "table4_ph") and r["role"] == "holdout_gating"]
            fig1 = {round(r["x"], 2): r for r in mine if r["case"] == "figure1_ph"}
            fig4 = sorted((r for r in mine if r["case"] == "figure4_temperature"), key=lambda r: r["x"])
            slope = math.log10(fig1[8.19]["model_d_li"] / fig1[3.07]["model_d_li"]) / (fig1[8.19]["model_ph"] - fig1[3.07]["model_ph"])
            worst = max(abs(r["log10_model_over_observed_d_li"]) for r in gating)
            fig4_worst = max(abs(r["log10_model_over_observed_d_li"]) for r in fig4)
            falls = all(b["model_d_li"] < a["model_d_li"] for a, b in zip(fig4, fig4[1:]))
            strict = all(r["status"] == "evaluated" for r in table if r["reaction_set"] == reaction_set and r["status"] != "not_representable")
            failed = [r["identity"] for r in table if r["reaction_set"] == reaction_set and r["status"] == "no_strict_state"]
            # A gating state that did not reach its printed pH is scored at the wrong pH: it fails criterion 4.
            unreached = [r["identity"] for r in mine if r["ph_unreachable"]
                         and r["case"] in ("figure1_ph", "table4_ph", "figure4_temperature")]
            slope_reached = not (fig1[3.07]["ph_unreachable"] or fig1[8.19]["ph_unreachable"])
            fig3 = {series: _slope([r["x"] for r in mine if r["case"] == f"figure3_{series}"],
                                   [math.log10(r["model_d_li"]) for r in mine if r["case"] == f"figure3_{series}"])
                    for series in ("log_hbta", "log_topo")}
            summary[reaction_set] = {
                "ln_k_298": ln_k, "log10_k_298": {k: v / math.log(10) for k, v in ln_k.items()},
                "fit_residuals_log10": residual, "fit_cost": cost, "starts_ln_k": starts,
                "worst_abs_log10_d_holdout_figure1_table4": worst,
                "slope_log10_d_per_ph_3.07_to_8.19": slope,
                "figure1_interval_slopes": {f"{a}-{b}": math.log10(fig1[b]["model_d_li"] / fig1[a]["model_d_li"]) / (b - a)
                                            for a, b in ((2.83, 3.07), (3.07, 4.73), (4.73, 8.19), (8.19, 11.17))},
                "figure4_falls_with_temperature": falls, "figure4_worst_abs_log10_d": fig4_worst,
                "figure3_model_slopes": fig3,
                "figure5_worst_abs_log10_d": max(abs(r["log10_model_over_observed_d_li"]) for r in mine if r["case"] == "figure5_isotherm"),
                "figure2_worst_abs_log10_d": max(abs(r["log10_model_over_observed_d_li"]) for r in mine if r["case"] == "figure2_ratio"),
                "table3_worst_abs_log10_d": max(abs(r["log10_model_over_observed_d_li"]) for r in mine if r["case"] == "table3_scrub_forward"),
                "table3_failed_states": sum(r.startswith("table3") for r in failed),
                "all_states_strict": strict, "failed_states": failed, "gating_states_off_target_ph": unreached,
                "criteria": {"worst_le_0.30": worst <= 0.30,
                             "slope_in_0.15_0.44": slope_reached and 0.15 <= slope <= 0.44,
                             "temperature_falls_and_worst_le_0.30": falls and fig4_worst <= 0.30,
                             "strict_at_printed_ph": strict and not unreached},
            }
            summary[reaction_set]["passes"] = all(summary[reaction_set]["criteria"].values())
            print(json.dumps({reaction_set: summary[reaction_set]}, indent=1), file=log, flush=True)
    for name, records in (("mechanism_candidates.csv", table), ("mechanism_candidates_ms2.csv", ms2)):
        fields = list(dict.fromkeys(k for r in records for k in r))
        with (RESULTS / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    document = {
        "engine": engine, "issue": "tannerpolley/Lithium_Extraction#158",
        "design": "issue #158 comment 5885287577 (fit/holdout split and criteria fixed before fitting)",
        "enthalpy_j_mol": ENTHALPY_J_MOL, "reference_temperature_k": REFERENCE_T,
        "sets": {k: list(v) for k, v in SETS.items()}, "summary": summary,
        "observed_figure1_interval_slopes": {"2.83-3.07": 0.59, "3.07-4.73": 0.44, "4.73-8.19": 0.24, "8.19-11.17": 0.15},
        "sources": {
            "zhang_2017": "Sep. Purif. Technol. 188:167-173; Figs. 1-5, Tables 2-4, Sections 3.1-3.7",
            "zhang_2020": "J. Hazard. Mater. 398:122840 (Zotero G8GUHCG4), Section 3.11 dH = -5.73 kJ/mol",
        },
        "limits": (
            "Figures 1-4 print no buffer, base or Na level; states are solved at the printed (Fig. 1, Table 4) or stated "
            "(Sec. 3.4 pH 9.5; Sec. 3.6 pH 8.0 for the brine) equilibrium pH with NaOH entering as NaA.T or HCl. "
            "Figures 2-3 print no pH and Fig. 3 no partner concentration (0.3 M assumed). BTA-(aq) ion parameters are "
            "provisional (chloride's). dH is Zhang 2020's apparent value applied to R1 and R3 as a standard enthalpy. "
            "Table 3's loaded organic (Sec. 3.6: 1.51 g/L Li, 0.81 g/L Na) is represented as chelates. MS-2 chloride is "
            "charge-closed to Li + Na (2.81 M; measured 4.91 M) and Ca, Mg, K, Sr are absent. Model pH is the Engine's "
            "H+ molal-activity scale. Dodecane stands in for kerosene."
        ),
    }
    (RESULTS / "mechanism_candidates.json").write_text(json.dumps(document, indent=1, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
