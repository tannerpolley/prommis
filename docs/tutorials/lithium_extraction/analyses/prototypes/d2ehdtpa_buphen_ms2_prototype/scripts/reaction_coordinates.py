"""Raiguel equations 5–6 on Held 2008; configured LLE and certificate follow #158.

Amounts use one litre aqueous and O/A litres organic; D uses that declared volume basis.
Organic parameters are the owner-approved ligand prior, not measured properties.
"""

from copy import deepcopy
from functools import lru_cache
import json
import math

import epcsaft
from epcsaft import equilibrium

from analyses.shared.hbta_topo_configured_reactive_lle.scripts.mechanism_candidates import (
    payload as ligand_payload, is_strict, SEEDS,
)
from analyses.shared.hbta_topo_configured_reactive_lle.scripts.independent_network import _pair, _rename_component
from .aqueous_activities import INPUTS, ION, T, P, WATER_MW

ERROR_LABEL = "Held gamma: MgCl2 ~0.10 decade low, CaCl2 ~0.09 low; Mg mixture remainder 0.40–0.49 decade unresolved"
EVIDENCE = "calibration plus a partial holdout; not physical validation of MS-2"
HELD_BASIS = True  # Held 2008 aqueous basis: Born off, permittivity fixed at 78.09
COMPLEX = {"Li": "lib2a", "Mg": "mga2", "Ca": "caa2", "Sr": "sra2"}
REACTIONS = {e: {ION[e][0]: -1, "naa": -ION[e][2], ION["Na"][0]: ION[e][2], c: 1,
                 **({"buphen": -2} if e == "Li" else {})} for e, c in COMPLEX.items()}
MOIETIES = {e: {ION[e][0]: 1, **({COMPLEX[e]: 1} if e in COMPLEX else {}),
               **({"naa": 1} if e == "Na" else {})} for e in ION}
MOIETIES.update(water={"water": 1}, diluent={"diluent": 1},
                A={"naa": 1, "lib2a": 1, "mga2": 2, "caa2": 2, "sra2": 2}, B={"buphen": 1, "lib2a": 2})


@lru_cache(maxsize=1)
def payload():
    document = json.loads((INPUTS / "held_2008_parameters.json").read_text())
    prior = ligand_payload()
    document["sources"] += [r for r in prior["sources"] if r["source_id"] not in {r["source_id"] for r in document["sources"]}]
    document["domains"] += [r for r in prior["domains"] if r["domain_id"] not in {r["domain_id"] for r in document["domains"]}]
    template = next(r for r in prior["components"] if r["component_id"] == "hbta")
    basis = json.loads((INPUTS / "reaction_basis.json").read_text())
    scale = basis["organic_segment_scale_mol_kg"]
    provenance = {"source_id": "lithium-188", "domain_id": "hbta-topo-independent-network",
                  "locator": "Owner-approved organic prior scale and Raiguel molecular masses, Lithium #188"}
    document["sources"].append({"source_id": "lithium-188", "citation": "Raiguel et al., doi:10.1039/d4gc04760e; Lithium #188 owner-approved priors"})
    masses = basis["molar_masses_kg_mol"]
    a, b = .35456 - .00100794, .29242  # Raiguel D2EHDTPA and BuPhen, kg/mol; remove acid H from A.
    organic = {"buphen": b, "naa": a + masses["Na"], "lib2a": a + 2*b + masses["Li"],
               "mga2": 2*a + masses["Mg"], "caa2": 2*a + masses["Ca"], "sra2": 2*a + masses["Sr"]}
    for name, mass in organic.items():
        record = deepcopy(template)
        _rename_component(record, "hbta", name)
        record["coefficients"] = [c for c in record["coefficients"] if c["family"] != "solvation_factor"]  # Born is disabled.
        record["fixed"]["molar_mass"]["value"]["magnitude"] = mass
        record["fixed"]["molar_mass"]["provenance"] = provenance
        for coefficient in record["coefficients"]:
            if coefficient["family"] == "segment_count":
                coefficient["value"]["magnitude"] = scale*mass
                coefficient["provenance"] = provenance
        document["components"].append(record)
    diluent = deepcopy(next(r for r in prior["components"] if r["component_id"] == "diluent"))
    diluent["coefficients"] = [c for c in diluent["coefficients"] if c["family"] != "solvation_factor"]
    document["components"].append(diluent)
    names = [r["component_id"] for r in document["components"]]
    pairs = {frozenset((r["component_id_a"], r["component_id_b"])) for r in document["pairs"]}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if (left in organic or right in organic or "diluent" in (left, right)) and frozenset((left, right)) not in pairs:
                document["pairs"].append(_pair(left, right, 0, source_id="hbta-topo-independent-network", locator="Lithium #188 zero cross-interaction prior"))
    return document


@lru_cache(maxsize=None)
def model(names):
    mixture = epcsaft.Mixture(epcsaft.Parameters.from_mapping(payload(), components=names))
    offsets = {}
    for solvent, members in (("water", names[:names.index("diluent")]), ("diluent", names[names.index("diluent"):])):
        x = [float(s == solvent) for s in names]
        pure = mixture.state(T, P=P, x=x, phase="liquid")
        reference = math.log(pure.molar_density*WATER_MW) if solvent == "water" else math.log(1000)
        for name in members:
            if name == solvent:
                continue
            trace = x.copy()
            trace[names.index(name)] = 1e-20
            state = mixture.state(T, P=P, x=trace, phase="liquid")
            offsets[name] = reference + state.residual_chemical_potential_over_rt[names.index(name)]
    diluent = mixture.state(T, P=P, x=[float(s == "diluent") for s in names], phase="liquid").molar_density/1000
    return mixture, offsets, diluent


def aqueous_feed(feed):
    """The existing stage aqueous inventory, independent of its equilibrium solve."""
    aq = {"water": 55.508, **{ION[e][0]: n for e, n in feed.items() if n > 0}}
    aq[ION["Na"][0]] = feed.get("Na", 0)
    aq[ION["Cl"][0]] = math.fsum(ION[e][2]*n for e, n in feed.items())
    return aq


def build_problem(feed, sites, buphen, ln_k, *, organic_to_aqueous=1., loading=None, sr_decade=0., fraction=SEEDS[0]):
    """Source-owned problem assembly shared by direct solves and the generic stage adapter."""
    v, loading = organic_to_aqueous, loading or {}
    active = {e: nu for e, nu in REACTIONS.items() if feed.get(e, 0) + loading.get(e, 0) > 0}
    aq = aqueous_feed(feed)
    for e in active:
        aq.setdefault(ION[e][0], 0.)
    org = {"diluent": 1, "buphen": buphen - 2*loading.get("Li", 0),
           "naa": sites - sum(ION[e][2]*loading.get(e, 0) for e in active),
           **{COMPLEX[e]: loading.get(e, 0) for e in active}}
    if min(org.values()) < 0 or v <= 0:
        raise ValueError(f"Invalid entering organic inventory: {org}, O/A={v}")
    names = tuple(aq) + tuple(org)
    mixture, offsets, diluent = model(names)
    org["diluent"] = diluent
    org = {s: n*v for s, n in org.items()}
    inventory = {**aq, **org}
    constants = {e: float(ln_k[2 if e == "Sr" else ("Li", "Mg", "Ca").index(e)])
                 + (sr_decade*math.log(10) if e == "Sr" else 0)
                 + sum(c*offsets[s] for s, c in nu.items()) for e, nu in active.items()}
    guess = [aq.copy(), org.copy()]
    for e, nu in active.items():
        extent = fraction*min(guess[int(s in org)][s]/abs(c) for s, c in nu.items() if c < 0)
        for s, c in nu.items():
            guess[int(s in org)][s] += c*extent
    phases = [equilibrium.Phase(label, kind="liquid", support=tuple(amounts), amount=equilibrium.Free(),
                composition_guess=[max(n, 1e-30)/sum(amounts.values()) for n in amounts.values()])
              for label, amounts in zip(("aqueous", "organic"), guess)]
    # The Engine target -RT ln K, passed directly: with Born diameters at their crystal bounds (#213 packet) the
    # Engine-basis ln K exceeds 709, beyond the floating-point range of K itself. Equivalent at fixed T only: a scalar
    # K implies constant ln K, an affinity constant Delta G; no D2EHDTPA caller uses temperature derivatives.
    rt = epcsaft.GAS_CONSTANT_J_PER_MOL_K*T
    problem = equilibrium.Problem(T=T, P=P, feed=equilibrium.Amounts(inventory), phases=phases, row_tolerance=1e-10,
              reactions=[equilibrium.Reaction(nu, affinity_j_per_mol=-rt*constants[e]) for e, nu in active.items()])
    return mixture, problem, aq, org, names, inventory, active, constants


def solve(feed, sites, buphen, ln_k, *, organic_to_aqueous=1., loading=None, sr_decade=0.):
    v = organic_to_aqueous
    failures = []
    for fraction in SEEDS:
        mixture, problem, aq, org, names, inventory, active, constants = build_problem(
            feed, sites, buphen, ln_k, organic_to_aqueous=v, loading=loading,
            sr_decade=sr_decade, fraction=fraction)
        compiled = equilibrium.compile_problem(mixture, problem)
        result = equilibrium.solve_equilibrium(mixture, problem)
        if not result.success:
            failures.append(str(result.message))
            continue
        amounts, potentials, states = [], {}, []
        for p in range(2):
            x = result.mole_fractions[p*len(names):(p + 1)*len(names)]
            state = mixture.state(T, rho=result.molar_densities[p], x=x)
            states.append(state)
            amounts.append({s: result.amounts[p]*x[i] for i, s in enumerate(names)})
            for i, s in enumerate(names):
                if (s in aq) == (p == 0):
                    potentials[s] = math.log(x[i]*state.molar_density) + state.residual_chemical_potential_over_rt[i]
        point = list(result.coordinates[:compiled.primal_size])
        gradient, jacobian = compiled.objective_gradient(point), compiled.jacobian(point)
        rows = [i for i, row in enumerate(compiled.rows) if str(row.kind).split(".")[-1] not in ("Chemical", "Pressure")]
        check = {"balance_inf_norm": max(abs(sum(w*(amounts[0].get(s, 0) + amounts[1].get(s, 0) - inventory.get(s, 0)) for s, w in row.items())) for row in MOIETIES.values()),
                 "charge_inf_norm": max(abs(sum(ION[e][2]*phase.get(ION[e][0], 0) for e in ION)) for phase in amounts),
                 "absence_inf_norm": max(abs(amounts[p][s]) for p in range(2) for s in names if (s in aq) != (p == 0)),
                 "reaction_affinity_inf_norm": max(abs(sum(c*potentials[s] for s, c in nu.items()) - constants[e]) for e, nu in active.items()),
                 "kkt_stationarity_inf_norm": max(abs(gradient[j] + sum(mu*jacobian[i*compiled.primal_size + j] for mu, i in zip(result.constraint_multipliers, rows, strict=True))) for j in range(compiled.primal_size)),
                 "pressure_relative_inf_norm": max(abs(s.pressure/P - 1) for s in states),
                 "reduced_hessian_min_eigenvalue": result.reduced_hessian_minimum, "direction_rank": len(names) - compiled.rank}
        if not is_strict(result, check, active):
            failures.append(str(check))
            continue
        assert not HELD_BASIS or (states[0].born == 0 and abs(states[0].bulk_relative_permittivity - 78.09) <= math.ulp(78.09))
        d = {e: amounts[1][COMPLEX[e]]/v/amounts[0][ION[e][0]] for e in active}
        return {"D": d, "alpha": {e: d["Li"]/d[e] for e in active if e != "Li"}, "amounts_mol": amounts,
                "Na_mol_kg_water": amounts[0][ION["Na"][0]]/(55.508*WATER_MW), "certificate": check,
                "evidence_type": EVIDENCE, "activity_error_label": ERROR_LABEL}
    raise RuntimeError(f"No strict state: {failures}")
