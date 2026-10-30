"""DES-X/S conditional F/3 chemistry: #216 DES-0 and amendments 1–2."""
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from epcsaft import Mixture, Parameters
from analyses.shared.hbta_topo_configured_reactive_lle.scripts import mechanism_candidates as mc

STUDY = Path(__file__).resolve().parents[1]
INPUT = STUDY / "inputs/des0"
AQ = ("water", "lithium-cation", "sodium-cation", "chloride-anion", "hydrogen-cation",
      "hydroxide-anion", "ammonia", "ammonium-cation")
ORG = ("des", "topo", "lithium-des", "sodium-des")
IDS = AQ + ORG
CHARGE = dict(zip(AQ, (0, 1, 1, -1, 1, -1, 0, 1), strict=True))
MW = np.array([.01801528, .006941, .02298977, .035453, .00100794, .01700734, .01703052, .01803846])


@lru_cache(None)
def document(t, candidate, placeholder=0.):
    d = json.loads((INPUT / "aqueous_parameter_probe.json").read_text().replace("hydronium-cation", "hydrogen-cation"))
    o = json.loads((INPUT / "organic_parameter_probe.json").read_text().replace("des-h", "des"))
    d["sources"].append({"source_id": "esper-2023", "citation": "Esper et al. 2023", "doi": "10.1021/acs.iecr.3c02255",
                         "use_basis": "NH3 pure-fluid prior; authors FEOS collection CAS 7664-41-7"})
    for coefficient in d["components"][6]["coefficients"]:
        coefficient["provenance"] = {"source_id": "esper-2023", "domain_id": "sensitivity-only",
                                     "locator": "FEOS esper2023.json CAS 7664-41-7; no brine mixture fit"}
    for coefficient in d["components"][4]["coefficients"]:
        coefficient["provenance"]["locator"] = "Held2008 Table2 H3O+ residual template for formal H+; #216 DES-0"
    d["components"] += o["components"]
    prior = {"source_id": "issue-77-model-form", "domain_id": "sensitivity-only",
             "locator": "#216 DES-0 amendments R1/R6: F/3, nonassociating equal-DES complex prior"}
    for name, metal in zip(ORG[2:], MW[1:3], strict=True):
        c = deepcopy(o["components"][0])
        mc._rename_component(c, "des", name)
        c["fixed"]["molar_mass"] = {"value": {"magnitude": .20748558 + (metal + (-MW[4] if candidate == "X" else MW[3])) / 3,
                                             "unit": "kilogram / mole"}, "provenance": prior}
        for value in c["coefficients"]:
            value["provenance"] = prior
        d["components"].append(c)
    d["topology"]["sites"] += o["topology"]["sites"]
    d["topology"]["edges"] += o["topology"]["edges"]
    associating = ("water", "ammonia", "des", "topo")
    d["topology"]["no_cross_association"] = [
        {"component_id_a": a, "component_id_b": b, "provenance": prior}
        for i, a in enumerate(associating) for b in associating[i+1:] if {a, b} != {"water", "ammonia"}]
    for i, a in enumerate(ORG):
        for b in ORG[i+1:]:
            d["pairs"].append(mc._pair(a, b, 0., source_id=prior["source_id"], locator=prior["locator"]))
    supports = (set(AQ), set(ORG))
    placeholders = [(a, b) for a in IDS for b in IDS if IDS.index(a) < IDS.index(b)
                    and not any(a in support and b in support for support in supports)]
    for a, b in placeholders:
        assert not any(a in support and b in support for support in supports)
        d["pairs"].append(mc._pair(a, b, placeholder, source_id=prior["source_id"],
                                      locator="#216 construction placeholders; disjoint phase supports; inertness checked"))
    # The frozen probe used the 298 K water diameter in its unlike-association volume.
    sigma = 2.7927 + 10.11 * math.exp(-.01775*t) - 1.417 * math.exp(-.01146*t)
    next(v for v in d["components"][0]["coefficients"] if v["family"] == "segment_diameter")["value"]["magnitude"] = sigma
    for edge in d["topology"]["edges"]:
        names = {edge["endpoint_a"]["component_id"], edge["endpoint_b"]["component_id"]}
        if names == {"water", "ammonia"}:
            edge["volume"]["value"]["magnitude"] = math.sqrt(.0451*.0001)*(math.sqrt(sigma*2.3709)/((sigma+2.3709)/2))**3
    return d


@lru_cache(None)
def model(t, candidate, placeholder=0.):
    return Mixture(Parameters.from_mapping(document(t, candidate, placeholder)))


def phase_state(mix, t, amounts):
    n = np.array([amounts.get(s, 0.) for s in IDS])
    state = mix.state(t, P=mc.PRESSURE_PA, x=n/n.sum(), phase="liquid")
    if state.molar_density <= 0:
        raise RuntimeError("nonpositive fluid density")
    return state


@lru_cache(None)
def references(t, candidate):
    mix = model(t, candidate)
    offset = {}
    diagnostics = {}
    for name in ("water",) + ORG:
        state = phase_state(mix, t, {name: 1.})
        i = IDS.index(name)
        offset[name] = math.log(state.molar_density) + state.residual_chemical_potential_over_rt[i]
        step = state.molar_density * 1e-5
        pressures = [mix.state(t, rho=state.molar_density+sign*step, x=[float(s == name) for s in IDS]).pressure for sign in (-1, 1)]
        slope = (pressures[1]-pressures[0])/(2*step)
        if slope <= 0:
            raise RuntimeError(f"unstable pure-fluid reference: {name}")
        diagnostics[name] = {"molar_density": state.molar_density, "dP_drho": slope}
    pure = phase_state(mix, t, {"water": 1.})
    for name in AQ[1:]:
        values = []
        for trace in (1e-18, 1e-20):
            state = phase_state(mix, t, {"water": 1-trace, name: trace})
            values.append(math.log(pure.molar_density*MW[0]) + state.residual_chemical_potential_over_rt[IDS.index(name)])
        offset[name] = values[1]
        diagnostics[name] = {"trace_offset_difference": abs(values[1]-values[0])}
        if abs(values[1]-values[0]) > 1e-6:
            raise RuntimeError(f"trace reference unresolved: {name}")
    return offset, diagnostics


def ln_ka(t):
    return -math.log(10)*(2835.76/t-.6322+.001225*t)


def prepare_inventory(t, ph, inventory):
    """Invert activity pH for a supplied initial aqueous inventory, in mol."""
    mix = model(t, "X")
    offset, _ = references(t, "X")
    off = np.array([offset[s] for s in AQ])
    def evaluate(y):
        h, oh, am = np.exp(y)
        nh = oh-h+inventory["hydrogen-cation"]
        if nh <= 0:
            raise RuntimeError("preparation has nonpositive ammonium")
        n = np.array([inventory["water"]-oh, inventory[AQ[1]], inventory[AQ[2]], inventory[AQ[3]], h, oh, am, nh])
        state = phase_state(mix, t, dict(zip(AQ, n, strict=True)))
        activity = np.log(state.molar_density*n/n.sum()) + np.array(state.residual_chemical_potential_over_rt[:8])-off
        residual = np.array([activity[4]+ph*math.log(10), activity[4]+activity[5]-activity[0]-mc.ln_k_at("RW", 0., t),
                             activity[4]+activity[6]-activity[7]-ln_ka(t)])
        return n, state, residual
    h, oh = 10**-ph, math.exp(mc.ln_k_at("RW", 0., t))*10**ph
    fit = least_squares(lambda y: evaluate(y)[2], np.log([h, oh, math.exp(ln_ka(t))*(oh-h+inventory["hydrogen-cation"])/h]),
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=100)
    n, state, residual = evaluate(fit.x)
    if max(abs(residual)) > 1e-6 or min(n) <= 0:
        raise RuntimeError(f"preparation affinity/pH unresolved: {residual}")
    return n, state, residual


@lru_cache(None)
def prepare(t, ph, ratio, basis="kg_water"):
    scale = 1.
    for _ in range(15):
        li, na = scale/6.941, scale*ratio/22.98977
        inventory = dict(zip(AQ, (1/MW[0], li, na, li+na, 0., 0., 0., 0.), strict=True))
        n, state, residual = prepare_inventory(t, ph, inventory)
        mass = float(n @ MW)
        volume = n.sum()/state.molar_density*1000
        target = 1. if basis == "kg_water" else mass if basis == "kg_solution" else volume
        if abs(target-scale) <= 1e-12:
            break
        scale = target
    else:
        raise RuntimeError("alternative feed-basis fixed point unresolved")
    return {"amounts": dict(zip(AQ, n.tolist(), strict=True)), "mass_kg": mass, "volume_L": volume,
            "scale": scale, "total_N_mol": float(n[6]+n[7]), "max_ln_residual": float(max(abs(residual)))}


def chemistry(candidate):
    reactions = [("RW", mc.REACTIONS["RW"]), ("RN", {"ammonium-cation": -1, "ammonia": 1, "hydrogen-cation": 1})]
    for metal, complex_name in zip(AQ[1:3], ORG[2:], strict=True):
        reactions.append((metal, {metal: -1, "des": -3, complex_name: 3,
                                 **({"hydrogen-cation": 1} if candidate == "X" else {"chloride-anion": -1})}))
    moieties = {"O_water": {"water": 1, "hydroxide-anion": 1}, "Li": {AQ[1]: 1, ORG[2]: 1/3},
                "Na": {AQ[2]: 1, ORG[3]: 1/3}, "Cl": {"chloride-anion": 1},
                "N_ammonia": {"ammonia": 1, "ammonium-cation": 1}, "DES_particles": dict.fromkeys(("des",)+ORG[2:], 1),
                "TOPO": {"topo": 1}, "charge": CHARGE}
    if candidate == "S":
        moieties["Cl"].update(dict.fromkeys(ORG[2:], 1/3))
    return reactions, moieties


@dataclass(frozen=True)
class State:
    identity: str
    temperature_k: float


def solve_inventory(candidate, t, ln_k, inventories, guess=None, placeholder=0., state=None):
    """DES declarations shared by source rows and loaded-organic process stages."""
    mix = model(t, candidate, placeholder)
    offset, _ = references(t, candidate)
    active, moieties = chemistry(candidate)
    constants = {"RW": mc.ln_k_at("RW", 0., t), "RN": ln_ka(t), AQ[1]: ln_k[0], AQ[2]: ln_k[1]}
    engine_k = {name: constants[name]+math.fsum(c*offset[s] for s, c in nu.items()) for name, nu in active}
    return mc.solve_declared(state or State("DES process stage", t), mix, IDS, active, offset, engine_k,
        inventories, guess, AQ, moieties, CHARGE, row_tolerance=1e-12)


def solve(candidate, row, ln_k, basis="kg_water", guess=None, placeholder=0.):
    t = float(row["temperature_c"])+273.15
    preparation = prepare(t, float(row["initial_pH"]), float(row["na_li_mass_ratio"]), basis)
    aq = preparation["amounts"]
    w, mass = float(row["topo_wt_pct"])/100, preparation["mass_kg"]
    org = {"des": mass*(1-w)/.20748558, "topo": mass*w/.3866349}
    mix = model(t, candidate, placeholder)
    offset, ref_check = references(t, candidate)
    state = State(str(row["experiment_no"]), t)
    normalization = 1.  # Physical mol per kg initial water; the normalization probe did not cure trace curvature.
    amounts, potentials, check, compiled, result, problem, seed, elapsed = solve_inventory(
        candidate, t, ln_k, (aq, org), guess, placeholder, state)
    transferred = np.array([amounts[1][name]/3 for name in ORG[2:]])
    remaining = np.array([amounts[0][name] for name in AQ[1:3]])
    if not np.all((transferred > 0) & (remaining > 0)):
        raise RuntimeError("nonpositive physical metal amount")
    extraction = transferred/(transferred+remaining)
    initial_org = phase_state(mix, t, org)
    output = {"row": row, "candidate": candidate, "ln_K": list(ln_k), "basis": basis, "preparation": preparation,
              "E": extraction.tolist(), "z": (np.log10(transferred)-np.log10(remaining)).tolist(), "normalization_mol": normalization,
              "pH": -(potentials["hydrogen-cation"]-offset["hydrogen-cation"])/math.log(10),
              "phase_amounts_mol": amounts, "certificate": check,
              "Engine_result": {"requested_tolerance_met": result.requested_tolerance_met, "topology_event": result.topology_event.name,
                                "jacobian_rank": result.jacobian_rank, "unknown_count": result.unknown_count}, "seed": seed, "seconds": elapsed,
              "initial_volume_ratio": (sum(org.values())/initial_org.molar_density*1000)/preparation["volume_L"],
              "final_volumes_L": [result.amounts[p]*normalization/result.molar_densities[p]*1000 for p in range(2)],
              "reference_checks": ref_check}
    output["formula_checks"] = formula_checks(candidate, mix, (aq, org), amounts)
    return output, compiled, result, problem


def formula_checks(candidate, mix, inventories, amounts):
    """Formula H/O/mass residuals and limits propagated from the eight conserved quantities."""
    incoming = np.array([inventories[0].get(name, 0.)+inventories[1].get(name, 0.) for name in IDS])
    delta = np.array([sum(phase[name] for phase in amounts) for name in IDS])-incoming
    c = np.array([[row.get(name, 0.) for name in IDS] for row in chemistry(candidate)[1].values()])
    formula_h = np.array([2, 0, 0, 0, 1, 1, 3, 4, 76/3, 51,
                          (75 if candidate == "X" else 76)/3, (75 if candidate == "X" else 76)/3])
    formula_o = np.array([1, 0, 0, 0, 0, 1, 0, 0, 4/3, 1, 4/3, 4/3])
    checks = {}
    for label, vector in (("H_mol", formula_h), ("O_mol", formula_o), ("mass_kg", np.array(mix.parameters.molar_masses))):
        error = float(abs(vector@delta))
        limit = float(np.linalg.norm(vector@np.linalg.pinv(c), 1)*1e-9)
        checks[label] = {"error": error, "propagated_limit": limit}
        if error > limit:
            raise RuntimeError(f"formula closure failed: {label}, {error} > {limit}")
    return checks


def response_bound(output, compiled, result):
    """R3: physical conservation bounds mapped to Engine 8298c602's square KKT rows."""
    point = result.coordinates[:compiled.primal_size]
    n = compiled.primal_size
    jac = np.array(compiled.jacobian(point)).reshape(-1, n)
    rows = [i for i, row in enumerate(compiled.rows) if str(row.kind).split(".")[-1] not in ("Chemical", "Pressure")]
    moiety_rows = [i for i in rows if str(compiled.rows[i].kind).endswith("Moiety")]
    hessian = np.array(compiled.hessian(point, result.constraint_multipliers)).reshape(n, n)
    a = jac[rows]
    kkt = np.block([[hessian, a.T], [a, np.zeros((len(rows), len(rows)))]])
    # Native coordinates: log(x_supported), free phase amount, log(packing fraction).
    # Engine 8298c602 request_compile.cpp phase layouts; no equilibrium derivatives are reconstructed.
    dn = np.zeros((len(IDS), n))
    b = np.zeros((len(moiety_rows), len(IDS)))
    cursor = 0
    for phase, support in enumerate((AQ, ORG)):
        amount = result.amounts[phase]
        for k, name in enumerate(support):
            i = IDS.index(name)
            ni = output["phase_amounts_mol"][phase][name]
            dn[i, cursor+k], dn[i, cursor+len(support)] = ni, ni/amount
            b[:, i] = jac[moiety_rows, cursor+k]*compiled.feed_total*output["normalization_mol"]/ni
        cursor += len(support)+2
    _, moieties = chemistry(output["candidate"])
    c = np.array([[row.get(s, 0.) for s in IDS] for row in moieties.values()])
    transform = b @ np.linalg.pinv(c)
    if not np.allclose(transform@c, b, rtol=1e-12, atol=1e-12):
        raise RuntimeError("Engine/physical conservation transformation does not close")
    tau_balance = abs(transform) @ np.full(len(c), 1e-9)/(compiled.feed_total*output["normalization_mol"])
    tau = np.r_[np.full(n, 1e-7), [tau_balance[moiety_rows.index(i)] if i in moiety_rows else 1e-12 for i in rows]]
    gain = np.linalg.solve(kkt.T, np.pad(dn, ((0, 0), (0, len(rows)))).T).T
    delta_n = abs(gain) @ tau
    inlet = np.array([output["preparation"]["amounts"][name] for name in AQ[1:3]])
    delta_e = delta_n[[IDS.index(s) for s in ORG[2:]]]/3/inlet
    remaining = np.array([output["phase_amounts_mol"][0][name] for name in AQ[1:3]])
    transfer = np.array([output["phase_amounts_mol"][1][name]/3 for name in ORG[2:]])
    delta_z = delta_e/(math.log(10)*transfer*remaining/(transfer+remaining)**2)
    if not np.all(np.isfinite(delta_z)):
        raise RuntimeError("nonfinite response uncertainty")
    output["response_error_bound_decade"] = delta_z.tolist()
    output["KKT_propagation"] = {"row_kinds": ["stationarity"]*n+[str(compiled.rows[i].kind) for i in rows],
                                 "physical_to_Engine_moieties": transform.tolist(), "tau": tau.tolist(),
                                 "feed_total_mol": compiled.feed_total, "delta_n_mol": delta_n.tolist(),
                                 "condition_number": float(np.linalg.cond(kkt))}
    return delta_z
