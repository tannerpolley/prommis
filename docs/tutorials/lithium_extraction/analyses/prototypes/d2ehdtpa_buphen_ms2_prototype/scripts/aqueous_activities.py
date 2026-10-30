"""Issue 188 F0: frozen Held 2008 versus Reaktoro 2.13 Pitzer activity transfers.

The original F0 negative result is preserved. Quantities are per kg water on the
issue's 1 kg/L and 55.508 mol/L approximation. Chloride closes charge.
The residual-potential offsets follow #158 mechanism_candidates.offsets.
"""

import csv
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path

import epcsaft
import numpy as np

from analyses.shared.hbta_topo_configured_reactive_lle.scripts.independent_network import PRESSURE_RELATIVE_LIMIT
from lithium_extraction.prototypes.produced_water_sx_prototype.flowsheet import installed_engine

ANALYSIS = Path(__file__).resolve().parents[1]
INPUTS, RESULTS = ANALYSIS / "inputs", ANALYSIS / "results"
ION = {"Li": ("lithium-cation", "Li+", 1), "Na": ("sodium-cation", "Na+", 1),
       "K": ("potassium-cation", "K+", 1), "Mg": ("magnesium-cation", "Mg+2", 2),
       "Ca": ("calcium-cation", "Ca+2", 2), "Sr": ("strontium-cation", "Sr+2", 2),
       "Cl": ("chloride-anion", "Cl-", -1)}
T, P, WATER_MW = 298.15, 101325.0, 0.01801528
F0_LIMIT, CLOSURE_LIMIT = 0.15, 1e-9  # issue 188, F0 and F3 (mol)


def write_csv(name, rows):
    with (RESULTS / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ratios(a):
    """Dimensionless molal activities; all three combinations have net charge zero."""
    return {"R_Mg": 2*a["Na"] - a["Mg"], "R_Ca": 2*a["Na"] - a["Ca"], "R_Li": a["Li"] - a["Na"]}


def held_model():
    """Verify the prescribed encoding before evaluating any frozen state."""
    provenance = installed_engine()
    document = json.loads((INPUTS / "held_2008_parameters.json").read_text())
    choices = {r["kind"]: r for r in document["model_families"]}
    assert choices["electrolyte"]["choice"] == "fully-dissociated-debye-huckel"
    assert choices["electrolyte"]["ionic_dispersion"] == "exclude-same-sign-ion-pairs"
    assert choices["permittivity"]["choice"] == "solvent-only"
    coefficients = {c["component_id"]: {r["family"]: r["value"]["magnitude"] for r in c["coefficients"]}
                    for c in document["components"]}
    for name, _, _ in ION.values():
        row = coefficients[name]
        assert row["segment_count"] == 1
        for diameter in ("packing_diameter", "debye_huckel_diameter"):
            assert math.isclose(row[diameter], .88*row["segment_diameter"], abs_tol=1e-12)
        assert "born_diameter" not in row
    pairs = {frozenset((r["component_id_a"], r["component_id_b"])): r["coefficients"][0]["value"]["magnitude"]
             for r in document["pairs"]}
    for name, _, charge in ION.values():
        assert pairs[frozenset((name, "water"))] == 0
        if charge > 0:
            assert pairs[frozenset((name, "chloride-anion"))] == 1
    mixture = epcsaft.Mixture(epcsaft.Parameters.from_mapping(document))
    pure = mixture.state(T, P=P, x=[float(i == "water") for i in mixture.component_ids], phase="liquid")
    assert pure.born == 0 and pure.bulk_relative_permittivity == 78.09
    offsets = {}
    for name, _, _ in ION.values():
        x = [0.0]*len(mixture.component_ids)
        x[mixture.component_ids.index("water")] = 1 - 1e-20
        index = mixture.component_ids.index(name)
        x[index] = 1e-20
        trace = mixture.state(T, P=P, x=x, phase="liquid")
        offsets[name] = math.log(pure.molar_density*WATER_MW) + trace.residual_chemical_potential_over_rt[index]
    (RESULTS / "held_encoding.json").write_text(json.dumps({"engine": provenance, "model_choices": choices,
        "ion_coefficients": coefficients, "pure_water_density_mol_m3": pure.molar_density,
        "bulk_relative_permittivity": pure.bulk_relative_permittivity, "born": pure.born,
        "reference_offsets": offsets, "evidence_type": "numerical verification"}, indent=2) + "\n")
    return mixture, offsets


def engine_activities(mixture, offsets, mol, water):
    amounts = {ION[e][0]: n for e, n in mol.items()} | {"water": water}
    total = math.fsum(amounts.values())
    x = [amounts[i]/total for i in mixture.component_ids]
    state = mixture.state(T, P=P, x=x, phase="liquid")
    assert state.born == 0 and abs(state.bulk_relative_permittivity - 78.09) <= math.ulp(78.09)
    assert abs(state.pressure/P - 1) <= PRESSURE_RELATIVE_LIMIT
    loga = {e: (math.log(x[j]*state.molar_density) + state.residual_chemical_potential_over_rt[j] - offsets[name])/math.log(10)
            for e, (name, _, _) in ION.items() if mol[e] > 0 for j in [mixture.component_ids.index(name)]}
    return loga, state


def database_coverage(contents):
    """Inventory explicit Pitzer records; absent records are not asserted to be measured zeros."""
    terms, section = [], ""
    for lineno, line in enumerate(contents.splitlines(), 1):
        text = line.split("#")[0].strip()
        if text.startswith("-"):
            section = text.split()[0][1:]
        tokens = text.split()
        if section in ("B0", "B1", "B2", "C0", "THETA", "PSI") and tokens and not text.startswith("-"):
            count = 3 if section == "PSI" else 2
            if len(tokens) > count:
                terms.append((section, frozenset(tokens[:count]), lineno, text))
    rows = []
    cations = [v[1] for v in ION.values() if v[2] > 0]
    requests = [(term, frozenset((cation, "Cl-"))) for cation in cations for term in ("B0", "B1", "C0")]
    requests += [(term, frozenset((a, b, "Cl-") if term == "PSI" else (a, b)))
                 for a, b in combinations(cations, 2) for term in ("THETA", "PSI")]
    for term, species in requests:
        records = [(lineno, text) for kind, names, lineno, text in terms if kind == term and names == species]
        rows.append({"term": term, "species": " ".join(sorted(species)), "explicit_record": bool(records),
                     "database_line": records[0][0] if records else "", "record": records[0][1] if records else "",
                     "evidence_type": "numerical verification"})
    write_csv("pitzer_coverage.csv", rows)


def main():
    import reaktoro as rkt

    RESULTS.mkdir(exist_ok=True)
    mixture, offsets = held_model()
    reference = mixture.reference(T, P, solvent_fractions=[float(i == "water") for i in mixture.component_ids],
                                  ions=("sodium-cation", "chloride-anion"), stoichiometry=(1, 1))
    convention = []
    for concentration in (1e-8, 0.1):
        mol = {e: concentration if e in ("Na", "Cl") else 0.0 for e in ION}
        loga, _ = engine_activities(mixture, offsets, mol, 1/WATER_MW)
        gamma = (loga["Na"] + loga["Cl"])/2 - math.log10(concentration)
        native = reference.activity(concentration).mean_ionic_log_molality_scale/math.log(10)
        convention.append({"NaCl_molality_mol_kg": concentration, "offset_log10_mean_gamma": gamma,
            "native_log10_mean_gamma": native, "difference_decade": gamma - native,
            "native_reference_molality_mol_kg": reference.molality, "evidence_type": "numerical verification"})
    write_csv("activity_convention.csv", convention)
    assert rkt.__version__ == "2.13.0"
    contents = rkt.PhreeqcDatabase.contents("pitzer.dat")
    database_coverage(contents)
    aqueous = rkt.AqueousPhase(rkt.speciate("H O Li Na K Mg Ca Sr Cl"))
    aqueous.set(rkt.ActivityModelPitzer())
    system = rkt.ChemicalSystem(rkt.PhreeqcDatabase("pitzer.dat"), aqueous)
    solver = rkt.EquilibriumSolver(system)
    activities, species_rows, inputs, numerical, values, refinements = [], [], [], [], {}, []
    with (INPUTS / "f0_states.csv").open() as stream:
        frozen = list(csv.DictReader(stream))
    for row in frozen:
        assert float(row["temperature_K"]) == T and float(row["pressure_Pa"]) == P
        assert float(row["water_kg"]) == 1.0 and row["unit"] == "mmol per kg water"
        name = row["state"]
        mol = {e: float(row[e])/1000 for e in ION if e != "Cl"}
        mol["Cl"] = math.fsum(n*ION[e][2] for e, n in mol.items())
        water = float(row["water_mol"])
        inputs.append({"state": name, "water_mol": water, "water_kg": float(row["water_kg"]),
            "temperature_K": T, "pressure_Pa": P, **{f"{e}_mol_kg": n for e, n in mol.items()},
            "Cl_printed_mol_kg": float(row["Cl_printed"])/1000,
            "Cl_rounding_correction_mol_kg": mol["Cl"] - float(row["Cl_printed"])/1000,
            "evidence_type": "numerical verification"})
        engine, state = engine_activities(mixture, offsets, mol, water)
        exact_water, _ = engine_activities(mixture, offsets, mol, 1/WATER_MW)
        for ratio, value in ratios(exact_water).items():
            refinements.append({"state": name, "variant": "engine_exact_kg_water", "ratio": ratio,
                "change_from_base_decade": value - ratios(engine)[ratio], "evidence_type": "numerical verification"})
        for variant, tolerance, water_unit, water_amount in (("base", 1e-8, "kg", 1.0),
                ("tighter_pitzer", 1e-11, "kg", 1.0), ("same_water_moles", 1e-11, "mol", water)):
            options = rkt.EquilibriumOptions()
            options.optima.convergence.tolerance = tolerance
            solver.setOptions(options)
            chem = rkt.ChemicalState(system)
            chem.temperature(T, "K")
            chem.pressure(P, "Pa")
            chem.set("H2O", water_amount, water_unit)
            for e, n in mol.items():
                if n > 0:
                    chem.set(ION[e][1], n, "mol")
            initial = np.array(rkt.ChemicalProps(chem).elementAmounts(), dtype=float)
            result = solver.solve(chem)
            if not result.succeeded():
                raise RuntimeError(f"{name}/{variant}: {result.optima.failure_reason}")
            props = rkt.ChemicalProps(chem)
            closure = float(np.max(np.abs(np.array(props.elementAmounts(), dtype=float) - initial)))
            charge = float(props.charge())
            assert max(closure, abs(charge)) <= CLOSURE_LIMIT
            pitzer = {e: float(props.speciesActivityLg(s)) for e, (_, s, _) in ION.items() if mol[e] > 0}
            numerical.append({"state": name, "variant": variant, "pitzer_tolerance": tolerance,
                "element_closure_mol": closure, "charge_mol": charge, "iterations": result.iterations(),
                "optimality_error": result.optima.error_optimality, "feasibility_error": result.optima.error_feasibility,
                "engine_pressure_relative_error": abs(state.pressure/P - 1), "engine_density_mol_m3": state.molar_density,
                "engine_density_stop": str(state.density_convergence.stop),
                "engine_density_residual_over_criterion": state.density_convergence.residual_over_criterion,
                "pitzer_water_kg": float(props.speciesAmount("H2O"))*system.species().get("H2O").molarMass(),
                "engine_water_kg": water*WATER_MW, "evidence_type": "numerical verification"})
            if variant != "base":
                for ratio, value in ratios(pitzer).items():
                    refinements.append({"state": name, "variant": variant, "ratio": ratio,
                        "change_from_base_decade": value - values[name]["Pitzer"][ratio], "evidence_type": "numerical verification"})
                continue
            values[name] = {"Engine": ratios(engine), "Pitzer": ratios(pitzer)}
            for method, loga in (("Engine", engine), ("Pitzer", pitzer)):
                for e, value in loga.items():
                    activities.append({"state": name, "method": method, "element": e, "log10_activity": value,
                        "convention": "dimensionless, molal infinite dilution", "evidence_type": "numerical verification"})
            for species in system.species():
                s = species.name()
                species_rows.append({"state": name, "species": s, "amount_mol": float(props.speciesAmount(s)),
                    "log10_activity": float(props.speciesActivityLg(s)), "evidence_type": "numerical verification"})
    rows = []
    for name in ("table1", "ms2"):
        for ratio in ("R_Mg", "R_Ca", "R_Li"):
            eng = values[name]["Engine"][ratio] - values["fig7"]["Engine"][ratio]
            pit = values[name]["Pitzer"][ratio] - values["fig7"]["Pitzer"][ratio]
            rows.append({"target_state": name, "ratio": ratio, "engine_delta_decade": eng, "pitzer_delta_decade": pit,
                "disagreement_decade": eng - pit, "tolerance_decade": F0_LIMIT, "passed": abs(eng - pit) <= F0_LIMIT,
                "evidence_type": "numerical verification"})
    for file, data in (("f0.csv", rows), ("f0_activities.csv", activities), ("f0_pitzer_species.csv", species_rows),
                       ("f0_inputs.csv", inputs), ("f0_numerical.csv", numerical), ("f0_refinement.csv", refinements)):
        write_csv(file, data)
    summary = {"F0_passed": all(r["passed"] for r in rows), "activity_ratios": values,
        "reaktoro_version": rkt.__version__, "pitzer_sha256": hashlib.sha256(contents.encode()).hexdigest(),
        "maximum_refinement_change_decade": max(abs(r["change_from_base_decade"]) for r in refinements),
        "maximum_element_closure_mol": max(r["element_closure_mol"] for r in numerical),
        "stopping_point": "F0 failed; calibration, F1, F2 and their fallback are not executed" if any(not r["passed"] for r in rows)
                          else "F0 passed; this script performs no calibration",
        "evidence_type": "numerical verification"}
    (RESULTS / "f0_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
