"""Parameter inputs of the retired manufactured Li/Na carrier-exchange oracle.

The oracle's solve, derivative and fitting checks used the superseded Engine interface and
were retired when the repository moved to one Engine (83ac1126). Their retained result files
remain historical evidence; the parameter builders below still feed the current producer.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any



ANALYSIS = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
SPEC_PATH = ANALYSIS / "data/input/hybrid_li_na_oracle.json"
YU_PARAMETERS_PATH = REPO / "analyses/engine_benchmarks/yu_2025_epcsaft_benchmark/data/input/yu_2025_parameters.json"


def _value(magnitude: float, unit: str) -> dict[str, object]:
    return {"magnitude": magnitude, "unit": unit}


def _temperature_value(record: dict[str, object], temperature_k: float) -> float:
    form = str(record["form"])
    if form == "constant":
        return float(record["value"])
    if form == "constant_plus_sum_of_exponentials":
        return float(record["constant"]) + math.fsum(
            float(term["amplitude"])
            * math.exp(float(term["exponent_coefficient_per_k"]) * temperature_k)
            for term in record["terms"]  # type: ignore[index]
        )
    if form == "a_ln_temperature_plus_b":
        return float(record["a"]) * math.log(temperature_k) + float(record["b"])
    raise ValueError(f"unsupported source correlation: {form}")


def load_spec() -> dict[str, Any]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def _provenance(source_id: str, locator: str, domain_id: str) -> dict[str, str]:
    return {"source_id": source_id, "locator": locator, "domain_id": domain_id}


def _component(
    component_id: str,
    *,
    molar_mass: float,
    charge: int,
    segment_count: float,
    segment_diameter: float,
    dispersion_energy: float,
    provenance: dict[str, str],
    relative_permittivity: float | None = None,
    packing_diameter: float | None = None,
    debye_huckel_diameter: float | None = None,
    born_diameter: float | None = None,
    solvation_factor: float | None = None,
    dipole_moment: float | None = None,
) -> dict[str, object]:
    fixed: dict[str, object] = {
        "molar_mass": {
            "value": _value(molar_mass, "kilogram / mole"),
            "provenance": provenance,
        },
        "charge_number": {
            "value": _value(charge, "elementary-charge"),
            "provenance": provenance,
        },
    }
    if dipole_moment is not None:
        fixed["dipole_moment"] = {
            "value": _value(dipole_moment, "debye"),
            "provenance": provenance,
        }
    coefficients: list[dict[str, object]] = []
    families = {
        "segment_count": (segment_count, "dimensionless"),
        "segment_diameter": (segment_diameter, "angstrom"),
        "dispersion_energy_over_k": (dispersion_energy, "kelvin"),
        "relative_permittivity": (relative_permittivity, "dimensionless"),
        "packing_diameter": (packing_diameter, "angstrom"),
        "debye_huckel_diameter": (debye_huckel_diameter, "angstrom"),
        "born_diameter": (born_diameter, "angstrom"),
        "solvation_factor": (solvation_factor, "dimensionless"),
    }
    for family, (magnitude, unit) in families.items():
        if magnitude is None:
            continue
        coefficients.append(
            {
                "identity": f"component/{component_id}/{family}",
                "family": family,
                "value": _value(float(magnitude), unit),
                "provenance": provenance,
            }
        )
    return {"component_id": component_id, "fixed": fixed, "coefficients": coefficients}


def _pair(
    component_a: str,
    component_b: str,
    value: float,
    provenance: dict[str, str],
) -> dict[str, object]:
    left, right = sorted((component_a, component_b))
    return {
        "component_id_a": left,
        "component_id_b": right,
        "coefficients": [
            {
                "identity": f"pair/{left}/{right}/k_ij",
                "family": "k_ij",
                "value": _value(value, "dimensionless"),
                "provenance": provenance,
            }
        ],
    }


def parameter_payload(spec: dict[str, Any]) -> dict[str, object]:
    """Build the exact mixed-provenance model used only by this oracle."""

    yu = json.loads(YU_PARAMETERS_PATH.read_text(encoding="utf-8"))
    temperature_k = float(spec["temperature_k"])
    yu_source = _provenance(
        "yu-2025",
        "Article Tables 1-2 and Supporting Information Table S4",
        "yu-2025-water-dcm-298-k",
    )
    figiel_source = _provenance(
        "figiel-yu-held-2025",
        "Tables 1, 3-5 and Eq. 11",
        "figiel-298-k-1-bar",
    )
    manufactured = _provenance(
        "manufactured-li-na-oracle",
        "hybrid_li_na_oracle.json manufactured_inputs",
        "manufactured-li-na-oracle-domain",
    )

    water_index = yu["component_ids"].index("water")
    dcm_index = yu["component_ids"].index("dcm")
    components = [
        _component(
            "water",
            molar_mass=float(yu["molar_mass_kg_mol"][water_index]),
            charge=0,
            segment_count=float(yu["segment_count"][water_index]),
            segment_diameter=_temperature_value(
                yu["segment_diameter_angstrom"][water_index], temperature_k
            ),
            dispersion_energy=float(yu["dispersion_energy_over_k_k"][water_index]),
            relative_permittivity=_temperature_value(
                yu["relative_permittivity"][water_index], temperature_k
            ),
            solvation_factor=1.5,
            provenance=yu_source,
        ),
        _component(
            "dcm",
            molar_mass=float(yu["molar_mass_kg_mol"][dcm_index]),
            charge=0,
            segment_count=float(yu["segment_count"][dcm_index]),
            segment_diameter=_temperature_value(
                yu["segment_diameter_angstrom"][dcm_index], temperature_k
            ),
            dispersion_energy=float(yu["dispersion_energy_over_k_k"][dcm_index]),
            relative_permittivity=float(spec["manufactured_inputs"]["dcm_relative_permittivity"]),
            solvation_factor=1.0,
            dipole_moment=float(yu["dipole_moment_debye"][dcm_index]),
            provenance=yu_source,
        ),
    ]
    for coefficient in components[0]["coefficients"]:  # type: ignore[index]
        if coefficient["family"] == "solvation_factor":
            coefficient["provenance"] = figiel_source
    for coefficient in components[1]["coefficients"]:  # type: ignore[index]
        if coefficient["family"] in {"relative_permittivity", "solvation_factor"}:
            coefficient["provenance"] = manufactured
    charges = {"water": 0, "dcm": 0}
    for component_id, record in spec["source_backed_ions"]["components"].items():
        provenance = _provenance(
            "figiel-yu-held-2025", str(record["locator"]), "figiel-298-k-1-bar"
        )
        components.append(
            _component(
                component_id,
                molar_mass=float(record["molar_mass_kg_mol"]),
                charge=int(record["charge_number"]),
                segment_count=float(record["segment_count"]),
                segment_diameter=float(record["segment_diameter_angstrom"]),
                dispersion_energy=float(record["dispersion_energy_over_k_k"]),
                packing_diameter=float(record["packing_diameter_angstrom"]),
                debye_huckel_diameter=float(record["debye_huckel_diameter_angstrom"]),
                born_diameter=float(record["born_diameter_angstrom"]),
                solvation_factor=float(record["solvation_factor"]),
                provenance=provenance,
            )
        )
        charges[component_id] = int(record["charge_number"])
    for component_id, record in spec["manufactured_inputs"]["carrier_components"].items():
        components.append(
            _component(
                component_id,
                molar_mass=float(record["molar_mass_kg_mol"]),
                charge=0,
                segment_count=float(record["segment_count"]),
                segment_diameter=float(record["segment_diameter_angstrom"]),
                dispersion_energy=float(record["dispersion_energy_over_k_k"]),
                relative_permittivity=float(record["relative_permittivity"]),
                solvation_factor=float(record["solvation_factor"]),
                provenance=manufactured,
            )
        )
        charges[component_id] = 0

    pairs = [
        _pair(
            "water",
            "dcm",
            float(yu["binary_k_ij"][water_index][dcm_index]),
            yu_source,
        )
    ]
    for identity, record in spec["source_backed_ions"]["binary_k_ij"].items():
        left, right = identity.split("/")
        pairs.append(
            _pair(
                left,
                right,
                float(record["value"]),
                _provenance(
                    "figiel-yu-held-2025",
                    str(record["locator"]),
                    "figiel-298-k-1-bar",
                ),
            )
        )
    for identity, value in spec["manufactured_inputs"]["binary_k_ij"].items():
        left, right = identity.split("/")
        pairs.append(_pair(left, right, float(value), manufactured))
    present_pairs = {
        (str(record["component_id_a"]), str(record["component_id_b"])) for record in pairs
    }
    for left_index, left in enumerate(spec["species_ids"]):
        for right in spec["species_ids"][left_index + 1 :]:
            key = tuple(sorted((left, right)))
            if key in present_pairs:
                continue
            if charges.get(left, 0) != 0 and charges.get(left, 0) == charges.get(right, 0):
                continue
            pairs.append(_pair(left, right, 0.0, manufactured))
            present_pairs.add(key)

    electrolyte = spec["source_backed_ions"]["electrolyte_model"]
    model_families: list[dict[str, object]] = []
    for kind, choice, source in (
        ("association", "general-site", yu_source),
        ("base", "pc-saft", yu_source),
        ("electrolyte", str(electrolyte["choice"]), figiel_source),
        ("permittivity", str(electrolyte["permittivity_choice"]), figiel_source),
        ("polar", "point-multipole", yu_source),
    ):
        record: dict[str, object] = {
            "family_id": {
                "association": "model/association",
                "base": "model/base",
                "electrolyte": "model/electrolyte",
                "permittivity": "model/relative_permittivity",
                "polar": "model/polar",
            }[kind],
            "kind": kind,
            "choice": choice,
            "provenance": source,
        }
        if kind == "electrolyte":
            record["c_shell"] = float(electrolyte["c_shell"])
            record["c_dielectric"] = float(electrolyte["c_dielectric"])
        model_families.append(record)
    model_coefficients = [
        {
            "identity": "model/relative_permittivity/ionic_region_relative_permittivity",
            "family": "ionic_region_relative_permittivity",
            "value": _value(
                float(electrolyte["ionic_region_relative_permittivity"]), "dimensionless"
            ),
            "provenance": figiel_source,
        },
        {
            "identity": "model/relative_permittivity/ion_fraction_suppression_coefficient",
            "family": "ion_fraction_suppression_coefficient",
            "value": _value(
                float(electrolyte["ion_fraction_suppression_coefficient"]), "dimensionless"
            ),
            "provenance": figiel_source,
        },
    ]
    water_energy = float(yu["association_energy_over_k_k"][water_index])
    water_volume = float(yu["association_volume"][water_index])
    topology = {
        "presets": [],
        "sites": [
            {
                "component_id": "water",
                "site_id": site_id,
                "site_role": role,
                "multiplicity": 1,
                "provenance": yu_source,
            }
            for site_id, role in (("a", "donor"), ("b", "acceptor"))
        ],
        "edges": [
            {
                "endpoint_a": {"component_id": "water", "site_id": "a"},
                "endpoint_b": {"component_id": "water", "site_id": "b"},
                "energy_over_k": {
                    "identity": "association/water/a/water/b/energy_over_k",
                    "value": _value(water_energy, "kelvin"),
                },
                "volume": {
                    "identity": "association/water/a/water/b/volume",
                    "value": _value(water_volume, "dimensionless"),
                },
                "source": {"kind": "explicit", "provenance": [yu_source]},
            }
        ],
    }
    return {
        "schema": "epcsaft.parameters",
        "schema_version": 1,
        "document_id": "hybrid-li-na-carrier-exchange-oracle",
        "document_version": 1,
        "purpose": "user-provided",
        "sources": [
            {
                "source_id": "yu-2025",
                "citation": str(spec["source_backed_neutral_lle"]["citation"]),
                "doi": "10.1002/aic.18907",
                "use_basis": "Water/DCM pure and binary parameters only.",
            },
            {
                "source_id": "figiel-yu-held-2025",
                "citation": str(spec["source_backed_ions"]["citation"]),
                "doi": "10.1021/acs.iecr.5c00475",
                "use_basis": "Aqueous Li+, Na+, and Cl- parameters and electrolyte model only.",
            },
            {
                "source_id": "manufactured-li-na-oracle",
                "citation": "Project-authored numerical capability oracle.",
                "use_basis": "Carrier complexes, missing DCM permittivity, cross interactions, feed, and K sweep; no physical prediction.",
            },
        ],
        "domains": [
            {"domain_id": "yu-2025-water-dcm-298-k", "kind": "unknown"},
            {"domain_id": "figiel-298-k-1-bar", "kind": "unknown"},
            {"domain_id": "manufactured-li-na-oracle-domain", "kind": "unknown"},
        ],
        "components": components,
        "pairs": pairs,
        "model_families": model_families,
        "model_coefficients": model_coefficients,
        "correlations": [],
        "topology": topology,
    }
