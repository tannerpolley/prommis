"""Source-bounded apparent HBTA/TOPO reactive-LLE calculations.

This module uses the installed Engine's configured finite-phase owner.  It does
not claim that the provisional organic pseudo-component parameters are unique
pure-component parameters.  The only calibrated thermodynamic coordinate in
the baseline model is the provider-basis Li/Na exchange ``ln K``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import epcsaft

from .hybrid_oracle import load_spec as load_oracle_spec
from .hybrid_oracle import parameter_payload as oracle_parameter_payload


ANALYSIS = Path(__file__).resolve().parents[1]
SPECIES = (
    "water",
    "diluent",
    "lithium-cation",
    "sodium-cation",
    "chloride-anion",
    "sodium-carrier",
    "lithium-carrier",
)
AQUEOUS_SUPPORT = (
    "water",
    "lithium-cation",
    "sodium-cation",
    "chloride-anion",
)
ORGANIC_SUPPORT = ("diluent", "sodium-carrier", "lithium-carrier")
PRESSURE_PA = 101_325.0
WATER_MOLAR_MASS = 0.01801528
LI_MOLAR_MASS = 0.00694
NA_MOLAR_MASS = 0.02298976928
CL_MOLAR_MASS = 0.035453
HBTA_MOLAR_MASS = 0.21616
TOPO_MOLAR_MASS = 0.3866349


@dataclass(frozen=True)
class Topology:
    identity: str
    hbta_per_carrier: float
    carrier_segment_count: float
    carrier_segment_diameter_angstrom: float
    carrier_dispersion_energy_over_k_k: float


TOPOLOGIES = {
    "shan_1_1_1": Topology("shan_1_1_1", 1.0, 20.60, 3.70, 360.0),
    "zhang_apparent_2_1_1": Topology("zhang_apparent_2_1_1", 2.0, 27.99, 3.67, 348.0),
}


DILUENTS = {
    "dodecane": {
        "molar_mass_kg_mol": 0.170338,
        "segment_count": 5.3060,
        "segment_diameter_angstrom": 3.8959,
        "dispersion_energy_over_k_k": 249.21,
        "locator": "Gross and Sadowski (2001), Table 2 dodecane row",
    },
    "hexadecane": {
        "molar_mass_kg_mol": 0.226446,
        "segment_count": 6.6485,
        "segment_diameter_angstrom": 3.9552,
        "dispersion_energy_over_k_k": 254.70,
        "locator": "Gross and Sadowski (2001), Table 2 hexadecane row",
    },
}


@dataclass(frozen=True)
class Condition:
    identity: str
    source_id: str
    role: str
    temperature_k: float
    aqueous_volume_l: float
    organic_volume_l: float
    li_mol_l_aqueous: float
    na_mol_l_aqueous: float
    hbta_mol_l_organic: float
    topo_mol_l_organic: float
    li_mol_l_organic: float | None = None
    na_mol_l_organic: float | None = None
    observed_li_extraction_percent: float | None = None
    observed_li_distribution: float | None = None
    observed_na_distribution: float | None = None
    initial_ph: float = 6.0
    saponification_fraction: float | None = None
    hydrogen_mol_l_aqueous: float | None = None
    chloride_mol_l_aqueous: float | None = None
    total_carbonate_mol_l_aqueous: float = 0.0


def shan_condition(sodium_mg_l: float, observed_percent: float, role: str) -> Condition:
    return Condition(
        identity=f"shan_na_{int(sodium_mg_l)}",
        source_id="shan_2025_water_17_2258",
        role=role,
        temperature_k=298.15,
        aqueous_volume_l=1.0,
        organic_volume_l=1.0,
        li_mol_l_aqueous=0.060 / 6.94,
        na_mol_l_aqueous=(sodium_mg_l / 1000.0) / 22.98976928,
        hbta_mol_l_organic=0.015,
        topo_mol_l_organic=0.015,
        observed_li_extraction_percent=observed_percent,
        initial_ph=6.0,
        saponification_fraction=1.0,
    )


SHAN_CONDITIONS = (
    shan_condition(0.0, 49.82, "calibration"),
    shan_condition(2000.0, 50.85, "calibration"),
    shan_condition(4000.0, 51.18, "holdout"),
    shan_condition(6000.0, 51.33, "calibration"),
    shan_condition(8000.0, 51.54, "holdout"),
    shan_condition(10900.0, 51.82, "calibration"),
)


# One-stage calibration point: Zhang (2017) Figure 5 at O/A 3:2 (digitized aqueous 0.13877, organic 0.96144 g/L
# Li). Feed Li 1.581 g/L is the isotherm's mass-balance feed and the Section 3.6 brine's Li; Na is that brine's
# 52 g/L. The Conclusions place the isotherm at pH 11.0, taken as the initial pH (Section 3.6's 8.0 is the
# equilibrium pH). D_Na = 0.016 is Table 2's two-stage countercurrent outlet, which approximates stage 1 and
# includes entrained Na; the source prints no one-stage Na value. Organic is fresh 0.3 M HBTA + 0.3 M TOPO.
ZHANG_2017 = Condition(
    identity="zhang_2017_figure5_oa_1.5",
    source_id="zhang_2017_spt_188_167_173",
    role="paired_calibration",
    temperature_k=298.15,
    aqueous_volume_l=1.0,
    organic_volume_l=1.5,
    li_mol_l_aqueous=1.581 / 6.941,
    na_mol_l_aqueous=2.263,
    hbta_mol_l_organic=0.3,
    topo_mol_l_organic=0.3,
    observed_li_distribution=0.96144 / 0.13877,
    observed_na_distribution=0.016,
    initial_ph=11.0,
    saponification_fraction=0.0,
)


SMACKOVER = Condition(
    identity="smackover_selected_feed",
    source_id="project_selected_smackover_feed",
    role="post_holdout_application",
    temperature_k=298.15,
    aqueous_volume_l=1.0,
    organic_volume_l=0.5,
    li_mol_l_aqueous=0.168 / 6.94,
    na_mol_l_aqueous=(0.168 / 6.94) * 214.5,
    hbta_mol_l_organic=0.15,
    topo_mol_l_organic=0.15,
    initial_ph=8.0,
    saponification_fraction=1.0,
)


def _rename_component(record: dict[str, Any], old: str, new: str) -> None:
    record["component_id"] = new
    for coefficient in record.get("coefficients", []):
        coefficient["identity"] = str(coefficient["identity"]).replace(f"component/{old}/", f"component/{new}/")


def _coefficient(record: dict[str, Any], family: str) -> dict[str, Any]:
    return next(item for item in record["coefficients"] if item["family"] == family)


def parameter_payload(topology: Topology, diluent_id: str) -> dict[str, Any]:
    """Return one explicitly provisional but fully traceable parameter document."""

    if diluent_id not in DILUENTS:
        raise ValueError(f"unsupported bounded diluent scenario: {diluent_id}")
    payload = oracle_parameter_payload(load_oracle_spec())
    payload["model_families"] = [model for model in payload["model_families"] if model["family_id"] != "model/polar"]
    for model in payload["model_families"]:
        if model["kind"] == "electrolyte":
            model["ionic_dispersion"] = "exclude-same-sign-ion-pairs"
    topology_document_id = topology.identity.replace("_", "-")
    payload["document_id"] = f"hbta-topo-{topology_document_id}-{diluent_id}-apparent"
    payload["purpose"] = "user-provided"
    payload["sources"].append(
        {
            "source_id": "gross-sadowski-2001",
            "citation": "Gross and Sadowski, Ind. Eng. Chem. Res. 40 (2001) 1244-1260",
            "use_basis": "Bounded normal-alkane diluent scenarios only; not kerosene identity.",
        }
    )
    payload["sources"].append(
        {
            "source_id": "hbta-topo-structural-prior",
            "citation": "Project structural prior constrained by HBTA/TOPO formulas and Rezaee TOPO tuple.",
            "use_basis": "Carrier size/dispersion prior only; not promoted pure-component evidence.",
        }
    )
    payload["domains"].extend(
        [
            {"domain_id": "gross-normal-alkane-scenario", "kind": "unknown"},
            {"domain_id": "hbta-topo-apparent-fit", "kind": "unknown"},
        ]
    )

    by_id = {record["component_id"]: record for record in payload["components"]}
    diluent = by_id.pop("dcm")
    _rename_component(diluent, "dcm", "diluent")
    diluent_prior = DILUENTS[diluent_id]
    diluent["fixed"] = {
        "molar_mass": {
            "value": {
                "magnitude": diluent_prior["molar_mass_kg_mol"],
                "unit": "kilogram / mole",
            },
            "provenance": {
                "source_id": "gross-sadowski-2001",
                "locator": diluent_prior["locator"],
                "domain_id": "gross-normal-alkane-scenario",
            },
        },
        "charge_number": {
            "value": {"magnitude": 0, "unit": "elementary-charge"},
            "provenance": {
                "source_id": "gross-sadowski-2001",
                "locator": diluent_prior["locator"],
                "domain_id": "gross-normal-alkane-scenario",
            },
        },
    }
    diluent["coefficients"] = []
    for family, magnitude, unit in (
        ("segment_count", diluent_prior["segment_count"], "dimensionless"),
        ("segment_diameter", diluent_prior["segment_diameter_angstrom"], "angstrom"),
        (
            "dispersion_energy_over_k",
            diluent_prior["dispersion_energy_over_k_k"],
            "kelvin",
        ),
        ("relative_permittivity", 2.0, "dimensionless"),
        ("solvation_factor", 1.0, "dimensionless"),
    ):
        diluent["coefficients"].append(
            {
                "identity": f"component/diluent/{family}",
                "family": family,
                "value": {"magnitude": magnitude, "unit": unit},
                "provenance": {
                    "source_id": "gross-sadowski-2001",
                    "locator": diluent_prior["locator"],
                    "domain_id": "gross-normal-alkane-scenario",
                },
            }
        )
    by_id["diluent"] = diluent

    carrier_mass = {
        "sodium-carrier": (topology.hbta_per_carrier * HBTA_MOLAR_MASS + TOPO_MOLAR_MASS + NA_MOLAR_MASS),
        "lithium-carrier": (topology.hbta_per_carrier * HBTA_MOLAR_MASS + TOPO_MOLAR_MASS + LI_MOLAR_MASS),
    }
    for identity in ("sodium-carrier", "lithium-carrier"):
        record = by_id[identity]
        record["fixed"]["molar_mass"]["value"]["magnitude"] = carrier_mass[identity]
        record["fixed"]["molar_mass"]["provenance"] = {
            "source_id": "hbta-topo-structural-prior",
            "locator": f"{topology.identity} formula-derived molar mass",
            "domain_id": "hbta-topo-apparent-fit",
        }
        for family, magnitude in (
            ("segment_count", topology.carrier_segment_count),
            ("segment_diameter", topology.carrier_segment_diameter_angstrom),
            (
                "dispersion_energy_over_k",
                topology.carrier_dispersion_energy_over_k_k,
            ),
        ):
            item = _coefficient(record, family)
            item["value"]["magnitude"] = magnitude
            item["provenance"] = {
                "source_id": "hbta-topo-structural-prior",
                "locator": f"shared ligand-derived {topology.identity} prior",
                "domain_id": "hbta-topo-apparent-fit",
            }
        for family, magnitude in (
            ("relative_permittivity", 4.0),
            ("solvation_factor", 1.0),
        ):
            item = _coefficient(record, family)
            item["value"]["magnitude"] = magnitude
            item["provenance"] = {
                "source_id": "hbta-topo-structural-prior",
                "locator": "bounded organic-phase dielectric/solvation prior",
                "domain_id": "hbta-topo-apparent-fit",
            }

    payload["components"] = [
        by_id[identity]
        for identity in (
            "water",
            "diluent",
            "lithium-cation",
            "sodium-cation",
            "chloride-anion",
            "sodium-carrier",
            "lithium-carrier",
        )
    ]
    remapped_pairs = []
    for pair in payload["pairs"]:
        left = "diluent" if pair["component_id_a"] == "dcm" else pair["component_id_a"]
        right = "diluent" if pair["component_id_b"] == "dcm" else pair["component_id_b"]
        left, right = sorted((left, right))
        pair["component_id_a"] = left
        pair["component_id_b"] = right
        for coefficient in pair["coefficients"]:
            coefficient["identity"] = f"pair/{left}/{right}/k_ij"
            if "diluent" in (left, right) or "carrier" in left or "carrier" in right:
                coefficient["value"]["magnitude"] = 0.0
                coefficient["provenance"] = {
                    "source_id": "hbta-topo-structural-prior",
                    "locator": "zero cross-interaction prior before extraction correction",
                    "domain_id": "hbta-topo-apparent-fit",
                }
        remapped_pairs.append(pair)
    payload["pairs"] = remapped_pairs
    return payload


@lru_cache(maxsize=None)
def build_model(topology_id: str, diluent_id: str) -> tuple[epcsaft.Parameters, epcsaft.Mixture]:
    parameters = epcsaft.Parameters.from_mapping(parameter_payload(TOPOLOGIES[topology_id], diluent_id))
    return parameters, epcsaft.Mixture(parameters)


def parameter_document(topology_id: str, diluent_id: str) -> dict[str, Any]:
    return parameter_payload(TOPOLOGIES[topology_id], diluent_id)
