"""Independent Li/Na HBTA/TOPO reaction networks on declared phase supports.

The model separates the source-declared initial saponification inventory from
two equilibrium coordinates: Li/Na carrier exchange and direct Na uptake by
the remaining protonated carrier.  The three molecular stoichiometries are
separate model-form hypotheses; they are never averaged or selected by the
Engine.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import math
import time
from typing import Any

import epcsaft
from epcsaft import equilibrium

from .apparent_application import (
    HBTA_MOLAR_MASS,
    LI_MOLAR_MASS,
    NA_MOLAR_MASS,
    PRESSURE_PA,
    TOPOLOGIES as LEGACY_TOPOLOGIES,
    TOPO_MOLAR_MASS,
    Condition,
    parameter_payload as legacy_parameter_payload,
)


H_MOLAR_MASS = 0.00100794
SPECIES = (
    "water",
    "diluent",
    "lithium-cation",
    "sodium-cation",
    "hydrogen-cation",
    "chloride-anion",
    "free-carrier",
    "sodium-carrier",
    "lithium-carrier",
)
# Dissolved inorganic carbon: HCO3-, CO3(2-) and H2CO3* (dissolved CO2, the species of the source K1). Only
# conditions with a declared total carbonate carry them; every other condition keeps the nine species above.
CARBONATE_SPECIES = ("bicarbonate-anion", "carbonate-anion", "carbon-dioxide")
ALL_SPECIES = SPECIES + CARBONATE_SPECIES
AQUEOUS_SUPPORT = (
    "water",
    "lithium-cation",
    "sodium-cation",
    "hydrogen-cation",
    "chloride-anion",
)
ORGANIC_SUPPORT = (
    "diluent",
    "free-carrier",
    "sodium-carrier",
    "lithium-carrier",
)
CHARGES = (0, 0, 1, 1, 1, -1, 0, 0, 0)
ALL_CHARGES = CHARGES + (-1, -2, 0)
INITIALIZATION_SEEDS = (
    (0.02, 0.20, 0.20),
    (0.001, 0.05, 0.20),
    (0.001, 0.10, 0.99),
    (0.005, 0.50, 0.80),
)
APPLICATION_AFFINITY_LIMIT = 2.0e-3
# Limit on |P_phase/P - 1|. The Engine's pressure row is (P_phase - P)/(rho RT); it counts as converged when
# that row times n_phase/n_feed is <= 1e-10. For the organic phase (rho RT/P ~ 100, n_org/n_feed >= 0.035 at
# O/A >= 0.5) the Engine therefore certifies |P_phase/P - 1| only to ~3e-7, and a lean-end stage with trace
# aqueous Li returned 1.7e-8 (four-stage MS-2 start). 1e-6 (0.1 Pa) sits above that certified precision and
# shifts each ln(activity) by v_i dP/RT <= 4e-8 for v_i <= 1 L/mol, far below the 1e-6 strict affinity limit.
PRESSURE_RELATIVE_LIMIT = 1.0e-6
# Bulow et al. (2021, Ind. Eng. Chem. Res. 60:6327, Tables 2-3) ePC-SAFT ions in Held's framework, entered like the
# Held (2014) H+ above: electrolyte diameters 0.88 sigma, Born diameter sigma. CO2(aq) uses their pure CO2 row.
CARBONATE_PARAMETERS = {
    "bicarbonate-anion": (0.06101684, -1, 1.0, 2.9296, 70.0),
    "carbonate-anion": (0.0600089, -2, 1.0, 2.4422, 249.26),
    "carbon-dioxide": (0.0440095, 0, 2.0729, 2.7852, 169.21),
}
CARBONATE_WATER_KIJ = {"bicarbonate-anion": 0.0, "carbonate-anion": -0.25, "carbon-dioxide": 0.004695}
CO2_RELATIVE_PERMITTIVITY = 1.6  # liquid CO2 (CRC Handbook); CO2(aq) is at most ~1e-3 mole fraction here
CARBONATE_STANDARD_STATE = "aqueous-molality-infinite-dilution-water-v1"


def carbonic_log10_k(temperature_k: float) -> tuple[float, float]:
    """Plummer and Busenberg (1982, GCA 46:1011) molal infinite-dilution fits: K1 for CO2(aq) + H2O = HCO3- + H+,
    K2 for HCO3- = CO3(2-) + H+ (6.352 and 10.329 at 298.15 K)."""
    t = temperature_k
    first = -356.3094 - 0.06091964 * t + 21834.37 / t + 126.8339 * math.log10(t) - 1684915.0 / t**2
    second = -107.8871 - 0.03252849 * t + 5151.79 / t + 38.92561 * math.log10(t) - 563713.9 / t**2
    return first, second


@dataclass(frozen=True)
class IndependentTopology:
    identity: str
    hbta_per_carrier: float
    topo_per_carrier: float
    carrier_segment_count: float
    carrier_segment_diameter_angstrom: float
    carrier_dispersion_energy_over_k_k: float
    source_basis: str


TOPOLOGIES = {
    "t1_mbta_topo": IndependentTopology(
        "t1_mbta_topo",
        1.0,
        1.0,
        20.60,
        3.70,
        360.0,
        "Shan 2025 NaBTA-TOPO/LiBTA-TOPO 1:1:1 mechanism",
    ),
    "t2_mbta_topo2": IndependentTopology(
        "t2_mbta_topo2",
        1.0,
        2.0,
        33.809,
        3.70,
        360.0,
        "Healy isolated Li(DBM)(TOPO)2 analogue; HBTA identity not asserted",
    ),
    "t3_m_bta_hbta_topo": IndependentTopology(
        "t3_m_bta_hbta_topo",
        2.0,
        1.0,
        27.99,
        3.67,
        348.0,
        "Zhang 2017 apparent 2 HBTA : 1 TOPO : 1 metal slope basis",
    ),
}


SAPONIFICATION_BY_SOURCE = {
    "shan_2025_water_17_2258": 1.0,
    "zhang_2017_spt_188_167_173": 0.0,
    "project_selected_smackover_feed": 1.0,
}


def saponification_fraction(condition: Condition) -> float:
    """Return only a source-declared initial organic inventory fraction."""

    if condition.saponification_fraction is not None:
        return condition.saponification_fraction
    try:
        return SAPONIFICATION_BY_SOURCE[condition.source_id]
    except KeyError as exc:
        raise ValueError(f"no source-declared saponification fraction for {condition.source_id}") from exc


def phase_supports(
    condition: Condition,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Use the smaller exact support when no protonated carrier was supplied."""

    carbonate = CARBONATE_SPECIES if condition.total_carbonate_mol_l_aqueous > 0.0 else ()
    if saponification_fraction(condition) == 1.0 and not carbonate:
        return (
            tuple(identity for identity in AQUEOUS_SUPPORT if identity != "hydrogen-cation"),
            tuple(identity for identity in ORGANIC_SUPPORT if identity != "free-carrier"),
        )
    if saponification_fraction(condition) == 1.0:
        return AQUEOUS_SUPPORT + carbonate, tuple(identity for identity in ORGANIC_SUPPORT if identity != "free-carrier")
    return AQUEOUS_SUPPORT + carbonate, ORGANIC_SUPPORT


def _coefficient(record: dict[str, Any], family: str) -> dict[str, Any]:
    return next(item for item in record["coefficients"] if item["family"] == family)


def _rename_component(record: dict[str, Any], old: str, new: str) -> None:
    record["component_id"] = new
    for coefficient in record.get("coefficients", []):
        coefficient["identity"] = str(coefficient["identity"]).replace(f"component/{old}/", f"component/{new}/")


def _pair(
    left: str,
    right: str,
    value: float,
    *,
    source_id: str,
    locator: str,
) -> dict[str, Any]:
    left, right = sorted((left, right))
    return {
        "component_id_a": left,
        "component_id_b": right,
        "coefficients": [
            {
                "identity": f"pair/{left}/{right}/k_ij",
                "family": "k_ij",
                "value": {"magnitude": value, "unit": "dimensionless"},
                "provenance": {
                    "source_id": source_id,
                    "locator": locator,
                    "domain_id": "hbta-topo-independent-network",
                },
            }
        ],
    }


def parameter_payload(topology: IndependentTopology, diluent_id: str, carbonate: bool = False) -> dict[str, Any]:
    """Build shared ligand-derived complex priors plus Held aqueous H+."""

    legacy = (
        LEGACY_TOPOLOGIES["zhang_apparent_2_1_1"]
        if topology.hbta_per_carrier == 2.0
        else LEGACY_TOPOLOGIES["shan_1_1_1"]
    )
    payload = deepcopy(legacy_parameter_payload(legacy, diluent_id))
    topology_document_id = topology.identity.replace("_", "-")
    payload["document_id"] = f"hbta-topo-{topology_document_id}-{diluent_id}-independent"
    payload["sources"].append(
        {
            "source_id": "held-2014-hydrogen",
            "citation": "Held et al., Chem. Eng. Res. Des. (2014)",
            "doi": "10.1016/j.cherd.2014.05.017",
            "use_basis": "Aqueous hydrogen-ion ePC-SAFT strategy-2 prior.",
        }
    )
    payload["sources"].append(
        {
            "source_id": "hbta-topo-independent-network",
            "citation": topology.source_basis,
            "use_basis": (
                "Declared model-form stoichiometry and shared ligand-derived "
                "free/Na/Li carrier priors; not unique pure-component parameters."
            ),
        }
    )
    payload["domains"].append({"domain_id": "hbta-topo-independent-network", "kind": "unknown"})

    by_id = {row["component_id"]: row for row in payload["components"]}
    structural = {
        "source_id": "hbta-topo-independent-network",
        "locator": f"shared ligand-derived {topology.identity} structural prior",
        "domain_id": "hbta-topo-independent-network",
    }
    ligand_mass = topology.hbta_per_carrier * HBTA_MOLAR_MASS + topology.topo_per_carrier * TOPO_MOLAR_MASS
    masses = {
        "sodium-carrier": ligand_mass + NA_MOLAR_MASS - H_MOLAR_MASS,
        "lithium-carrier": ligand_mass + LI_MOLAR_MASS - H_MOLAR_MASS,
    }
    for identity, mass in masses.items():
        record = by_id[identity]
        record["fixed"]["molar_mass"] = {
            "value": {"magnitude": mass, "unit": "kilogram / mole"},
            "provenance": structural,
        }
        for family, value in (
            ("segment_count", topology.carrier_segment_count),
            ("segment_diameter", topology.carrier_segment_diameter_angstrom),
            (
                "dispersion_energy_over_k",
                topology.carrier_dispersion_energy_over_k_k,
            ),
        ):
            coefficient = _coefficient(record, family)
            coefficient["value"]["magnitude"] = value
            coefficient["provenance"] = structural

    free = deepcopy(by_id["sodium-carrier"])
    _rename_component(free, "sodium-carrier", "free-carrier")
    free["fixed"]["molar_mass"] = {
        "value": {"magnitude": ligand_mass, "unit": "kilogram / mole"},
        "provenance": structural,
    }
    free["fixed"]["charge_number"] = {
        "value": {"magnitude": 0, "unit": "elementary-charge"},
        "provenance": structural,
    }
    by_id["free-carrier"] = free

    hydrogen = deepcopy(by_id["lithium-cation"])
    _rename_component(hydrogen, "lithium-cation", "hydrogen-cation")
    hydrogen["fixed"]["molar_mass"]["value"]["magnitude"] = H_MOLAR_MASS
    hydrogen["fixed"]["charge_number"]["value"]["magnitude"] = 1
    h_provenance = {
        "source_id": "held-2014-hydrogen",
        "locator": "strategy-2 H+ row; electrolyte diameters 0.88 sigma",
        "domain_id": "hbta-topo-independent-network",
    }
    hydrogen["fixed"]["molar_mass"]["provenance"] = h_provenance
    hydrogen["fixed"]["charge_number"]["provenance"] = h_provenance
    h_values = {
        "segment_count": 1.0,
        "segment_diameter": 3.4654,
        "dispersion_energy_over_k": 500.0,
        "packing_diameter": 0.88 * 3.4654,
        "debye_huckel_diameter": 0.88 * 3.4654,
        "born_diameter": 3.4654,
        "solvation_factor": 1.0,
    }
    for coefficient in hydrogen["coefficients"]:
        coefficient["value"]["magnitude"] = h_values[coefficient["family"]]
        coefficient["provenance"] = h_provenance
    by_id["hydrogen-cation"] = hydrogen
    species = ALL_SPECIES if carbonate else SPECIES
    carbonate_provenance = {
        "source_id": "bulow-2021-carbonate",
        "locator": "Tables 2-3 HCO3-, CO3(2-) and CO2 rows; electrolyte diameters 0.88 sigma as for H+",
        "domain_id": "hbta-topo-independent-network",
    }
    if carbonate:
        payload["sources"].append(
            {
                "source_id": "bulow-2021-carbonate",
                "citation": "Bulow, Ince, Hirohama, Sadowski and Held, Ind. Eng. Chem. Res. 60 (2021) 6327-6336",
                "doi": "10.1021/acs.iecr.1c00176",
                "use_basis": "Aqueous carbonate ePC-SAFT priors (Held framework), as for the Held 2014 H+.",
            }
        )
    for identity, (mass, charge, segments, diameter, energy) in (CARBONATE_PARAMETERS.items() if carbonate else ()):
        template = hydrogen if charge else by_id["diluent"]
        record = deepcopy(template)
        _rename_component(record, template["component_id"], identity)
        record["fixed"]["molar_mass"] = {"value": {"magnitude": mass, "unit": "kilogram / mole"}, "provenance": carbonate_provenance}
        record["fixed"]["charge_number"] = {"value": {"magnitude": charge, "unit": "elementary-charge"}, "provenance": carbonate_provenance}
        values = {"segment_count": segments, "segment_diameter": diameter, "dispersion_energy_over_k": energy, "solvation_factor": 1.0}
        if charge:
            values.update(packing_diameter=0.88 * diameter, debye_huckel_diameter=0.88 * diameter, born_diameter=diameter)
        else:
            values["relative_permittivity"] = CO2_RELATIVE_PERMITTIVITY
        record["coefficients"] = [item for item in record["coefficients"] if item["family"] in values]
        for coefficient in record["coefficients"]:
            coefficient["value"]["magnitude"] = values[coefficient["family"]]
            coefficient["provenance"] = carbonate_provenance
        by_id[identity] = record

    payload["components"] = [by_id[identity] for identity in species]
    pairs = {frozenset((row["component_id_a"], row["component_id_b"])): row for row in payload["pairs"]}
    for identity in SPECIES:
        if identity == "free-carrier":
            continue
        pairs.setdefault(
            frozenset(("free-carrier", identity)),
            _pair(
                "free-carrier",
                identity,
                0.0,
                source_id="hbta-topo-independent-network",
                locator="zero cross-interaction prior before model-form correction",
            ),
        )
    for identity in SPECIES:
        if identity in {"hydrogen-cation", "lithium-cation", "sodium-cation"}:
            continue
        value = 0.0
        locator = "zero cross-interaction prior outside validated aqueous pairs"
        if identity == "water":
            value = 0.25
            locator = "Held 2014 strategy-2 water/H+ pair"
        elif identity == "chloride-anion":
            value = 0.654
            locator = "Held 2014 strategy-2 H+/Cl- pair"
        pairs.setdefault(
            frozenset(("hydrogen-cation", identity)),
            _pair(
                "hydrogen-cation",
                identity,
                value,
                source_id=(
                    "held-2014-hydrogen" if identity in {"water", "chloride-anion"} else "hbta-topo-independent-network"
                ),
                locator=locator,
            ),
        )
    for identity in species[len(SPECIES):]:
        for other in species:
            # exclude-same-sign-ion-pairs: anion/anion rows are inactive and must be omitted.
            if other != identity and ALL_CHARGES[ALL_SPECIES.index(identity)] * ALL_CHARGES[ALL_SPECIES.index(other)] <= 0:
                pairs.setdefault(
                    frozenset((identity, other)),
                    _pair(identity, other, CARBONATE_WATER_KIJ[identity] if other == "water" else 0.0,
                          source_id="bulow-2021-carbonate" if other == "water" else "hbta-topo-independent-network",
                          locator="Table 3 water pair" if other == "water" else "zero cross-interaction prior"),
                )
    payload["pairs"] = list(pairs.values())
    return payload


@lru_cache(maxsize=None)
def build_model(
    topology_id: str,
    diluent_id: str,
    components: tuple[str, ...] = SPECIES,
    complex_diluent_kij: float = 0.0,
) -> tuple[epcsaft.Parameters, epcsaft.Mixture]:
    payload = parameter_payload(TOPOLOGIES[topology_id], diluent_id, carbonate=bool(set(components) & set(CARBONATE_SPECIES)))
    pair = next(
        row
        for row in payload["pairs"]
        if {row["component_id_a"], row["component_id_b"]} == {"diluent", "lithium-carrier"}
    )
    coefficient = _coefficient(pair, "k_ij")
    coefficient["value"]["magnitude"] = complex_diluent_kij
    coefficient["provenance"]["locator"] = "caller-supplied complex/diluent interaction"
    parameters = epcsaft.Parameters.from_mapping(payload, components=components)
    return parameters, epcsaft.Mixture(parameters)


def _species(condition: Condition) -> tuple[str, ...]:
    return ALL_SPECIES if condition.total_carbonate_mol_l_aqueous > 0.0 else SPECIES


def _basis_matrix(species: tuple[str, ...] = SPECIES) -> tuple[tuple[float, ...], ...]:
    rows = (
        {"water": 1.0, "bicarbonate-anion": 3.0, "carbonate-anion": 3.0, "carbon-dioxide": 2.0},  # oxygen
        {"diluent": 1.0},
        {"lithium-cation": 1.0, "lithium-carrier": 1.0},
        {"sodium-cation": 1.0, "sodium-carrier": 1.0},
        {"chloride-anion": 1.0},
        {"free-carrier": 1.0, "sodium-carrier": 1.0, "lithium-carrier": 1.0},
        {"hydrogen-cation": 1.0, "free-carrier": 1.0, "bicarbonate-anion": 1.0, "carbon-dioxide": 2.0},  # exchangeable protons (H - 2 O + 6 C)
        {"bicarbonate-anion": 1.0, "carbonate-anion": 1.0, "carbon-dioxide": 1.0},  # carbon
    )
    return tuple(tuple(row.get(name, 0.0) for name in species) for row in rows[: 8 if species != SPECIES else 7])


def _carrier_amount(condition: Condition, topology: IndependentTopology) -> float:
    concentration = min(
        condition.hbta_mol_l_organic / topology.hbta_per_carrier,
        condition.topo_mol_l_organic / topology.topo_per_carrier,
    )
    return concentration * condition.organic_volume_l


def _declared_carrier_amounts(
    condition: Condition,
    topology: IndependentTopology,
) -> tuple[float, float, float]:
    """Return the incoming free-, sodium-, and lithium-carrier inventories."""

    carrier = _carrier_amount(condition, topology)
    explicit = (condition.li_mol_l_organic, condition.na_mol_l_organic)
    if explicit == (None, None):
        sodium_carrier = saponification_fraction(condition) * carrier
        return carrier - sodium_carrier, sodium_carrier, 0.0
    if any(value is None for value in explicit):
        raise ValueError("incoming organic Li and Na loadings must be supplied together")
    lithium_carrier = float(condition.li_mol_l_organic) * condition.organic_volume_l
    sodium_carrier = float(condition.na_mol_l_organic) * condition.organic_volume_l
    if lithium_carrier < 0.0 or sodium_carrier < 0.0:
        raise ValueError("incoming organic metal loadings must be nonnegative")
    if lithium_carrier + sodium_carrier > carrier + 1.0e-12:
        raise ValueError("incoming organic metal loading exceeds the declared carrier inventory")
    free_carrier = max(carrier - lithium_carrier - sodium_carrier, 0.0)
    if saponification_fraction(condition) == 1.0 and free_carrier > 1.0e-12:
        raise ValueError("fully saponified organic inventory cannot contain free carrier")
    return free_carrier, sodium_carrier, lithium_carrier


def _initial_amounts(
    condition: Condition,
    topology: IndependentTopology,
    model: epcsaft.Mixture,
    initialization_seed: tuple[float, float, float] = INITIALIZATION_SEEDS[0],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    li = condition.li_mol_l_aqueous * condition.aqueous_volume_l
    na = condition.na_mol_l_aqueous * condition.aqueous_volume_l
    free_carrier, sodium_carrier, lithium_carrier = _declared_carrier_amounts(condition, topology)
    saponification = saponification_fraction(condition)
    fully_saponified = saponification == 1.0
    hydrogen = (
        0.0
        if fully_saponified
        else (
            condition.hydrogen_mol_l_aqueous * condition.aqueous_volume_l
            if condition.hydrogen_mol_l_aqueous is not None
            else 10.0 ** (-condition.initial_ph) * condition.aqueous_volume_l
        )
    )

    sodium_fraction, free_carrier_fraction, lithium_exchange_fraction = initialization_seed
    sodium_seed = min(sodium_fraction * na, free_carrier_fraction * free_carrier)
    exchange_seed = min(
        lithium_exchange_fraction * li,
        0.99 * (sodium_carrier + sodium_seed),
    )
    li_aq = li - exchange_seed
    na_aq = na + exchange_seed - sodium_seed
    hydrogen += sodium_seed
    free_carrier -= sodium_seed
    sodium_carrier += sodium_seed - exchange_seed
    lithium_carrier += exchange_seed
    # Inorganic carbon enters speciated ideally at the initial pH; the Engine re-equilibrates it.
    carbon = condition.total_carbonate_mol_l_aqueous * condition.aqueous_volume_l
    k1, k2 = (10.0**value for value in carbonic_log10_k(condition.temperature_k))
    h = 10.0 ** (-condition.initial_ph)
    denominator = h * h + k1 * h + k1 * k2
    carbonate = tuple(carbon * term / denominator for term in (k1 * h, k1 * k2, h * h))
    charge_closed_chloride = li_aq + na_aq + hydrogen - carbonate[0] - 2.0 * carbonate[1]
    if condition.chloride_mol_l_aqueous is not None:
        supplied_chloride = condition.chloride_mol_l_aqueous * condition.aqueous_volume_l
        if abs(supplied_chloride - charge_closed_chloride) > 1.0e-8:
            raise ValueError("explicit aqueous chloride does not close the declared charge inventory")
        chloride = charge_closed_chloride
    else:
        chloride = charge_closed_chloride

    diluent_state = model.state(
        T=condition.temperature_k,
        P=PRESSURE_PA,
        x=tuple(float(name == "diluent") for name in model.component_ids),
        phase="liquid",
    )
    diluent = diluent_state.molar_density * condition.organic_volume_l * 1.0e-3
    extra = len(_species(condition)) - len(SPECIES)
    return (
        (55.508 * condition.aqueous_volume_l, 0.0, li_aq, na_aq, hydrogen, chloride, 0.0, 0.0, 0.0, *carbonate[:extra]),
        (0.0, diluent, 0.0, 0.0, 0.0, 0.0, free_carrier, sodium_carrier, lithium_carrier, *(0.0,) * extra),
    )


def _reactions(condition: Condition, ln_k_exchange: float, ln_k_na_uptake: float):
    """(stoichiometry, ln K, molal-referenced) triples; fitted constants are on the Engine's own basis."""
    reactions = [
        ({"lithium-cation": -1.0, "sodium-cation": 1.0, "sodium-carrier": -1.0, "lithium-carrier": 1.0}, ln_k_exchange, False)
    ]
    if saponification_fraction(condition) < 1.0:
        reactions.append(
            (
                {"sodium-cation": -1.0, "hydrogen-cation": 1.0, "free-carrier": -1.0, "sodium-carrier": 1.0},
                ln_k_na_uptake,
                False,
            )
        )
    if condition.total_carbonate_mol_l_aqueous > 0.0:
        first, second = (math.log(10.0) * value for value in carbonic_log10_k(condition.temperature_k))
        reactions += [
            ({"carbon-dioxide": -1.0, "water": -1.0, "bicarbonate-anion": 1.0, "hydrogen-cation": 1.0}, first, True),
            ({"bicarbonate-anion": -1.0, "carbonate-anion": 1.0, "hydrogen-cation": 1.0}, second, True),
        ]
    return reactions


def _engine_reaction(nu, ln_k, referenced, temperature_k):
    if not referenced:
        return equilibrium.Reaction(nu, equilibrium_constant=math.exp(ln_k))
    # Constant at the condition temperature only; carbonate states carry no temperature dependence.
    constant = equilibrium.ReactionLogPolynomial(
        ln_k, 0.0, 0.0, 0.0, temperature_k, temperature_k, temperature_k, CARBONATE_STANDARD_STATE
    )
    basis = equilibrium.ReferenceSourceBasis.CommonMolalityInfiniteDilution
    return equilibrium.Reaction(nu, correlation=constant, reference=equilibrium.ReactionReference(basis))


def _neutral_reference(model: epcsaft.Mixture) -> equilibrium.NeutralReference:
    """Charge-neutral dilution path in water along which the Engine converts the molal source constants: every
    other component at one ratio, with H+ closing the charge (exactly, in binary). The 2**-13 (1.2e-4) ratio runs the Engine's fixed 1e-6 to 1e-14
    mol/kg ladder at 1e-10 to 1e-18 mol/kg per solute; at ratio 1 the z = 2 carbonate Debye-Hueckel term,
    which falls only as sqrt(m), still exceeds the ladder's terminal-change tolerance. The infinite-dilution limit
    does not depend on the path; _molal_offsets checks it independently."""
    charges = dict(zip(ALL_SPECIES, ALL_CHARGES))
    ratios = {name: 2.0**-13 for name in model.component_ids if name != "water"}
    ratios["hydrogen-cation"] -= math.fsum(charges[name] * ratio for name, ratio in ratios.items())
    return equilibrium.NeutralReference(solvent="water", solute_molality_ratios=ratios, standard_molality_mol_per_kg=1.0)


def _molal_offsets(model: epcsaft.Mixture, temperature_k: float) -> dict[str, float]:
    """ln a_i = ln(c_i) + mu_res_i/RT - offset_i: pure-water basis for water; molal infinite-dilution basis
    (single-ion trace at mole fraction 1e-20, where the Debye-Hueckel residue is below 5e-9) for H+ and the carbonate species."""
    ids = model.component_ids
    water_index = ids.index("water")
    water = model.state(temperature_k, P=PRESSURE_PA, x=tuple(float(i == water_index) for i in range(len(ids))), phase="liquid")
    offsets = {"water": math.log(water.molar_density) + water.residual_chemical_potential_over_rt[water_index]}
    ln_mass_density = math.log(water.molar_density * 0.01801528)  # c_i/m_i at infinite dilution, kg/m3
    for name in ("hydrogen-cation", *CARBONATE_SPECIES):
        i = ids.index(name)
        x = [0.0] * len(ids)
        x[water_index], x[i] = 1.0 - 1.0e-20, 1.0e-20
        trace = model.state(temperature_k, P=PRESSURE_PA, x=tuple(x), phase="liquid")
        offsets[name] = ln_mass_density + trace.residual_chemical_potential_over_rt[i]
    return offsets


def _problem(condition, topology, model, reactions, initialization_seed):
    starts = _initial_amounts(condition, topology, model, initialization_seed)
    species = _species(condition)
    feed = {name: math.fsum(row[species.index(name)] for row in starts) for name in model.component_ids}
    phases = []
    for name, amounts, support in zip(("aqueous", "organic"), starts, phase_supports(condition), strict=True):
        total = math.fsum(amounts)
        phases.append(
            equilibrium.Phase(
                name,
                kind="liquid",
                support=support,
                amount=equilibrium.Free(),
                composition_guess=[amounts[species.index(s)] / total for s in support],
            )
        )
    return equilibrium.Problem(
        T=condition.temperature_k,
        P=PRESSURE_PA,
        feed=equilibrium.Amounts(feed),
        phases=phases,
        row_tolerance=1e-10,
        reactions=[_engine_reaction(*reaction, condition.temperature_k) for reaction in reactions],
        neutral_reference=_neutral_reference(model) if any(referenced for *_, referenced in reactions) else None,
    )


def _physical_state(condition, model, problem, compiled, result, reactions):
    """Recompute intensive chemistry and extensive stationarity on the returned state."""
    count = len(model.component_ids)
    species = _species(condition)
    amounts, potentials, states = [], {}, []
    for p, support in enumerate(phase_supports(condition)):
        x = tuple(result.mole_fractions[p * count : (p + 1) * count])
        state = model.state(condition.temperature_k, rho=result.molar_densities[p], x=x)
        states.append(state)
        amounts.append(
            {
                name: result.amounts[p] * x[model.component_ids.index(name)] if name in model.component_ids else 0.0
                for name in species
            }
        )
        for name in support:
            i = model.component_ids.index(name)
            potentials[name] = math.log(x[i] * state.molar_density) + state.residual_chemical_potential_over_rt[i]
    balance = max(
        abs(
            math.fsum(
                b * (amounts[0][name] + amounts[1][name] - problem.feed.values.get(name, 0.0))
                for b, name in zip(row, species)
            )
        )
        for row in _basis_matrix(species)
    )
    affinity = max(abs(math.fsum(nu[name] * potentials[name] for name in nu) - k) for nu, k, ref in reactions if not ref)
    carbonate = {}
    if any(referenced for *_, referenced in reactions):
        # Independent of the Engine's reference conversion: molal activities from trace-dilution offsets.
        offsets = _molal_offsets(model, condition.temperature_k)
        carbonate = {
            "referenced_reaction_affinity_inf_norm": max(
                abs(math.fsum(nu[name] * (potentials[name] - offsets[name]) for name in nu) - k)
                for nu, k, referenced in reactions
                if referenced
            ),
            "ph_molal_activity": -(potentials["hydrogen-cation"] - offsets["hydrogen-cation"]) / math.log(10.0),
            "engine_reference_status": [str(item.status) for item in result.neutral_reference_diagnostics],
        }
    point = list(result.coordinates[: compiled.primal_size])
    gradient, jacobian = compiled.objective_gradient(point), compiled.jacobian(point)
    constraint_rows = [
        i for i, row in enumerate(compiled.rows) if str(row.kind).split(".")[-1] not in ("Chemical", "Pressure")
    ]
    kkt = max(
        abs(
            gradient[j]
            + math.fsum(
                mu * jacobian[i * compiled.primal_size + j]
                for mu, i in zip(result.constraint_multipliers, constraint_rows, strict=True)
            )
        )
        for j in range(compiled.primal_size)
    )
    absent = max(
        abs(amounts[p][name])
        for p, support in enumerate(phase_supports(condition))
        for name in species
        if name not in support
    )
    charge = max(abs(math.fsum(z * phase[name] for z, name in zip(ALL_CHARGES, species))) for phase in amounts)
    distinctness = max(
        abs(amounts[0][name] / result.amounts[0] - amounts[1][name] / result.amounts[1]) for name in species
    )
    return amounts, {
        "objective_over_rt": compiled.evaluate(point).objective,
        "balance_inf_norm": balance,
        "reaction_affinity_inf_norm": affinity,
        # Historical phase-direction alias; disjoint supports add no shared-species transfer row.
        "transfer_chemical_potential_inf_norm": affinity,
        "pressure_relative_inf_norm": max(abs(state.pressure / PRESSURE_PA - 1.0) for state in states),
        "kkt_stationarity_inf_norm": kkt,
        "charge_inf_norm": charge,
        "reduced_hessian_min_eigenvalue": result.reduced_hessian_minimum,
        "phase_distinctness": distinctness,
        "declared_direction_rank": count - compiled.rank,
        "exact_absence_inf_norm": absent,
        "aqueous_volume_model_l": result.amounts[0] / result.molar_densities[0] * 1000,
        "organic_volume_model_l": result.amounts[1] / result.molar_densities[1] * 1000,
        **carbonate,
    }


def _evaluate_result(condition, model, problem, compiled, result, reactions, attempt):
    attempt.update(
        {
            "solver_status": result.solver_status,
            "engine_success": result.success,
            "requested_tolerance_met": result.requested_tolerance_met,
            "residuals": list(result.residuals),
            "classified_residuals": list(result.classified_residuals),
            "rows": [
                {"kind": str(row.kind), "phase": row.phase, "index": row.index, "eqid": row.eqid} for row in result.rows
            ],
            "convergence": {
                "stop": str(result.convergence.stop),
                "residual": result.convergence.residual,
                "floor": result.convergence.floor,
                "iterations": result.iterations,
            },
            "jacobian_rank": result.jacobian_rank,
            "unknown_count": result.unknown_count,
            "topology_event": str(result.topology_event),
            "topology_phase": result.topology_phase,
            "topology_pair": result.topology_pair,
            "diagnostic": result.message,
        }
    )
    failure = {"code": result.solver_status, "diagnostic": result.message}
    if not result.success and result.topology_event.name != "None_":
        failure["code"] = result.topology_event.name
        attempt["stopped_by"] = "certificate_check"
    elif not result.success and (
        result.solver_status == "SUCCESS"
        or (result.solver_status == "STOP_AT_TINY_STEP" and result.convergence.stop.name == "converged")
    ):
        failure["code"] = (
            "physical_rejection"
            if result.convergence.stop.name in ("converged", "converged_to_floor")
            else "not_converged"
        )
        attempt["stopped_by"] = "certificate_check"
    amounts, diagnostics = None, None
    if result.success:
        amounts, diagnostics = _physical_state(condition, model, problem, compiled, result, reactions)
        attempt.update(diagnostics)
        regular = (
            all(math.isfinite(value) for value in diagnostics.values() if isinstance(value, float))
            and result.topology_event.name == "None_"
            and result.jacobian_rank == result.unknown_count
            and diagnostics["declared_direction_rank"] == len(reactions)
            and diagnostics["charge_inf_norm"] <= 1e-9
            and diagnostics["exact_absence_inf_norm"] <= 1e-14
            and diagnostics["pressure_relative_inf_norm"] <= PRESSURE_RELATIVE_LIMIT
            and diagnostics["kkt_stationarity_inf_norm"] <= 1e-7  # curvature: the Engine's band test (#220)
            and diagnostics["phase_distinctness"] > 1e-6
        )
        strict = (
            regular
            and result.requested_tolerance_met
            and diagnostics["balance_inf_norm"] <= 1e-9
            and diagnostics["reaction_affinity_inf_norm"] <= 1e-6
            and diagnostics.get("referenced_reaction_affinity_inf_norm", 0.0) <= 1e-6
        )
        candidate = (
            regular
            and diagnostics["balance_inf_norm"] <= 1e-8
            and diagnostics["reaction_affinity_inf_norm"] <= APPLICATION_AFFINITY_LIMIT
        )
        if strict or candidate:
            attempt["status"] = "evaluated" if strict else "application_accepted_candidate"
            attempt["stopped_by"] = "none" if strict else "certificate_check"
            failure = (
                None
                if strict
                else {
                    "code": "requested_tolerance_unmet"
                    if not result.requested_tolerance_met
                    else "application_strict_criteria_unmet",
                    "diagnostic": "finite state passed candidate limits but failed strict acceptance; see requested tolerance and physical residuals",
                }
            )
        else:
            attempt["stopped_by"] = "certificate_check"
            failure = {"code": "physical_rejection", "diagnostic": "returned state failed application criteria"}
    attempt["failure"] = failure
    return amounts, diagnostics


@lru_cache(maxsize=None)
def solve_condition(
    condition: Condition,
    topology_id: str,
    diluent_id: str,
    ln_k_exchange: float,
    ln_k_na_uptake: float,
    *,
    complex_diluent_kij: float = 0.0,
    initialization_seeds: tuple[tuple[float, float, float], ...] = INITIALIZATION_SEEDS,
) -> dict[str, Any]:
    if not initialization_seeds:
        raise ValueError("at least one finite initialization is required")
    topology = TOPOLOGIES[topology_id]
    components = tuple(name for name in _species(condition) if any(name in support for support in phase_supports(condition)))
    reactions = _reactions(condition, ln_k_exchange, ln_k_na_uptake)
    attempts, accepted = [], []
    for seed in initialization_seeds:
        started = time.perf_counter()
        attempt = {
            "initialization_seed": seed,
            "status": "non_evaluable",
            "attempted": "yes",
            "solver_status": "exception",
            "stopped_by": "solver",
        }
        try:
            _, model = build_model(topology_id, diluent_id, components, complex_diluent_kij)
            problem = _problem(condition, topology, model, reactions, seed)
            compiled = equilibrium.compile_problem(model, problem)
            result = equilibrium.solve_equilibrium(model, problem)
            amounts, diagnostics = _evaluate_result(condition, model, problem, compiled, result, reactions, attempt)
            if attempt["status"] != "non_evaluable":
                strict = attempt["status"] == "evaluated"
                rank = (
                    not strict,
                    diagnostics["objective_over_rt"] if strict else diagnostics["reaction_affinity_inf_norm"],
                    diagnostics["kkt_stationarity_inf_norm"],
                )
                accepted.append((rank, len(attempts), amounts, diagnostics))
        except Exception as error:
            attempt["failure"] = {"code": type(error).__name__, "diagnostic": str(error)}
            attempt["diagnostic"] = str(error)
        attempt["wall_time_seconds"] = time.perf_counter() - started
        attempts.append(attempt)
    chosen = min(accepted, key=lambda item: item[0]) if accepted else None
    index = chosen[1] if chosen else len(attempts) - 1
    selected = attempts[index]
    receipt = {
        **selected,
        "condition_id": condition.identity,
        "source_id": condition.source_id,
        "application_status": selected["status"],
        "engine_status": "evaluated" if selected["status"] == "evaluated" else "non_evaluable",
        "engine_failure": selected["failure"],
        "engine_diagnostic": selected["diagnostic"],
        "initialization_attempt_count": len(attempts),
        "attempts": attempts,
        "accepted_initialization_seed": selected["initialization_seed"] if chosen else None,
        "wall_time_seconds": sum(item["wall_time_seconds"] for item in attempts),
        "next_probe": "" if chosen else "inspect the retained start failures and physical residuals",
    }
    if chosen is None:
        return receipt
    _, _, phase_amounts, diagnostics = chosen
    li_org, na_org = phase_amounts[1]["lithium-carrier"], phase_amounts[1]["sodium-carrier"]
    li_aq, na_aq = phase_amounts[0]["lithium-cation"], phase_amounts[0]["sodium-cation"]
    h_aq, chloride_aq = phase_amounts[0]["hydrogen-cation"], phase_amounts[0]["chloride-anion"]
    initial_free_carrier, initial_na_org, initial_li_org = _declared_carrier_amounts(condition, topology)
    carrier_total = initial_free_carrier + initial_na_org + initial_li_org
    li_total = li_org + li_aq
    li_aq_feed = condition.li_mol_l_aqueous * condition.aqueous_volume_l
    na_aq_feed = condition.na_mol_l_aqueous * condition.aqueous_volume_l
    net_li = li_org - initial_li_org
    net_na = na_aq_feed - na_aq
    li_distribution = (
        (li_org / condition.organic_volume_l) / (li_aq / condition.aqueous_volume_l) if li_aq > 0.0 else math.inf
    )
    na_distribution = (
        (net_na / condition.organic_volume_l) / (na_aq / condition.aqueous_volume_l) if na_aq > 0.0 else math.inf
    )
    return {
        **receipt,
        "role": condition.role,
        "phase_amounts_mol": phase_amounts,
        "topology_id": topology_id,
        "diluent_id": diluent_id,
        "saponification_fraction": saponification_fraction(condition),
        "ln_k_exchange": ln_k_exchange,
        "ln_k_na_uptake": ln_k_na_uptake,
        "complex_diluent_kij": complex_diluent_kij,
        "li_extraction_percent": 100.0 * li_org / li_total,
        "na_extraction_percent": (100.0 * net_na / na_aq_feed if na_aq_feed > 0.0 else None),
        "li_distribution": li_distribution,
        "na_distribution": na_distribution,
        "li_na_selectivity": (li_distribution / na_distribution if na_distribution > 0.0 else None),
        "loaded_organic_li_fraction": li_org / (li_org + na_org),
        "incoming_li_org_mol": initial_li_org,
        "incoming_na_org_mol": initial_na_org,
        "incoming_free_carrier_org_mol": initial_free_carrier,
        "li_org_mol": li_org,
        "na_org_mol": na_org,
        "free_carrier_org_mol": max(carrier_total - li_org - na_org, 0.0),
        "carrier_total_org_mol": carrier_total,
        "li_net_transferred_mol": net_li,
        "li_net_transfer_fraction_aq_feed": (net_li / li_aq_feed if li_aq_feed > 0.0 else None),
        "na_net_extracted_mol": net_na,
        "na_net_transfer_fraction_aq_feed": (net_na / na_aq_feed if na_aq_feed > 0.0 else None),
        "li_aq_mol": li_aq,
        "na_aq_mol": na_aq,
        "hydrogen_aq_mol": h_aq,
        "chloride_aq_mol": chloride_aq,
        "d_li_extraction_d_kij_exact": None,
        "parameter_action_availability": "not_requested_no_current_consumer",
        **diagnostics,
    }


def parameter_document(topology_id: str, diluent_id: str) -> dict[str, Any]:
    return parameter_payload(TOPOLOGIES[topology_id], diluent_id)
