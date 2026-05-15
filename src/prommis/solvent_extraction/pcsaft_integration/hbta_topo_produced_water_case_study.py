#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2024 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""
HBTA/TOPO produced-water case-study artifact for the PrOMMiS ePC-SAFT integration lane.

This generator builds a staged MSContactor-style mass-balance artifact for a non-ionic HBTA/TOPO
produced-water process and labels model-support gaps. The primary bridge path is epcsaft_pse; when
bridge outputs are unavailable, it produces deterministic, documented fallbacks so downstream
artifact consumers still receive a complete structure.
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

_SRC_ROOT = Path(__file__).resolve().parents[3]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from prommis.solvent_extraction.pcsaft_integration._epcsaft_pse_bridge import ensure_epcsaft_pse_imports

OUT_DIR = Path(__file__).resolve().parent
OUT_CSV = OUT_DIR / "hbta_topo_produced_water_case_study.csv"
OUT_MD = OUT_DIR / "hbta_topo_produced_water_case_study.md"

PAPER_TITLE = "Influence Mechanism of Coexisting Ions on the Extraction Efficiency of Lithium from Oil and Gas Field Water"
PAPER_DOI = "10.3390/w17152258"

SIMULATED_FEED_COMPOSITION_MG_L = {
    "Li+": 60.0,
    "K+": 560.0,
    "Na+": 10900.0,
    "Ca2+": 8600.0,
    "Mg2+": 820.0,
    "Ba2+": 560.0,
    "Cl-": 47220.0,
}
SIMULATED_FEED_PH = 6.0

# Internal map used throughout by the HBTA/TOPO model
SIMULATED_FEED_SPECIES_MG_L = {
    "Li+": 60.0,
    "K+": 560.0,
    "Na+": 10900.0,
    "Mg+2": 820.0,
    "Ca+2": 8600.0,
    "Ba+2": 560.0,
    "Cl-": 47220.0,
}

TRACKED_IONS = ("Li+", "Na+", "K+", "Mg+2", "Ca+2", "Ba+2", "Cl-")
SOLVENT_SPECIES = ("TOP", "Hexane")
ALL_ARTIFACT_SPECIES = TRACKED_IONS + SOLVENT_SPECIES
BASELINE_SPECIES = ("Li+", "Na+", "K+", "Mg+2", "Ca+2", "Ba+2", "Cl-", "TOP", "Hexane")

MW_KG_PER_MOL = {
    "Li+": 6.94e-3,
    "K+": 39.0983e-3,
    "Na+": 22.98976928e-3,
    "Mg+2": 24.305e-3,
    "Ca+2": 40.078e-3,
    "Ba+2": 137.327e-3,
    "Cl-": 35.45e-3,
    "TOP": 0.38294,
    "Hexane": 0.08617848,
}
LI2CO3_MW_KG_MOL = 73.89e-3

AQ_FLOW_L_H = 1000.0
OA_RATIO = 1.0
TOP_MOL_L = 0.015
HEXANE_MOL_L = 0.6548 / 0.08617848

DEFAULT_CA_REMOVAL_FRACTION = 0.98
DEFAULT_MG_REMOVAL_FRACTION = 0.95

FALLBACK_ETA_PCT_BY_SPECIES = {
    "Li+": 75.2481,
    "Na+": 0.0904,
    "K+": 3.2648,
    "Mg+2": 5.0820,
    "Ca+2": 0.6537,
    "Ba+2": 17.2005,
    "Cl-": 0.0007,
    "TOP": 0.0,
    "Hexane": 0.0,
}

LITERATURE_RESULTS = {
    "three_stage_li_pct": 97.17,
    "average_transfer_efficiency_pct": 91.35,
    "li2co3_purity_pct": 99.28,
    "na_effect_note": "Na+ can promote Li extraction; Li/K separation factor stayed above 13 in the paper's K+ sweep.",
    "ca_mg_note": "Ca2+ and Mg2+ strongly suppress Li extraction and should be removed before extraction.",
}

PRETREATMENT_BASIS = (
    "The paper filters produced water, removes suspended matter, and applies impurity-removal "
    "(especially Ca/Mg control) before solvent extraction."
)
SOLVENT_BASIS_HIGH = (
    "0.15 mol/L HBTA + 0.15 mol/L TOPO in sulfonated kerosene, 100% saponification with 2 mol/L NaOH, "
    "O/A = 1:1, 6 min extraction."
)
SOLVENT_BASIS_POLISH = (
    "After impurity removal, the paper also reports a three-stage cross-flow polishing case using "
    "0.015 mol/L HBTA + 0.015 mol/L TOPO, 100% saponification, and O/A = 1:1."
)


def _mg_L_to_mol_L(value_mg_L: float, molecular_weight_kg_mol: float) -> float:
    return float(value_mg_L) * 1e-6 / molecular_weight_kg_mol


def _mol_L_to_mg_L(value_mol_L: float, molecular_weight_kg_mol: float) -> float:
    return float(value_mol_L) * molecular_weight_kg_mol * 1e6


def _sanitize_mg_map(raw: Mapping[str, float]) -> dict[str, float]:
    return {species: max(0.0, float(raw.get(species, 0.0))) for species in BASELINE_SPECIES}


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _format_species_map(values: Mapping[str, float], *, species: tuple[str, ...] = BASELINE_SPECIES) -> str:
    return ", ".join(f"{key}={values.get(key, 0.0):.6g}" for key in species)


def _format_competing_ion_preferences(preferences: Mapping[str, Any]) -> str:
    ordered_keys = ["Li+", "Na+", "K+", "Mg+2", "Ca+2", "Ba+2"]
    return ", ".join(f"{key}={float(preferences[key]):.2f}" for key in ordered_keys if key in preferences)


def _format_gap_status(is_bridge: bool) -> str:
    return "bridge_backed" if is_bridge else "deterministic_fallback"


def _species_in_g_per_L(value_mg_L: float, species: str) -> float:
    return float(value_mg_L) * 1e-3 * MW_KG_PER_MOL[species]


@dataclass(frozen=True)
class StageBalance:
    stage: int
    source_model: str
    source_status: str
    feed_aq_mg_L: dict[str, float]
    feed_org_mg_L: dict[str, float]
    aqueous_out_mg_L: dict[str, float]
    organic_out_mg_L: dict[str, float]
    eta_pct: dict[str, float]
    feed_mass_mg: dict[str, float]
    out_mass_mg: dict[str, float]
    material_balance_gap: dict[str, float]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CostingHook:
    unit: str
    input_proxy: str
    formula: str
    units: str
    status: str
    notes: str


@dataclass(frozen=True)
class PretreatmentProfile:
    inlet_mg_L: dict[str, float]
    outlet_mg_L: dict[str, float]
    removed_mg_L: dict[str, float]
    removal_fraction: dict[str, float]
    status: str
    notes: str


@dataclass(frozen=True)
class ArtifactRow:
    section: str
    item: str
    bridge_proxy: str
    literature: str
    units: str
    notes: str


def _solvent_stage_feed_mg_L(*, top_mol_L: float = TOP_MOL_L, hexane_mol_L: float = HEXANE_MOL_L) -> dict[str, float]:
    return {
        "TOP": _mol_L_to_mg_L(top_mol_L, MW_KG_PER_MOL["TOP"]),
        "Hexane": _mol_L_to_mg_L(hexane_mol_L, MW_KG_PER_MOL["Hexane"]),
    }


def _stage_mass_balance(
    feed_aq_mg_L: Mapping[str, float],
    feed_org_mg_L: Mapping[str, float],
    aq_out_mg_L: Mapping[str, float],
    org_out_mg_L: Mapping[str, float],
    *,
    aq_flow_L_h: float = AQ_FLOW_L_H,
    org_flow_L_h: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    feed_mass = {}
    out_mass = {}
    gap = {}
    for species in BASELINE_SPECIES:
        feed_mass[species] = float(feed_aq_mg_L.get(species, 0.0)) * aq_flow_L_h + float(
            feed_org_mg_L.get(species, 0.0)
        ) * org_flow_L_h
        out_mass[species] = float(aq_out_mg_L.get(species, 0.0)) * aq_flow_L_h + float(
            org_out_mg_L.get(species, 0.0)
        ) * org_flow_L_h
        denom = max(feed_mass[species], 1e-30)
        gap[species] = out_mass[species] / denom
    return feed_mass, out_mass, gap


def _run_stage_fallback(
    feed_aq_mg_L: Mapping[str, float],
    *,
    stage: int,
    top_mg_L: float,
    hexane_mg_L: float,
    org_flow_L_h: float,
    stage_index_scale: float = 1.0,
) -> StageBalance:
    feed = _sanitize_mg_map(feed_aq_mg_L)
    org_in = {
        "TOP": top_mg_L,
        "Hexane": hexane_mg_L,
    }
    aq_out: dict[str, float] = {}
    org_out: dict[str, float] = dict(org_in)
    eta_pct = {}
    source_scale = min(1.0, 0.85 + 0.05 * stage)

    for species in TRACKED_IONS:
        eta = min(99.999, max(0.0, FALLBACK_ETA_PCT_BY_SPECIES.get(species, 0.0)))
        eta_pct[species] = eta
        eta_frac = (eta * stage_index_scale * source_scale) / 100.0
        eta_frac = max(0.0, min(0.999999, eta_frac))
        aq_out[species] = feed[species] * (1.0 - eta_frac)
        org_out[species] = feed[species] * eta_frac
    for species in SOLVENT_SPECIES:
        aq_out.setdefault(species, 0.0)
        org_out.setdefault(species, org_in[species] * org_flow_L_h / max(AQ_FLOW_L_H, 1e-30))
        eta_pct.setdefault(species, 0.0)

    feed_mass, out_mass, gap = _stage_mass_balance(
        feed_aq_mg_L=feed,
        feed_org_mg_L=org_in,
        aq_out_mg_L=aq_out,
        org_out_mg_L=org_out,
        org_flow_L_h=org_flow_L_h,
    )
    return StageBalance(
        stage=stage,
        source_model="deterministic_stage_fallback",
        source_status="deterministic_fallback",
        feed_aq_mg_L=feed,
        feed_org_mg_L=org_in,
        aqueous_out_mg_L=aq_out,
        organic_out_mg_L=org_out,
        eta_pct=eta_pct,
        feed_mass_mg=feed_mass,
        out_mass_mg=out_mass,
        material_balance_gap=gap,
        metadata={
            "fallback_reason": "bridge_unavailable_or_unfit",
            "stage_fraction_scale": source_scale,
            "bridge_backed": False,
            "source_basis_note": (
                "No ePC-SAFT-backed multi-stage bridge output was available for all requested attempts. "
                "This deterministic proxy keeps unit-model I/O structure and explicit stage-level assumptions."
            ),
        },
    )


def _run_stage_bridge(
    feed_aq_mg_L: Mapping[str, float],
    *,
    stage: int,
    run_lle_stage: Callable[[Any], Any],
    StageInputs: type[Any],
    top_mol_L: float,
    hexane_mol_L: float,
    oa_ratio: float,
    cl_mg_L: float | None = None,
) -> StageBalance:
    feed = _sanitize_mg_map(feed_aq_mg_L)
    attempts = (
        {
            "label": "requested_hbta_topo",
            "stage_model": "shan_2025_hbta_topo",
            "parameter_overlay": "shan_2025_hbta_topo",
            "reaction_model": "shan_2025_hbta_topo",
            "top_basis_note": "requested HBTA/TOPO non-ionic stage model",
            "attempted": True,
        },
        {
            "label": "current_bridge_fallback",
            "stage_model": "case_study_selective",
            "parameter_overlay": "gando_2025_stage1",
            "reaction_model": None,
            "top_basis_note": "bridge fallback stage model (existing PrOMMiS-compatible case-study mode)",
            "attempted": True,
        },
    )

    org_in = _solvent_stage_feed_mg_L(top_mol_L=top_mol_L, hexane_mol_L=hexane_mol_L)
    aq_flow = AQ_FLOW_L_H
    org_flow = aq_flow * max(float(oa_ratio), 1e-30)

    last_error = None
    for attempt in attempts:
        inputs = StageInputs(
            li_mg_L=feed["Li+"],
            na_mg_L=feed["Na+"],
            k_mg_L=feed["K+"],
            mg_mg_L=feed["Mg+2"],
            ca_mg_L=feed["Ca+2"],
            ba_mg_L=feed["Ba+2"],
            cl_mg_L=cl_mg_L if cl_mg_L is not None else feed["Cl-"],
            aq_water_mol_L=55.34,
            top_mol_L=top_mol_L,
            hexane_mol_L=hexane_mol_L,
            o_to_a_ratio=oa_ratio,
            stage_model=attempt["stage_model"],
            reaction_model=attempt["reaction_model"],
            parameter_overlay=attempt["parameter_overlay"],
            compute_raw_diagnostics=True,
            ph=SIMULATED_FEED_PH,
        )
        try:
            out = run_lle_stage(inputs)
            aq_out = {species: _mol_L_to_mg_L(out.aqueous_out_mol_L.get(species, 0.0), MW_KG_PER_MOL[species]) for species in BASELINE_SPECIES}
            org_out = {species: _mol_L_to_mg_L(out.organic_out_mol_L.get(species, 0.0), MW_KG_PER_MOL[species]) for species in BASELINE_SPECIES}
            feed_mass, out_mass, gap = _stage_mass_balance(feed, org_in, aq_out, org_out, org_flow_L_h=org_flow)
            eta_pct = {}
            for species in TRACKED_IONS:
                in_mg = feed[species]
                if in_mg > 0:
                    eta_pct[species] = max(
                        0.0,
                        100.0 * max(0.0, in_mg - aq_out[species]) / in_mg,
                    )
                else:
                    eta_pct[species] = 0.0

            for species in ("K+", "Mg+2", "Ca+2", "Ba+2"):
                if species == "K+" and _is_finite(out.diagnostics.get("eta_k_pct")):
                    eta_pct[species] = float(out.diagnostics.get("eta_k_pct"))
                if species == "Mg+2" and _is_finite(out.diagnostics.get("eta_mg_pct")):
                    eta_pct[species] = float(out.diagnostics.get("eta_mg_pct"))
                if species == "Ca+2" and _is_finite(out.diagnostics.get("eta_ca_pct")):
                    eta_pct[species] = float(out.diagnostics.get("eta_ca_pct"))
                if species == "Ba+2" and _is_finite(out.diagnostics.get("eta_ba_pct")):
                    eta_pct[species] = float(out.diagnostics.get("eta_ba_pct"))
            for species in SOLVENT_SPECIES:
                eta_pct.setdefault(species, 0.0)

            source_status = "bridge_backed"
            metadata = {
                "bridge_backed": True,
                "stage_model": attempt["stage_model"],
                "reaction_model": attempt["reaction_model"],
                "attempt": attempt["label"],
                "selection_preference": out.diagnostics.get("selection_preference"),
                "kappa_li": _is_finite(out.kappa_li) and float(out.kappa_li),
                "kappa_na": _is_finite(out.kappa_na) and float(out.kappa_na),
                "kappa_cl": _is_finite(out.kappa_cl) and float(out.kappa_cl),
                "D_li": _is_finite(out.D_li) and float(out.D_li),
                "li_na_selectivity": _is_finite(out.li_na_selectivity) and float(out.li_na_selectivity),
                "eta_li_pct": _is_finite(out.eta_li_pct) and float(out.eta_li_pct),
                "eta_na_pct": _is_finite(out.eta_na_pct) and float(out.eta_na_pct),
                "feed_diagnostics": out.diagnostics,
                "top_basis_note": attempt["top_basis_note"],
            }
            return StageBalance(
                stage=stage,
                source_model=str(attempt["stage_model"]),
                source_status=source_status,
                feed_aq_mg_L=feed,
                feed_org_mg_L=org_in,
                aqueous_out_mg_L=aq_out,
                organic_out_mg_L=org_out,
                eta_pct=eta_pct,
                feed_mass_mg=feed_mass,
                out_mass_mg=out_mass,
                material_balance_gap=gap,
                metadata=metadata,
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

    fallback = _run_stage_fallback(
        feed,
        stage=stage,
        top_mg_L=org_in["TOP"],
        hexane_mg_L=org_in["Hexane"],
        org_flow_L_h=org_flow,
        stage_index_scale=1.0,
    )
    fallback_metadata = dict(fallback.metadata)
    fallback_metadata["bridge_backed"] = False
    fallback_metadata["bridge_failures"] = str(last_error)
    fallback_metadata["stage_model"] = "deterministic_stage_fallback"
    return StageBalance(
        stage=fallback.stage,
        source_model="deterministic_stage_fallback",
        source_status=fallback.source_status,
        feed_aq_mg_L=fallback.feed_aq_mg_L,
        feed_org_mg_L=fallback.feed_org_mg_L,
        aqueous_out_mg_L=fallback.aqueous_out_mg_L,
        organic_out_mg_L=fallback.organic_out_mg_L,
        eta_pct=fallback.eta_pct,
        feed_mass_mg=fallback.feed_mass_mg,
        out_mass_mg=fallback.out_mass_mg,
        material_balance_gap=fallback.material_balance_gap,
        metadata=fallback_metadata,
    )


def _apply_pretreatment(
    feed_mg_L: Mapping[str, float],
    *,
    ca_removal_fraction: float,
    mg_removal_fraction: float,
    status: str = "assumed",
) -> PretreatmentProfile:
    inlet = _sanitize_mg_map(feed_mg_L)
    outlet = dict(inlet)
    removed = {"Ca+2": 0.0, "Mg+2": 0.0}
    removal_fraction = {"Ca+2": max(0.0, min(1.0, ca_removal_fraction)), "Mg+2": max(0.0, min(1.0, mg_removal_fraction))}
    for species in ("Ca+2", "Mg+2"):
        removed_amt = inlet[species] * removal_fraction[species]
        removed[species] = removed_amt
        outlet[species] = max(0.0, inlet[species] - removed_amt)
    if status == "assumed":
        notes = "Assumed staged ion-reduction pretreatment: Ca/Mg reduction implemented to mirror paper direction; no unit model is integrated in this artifact yet."
    else:
        notes = "Pretreatment feed conditioning from bridge-fed diagnostics."
    return PretreatmentProfile(
        inlet_mg_L=inlet,
        outlet_mg_L=outlet,
        removed_mg_L=removed,
        removal_fraction=removal_fraction,
        status="implemented" if status == "assumed" else "bridge_driven",
        notes=notes,
    )


def _build_staging(
    *,
    n_stages: int,
    run_stage: Callable[[Mapping[str, float], int], StageBalance],
) -> list[StageBalance]:
    pretreated = _apply_pretreatment(
        SIMULATED_FEED_SPECIES_MG_L,
        ca_removal_fraction=DEFAULT_CA_REMOVAL_FRACTION,
        mg_removal_fraction=DEFAULT_MG_REMOVAL_FRACTION,
    )
    current_feed = pretreated.outlet_mg_L
    stages: list[StageBalance] = []
    for stage_idx in range(1, max(1, int(n_stages)) + 1):
        stage_result = run_stage(current_feed, stage_idx)
        stages.append(stage_result)
        current_feed = stage_result.aqueous_out_mg_L
    return stages


def _build_costing_hooks(
    stages: Sequence[StageBalance],
    pretreatment: PretreatmentProfile,
    *,
    aq_flow_L_h: float = AQ_FLOW_L_H,
    oa_ratio: float = OA_RATIO,
) -> list[CostingHook]:
    final_stage = stages[-1]
    first_stage = stages[0]
    final_li_mg_L = max(final_stage.aqueous_out_mg_L["Li+"], 1e-30)
    initial_li_mg_L = max(pretreatment.outlet_mg_L["Li+"], 1e-30)
    extracted_li_mg_per_h = (initial_li_mg_L - final_li_mg_L) * aq_flow_L_h
    extracted_li_mol_per_h = max(0.0, extracted_li_mg_per_h) * 1e-6 / MW_KG_PER_MOL["Li+"]
    li2co3_mol_per_h = 0.5 * extracted_li_mol_per_h * (LITERATURE_RESULTS["li2co3_purity_pct"] / 100.0)
    li2co3_kg_h = li2co3_mol_per_h * LI2CO3_MW_KG_MOL

    # Rough proxy hooks: these are intentionally documented as bridge-level or staged-gap placeholders.
    return [
        CostingHook(
            unit="pretreatment",
            input_proxy=(
                f"Q={aq_flow_L_h:.0f} L/h, Ca removal fraction={pretreatment.removal_fraction['Ca+2']:.3f}, "
                f"Mg removal fraction={pretreatment.removal_fraction['Mg+2']:.3f}"
            ),
            formula="m_removal = Q * C_in * x_remove * MW * 1e-6",
            units="kg/h (for Ca, Mg as separate species)",
            status="bridge_gap",
            notes="No integrated pretreatment reactor is wired yet; this row preserves handoff inputs for the unit model.",
        ),
        CostingHook(
            unit="extraction_contacting",
            input_proxy=(
                f"Stages={len(stages)}, oa_ratio={oa_ratio:.3f}, Q={aq_flow_L_h:.0f} L/h, "
                f"Li extracted={extracted_li_mg_per_h:.4g} mg/h"
            ),
            formula="η_stage from bridge/fallback + stage mass balance to populate extracted loads",
            units="kg/h equivalent species loads",
            status=_format_gap_status(all(stage.source_status == "bridge_backed" for stage in stages)),
            notes="This is staged MSContactor extraction/contacting handoff data; refine unit-level utility with solvent circulation model.",
        ),
        CostingHook(
            unit="stripping",
            input_proxy="No explicit strip solvent, pH swing, or temperature policy yet represented",
            formula="m_strip_agent and energy = model placeholders",
            units="kg/h, MJ/h",
            status="gap",
            notes="Gap: back-extraction/recover-and-regenerate flows are required before product solvent recycle claims.",
        ),
        CostingHook(
            unit="solvent_makeup_loss",
            input_proxy=(
                f"Extracted TOP proxy={first_stage.organic_out_mg_L.get('TOP', 0.0):.6g} mg/L, "
                f"final TOP proxy={final_stage.organic_out_mg_L.get('TOP', 0.0):.6g} mg/L"
            ),
            formula="loss proxy = max(0, TOP_in - TOP_out) * flow",
            units="kg/h",
            status="bridge_gap",
            notes="No solvent makeup/loss correlations are embedded; use this as explicit CAPEX/OPEX hook inputs.",
        ),
        CostingHook(
            unit="concentration",
            input_proxy=(
                f"captured Li proxy={extracted_li_mg_per_h:.6g} mg/h across {len(stages)} stages; "
                f"final raffinate Li={final_li_mg_L:.6g} mg/L"
            ),
            formula="concentrator duty and recycle ratios require explicit strip liquor flow and target Li concentration",
            units="kg/h, L/h",
            status="bridge_gap",
            notes="Concentrator/crystallizer block is outside current bridge path; hook captures captured-Li basis for later equipment sizing.",
        ),
        CostingHook(
            unit="li2co3_precipitation",
            input_proxy=f"Target purity={LITERATURE_RESULTS['li2co3_purity_pct']:.2f}%; Li2CO3 proxy={li2co3_kg_h:.6g} kg/h",
            formula="n_Li2CO3 = 0.5 * n_Li,captured; m_Li2CO3 = n_Li2CO3 * 73.89e-3",
            units="kg/h",
            status="bridge_gap",
            notes=f"Paper anchor: Li2CO3 purity {LITERATURE_RESULTS['li2co3_purity_pct']:.2f}% from field train; precipitation yields not yet coupled.",
        ),
    ]


def _build_rows(
    stages: Sequence[StageBalance],
    pretreatment: PretreatmentProfile,
    costing_hooks: Sequence[CostingHook],
    bridge_status: Mapping[str, Any],
) -> list[ArtifactRow]:
    rows: list[ArtifactRow] = []
    rows.append(
        ArtifactRow(
            section="feed",
            item="Simulated oil-and-gas field water basis",
            bridge_proxy=_format_species_map(pretreatment.inlet_mg_L),
            literature=f"Li 60, K 560, Na 10900, Ca 8600, Mg 820, Ba 560, Cl 47220; pH {SIMULATED_FEED_PH:.1f}",
            units="mg/L",
            notes="HBTA/TOPO reported composition for the Shan paper feed case, retained as primary literature anchor.",
        )
    )
    rows.append(
        ArtifactRow(
            section="pretreatment",
            item="Pretreatment / impurity removal (Ca+2, Mg+2)",
            bridge_proxy=f"Ca+2 removed {pretreatment.removal_fraction['Ca+2']:.1%}, Mg+2 removed {pretreatment.removal_fraction['Mg+2']:.1%}",
            literature=PRETREATMENT_BASIS,
            units="mg/L",
            notes=pretreatment.notes,
        )
    )
    rows.append(
        ArtifactRow(
            section="pretreatment",
            item="Feed after pretreatment",
            bridge_proxy=_format_species_map(pretreatment.outlet_mg_L),
            literature=(
                f"Assumed retained composition for stage 1 feed; Ca and Mg reductions are explicitly labeled as process assumptions."
            ),
            units="mg/L",
            notes="These values are now passed into Stage 1 as aqueous feed for staged MSContactor bridging.",
        )
    )
    rows.append(
        ArtifactRow(
            section="solvent",
            item="Extraction solvent basis",
            bridge_proxy="TOP proxy at 0.015 mol/L with O/A = 1:1",
            literature=SOLVENT_BASIS_HIGH,
            units="mol/L",
            notes=(
                "The paper first reports the 0.15 mol/L HBTA + 0.15 mol/L TOPO extraction basis, "
                "then a dilute three-stage polishing case at 0.015 mol/L each."
            ),
        )
    )
    rows.append(
        ArtifactRow(
            section="solvent",
            item="Polishing mode basis",
            bridge_proxy=(
                f"Top/TOPO in each polishing step={_solvent_stage_feed_mg_L()['TOP'] / 1e6 / MW_KG_PER_MOL['TOP']:.6g} mol/L"
            ),
            literature=SOLVENT_BASIS_POLISH,
            units="mol/L",
            notes="Stage-wise artifact is generated as three fresh-organic, equal-o/a cross-flow passes.",
        )
    )

    for stage in stages:
        rows.append(
            ArtifactRow(
                section="mscontactor",
                item=f"Stage {stage.stage} source",
                bridge_proxy=f"{stage.source_model} [{_format_gap_status(stage.source_status == 'bridge_backed')}]",
                literature=f"source_status={stage.source_status}, model={stage.source_model}",
                units="status",
                notes=f"Aqueous/organic coupling uses stage-wise bridge handoff; metadata key={bridge_status.get('selected_attempt', 'bridge')}.",
            )
        )
        rows.append(
            ArtifactRow(
                section="mscontactor",
                item=f"Stage {stage.stage} aqueous feed (mg/L)",
                bridge_proxy=_format_species_map(stage.feed_aq_mg_L, species=BASELINE_SPECIES),
                literature="MSContactor input profile for stage feed",
                units="mg/L",
                notes="Feeds are passed stage-to-stage from preceding stage aqueous outlet.",
            )
        )
        rows.append(
            ArtifactRow(
                section="mscontactor",
                item=f"Stage {stage.stage} aqueous outlet (mg/L)",
                bridge_proxy=_format_species_map(stage.aqueous_out_mg_L, species=BASELINE_SPECIES),
                literature="Stage 1 and subsequent aqueous outlet states available for stage chaining.",
                units="mg/L",
                notes="Used as feed for the next stage when building the staged profile.",
            )
        )
        rows.append(
            ArtifactRow(
                section="mscontactor",
                item=f"Stage {stage.stage} organic outlet loadings (mg/L)",
                bridge_proxy=_format_species_map(stage.organic_out_mg_L, species=BASELINE_SPECIES),
                literature="Organic loadings carry stage-wise competition across ions and are bridge-back when available.",
                units="mg/L",
                notes=f"Solvent basis source note: {stage.metadata.get('top_basis_note', 'unknown')}.",
            )
        )
        rows.append(
            ArtifactRow(
                section="mscontactor",
                item=f"Stage {stage.stage} extraction performance",
                bridge_proxy=(
                    f"Li {stage.eta_pct['Li+']:.4f}%, Na {stage.eta_pct['Na+']:.4f}%, "
                    f"K {stage.eta_pct.get('K+', 0.0):.4f}%, Mg {stage.eta_pct.get('Mg+2', 0.0):.4f}%, "
                    f"Ca {stage.eta_pct.get('Ca+2', 0.0):.4f}%, Ba {stage.eta_pct.get('Ba+2', 0.0):.4f}%"
                ),
                literature="Published benchmark context: >90% Li after pretreatment and 97.17% across 3 stages.",
                units="%",
                notes=f"Selection preference vector: {_format_competing_ion_preferences(stage.metadata.get('selection_preference', {})) or 'N/A'}.",
            )
        )
        rows.append(
            ArtifactRow(
                section="mscontactor",
                item=f"Stage {stage.stage} mass balance gap",
                bridge_proxy=_format_species_map(stage.material_balance_gap, species=("Li+", "Na+", "K+", "Mg+2", "Ca+2")),
                literature="Gap metric = (m_out,mg/h)/(m_in,mg/h) for species-by-species mass closure check.",
                units="dimensionless",
                notes=f"Absolute mass-balance deviations are expected at this stage where species-specific fallback assumptions apply.",
            )
        )

    rows.append(
        ArtifactRow(
            section="cumulative",
            item="Cumulative Li extraction",
            bridge_proxy=f"{100.0*(1.0 - stages[-1].aqueous_out_mg_L['Li+'] / max(pretreatment.outlet_mg_L['Li+'], 1e-30)):.8f}%",
            literature="Paper 3-stage polishing anchor: 97.17%",
            units="%",
            notes="Computed from staged bridge outputs; if this differs from the paper anchor, treat it as a bridge-calibration gap rather than a final process claim.",
        )
    )
    rows.append(
        ArtifactRow(
            section="costing",
            item="Costing readiness summary",
            bridge_proxy="Pretreatment/extraction/contact/stripping/solvent/co-precip hooks populated as explicit placeholders.",
            literature="Paper also includes concentration and Li2CO3 precipitation references.",
            units="status",
            notes="Hook tables in this artifact are intentionally labeled with bridge/knowledge gaps.",
        )
    )

    for hook in costing_hooks:
        rows.append(
            ArtifactRow(
                section="costing",
                item=f"{hook.unit}",
                bridge_proxy=hook.input_proxy,
                literature=hook.formula,
                units=hook.units,
                notes=f"[{hook.status}] {hook.notes}",
            )
        )
    return rows


def _solve_stages(
    *,
    n_stages: int = 3,
    stage_runner: Callable[[Mapping[str, float], int], StageBalance] | None = None,
    stage_model: str = "shan_2025_hbta_topo",
) -> tuple[list[StageBalance], PretreatmentProfile, dict[str, Any], Mapping[str, Any]]:
    try:
        if stage_runner is None:
            _, run_lle_stage, _, _, StageInputs = ensure_epcsaft_pse_imports()

            def stage_runner_default(feed_state: Mapping[str, float], stage_idx: int) -> StageBalance:
                return _run_stage_bridge(
                    feed_state,
                    stage=stage_idx,
                    run_lle_stage=run_lle_stage,
                    StageInputs=StageInputs,
                    top_mol_L=TOP_MOL_L,
                    hexane_mol_L=HEXANE_MOL_L,
                    oa_ratio=OA_RATIO,
                )

            stage_runner = stage_runner_default
            bridge_used = {"mode": "bridge_attempted", "requested_stage_model": stage_model}
        else:
            bridge_used = {"mode": "injectable_stage_runner", "requested_stage_model": stage_model}
    except Exception as exc:
        bridge_used = {"mode": "deterministic_fallback", "reason": f"{type(exc).__name__}: {exc}"}

        def _fallback_runner(feed_state: Mapping[str, float], stage_idx: int) -> StageBalance:
            org_in = _solvent_stage_feed_mg_L(top_mol_L=TOP_MOL_L, hexane_mol_L=HEXANE_MOL_L)
            return _run_stage_fallback(
                feed_state,
                stage=stage_idx,
                top_mg_L=org_in["TOP"],
                hexane_mg_L=org_in["Hexane"],
                org_flow_L_h=AQ_FLOW_L_H * OA_RATIO,
            )

        stage_runner = _fallback_runner

    pretreatment = _apply_pretreatment(
        SIMULATED_FEED_SPECIES_MG_L,
        ca_removal_fraction=DEFAULT_CA_REMOVAL_FRACTION,
        mg_removal_fraction=DEFAULT_MG_REMOVAL_FRACTION,
    )
    current_feed = pretreatment.outlet_mg_L
    balances: list[StageBalance] = []
    for stage_idx in range(1, int(max(1, n_stages)) + 1):
        assert stage_runner is not None
        balances.append(stage_runner(current_feed, stage_idx))
        current_feed = balances[-1].aqueous_out_mg_L

    bridge_status: dict[str, Any] = {"mode": bridge_used["mode"], "selected_attempt": bridge_used.get("requested_stage_model")}
    if "reason" in bridge_used:
        bridge_status["reason"] = bridge_used["reason"]
    return balances, pretreatment, bridge_status, {"n_stages": n_stages}


def build_case_study_artifact(
    *,
    n_stages: int = 3,
    stage_runner: Callable[[Mapping[str, float], int], StageBalance] | None = None,
) -> dict[str, Any]:
    stages, pretreatment, bridge_status, debug = _solve_stages(n_stages=n_stages, stage_runner=stage_runner)
    costing_hooks = _build_costing_hooks(stages, pretreatment)
    rows = _build_rows(stages=stages, pretreatment=pretreatment, costing_hooks=costing_hooks, bridge_status=bridge_status)
    return {
        "pretreatment": {
            "basis": PRETREATMENT_BASIS,
            "inlet_mg_L": pretreatment.inlet_mg_L,
            "outlet_mg_L": pretreatment.outlet_mg_L,
            "removed_mg_L": pretreatment.removed_mg_L,
            "removal_fraction": pretreatment.removal_fraction,
            "status": pretreatment.status,
            "notes": pretreatment.notes,
        },
        "bridge_status": bridge_status,
        "debug": debug,
        "rows": rows,
        "stages": stages,
        "costing_hooks": costing_hooks,
        "literature": LITERATURE_RESULTS,
        "total_ionic_basis_mg_L": sum(SIMULATED_FEED_SPECIES_MG_L.values()),
    }


def _write_csv(artifact: dict[str, Any], out_csv: Path) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "item", "bridge_proxy", "literature", "units", "notes"])
        for row in artifact["rows"]:
            writer.writerow([row.section, row.item, row.bridge_proxy, row.literature, row.units, row.notes])
    return out_csv


def _write_markdown(artifact: dict[str, Any], out_md: Path) -> Path:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    pretreatment = artifact["pretreatment"]
    stages: list[StageBalance] = artifact["stages"]
    costing_hooks: list[CostingHook] = artifact["costing_hooks"]
    rows: list[ArtifactRow] = artifact["rows"]

    lines: list[str] = []
    lines.append("# HBTA/TOPO Produced-Water Case Study for PrOMMiS")
    lines.append("")
    lines.append("## Basis")
    lines.append("")
    lines.append(f"- Paper: {PAPER_TITLE}")
    lines.append(f"- DOI: {PAPER_DOI}")
    lines.append(f"- Basis feed (raw): {_format_species_map(SIMULATED_FEED_SPECIES_MG_L)}")
    lines.append(f"- Process scale anchor: 15 L actual oil-and-gas field water train with filtering, pretreatment, extraction, back-extraction, concentration, and Li2CO3 precipitation")
    lines.append("")
    lines.append("## Pretreatment")
    lines.append("")
    lines.append(f"- {PRETREATMENT_BASIS}")
    lines.append(
        f"- Applied assumptions: Ca+2 removal {pretreatment['removal_fraction']['Ca+2']:.1%}, "
        f"Mg+2 removal {pretreatment['removal_fraction']['Mg+2']:.1%}"
    )
    lines.append(f"- Inlet: {_format_species_map(pretreatment['inlet_mg_L'])}")
    lines.append(f"- Outlet: {_format_species_map(pretreatment['outlet_mg_L'])}")
    lines.append("")
    lines.append("## Stage-wise MSContactor Material Balance")
    lines.append("")
    lines.append("| Stage | Species | Aq feed (mg/L) | Aq outlet (mg/L) | Org outlet (mg/L) | ηstage (%) | MB ratio |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for stage in stages:
        for species in BASELINE_SPECIES:
            lines.append(
                f"| {stage.stage} | {species} | {stage.feed_aq_mg_L.get(species, 0.0):.6g} | "
                f"{stage.aqueous_out_mg_L.get(species, 0.0):.6g} | {stage.organic_out_mg_L.get(species, 0.0):.6g} | "
                f"{stage.eta_pct.get(species, 0.0):.4f} | {stage.material_balance_gap.get(species, 0.0):.6f} |"
            )
    lines.append("")
    lines.append("## Costing Hooks")
    lines.append("")
    lines.append("| Unit | Input proxy | Formula | Units | Status | Notes |")
    lines.append("|---|---|---|---|---|---|")
    for hook in costing_hooks:
        lines.append(f"| {hook.unit} | {hook.input_proxy} | {hook.formula} | {hook.units} | {hook.status} | {hook.notes} |")
    lines.append("")
    lines.append("## Paper Anchors")
    lines.append("")
    lines.append(f"- Three-stage cross-flow Li extraction after impurity removal: {artifact['literature']['three_stage_li_pct']:.2f}%")
    lines.append(
        f"- Average Li transfer efficiency in 15 L process: {artifact['literature']['average_transfer_efficiency_pct']:.2f}%"
    )
    lines.append(f"- Li2CO3 product purity: {artifact['literature']['li2co3_purity_pct']:.2f}%")
    lines.append(f"- {artifact['literature']['na_effect_note']}")
    lines.append(f"- {artifact['literature']['ca_mg_note']}")
    lines.append("")
    lines.append("## Bridge Status")
    lines.append("")
    lines.append(f"- Bridge mode: {artifact['bridge_status']['mode']}")
    if "reason" in artifact["bridge_status"]:
        lines.append(f"- Bridge reason: {artifact['bridge_status']['reason']}")
    lines.append(f"- Requested attempt label: {artifact['bridge_status'].get('selected_attempt', 'requested attempt not exposed')}")
    lines.append("")
    lines.append("## Table Notes")
    lines.append("")
    lines.append("Rows in this table mirror the stage-level unit inputs/outputs and explicitly retain assumptions where support is missing.")
    lines.append("- Bridge-backed rows are those emitted from `epcsaft_pse` stage evaluation.")
    lines.append("- Rows marked with `gap`/`bridge_gap` rely on explicit deterministic fallback assumptions.")
    lines.append("- A bridge-derived cumulative recovery above the paper anchor is a calibration signal, not a claim that the process beats the reported experiment.")
    lines.append("")
    for row in rows:
        pass
    lines.append("## Transfer and Stage Summary")
    lines.append("")
    lines.append("| Section | Item | Bridge proxy | Literature | Units | Notes |")
    lines.append("|---|---|---|---|---|---|")
    for row in rows:
        lines.append(f"| {row.section} | {row.item} | {row.bridge_proxy} | {row.literature} | {row.units} | {row.notes} |")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_md


def export_hbta_topo_case_study_artifacts(
    artifact: dict[str, Any],
    *,
    out_csv: str | Path = OUT_CSV,
    out_md: str | Path = OUT_MD,
) -> tuple[Path, Path]:
    csv_path = _write_csv(artifact, Path(out_csv))
    md_path = _write_markdown(artifact, Path(out_md))
    return csv_path, md_path


def main() -> None:
    artifact = build_case_study_artifact()
    csv_path, md_path = export_hbta_topo_case_study_artifacts(artifact)
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
