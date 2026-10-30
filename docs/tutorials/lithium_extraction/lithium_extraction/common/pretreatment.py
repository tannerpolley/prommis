"""Divalent removal ahead of extraction (Lithium #162): NaOH then Na2CO3 precipitation softening of MS-2.

Neither #158 reaction set represents Ca, Mg, Sr or Ba, and beta-diketone/TOPO systems load them before Li, so they
are removed first, by conventional softening: NaOH precipitates Mg(OH)2, then Na2CO3 precipitates CaCO3, SrCO3 and
BaCO3. Shan et al. (2025, Section 3.5.1) use this sequence ahead of HBTA/TOPO extraction of oil and gas field water
(Mg about 93 %, Ca 99 %, Ba 92 % removed) and state that the Li content is unchanged (no number given). The excess base is neutralized with HCl back to the
measured feed pH.

- **Removal fractions and Li loss.** Wang et al. (2024, Appl. Geochem., doi:10.1016/j.apgeochem.2024.106126),
  carbonate precipitation of the Montney well-composite produced water.
  - Before (Table 1): Ca 18,200, Mg 1,600, Sr 1,320 mg/L. After (Table 4): 7.9, 10.5 and 1.2 mg/L.
  - Section 3.4.1.1 states a Li loss of about 25 %. The Montney row itself is 61.1 -> 40.0 mg/L (35 %), and the
    Duvernay row 22 %.
  - Ba is not reported and takes Sr's fraction (8.39 mg/L in MS-2).
  - Wang's process is two-step: ammonium carbonate (1.3 mol per mol divalent) precipitates the divalents, then ammonium bicarbonate removes about 30 % of the Na as NaHCO3 (Section 2.4; Section 3.4.1.1 also lists
    Na2CO3 among the chemicals added); the source gives no mechanism for the Li loss. The removal fractions are
    used as a transferable basis; the Li loss is not specific to NaOH/Na2CO3 softening.
  - Here the dose is Na reagents at 1.1 times
    stoichiometric, the repository's pretreatment register value (`naoh_excess_factor`, `na2co3_excess_factor`),
    a screening assumption.
- **Reagent prices.** Zhang et al. (2026) Table 8, 2024-2026 averages: NaOH 285, Na2CO3 200 and HCl (31 %) 100 USD
  per tonne. NaOH and HCl are also the register's `naoh_price` and `hcl_31pct_solution_price`.
- **Na.** Softening with Na reagents trades each removed Ca, Sr or Ba for two Na and each Mg for two Na. The
  extraction feed therefore carries about 5.2 mol/L Na instead of MS-2's 2.8, which strengthens R2's Na competition.
  That is close to NaCl saturation (6.1 mol/kg of water at 25 C). Saturation is not checked. Wang's 1.3 dose would
  add another 0.4 mol/L of Na. Reagents are dosed without dilution.
- **Sludge.** Disposal is reported separately at the register's screening values (70 % moisture, 0.05 USD/kg wet)
  and is not in the reagent cost.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv

from lithium_extraction.common.economics.hbta_topo import load_screening_cost_assumptions
from lithium_extraction.common.thermodynamics.hbta_topo.constants import NA_MOLAR_MASS_G_MOL
from lithium_extraction.common.support.paths import REPO_ROOT

MS2_SOURCE = REPO_ROOT / "data/reference/produced_water/smackover_selected_base_feed_ms2.csv"
PRETREATMENT_REGISTER = REPO_ROOT / "data/reference/tea/pretreatment_cost_assumptions.csv"
MOLAR_MASS_G_MOL = {"Ca": 40.078, "Mg": 24.305, "Sr": 87.62, "Ba": 137.327, "Cl": 35.453, "Li": 6.94}
# Wang et al. (2024) Montney well composite, Table 1 -> Table 4 (mg/L); Ba takes Sr's fraction (not reported).
WANG_2024_BEFORE_AFTER_MG_L = {"Ca": (18_200.0, 7.9), "Mg": (1_600.0, 10.5), "Sr": (1_320.0, 1.2)}
WANG_2024_LI_LOSS_FRACTION = 0.25  # Section 3.4.1.1 "about 25 %"; rows give 22 % (Duvernay) and 35 % (Montney)
# Zhang et al. (2026, J. Environ. Chem. Eng. 14:123500) Table 8, 2024-2026 average; not in the screening register.
NA2CO3_USD_KG = 0.200
NAOH_KG_MOL, NA2CO3_KG_MOL, HCL_KG_MOL = 0.039997, 0.105988, 0.036461
SLUDGE_KG_MOL = {"Ca": 0.100087, "Mg": 0.058320, "Sr": 0.147630, "Ba": 0.197336}  # CaCO3, Mg(OH)2, SrCO3, BaCO3


@dataclass(frozen=True)
class Pretreatment:
    """Per cubic metre of MS-2 brine; mol/L are of the softened extraction feed."""

    li_feed_mol_l: float
    na_feed_mol_l: float
    li_yield: float
    removal_fraction: dict[str, float]
    residual_divalent_mol_l: dict[str, float]
    naoh_kg_m3: float
    na2co3_kg_m3: float
    hcl_31pct_kg_m3: float
    reagent_cost_usd_m3: float
    sludge_dry_kg_m3: float
    sludge_disposal_usd_m3: float
    na_added_mol_l: float


def _register(name: str) -> float:
    """Base value of a pretreatment register row (screening assumptions)."""
    with PRETREATMENT_REGISTER.open(newline="", encoding="utf-8") as stream:
        return next(float(row["base_value"]) for row in csv.DictReader(stream) if row["parameter"] == name)


def _ms2_mg_l() -> dict[str, float]:
    with MS2_SOURCE.open(newline="", encoding="utf-8") as stream:
        return {row["analyte"]: float(row["value"]) for row in csv.DictReader(stream)
                if row["analyte"] in ("Li", "Na", "Ca", "Mg", "Sr", "Ba")}


def softening() -> Pretreatment:
    brine = _ms2_mg_l()
    prices = load_screening_cost_assumptions()  # NaOH and HCl (Zhang et al. 2026 Table 8), as for saponification
    carbonate_dose, hydroxide_dose = _register("na2co3_excess_factor"), _register("naoh_excess_factor")
    moisture, disposal = _register("sludge_moisture_fraction"), _register("sludge_disposal_usd_kg")
    removal = {m: 1.0 - after / before for m, (before, after) in WANG_2024_BEFORE_AFTER_MG_L.items()}
    removal["Ba"] = removal["Sr"]
    divalent = {m: brine[m] / 1000.0 / MOLAR_MASS_G_MOL[m] for m in ("Ca", "Mg", "Sr", "Ba")}
    carbonate_metals = ("Ca", "Sr", "Ba")
    na2co3 = carbonate_dose * sum(divalent[m] for m in carbonate_metals)  # mol/L
    naoh = hydroxide_dose * 2.0 * divalent["Mg"]
    # Excess carbonate and hydroxide neutralized back to the feed pH: CO3(2-) + 2 H+ -> CO2 + H2O; OH- + H+ -> H2O.
    hcl = 2.0 * (carbonate_dose - 1.0) * sum(divalent[m] for m in carbonate_metals) + (hydroxide_dose - 1.0) * 2.0 * divalent["Mg"]
    na_added = 2.0 * na2co3 + naoh
    sludge = sum(removal[m] * divalent[m] * SLUDGE_KG_MOL[m] for m in divalent) * 1000.0  # kg/m3
    naoh_kg, na2co3_kg = naoh * NAOH_KG_MOL * 1000.0, na2co3 * NA2CO3_KG_MOL * 1000.0
    hcl_kg = hcl * HCL_KG_MOL * 1000.0 / 0.31
    li_yield = 1.0 - WANG_2024_LI_LOSS_FRACTION
    return Pretreatment(
        li_feed_mol_l=li_yield * brine["Li"] / 1000.0 / MOLAR_MASS_G_MOL["Li"],
        na_feed_mol_l=brine["Na"] / 1000.0 / NA_MOLAR_MASS_G_MOL + na_added,
        li_yield=li_yield,
        removal_fraction=removal,
        residual_divalent_mol_l={m: (1.0 - removal[m]) * divalent[m] for m in divalent},
        naoh_kg_m3=naoh_kg,
        na2co3_kg_m3=na2co3_kg,
        hcl_31pct_kg_m3=hcl_kg,
        reagent_cost_usd_m3=naoh_kg * prices.naoh_price + na2co3_kg * NA2CO3_USD_KG
        + hcl_kg * prices.hcl_31pct_solution_price,
        sludge_dry_kg_m3=sludge,
        sludge_disposal_usd_m3=sludge / (1.0 - moisture) * disposal,
        na_added_mol_l=na_added,
    )


def softening_record() -> dict:
    return asdict(softening())
