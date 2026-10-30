"""Stage inventory and countercurrent cascade on a saponified HBTA/TOPO organic (Lithium #166).

The reaction set (#158 A: R1, R2; C: R1, R2, R5) is an input, with water autoprotolysis RW whenever the stage is
above pH 7. R1 Li+ + HA + T = LiA.T + H+, R2 Na+ + HA + T = NaA.T + H+, R5 HA(org) = H+ + BTA-(aq). The ln K are
inputs; the mixture, reference states, reactions and strict certificate are those of
``analyses/shared/hbta_topo_configured_reactive_lle/scripts/mechanism_candidates.py``.

An aqueous stream is carried as Li+, Na+, BTA- and its net acidity H+ - OH- (mol per litre of aqueous feed);
chloride closes its charge. The organic is carried as LiA.T, NaA.T and its BTA-bearing extractant HA + LiA.T + NaA.T
(which R5 depletes); TOPO stays in the organic. These fix every conserved row of the next stage, so OH- and BTA-
leaving one stage enter the next. A negative acidity enters as NaOH: ``mc.inventory`` places it on free extractant as
NaA.T (NaOH + HA + T = NaA.T + H2O, the same element inventory) or, without free extractant, as free OH-.
"""

from __future__ import annotations

import math

from analyses.shared.hbta_topo_configured_reactive_lle.scripts import mechanism_candidates as mc
from lithium_extraction.prototypes.produced_water_sx_prototype.cascade import sweep_stages

TEMPERATURE_K = 298.15
TRACE_H = 1.0e-10  # mc.State's default H+ before base, mol/L
CLOSURE_MOL = 1.0e-9


def acidity_at_ph(ph: float) -> float:
    """Net acidity H+ - OH- (mol/L) of an unbuffered solution at ``ph``, with the Harned-Robinson molal Kw."""
    h = 10.0 ** -ph
    return h - math.exp(mc.ln_k_at("RW", 0.0, TEMPERATURE_K)) / h


def stage_state(identity: str, *, organic_to_aqueous: float, extractant: float, li: float, na: float, acidity: float,
                li_org: float = 0.0, na_org: float = 0.0, hbta: float | None = None, bta: float = 0.0) -> mc.State:
    """One stage per litre of aqueous. ``extractant`` is TOPO, mol/L of organic; ``hbta`` is the organic's
    BTA-bearing extractant HA + LiA.T + NaA.T (``extractant`` when omitted); ``li_org`` and ``na_org`` are the
    entering LiA.T and NaA.T, mol/L of organic; ``bta`` is the entering aqueous BTA-, mol/L."""
    hbta = extractant if hbta is None else hbta
    if acidity > 0.0:
        return mc.State(identity, TEMPERATURE_K, organic_to_aqueous, li, na, hbta, extractant,
                        hydrogen=acidity, li_org=li_org, na_org=na_org, bta=bta)
    base = TRACE_H - acidity  # NaOH; its Na is taken from the stream's Na so the Na row is unchanged
    return mc.State(identity, TEMPERATURE_K, organic_to_aqueous, li, na - base, hbta, extractant,
                    base=base, hydrogen=TRACE_H, li_org=li_org, na_org=na_org, bta=bta)


def outlets(result: dict, organic_to_aqueous: float, hbta_in: float) -> tuple[dict[str, float], dict[str, float]]:
    """Aqueous (Li, Na, acidity, BTA-; mol per L aqueous) and organic (LiA.T, NaA.T, HA + LiA.T + NaA.T; mol per L
    organic) leaving a stage. Without R5 no BTA- forms and the organic's BTA inventory ``hbta_in`` is unchanged."""
    aq, org = result["phase_amounts_mol"]
    v = organic_to_aqueous
    aqueous = {"li": aq["lithium-cation"], "na": aq["sodium-cation"],
               "acidity": aq.get("hydrogen-cation", 0.0) - aq.get("hydroxide-anion", 0.0),
               "bta": aq.get("bta-anion", 0.0)}
    li, na = org.get("lithium-carrier", 0.0) / v, org.get("sodium-carrier", 0.0) / v
    hbta = org.get("hbta", 0.0) / v + li + na if "R5" in result["active_reactions"] else hbta_in
    return aqueous, {"li": li, "na": na, "hbta": hbta}


def countercurrent(ln_k: dict[str, float], *, reaction_set: str, stage_count: int, organic_to_aqueous: float,
                   extractant: float, saponification_fraction: float, feed_li: float, feed_na: float,
                   feed_acidity: float, tolerance: float = 1.0e-11, max_sweeps: int = 1000,
                   start: dict | None = None) -> dict:
    """Converged cascade: aqueous enters stage 1, saponified organic (``extractant`` HBTA and TOPO) enters stage N.

    Gauss-Seidel: the aqueous is swept from stage 1 to N against the organic leaving each stage in the previous
    sweep, until no organic outlet changes by more than ``tolerance`` mol/L. Every stage of the final sweep is a
    strict state (mc.solve). Li, Na, H (H+ + HA - OH-; BTA- carries no H) and BTA (HA + LiA.T + NaA.T + BTA-) must
    close to 1e-9 mol per litre of aqueous feed.

    ``start``: the ``start`` of a neighbouring converged cascade (organic outlets and stage compositions), used as the
    first sweep's organic profile and composition guesses in place of fresh organic; a warm start only.
    """
    v, c = organic_to_aqueous, extractant
    fresh = {"li": 0.0, "na": saponification_fraction * c, "hbta": c}
    def solve_stage(k, aqueous, entering, guess):
        state = stage_state(f"stage_{k + 1}", organic_to_aqueous=v, extractant=c, li=aqueous["li"],
                            na=aqueous["na"], acidity=aqueous["acidity"], li_org=entering["li"],
                            na_org=entering["na"], hbta=entering["hbta"], bta=aqueous["bta"])
        result = mc.solve(state, reaction_set, ln_k, guess=guess)
        out_aq, out_org = outlets(result, v, entering["hbta"])
        row = {
            "stage": k + 1, "li_in_mol_l": aqueous["li"], "na_in_mol_l": aqueous["na"],
            "acidity_in_mol_l": aqueous["acidity"], "li_org_in_mol_l": entering["li"],
            "na_org_in_mol_l": entering["na"], "li_out_mol_l": out_aq["li"], "na_out_mol_l": out_aq["na"],
            "acidity_out_mol_l": out_aq["acidity"], "li_org_out_mol_l": out_org["li"],
            "na_org_out_mol_l": out_org["na"], "bta_in_mol_l": aqueous["bta"], "bta_out_mol_l": out_aq["bta"],
            "hbta_org_in_mol_l": entering["hbta"], "hbta_org_out_mol_l": out_org["hbta"],
            "li_distribution": result["li_distribution"],
            "stage_ph": result["ph"], "water_autoprotolysis_active": "RW" in result["active_reactions"],
            **{key: result[key] for key in ("balance_inf_norm", "charge_inf_norm", "reaction_affinity_inf_norm",
                                            "kkt_stationarity_inf_norm", "reduced_hessian_min_eigenvalue")},
        }
        return out_aq, out_org, row, result["phase_amounts_mol"]
    feed = {"li": feed_li, "na": feed_na, "acidity": feed_acidity, "bta": 0.0}
    cascade = sweep_stages(stage_count, feed, fresh, solve_stage, start=start,
                           tolerance=tolerance, max_sweeps=max_sweeps)
    aqueous, loaded = cascade["raffinate"], cascade["loaded_organic"]
    closure = {  # in - out, mol per litre of aqueous feed; free HA is HBTA - LiA.T - NaA.T
        "li": feed_li - aqueous["li"] - v * loaded["li"],
        "na": feed_na + v * fresh["na"] - aqueous["na"] - v * loaded["na"],
        "h": feed_acidity + v * (fresh["hbta"] - fresh["na"]) - aqueous["acidity"]
        - v * (loaded["hbta"] - loaded["li"] - loaded["na"]),
        "bta": v * fresh["hbta"] - aqueous["bta"] - v * loaded["hbta"],
    }
    if max(map(abs, closure.values())) > CLOSURE_MOL:
        raise RuntimeError(f"cascade balance does not close to {CLOSURE_MOL} mol: {closure}")
    return {**cascade, "li_extraction_fraction": 1.0 - aqueous["li"] / feed_li,
            "hbta_lost_fraction": aqueous["bta"] / (v * c), "closure_mol": closure}
