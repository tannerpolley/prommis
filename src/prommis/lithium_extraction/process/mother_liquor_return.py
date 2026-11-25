"""External Ge mother-liquor return with stoichiometric chloride conditioning."""

import math

from pyomo.environ import Block, Constraint, Expression, Var, value

from prommis.lithium_extraction.process import (
    waters as brines,
    strip_and_carbonation as downstream,
    flowsheet,
    feeds as records,
)

REACTION = "CO3^2- + 2 HCl → CO2 + H2O + 2 Cl^-"


def condition(stream):
    """Per raw litre. Ge represents the dissolved carbon as carbonate equivalents."""
    acid = 2 * stream["C"]
    stock_kg = acid * flowsheet.HCL_MOLAR_MASS_KG / flowsheet.hcl_stock_fraction()
    water = (
        stream["H2O"]
        + stream["C"]
        + (stock_kg - acid * flowsheet.HCL_MOLAR_MASS_KG) / brines.WATER_MOLAR_MASS_KG
    )
    return {
        "H2O": water,
        "Li": stream["Li"],
        "Na": stream["Na"],
        "Cl": stream["Cl"] + acid,
        "CO2": stream["C"],
        "HCl31_kg": stock_kg,
    }


def mix(model, fraction, initial, *, stages_per_nominal_litre=True):
    """Compose the mixed nominal feed before attaching the Ge residual splitter."""
    if not 0 <= fraction <= 1:
        raise ValueError("Mother-liquor return fraction must lie in [0, 1]")
    fs = model.fs
    fs.mother_liquor_return = loop = Block()
    loop.fraction = Var(bounds=(0.0, 1.0), initialize=fraction)
    loop.fraction.fix()
    purge = initial["mother_purge_mol"] | {
        "Cl": initial["mother_nacl_mol_kg"] * initial["fresh_water_kg"],
        "H2O": initial["fresh_water_kg"] / brines.WATER_MOLAR_MASS_KG,
    }
    loop.returned = Var(
        downstream.LIQUOR,
        bounds=(0.0, None),
        initialize={e: fraction * n for e, n in purge.items()},
    )
    pretreated = condition(loop.returned)
    loop.conditioned = Expression(tuple(pretreated), initialize=pretreated)
    feed = model._feeds[0]
    if stages_per_nominal_litre:
        fresh_v = value(fs.washed_feed_l_per_raw_l)
        fs.del_component("washed_feed_l_per_raw_l")
        fs.washed_feed_l_per_raw_l = Expression(
            expr=fresh_v + loop.conditioned["H2O"] * brines.WATER_MOLAR_MASS_KG
        )
        volume = fs.washed_feed_l_per_raw_l
        additions = {
            s: fresh_v * n + (loop.conditioned[e] if e != "H" else 0.0)
            for s, n in feed.items()
            if (e := records.SPECIES_LABELS[s]) in ("Li", "Na", "Cl", "H")
        }
    else:
        volume = 1.0
        additions = {
            s: n + loop.conditioned[e]
            for s, n in feed.items()
            if (e := {"water": "H2O"}.get(s, records.SPECIES_LABELS[s]))
            in ("Li", "Na", "Cl", "H2O")
        }
    loop.mixer = Constraint(
        tuple(additions), rule=lambda _, s: volume * fs.boundary[s] == additions[s]
    )
    for s, amount in additions.items():
        fs.boundary[s].unfix()
        fs.boundary[s].set_value(value(amount) / value(volume))


def split(fs):
    loop = fs.mother_liquor_return
    purge = fs.carbonation.splitter.purge_state[0].flow_mol_comp
    loop.split = Constraint(
        downstream.LIQUOR,
        rule=lambda _, e: loop.returned[e] == loop.fraction * purge[e],
    )


def report(fs):
    loop = fs.mother_liquor_return
    f = value(loop.fraction)
    purge = {
        e: value(n)
        for e, n in fs.carbonation.splitter.purge_state[0].flow_mol_comp.items()
    }
    returned = {e: f * n for e, n in purge.items()}
    bleed = {e: n - returned[e] for e, n in purge.items()}
    conditioned = condition(returned)
    residuals = {f"split/{e}": value(loop.returned[e]) - returned[e] for e in purge}
    residuals["returned charge"] = math.fsum(
        (returned["Na"], returned["Li"], -returned["Cl"], -2 * returned["C"])
    )
    residuals["conditioned charge"] = math.fsum(
        (conditioned["Na"], conditioned["Li"], -conditioned["Cl"])
    )
    residuals["carbon"] = returned["C"] - conditioned["CO2"]
    residuals["chloride"] = conditioned["Cl"] - returned["Cl"] - 2 * returned["C"]
    residuals["water"] = (
        conditioned["H2O"]
        - returned["H2O"]
        - returned["C"]
        - (
            conditioned["HCl31_kg"]
            * (1 - flowsheet.hcl_stock_fraction())
            / brines.WATER_MOLAR_MASS_KG
        )
    )
    for s, equation in loop.mixer.items():
        residuals[f"mix/{s}"] = value(equation.body - equation.lower)
    return {
        "return_fraction": f,
        "bleed_fraction": 1 - f,
        "returned_mol_per_raw_l": returned,
        "bleed_mol_per_raw_l": bleed,
        "conditioned": conditioned,
        "residuals_mol_per_raw_l": residuals,
        "conditioning_reaction": REACTION,
        "volume_basis": "1 kg water/L nominal return volume",
    }


def mixed_feed(model, *, stages_per_nominal_litre=True):
    """Reconstruct the raw-litre physical and model inventories from both inlet streams."""
    fs = model.fs
    fresh = model._physical_feed
    if not hasattr(fs, "mother_liquor_return"):
        return fresh
    loop = report(fs)
    conditioned = loop["conditioned"]
    v = value(fs.washed_feed_l_per_raw_l)
    elemental = {
        e: fresh["physical_feed"].get(e, 0.0) + conditioned[e]
        for e in ("Li", "Na", "Cl")
    }
    elemental["water"] = fresh["physical_feed"].get("water", 0.0) + conditioned["H2O"]
    physical = fresh["physical_feed"] | elemental
    projected = {
        records.SPECIES_LABELS[s]: v * value(fs.boundary[s]) for s in model._feeds[0]
    }
    adjustments = {
        e: projected.get(e, 0.0) - physical.get(e, 0.0)
        for e in projected.keys() | physical.keys()
    }
    return (
        fresh
        | {
            "physical_feed": physical,
            "model_feed": projected,
            "model_composition_adjustment": adjustments,
        }
        | ({"washed_feed_l_per_raw_l": v} if stages_per_nominal_litre else {})
    )
