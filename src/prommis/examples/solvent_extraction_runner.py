#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""
Solvent extraction loading circuit as a Flowsheet Inspector ``FlowsheetRunner``.

Wraps the steady-state ``SolventExtraction`` model from
``prommis.solvent_extraction.solvent_extraction_steady`` (same feed, dosage and
scaling) and adds feed and product blocks, an organic make-up mixer, a
raffinate splitter and mixer-settler equipment costing. Declared inputs and
outputs are marked with ``FS.annotate_var``.

The number of stages (``NUMBER_OF_STAGES``) and the extractant dosage
(``DOSAGE``) are structural: they are fixed when the model is built, so
changing them requires a rebuild.

Requires the ``idaes-fi`` package (``prommis/flowsheet-inspector-lib``).

Usage:
    python -m prommis.examples.solvent_extraction_runner
"""

from pyomo.environ import (
    Constraint,
    Expression,
    TransformationFactory,
    Var,
    units,
    value,
)
from pyomo.network import Arc

from idaes.core import MaterialBalanceType, MomentumBalanceType, UnitModelBlock
from idaes.core.solvers import get_solver
from idaes.core.util.initialization import propagate_state
from idaes.models.unit_models import (
    EnergySplittingType,
    Feed,
    Mixer,
    MixingType,
    MomentumMixingType,
    Product,
    Separator,
    SplittingType,
)

from idaes_fi.structfs import FlowsheetRunner, Steps  # pylint: disable=import-error

from prommis.costing.ree_costing import (
    REECosting,
    REECostingData,
    REEUnitModelCostingBlock,
)
from prommis.properties.mixed_acid_properties import get_aliases
from prommis.solvent_extraction.ree_og_distribution import ree_list
from prommis.solvent_extraction.solvent_extraction_steady import (
    build_model,
    scale_model,
    set_inputs,
)

NUMBER_OF_STAGES = 3
DOSAGE = 5  # DEHPA, volume percent

AQ = get_aliases(include_sulfates=True)
REES = [j[:-2] for j in ree_list]  # "Sc_o" -> "Sc"
IMPURITIES = ["Al", "Ca", "Fe"]

FS = FlowsheetRunner(
    name="PrOMMiS solvent extraction",
    tags="prommis demo",
    solver=get_solver("ipopt_v2"),
)


@FS.step(Steps.build)
def build(ctx):
    """Build the loading circuit around the steady SolventExtraction model."""
    m = build_model(DOSAGE, NUMBER_OF_STAGES, has_holdup=False, use_mixed_acid=True)
    fs = m.fs

    fs.aqueous_feed = Feed(property_package=fs.leach_soln)
    fs.makeup_organic = Feed(property_package=fs.prop_o)
    fs.barren_organic = Feed(property_package=fs.prop_o)
    fs.organic_mixer = Mixer(
        property_package=fs.prop_o,
        inlet_list=["makeup", "barren"],
        material_balance_type=MaterialBalanceType.componentTotal,
        energy_mixing_type=MixingType.none,
        momentum_mixing_type=MomentumMixingType.none,
    )
    fs.raffinate_splitter = Separator(
        property_package=fs.leach_soln,
        outlet_list=["recycle", "purge"],
        split_basis=SplittingType.totalFlow,
        material_balance_type=MaterialBalanceType.componentTotal,
        momentum_balance_type=MomentumBalanceType.pressureTotal,
        energy_split_basis=EnergySplittingType.equal_temperature,
    )
    fs.loaded_organic = Product(property_package=fs.prop_o)
    fs.raffinate_recycle = Product(property_package=fs.leach_soln)
    fs.raffinate_purge = Product(property_package=fs.leach_soln)

    fs.s01 = Arc(source=fs.makeup_organic.outlet, destination=fs.organic_mixer.makeup)
    fs.s02 = Arc(source=fs.barren_organic.outlet, destination=fs.organic_mixer.barren)
    fs.s03 = Arc(source=fs.organic_mixer.outlet, destination=fs.solex.organic_inlet)
    fs.s04 = Arc(source=fs.aqueous_feed.outlet, destination=fs.solex.aqueous_inlet)
    fs.s05 = Arc(source=fs.solex.organic_outlet, destination=fs.loaded_organic.inlet)
    fs.s06 = Arc(
        source=fs.solex.aqueous_outlet, destination=fs.raffinate_splitter.inlet
    )
    fs.s07 = Arc(
        source=fs.raffinate_splitter.recycle, destination=fs.raffinate_recycle.inlet
    )
    fs.s08 = Arc(
        source=fs.raffinate_splitter.purge, destination=fs.raffinate_purge.inlet
    )
    TransformationFactory("network.expand_arcs").apply_to(m)

    feed, raff, org = (
        fs.aqueous_feed.outlet,
        fs.solex.aqueous_outlet,
        fs.solex.organic_outlet,
    )
    fs.ree_recovery = Expression(
        expr=100
        * (
            1
            - sum(raff.flow_vol[0] * raff.conc_mass_comp[0, AQ[j]] for j in REES)
            / sum(feed.flow_vol[0] * feed.conc_mass_comp[0, AQ[j]] for j in REES)
        ),
        doc="REE recovery from the aqueous feed into the loaded organic, percent",
    )
    fs.ree_purity = Expression(
        expr=100
        * sum(org.conc_mass_comp[0, f"{j}_o"] for j in REES)
        / sum(org.conc_mass_comp[0, f"{j}_o"] for j in REES + IMPURITIES),
        doc="REE share of the metals in the loaded organic, percent",
    )

    # Mixer-settler equipment for costing (UKy account 5.4), sized by residence time.
    # shortcut: fixed residence time, replace with a settler design model if sizing matters
    fs.costing = REECosting()
    fs.mixer_settler = UnitModelBlock()
    ms = fs.mixer_settler
    ms.residence_time = Var(initialize=0.5, units=units.hour)
    ms.residence_time.fix()
    ms.volume = Var(initialize=50, units=units.gal, bounds=(0, None))
    ms.volume_eq = Constraint(
        expr=ms.volume
        == units.convert(
            NUMBER_OF_STAGES
            * ms.residence_time
            * (fs.solex.aqueous_inlet.flow_vol[0] + fs.solex.organic_inlet.flow_vol[0]),
            to_units=units.gal,
        )
    )
    ms.costing = REEUnitModelCostingBlock(
        flowsheet_costing_block=fs.costing,
        costing_method=REECostingData.get_equipment_costing,
        costing_method_arguments={"cost_accounts": ["5.4"], "scaled_param": ms.volume},
    )

    ctx.model = m


@FS.step(Steps.set_operating_conditions)
def set_operating_conditions(ctx):
    """Move the steady model's inlet specification onto the feed blocks."""
    m = ctx.model
    fs = m.fs
    set_inputs(m, DOSAGE, has_holdup=False, use_mixed_acid=True)

    for feed, port in (
        (fs.aqueous_feed, fs.solex.aqueous_inlet),
        (fs.barren_organic, fs.solex.organic_inlet),
        (fs.makeup_organic, fs.solex.organic_inlet),
    ):
        for name, var in port.vars.items():
            for idx in var:
                feed.outlet.vars[name][idx].fix(value(var[idx]))
    for var in fs.solex.organic_inlet.vars.values():
        var.unfix()
    for var in fs.solex.aqueous_inlet.vars.values():
        var.unfix()

    # Fresh extractant carries no metals; it replaces part of the barren organic.
    for j in ree_list + [f"{j}_o" for j in IMPURITIES]:
        fs.makeup_organic.conc_mass_comp[0, j].fix(1e-7)
    fs.makeup_organic.flow_vol.fix(2.01)
    fs.barren_organic.flow_vol.fix(60)

    fs.organic_mixer.mixed_state[0].temperature.fix(305.15)
    fs.organic_mixer.mixed_state[0].pressure.fix(101300)
    fs.raffinate_splitter.split_fraction[0, "recycle"].fix(0.9)

    scale_model(m)

    ann = FS.annotate_var
    ann(
        fs.aqueous_feed.flow_vol[0],
        key="aqueous_feed_flow",
        title="Aqueous feed flow",
        is_output=False,
        input_category="Feed",
    )
    ann(
        fs.aqueous_feed.conc_mass_comp[0, AQ["Nd"]],
        key="feed_nd",
        title="Feed Nd",
        is_output=False,
        input_category="Feed",
    )
    ann(
        fs.aqueous_feed.conc_mass_comp[0, AQ["Fe"]],
        key="feed_fe",
        title="Feed Fe",
        is_output=False,
        input_category="Feed",
    )
    ann(
        fs.barren_organic.flow_vol[0],
        key="barren_organic_flow",
        title="Barren organic flow (sets O/A ratio)",
        is_output=False,
        input_category="Organic",
    )
    ann(
        fs.makeup_organic.flow_vol[0],
        key="makeup_organic_flow",
        title="Make-up organic flow",
        is_output=False,
        input_category="Organic",
    )
    ann(
        fs.raffinate_splitter.split_fraction[0, "recycle"],
        key="raffinate_recycle",
        title="Raffinate recycle fraction",
        is_output=False,
        input_category="Raffinate",
    )
    ann(
        fs.ree_recovery,
        key="ree_recovery",
        title="REE recovery",
        units="%",
        is_input=False,
        output_category="Performance",
    )
    ann(
        fs.ree_purity,
        key="ree_purity",
        title="REE purity in loaded organic",
        units="%",
        is_input=False,
        output_category="Performance",
    )
    ann(
        fs.mixer_settler.costing.bare_erected_cost["5.4"],
        key="mixer_settler_cost",
        title="Mixer-settler bare erected cost",
        is_input=False,
        output_category="Economics",
    )


@FS.step(Steps.initialize)
def initialize(ctx):
    """Initialize units in flow order."""
    fs = ctx.model.fs
    for arc, unit in (
        ((fs.s01, fs.s02), fs.organic_mixer),
        ((fs.s03, fs.s04), fs.solex),
        ((fs.s06,), fs.raffinate_splitter),
    ):
        for a in arc:
            propagate_state(a)
        unit.default_initializer().initialize(unit)
    for a in (fs.s05, fs.s07, fs.s08):
        propagate_state(a)


@FS.step(Steps.solve_initial)
def solve_initial(ctx):
    """Solve the flowsheet, including costing."""
    ctx.solve()


if __name__ == "__main__":
    FS.run_steps()
    for key, ann in FS.annotated_vars.items():
        print(f"{ann['title']}: {value(ann['var']):.4g} {ann['units']}")
