"""P-D0 N1–N3/O1–O2: unit-model acid recovery, neutralization and open organic preparation."""

from idaes.core import FlowDirection, MomentumBalanceType
from idaes.models.unit_models import Mixer, Separator, Translator
from idaes.models.unit_models.mixer import MixingType, MomentumMixingType
from idaes.models.unit_models.separator import EnergySplittingType, SplittingType
from prommis.solvent_extraction.solvent_extraction import SolventExtraction
from pyomo.environ import Block, Constraint, Expression, Expr_if, Var, is_fixed, units as u, value
from pyomo.network import Arc

from . import brines, softening, stage


def separator(fs, name, properties, outlets):
    fs.add_component(name, Separator(property_package=properties, outlet_list=outlets,
        split_basis=SplittingType.componentFlow, energy_split_basis=EnergySplittingType.none,
        momentum_balance_type=MomentumBalanceType.none))
    return getattr(fs, name)


def mixer(fs, name, properties, inlets):
    fs.add_component(name, Mixer(property_package=properties, inlet_list=inlets,
        energy_mixing_type=MixingType.none, momentum_mixing_type=MomentumMixingType.none))
    return getattr(fs, name)


def aqueous(fs, assumptions, volume, concentrate_li):
    """Neutralization water is balanced locally, then explicitly projected out of the retained account."""
    recovery = separator(fs, 'acid_recovery', fs.acid_properties, ['recovered', 'liquor'])
    recovery.feed = Arc(source=fs.strip.unit.aqueous_outlet, destination=recovery.inlet)
    inlet = recovery.mixed_state[0].flow_mol_comp
    recovery.fractions = Constraint(fs.acid_properties.component_list, rule=lambda b, j:
        b.split_fraction[0, 'recovered', j] == (assumptions['acid_recovery'] if j == 'hydrogen-cation' else
        assumptions['acid_recovery'] * inlet['hydrogen-cation'] / Expr_if(IF=inlet[j] > 0, THEN=inlet[j], ELSE=1.)
        if j == 'chloride-anion' else 0.))
    liquid = recovery.liquor_state[0].flow_mol_comp
    fs.strip.free_acid_mol = Expression(expr=liquid['hydrogen-cation'])
    fs.strip.fresh_acid_mol = Expression(expr=liquid['chloride-anion'])
    masses = {'H2O': brines.WATER_MOLAR_MASS_KG, 'H': .00100745142009112, 'OH': .01700854857990888,
              'Li': .00694045142009112, 'Na': .022989251420091117, 'Cl': .035453548579908886, 'C': .06001219715981776}
    charges = {'H2O': 0, 'H': 1, 'OH': -1, 'Li': 1, 'Na': 1, 'Cl': -1, 'C': -2}
    fs.neutral_properties = softening.reactive_properties(masses, masses, charges)
    softening.inherent_reaction(fs.neutral_properties)
    fs.strip_liquid = Translator(inlet_property_package=fs.acid_properties, outlet_property_package=fs.neutral_properties)
    t = fs.strip_liquid
    t.feed = Arc(source=recovery.liquor, destination=t.inlet)
    labels = {'H': 'hydrogen-cation', 'Li': 'lithium-cation', 'Na': 'sodium-cation', 'Cl': 'chloride-anion'}
    t.mapping = Constraint(tuple(labels) + ('H2O',), rule=lambda b, j: b.properties_out[0].flow_mol_phase_comp['Liq', j] ==
        u.convert((b.properties_in[0].flow_mol_comp[labels[j]] if j in labels else
                   volume / 10 / masses['H2O'] if j == 'H2O' else 0.) * u.mol / u.hour, to_units=u.mol / u.s))
    for j in ('OH', 'C'):
        t.properties_out[0].flow_mol_phase_comp['Liq', j].fix(0.)
    neutral = mixer(fs, 'neutralization', fs.neutral_properties, ['acid', 'caustic'])
    neutral.feed = Arc(source=t.outlet, destination=neutral.acid)
    caustic = neutral.caustic_state[0].flow_mol_phase_comp
    acid = neutral.acid_state[0].flow_mol_phase_comp
    neutral.dose = Constraint(('Na', 'OH'), rule=lambda b, j: caustic['Liq', j] ==
                               (acid['Liq', 'H'] if j in ('Na', 'OH') else 0. * u.mol / u.s))
    for j in ('H2O', 'H', 'Li', 'Cl', 'C'):
        caustic['Liq', j].fix(0.)
    neutral.reaction = Constraint(expr=neutral.inherent_reaction_extent[0, 'H_OH'] == acid['Liq', 'H'])
    neutral.naoh_mol = Expression(expr=u.convert(caustic['Liq', 'Na'], to_units=u.mol / u.hour) / (u.mol / u.hour))
    neutral.excluded_reaction_water_kg = Expression(expr=neutral.naoh_mol * masses['H2O'])
    fs.neutral_projection = Translator(inlet_property_package=fs.neutral_properties,
                                       outlet_property_package=fs.liquor_properties)
    t = fs.neutral_projection
    t.feed = Arc(source=neutral.outlet, destination=t.inlet)
    t.mapping = Constraint(fs.liquor_properties.component_list, rule=lambda b, j: b.properties_out[0].flow_mol_comp[j] ==
        u.convert(b.properties_in[0].flow_mol_phase_comp['Liq', j] -
                  (acid['Liq', 'H'] if j == 'H2O' else 0. * u.mol / u.s), to_units=u.mol / u.hour) / (u.mol / u.hour))
    concentration = separator(fs, 'concentration', fs.liquor_properties, ['concentrate', 'recycle'])
    concentration.feed = Arc(source=t.outlet, destination=concentration.inlet)
    concentration.fractions = Constraint(fs.liquor_properties.component_list, rule=lambda b, j:
        b.split_fraction[0, 'concentrate', j] == (Expr_if(IF=volume > 0,
            THEN=concentrate_li / assumptions['concentrate_li_mol_kg'] / Expr_if(IF=volume > 0, THEN=volume / 10, ELSE=1.), ELSE=0.) if j == 'H2O' else 1.))
    concentration.available_strip_water = Constraint(expr=concentrate_li / assumptions['concentrate_li_mol_kg'] <= volume / 10)
    concentration.net_water_kg = Expression(expr=concentration.concentrate_state[0].flow_mol_comp['H2O'] * masses['H2O'])
    for b in (fs.strip_liquid, neutral, fs.neutral_projection):
        for variable in b.component_data_objects(Var):
            if variable.parent_component().local_name in ('temperature', 'pressure'):
                variable.fix(298.15 if variable.parent_component().local_name == 'temperature' else 101325)
    for state in (neutral.acid_state[0], fs.neutral_projection.properties_in[0]):
        state.temperature.unfix()
        state.pressure.unfix()
    return concentration.concentrate_state[0].flow_mol_comp


def organic(fs, model, assumptions, v, free):
    """The preparation outlet is checked against extraction, with no organic recycle arc."""
    org = fs.strip.organic_out
    fs.organic_total_properties = stage.MolarFlowParameters(phase='organic', species=('BTA_total', 'TOPO_total', 'diluent', 'Li'))
    fs.regeneration = Translator(inlet_property_package=fs.loaded_properties, outlet_property_package=fs.organic_total_properties)
    t = fs.regeneration
    t.feed = Arc(source=fs.strip.unit.organic_outlet, destination=t.inlet)
    total_b = sum(org[j] for j in (free, 'lithium-carrier', 'sodium-carrier'))
    total_t = sum(org[j] for j in ('topo', 'lithium-carrier', 'sodium-carrier')) if free == 'hbta' else total_b
    amounts = {'BTA_total': total_b, 'TOPO_total': total_t, 'diluent': org['diluent'], 'Li': org['lithium-carrier']}
    t.mapping = Constraint(fs.organic_total_properties.component_list, rule=lambda b, j:
                           b.properties_out[0].flow_mol_comp[j] == amounts[j])
    regenerated = separator(fs, 'regenerated', fs.organic_total_properties, ['organic', 'lithium_discharge'])
    regenerated.feed = Arc(source=t.outlet, destination=regenerated.inlet)
    for j in fs.organic_total_properties.component_list:
        regenerated.split_fraction[0, 'lithium_discharge', j].fix(float(j == 'Li'))
    fs.strip.regeneration_li = Expression(expr=regenerated.lithium_discharge_state[0].flow_mol_comp['Li'])
    loss = separator(fs, 'organic_loss', fs.organic_total_properties, ['retained', 'loss'])
    loss.feed = Arc(source=regenerated.organic, destination=loss.inlet)
    for j in fs.organic_total_properties.component_list:
        loss.split_fraction[0, 'loss', j].fix(assumptions['organic_loss'])
    makeup = mixer(fs, 'organic_makeup', fs.organic_total_properties, ['retained', 'fresh'])
    makeup.feed = Arc(source=loss.retained, destination=makeup.retained)
    raffinate = fs.stage[len(fs.stages)].outlet
    dissolved = v * (raffinate['bta-anion'] if 'bta-anion' in raffinate else 0.)
    makeup.replacement = Constraint(fs.organic_total_properties.component_list, rule=lambda b, j:
        b.fresh_state[0].flow_mol_comp[j] == loss.loss_state[0].flow_mol_comp[j] + (dissolved if j == 'BTA_total' else 0.))
    for label, j in (('hbta', 'BTA_total'), ('topo', 'TOPO_total'), ('diluent', 'diluent')):
        loss.add_component(label + '_mol', Expression(expr=makeup.fresh_state[0].flow_mol_comp[j]))
    fs.unloaded_organic = Translator(inlet_property_package=fs.organic_total_properties, outlet_property_package=fs.loaded_properties)
    t = fs.unloaded_organic
    t.feed = Arc(source=makeup.outlet, destination=t.inlet)
    labels = {free: 'BTA_total', 'topo': 'TOPO_total', 'diluent': 'diluent'}
    t.mapping = Constraint(fs.loaded_properties.component_list, rule=lambda b, j:
        b.properties_out[0].flow_mol_comp[j] == (b.properties_in[0].flow_mol_comp[labels[j]] if j in labels else 0.))
    fs.saponification = Block()
    sap = fs.saponification
    sap.aqueous_properties = stage.MolarFlowParameters(phase='aqueous', species=('Na', 'OH', 'H2O'))
    stoich = {('saponification', 'aqueous', j): n for j, n in {'Na': -1, 'OH': -1, 'H2O': 1}.items()}
    stoich |= {('saponification', 'organic', j): n for j, n in
               ({free: -1, 'sodium-carrier': 1} | ({'topo': -1} if free == 'hbta' else {})).items()}
    sap.reactions = stage.ExtentReactions(stoichiometry=stoich)
    sap.unit = SolventExtraction(number_of_finite_elements=1, heterogeneous_reaction_package=sap.reactions,
        aqueous_stream={'property_package': sap.aqueous_properties},
        organic_stream={'property_package': fs.loaded_properties, 'flow_direction': FlowDirection.backward})
    sap.feed = Arc(source=t.outlet, destination=sap.unit.organic_inlet)
    unit = sap.unit.mscontactor
    sap.extent = Constraint(expr=unit.heterogeneous_reaction_extent[0, 1, 'saponification'] ==
                           v * fs.boundary['sodium-carrier'] * u.mol / u.hour)
    zero = ('Na', 'OH', 'H2O') if is_fixed(fs.boundary['sodium-carrier']) and value(fs.boundary['sodium-carrier']) == 0 else ()
    for j in zero:
        unit.aqueous_inlet_state[0].flow_mol_comp[j].fix(0.)
    sap.caustic = Constraint([j for j in ('Na', 'OH', 'H2O') if j not in zero], rule=lambda b, j:
        unit.aqueous_inlet_state[0].flow_mol_comp[j] == (v * fs.boundary['sodium-carrier'] if j != 'H2O' else 0.))
    sap.naoh_mol = Expression(expr=unit.aqueous_inlet_state[0].flow_mol_comp['Na'])
    sap.prepared_difference = Expression(fs.loaded_properties.component_list, rule=lambda b, j:
        unit.organic[0, 1].flow_mol_comp[j] - v * fs.boundary[j])
