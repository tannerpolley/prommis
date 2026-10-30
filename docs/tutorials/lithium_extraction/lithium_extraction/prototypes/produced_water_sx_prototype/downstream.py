"""Shared fixed-fraction acid strip and Ge 2025 concentration/carbonation, per raw litre."""


import math

from idaes.core import FlowDirection, MomentumBalanceType
from idaes.models.unit_models import Mixer, Separator
from idaes.models.unit_models.mixer import MixingType, MomentumMixingType
from idaes.models.unit_models.separator import EnergySplittingType, SplittingType
from prommis.precipitate.precipitator import Precipitator
from prommis.solvent_extraction.solvent_extraction import SolventExtraction
from pyomo.environ import (Block, Constraint, Expression, NonNegativeReals, Reference, Set, TransformationFactory, Var,
                           value)
from pyomo.network import Arc

from . import brines, carbonation, stage

LIQUOR = ('H2O', 'Li', 'Na', 'Cl', 'C')


def held_liquid_loss(dry_kg, liquor_kg, moisture, wash_volumes):
    """Fully mixed displacement wash used by the softening owner (#166)."""
    held = dry_kg * moisture / (1 - moisture)
    fraction = held / liquor_kg
    if not 0 <= value(fraction) < 1:
        raise ValueError('Cake liquid exceeds available stage liquor')
    return held, fraction, fraction * math.exp(-wash_volumes)


def accounts(liquor, assumptions):
    """Ge Table 4 NaCl series at 0.5 mol/kg added soda ash; no extrapolation."""
    net_water = liquor['Li'] / assumptions['concentrate_li_mol_kg']
    recycle = assumptions['mother_recycle']
    water = net_water / (1 - recycle)
    nacl = liquor['Cl'] / net_water
    s = carbonation.empirical_solubility(assumptions['carbonation_temperature_k'], nacl)
    mother = {'Li': 2 * s * water, 'Na': (nacl + 1) * water, 'C': (.5 + s) * water}
    purge = {e: n * (1 - recycle) for e, n in mother.items()}
    product = (liquor['Li'] - purge['Li']) / 2
    carbonate = product + purge['C'] - liquor['C']
    if product <= 0:
        raise ValueError('Concentration input leaves no Li2CO3 product')
    if carbonate < 0:
        raise ValueError('Carbonation soda ash is negative')
    return {'product_li2co3_mol': product, 'ge_li2co3_mol_kg': s, 'mother_liquor_mol': mother,
            'mother_purge_mol': purge, 'mother_nacl_mol_kg': nacl, 'reactor_water_kg': water,
            'fresh_water_kg': net_water, 'neutralization_naoh_mol': liquor['H'],
            'carbonation_na2co3_mol': carbonate}


def liquor_properties(fs):
    water = brines.WATER_MOLAR_MASS_KG
    fs.liquor_properties = stage.MolarFlowParameters(phase='aqueous', species=LIQUOR, molar_mass_kg_mol={
        'H2O': water, **{e: brines.MOLAR_MASS[e] / 1000 for e in LIQUOR[1:]}})
    fs.liquor_properties.dissolved_elements = Set(initialize=LIQUOR)
    fs.liquor_properties.split = Var(LIQUOR, initialize=0.)  # percent of entering component precipitated
    for e in ('H2O', 'Na', 'Cl'):
        fs.liquor_properties.split[e].fix(0.)
    fs.product_properties = stage.MolarFlowParameters(phase='solid', species=('Li2CO3',))
    fs.product_properties.react, fs.product_properties.stoich = {'Li2CO3': 'Li'}, {'Li2CO3': 2}


def connect(fs, liquor, assumptions, candidate, owner):
    """Connected concentrate → Mixer/Precipitator/Separator with a fixed Ge interval."""
    # The Precipitator is the physical reactor: concentrate, soda ash and recycled mother liquor enter;
    # 2 mol Li and 1 mol C leave per mol Li2CO3; the purge closes steady-state Li, Na, Cl, C and water.
    water = brines.WATER_MOLAR_MASS_KG
    if not hasattr(fs, 'liquor_properties'):
        liquor_properties(fs)
    fs.carbonation = Block()
    carbon = fs.carbonation
    carbon.mixer = Mixer(property_package=fs.liquor_properties, inlet_list=['concentrate', 'soda_ash', 'recycle'],
                         energy_mixing_type=MixingType.none, momentum_mixing_type=MomentumMixingType.none)
    carbon.reactor = Precipitator(property_package_aqueous=fs.liquor_properties,
                                  property_package_precipitate=fs.product_properties)
    carbon.splitter = Separator(property_package=fs.liquor_properties, outlet_list=['purge', 'recycle'],
        split_basis=SplittingType.totalFlow, energy_split_basis=EnergySplittingType.none,
        momentum_balance_type=MomentumBalanceType.none)
    carbon.feed = Arc(source=carbon.mixer.outlet, destination=carbon.reactor.aqueous_inlet)
    carbon.mother_liquor = Arc(source=carbon.reactor.aqueous_outlet, destination=carbon.splitter.inlet)
    carbon.recycle = Arc(source=carbon.splitter.recycle, destination=carbon.mixer.recycle)
    carbon.splitter.split_fraction[0, 'recycle'].fix(assumptions['mother_recycle'])
    if hasattr(fs, 'acid_recovery'):
        carbon.concentrate_feed = Arc(source=fs.concentration.concentrate, destination=carbon.mixer.concentrate)
    concentrate = carbon.mixer.concentrate_state[0].flow_mol_comp
    carbon.concentrate = Constraint(() if hasattr(fs, 'acid_recovery') else LIQUOR, rule=lambda b, e: concentrate[e] == {
        'H2O': liquor['Li'] / assumptions['concentrate_li_mol_kg'] / water, 'Li': liquor['Li'],
        'Na': liquor['Na'], 'Cl': liquor['Cl'], 'C': liquor['C']}[e])
    soda = carbon.mixer.soda_ash_state[0].flow_mol_comp
    for e in ('H2O', 'Li', 'Cl'):
        soda[e].fix(0.)
    carbon.soda_ash = Constraint(expr=soda['Na'] == 2 * soda['C'])
    inlet, mother = (getattr(carbon.reactor.cv_aqueous, f'properties_{n}')[0].flow_mol_comp for n in ('in', 'out'))
    product = carbon.reactor.precipitate_state_block[0].flow_mol_comp['Li2CO3']
    product.setlb(0.)
    carbon.water_kg = Expression(expr=mother['H2O'] * water)
    lo, hi = candidate['carbonation_interval_mol_kg']
    carbon.nacl_mol_kg = Var(bounds=(0., 3.), initialize=owner['mother_nacl_mol_kg'])
    carbon.salt_balance = Constraint(expr=carbon.nacl_mol_kg * carbon.water_kg == mother['Cl'])
    carbon.interval = Constraint(expr=(lo, carbon.nacl_mol_kg, hi))
    s0, s1 = (carbonation.empirical_solubility(assumptions['carbonation_temperature_k'], salt) for salt in (lo, hi))
    carbon.solubility_mol_kg = Var(domain=NonNegativeReals, initialize=owner['ge_li2co3_mol_kg'])
    carbon.interpolation = Constraint(expr=carbon.solubility_mol_kg == s0 +
                                     (s1 - s0) * (carbon.nacl_mol_kg - lo) / (hi - lo))
    # Ge 2025 Table 4 NaCl series: residual Li is twice the Li2CO3 solubility at 0.5 mol/kg added Na2CO3.
    carbon.lithium_solubility = Constraint(expr=mother['Li'] == 2 * carbon.solubility_mol_kg * carbon.water_kg)
    carbon.excess_carbonate = Constraint(expr=mother['C'] == (.5 + carbon.solubility_mol_kg) * carbon.water_kg)
    carbon.carbonate_depletion = Constraint(expr=inlet['C'] - mother['C'] == product)
    carbon.product_mol = Expression(expr=product)
    carbon.carbonate_mol = Expression(expr=soda['C'])
    carbon.mother = Expression(('Li', 'Na', 'C'), rule=lambda b, e: mother[e])
    carbon.purge = Expression(('Li', 'Na', 'C'), rule=lambda b, e: carbon.splitter.purge_state[0].flow_mol_comp[e])
    carbon.recovered_li_nonnegative = Constraint(expr=liquor['Li'] - carbon.purge['Li'] >= 0)
    TransformationFactory('network.expand_arcs').apply_to(fs)


def initialize(fs, owner, recycle):
    """Initialize the common recycle loop from its closed-form material balance."""
    carbon, water = fs.carbonation, brines.WATER_MOLAR_MASS_KG
    mother = dict(owner['mother_liquor_mol'], H2O=owner['reactor_water_kg'] / water,
                  Cl=owner['mother_nacl_mol_kg'] * owner['reactor_water_kg'])
    carbonate = (owner['product_li2co3_mol'] + owner['mother_purge_mol']['C'] -
                 value(carbon.mixer.concentrate_state[0].flow_mol_comp['C']))
    soda = {'H2O': 0., 'Li': 0., 'Na': 2 * carbonate, 'Cl': 0., 'C': carbonate}
    feed = {e: value(carbon.mixer.concentrate_state[0].flow_mol_comp[e]) + soda[e] + recycle * mother[e] for e in LIQUOR}
    for state, amounts in ((carbon.mixer.soda_ash_state, soda), (carbon.mixer.mixed_state, feed),
            (carbon.reactor.cv_aqueous.properties_in, feed), (carbon.reactor.cv_aqueous.properties_out, mother),
            (carbon.splitter.mixed_state, mother), (carbon.mixer.recycle_state, {e: recycle * n for e, n in mother.items()}),
            (carbon.splitter.recycle_state, {e: recycle * n for e, n in mother.items()}),
            (carbon.splitter.purge_state, {e: (1 - recycle) * n for e, n in mother.items()})):
        for e, n in amounts.items():
            state[0].flow_mol_comp[e].set_value(n)
    carbon.reactor.precipitate_state_block[0].flow_mol_comp['Li2CO3'].set_value(owner['product_li2co3_mol'])
    carbon.splitter.split_fraction[0, 'purge'].set_value(1 - recycle)
    for e in ('Li', 'C'):
        fs.liquor_properties.split[e].set_value(100 * (1 - mother[e] / feed[e]))



def fixed_fraction_strip(fs, name, complexes, ions, charges, released, fractions, spectators=()):
    """One acid-exchange stage; chemistry owners supply charge and released ligand stoichiometry."""
    if not hasattr(fs, 'acid_properties'):
        fs.acid_properties = stage.MolarFlowParameters(phase='aqueous', species=(
            'hydrogen-cation', *ions.values(), 'chloride-anion'))
        organic = tuple(dict.fromkeys((*complexes.values(), *(s for r in released.values() for s in r), *spectators)))
        fs.loaded_properties = stage.MolarFlowParameters(phase='organic', species=organic)
        stoich = {(e, 'organic', c): -1. for e, c in complexes.items()}
        for e in complexes:
            stoich.update({(e, 'aqueous', ions[e]): 1., (e, 'aqueous', 'hydrogen-cation'): -charges[e]})
            stoich.update({(e, 'organic', s): n for s, n in released[e].items()})
        fs.acid_exchange = stage.ExtentReactions(stoichiometry=stoich)
    fs.add_component(name, Block())
    b = getattr(fs, name)
    b.unit = SolventExtraction(number_of_finite_elements=1, heterogeneous_reaction_package=fs.acid_exchange,
        aqueous_stream={'property_package': fs.acid_properties},
        organic_stream={'property_package': fs.loaded_properties, 'flow_direction': FlowDirection.backward})
    unit = b.unit.mscontactor
    for s in ions.values():
        unit.aqueous_inlet_state[0].flow_mol_comp[s].fix(0.)
    b.fixed_fraction = Constraint(tuple(complexes), rule=lambda b, e:
        unit.heterogeneous_reaction_extent[0, 1, e] == fractions[e] *
        unit.organic_inlet_state[0].get_material_flow_terms('organic', complexes[e]))
    b.aqueous_out = Reference(unit.aqueous[0, 1].flow_mol_comp)
    b.organic_out = Reference(unit.organic[0, 1].flow_mol_comp)
    return b
