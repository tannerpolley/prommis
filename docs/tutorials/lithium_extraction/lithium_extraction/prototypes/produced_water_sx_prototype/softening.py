"""P-D0 S1–S6: Reaktoro states carried by WaterTAP reactors and IDAES cake washing."""

import math

import numpy as np
from idaes.core import FlowsheetBlock, MaterialFlowBasis, MomentumBalanceType
from idaes.models.unit_models import Mixer, Separator
from idaes.models.unit_models.mixer import MixingType, MomentumMixingType
from idaes.models.unit_models.separator import EnergySplittingType, SplittingType
from pyomo.core.expr.visitor import identify_variables
from pyomo.environ import ConcreteModel, Constraint, Set, Var, units as u, value
from pyomo.repn import generate_standard_repn
from pyomo.util.check_units import assert_units_consistent
from watertap.property_models.multicomp_aq_sol_prop_pack import MCASParameterBlock
from watertap.unit_models.stoichiometric_reactor import StoichiometricReactor


def components(ps, elements):
    """S1: analytical H is distinct from free H+; the negative branch is OH minus water."""
    species = {'H': 'H+', 'H2O': 'H2O', 'C': 'CO3-2', 'B': 'B(OH)4-', 'S': 'SO4-2',
               **{e: ps.ION[e] for e in elements if e not in ('H', 'O', 'C', 'B', 'S')}}
    matrix = np.array([[ps.DB.species(s).elements().coefficient(e) for s in species.values()] for e in elements])
    amounts = dict(zip(species, map(float, np.linalg.solve(matrix, list(elements.values()))), strict=True))
    amounts['OH'] = max(-amounts['H'], 0.)
    amounts['H'] = max(amounts['H'], 0.)
    amounts['H2O'] -= amounts['OH']
    return amounts, species | {'OH': 'OH-'}


def reactive_properties(species, masses, charges):
    p = MCASParameterBlock(solute_list=[s for s in species if s != 'H2O'], mw_data=masses,
                           charge={s: charges[s] for s in species if s != 'H2O'},
                           material_flow_basis=MaterialFlowBasis.molar)
    return p


def inherent_reaction(p):
    p._has_inherent_reactions = True
    p.inherent_reaction_idx = Set(initialize=['H_OH'])
    p.inherent_reaction_stoichiometry = {('H_OH', 'Liq', j): {'H': -1, 'OH': -1, 'H2O': 1}.get(j, 0.)
                                       for j in p.component_list}


def set_state(state, amounts, density):
    for j in state.params.component_list:
        state.flow_mol_phase_comp['Liq', j].fix(amounts.get(j, 0.) / 3600)
    state.temperature.fix(298.15)
    state.pressure.fix(101325)
    state.dens_mass_phase['Liq'].fix(density)
    state.eq_dens_mass_phase['Liq'].deactivate()
    state.flow_vol_phase['Liq'].unfix()


def evaluate(block):
    """Evaluate square scalar unit/property equations; fixed Reaktoro balances retain their residuals."""
    equations = list(block.component_data_objects(Constraint, active=True))
    while True:
        changed = False
        for c in equations:
            variables = list(identify_variables(c.body, include_fixed=False))
            if c.equality and len(variables) == 1:
                repn = generate_standard_repn(c.body - c.lower)
                if not repn.is_linear():
                    raise RuntimeError(f'Nonlinear fixed-state property equation: {c.name}')
                variables[0].fix(-repn.constant / repn.linear_coefs[0])
                changed = True
        if not changed:
            break
    remaining = [v.name for v in block.component_data_objects(Var) if not v.fixed]
    if remaining:
        raise RuntimeError(f'Unspecified fixed-state unit quantities: {remaining}')


class Softening:
    """One raw L/h; equilibrium chemistry stays in ps, wash operations read native unit outlets."""

    def __init__(self, ps, initial, model=None):
        self.ps = ps
        self.model = model if model is not None else ConcreteModel()
        self.model.fs = FlowsheetBlock(dynamic=False)
        _, self.species = components(ps, initial)
        self.mass = {j: float(ps.DB.species(s).molarMass()) for j, s in self.species.items()}
        self.charge = {j: ps.DB.species(s).charge() for j, s in self.species.items()}
        self.model.fs.properties = reactive_properties(self.species, self.mass, self.charge)
        inherent_reaction(self.model.fs.properties)
        self.stages = []

    def amounts(self, state):
        return {j: value(n) * 3600 for (p, j), n in state.flow_mol_phase_comp.items()}

    def stage(self, name, before, after, reagent, dose, moisture, washes, actual_water, *, floor_density=None):
        ps, fs = self.ps, self.model.fs
        incoming, _ = components(ps, before['aq'])
        liquid, _ = components(ps, after['aq'])
        def coefficients(elements):
            return components(ps, {e: elements.coefficient(e) for e in before['aq']})[0]
        precipitates = {p: {'mw': float(ps.DB.species(p).molarMass()) * u.kg / u.mol,
                           'precipitation_stoichiometric': coefficients(ps.DB.species(p).elements())}
                        for p in after['solids_mol']}
        reagent_stoich = {dict((s, j) for j, s in self.species.items())[s]: n for s, n in ps.REAGENT[reagent].items()}
        fs.add_component(name, StoichiometricReactor(property_package=fs.properties,
            reagent={reagent: {'mw': sum(self.mass[j] * n for j, n in reagent_stoich.items()) * u.kg / u.mol,
                              'density_reagent': 1000 * u.kg / u.m**3, 'dissolution_stoichiometric': reagent_stoich}},
            precipitate=precipitates, momentum_balance_type=MomentumBalanceType.none))
        unit = getattr(fs, name)
        rho_in = before['liquor_kg'] / (before['liquor_l'] / 1000)
        rho = floor_density if floor_density is not None else after['liquor_kg'] / (after['liquor_l'] / 1000)
        dissolved = {j: incoming[j] + dose * reagent_stoich.get(j, 0.) for j in incoming}
        xi = min(dissolved['H'], dissolved['OH'])
        for j, nu in {'H': -1, 'OH': -1, 'H2O': 1}.items():
            dissolved[j] += nu * xi
        cv = unit.dissolution_reactor
        set_state(cv.properties_in[0], incoming, rho_in)
        set_state(cv.properties_out[0], dissolved if precipitates else liquid, rho_in if precipitates else rho)
        cv.inherent_reaction_extent[0, 'H_OH'].fix(xi / 3600)
        unit.flow_mol_reagent[reagent].fix(dose / 3600)
        if not precipitates:
            evaluate(unit)
            self.stages.append(unit)
            return unit, self.amounts(cv.properties_out[0]), {}, 0.
        cv = unit.precipitation_reactor
        set_state(cv.properties_in[0], dissolved, rho_in)
        set_state(cv.properties_out[0], liquid, rho)
        provisional = {j: dissolved[j] - sum(after['solids_mol'][p] *
            precipitates[p]['precipitation_stoichiometric'][j] for p in precipitates) for j in incoming}
        cv.inherent_reaction_extent[0, 'H_OH'].fix(min(provisional['H'], provisional['OH']) / 3600)
        for p, n in after['solids_mol'].items():
            unit.flow_mol_precipitate[p].fix(n / 3600)
        unit.waste_mass_frac_precipitate.fix(1 - moisture)
        dry = sum(value(unit.flow_mol_precipitate[p]) * 3600 * float(ps.DB.species(p).molarMass()) for p in precipitates)
        held = dry * moisture / (1 - moisture)
        f = held / after['liquor_kg']
        if not 0 <= f < 1:
            raise ValueError('Cake liquid exceeds available stage liquor')
        unit.separator.split_fraction[0, 'waste', 'Liq'].fix(f)
        for state, fraction in ((unit.separator.mixed_state[0], 1.), (unit.separator.waste_state[0], f),
                                (unit.separator.treated_state[0], 1 - f)):
            set_state(state, {j: fraction * n for j, n in liquid.items()}, rho)
        evaluate(unit)
        fs.add_component(name + '_wash', Mixer(property_package=fs.properties, inlet_list=['cake', 'water'],
            energy_mixing_type=MixingType.none, momentum_mixing_type=MomentumMixingType.none))
        mixer = getattr(fs, name + '_wash')
        wash = washes * held * after['liquor_l'] / after['liquor_kg'] / self.mass['H2O']
        set_state(mixer.cake_state[0], self.amounts(unit.separator.waste_state[0]), rho)
        set_state(mixer.water_state[0], {'H2O': wash}, 1000.)
        mixer.inherent_reaction_extent[0, 'H_OH'].fix(0.)
        mixer.mixed_state[0].temperature.fix(298.15)
        mixer.mixed_state[0].pressure.fix(101325)
        evaluate(mixer)
        fs.add_component(name + '_filter', Separator(property_package=fs.properties, outlet_list=['cake', 'recovered'],
            split_basis=SplittingType.componentFlow, energy_split_basis=EnergySplittingType.none,
            momentum_balance_type=MomentumBalanceType.none))
        splitter = getattr(fs, name + '_filter')
        set_state(splitter.mixed_state[0], self.amounts(mixer.mixed_state[0]), rho)
        ell = f * math.exp(-washes)
        for j in fs.properties.component_list:
            fraction = math.exp(-washes) if j != 'H2O' else (
                (ell * liquid[j] + (f - ell) * actual_water) / (f * liquid[j] + wash) if f * liquid[j] + wash else 0.)
            splitter.split_fraction[0, 'cake', j].fix(fraction)
        splitter.inherent_reaction_extent.fix(0.)
        for state in (splitter.cake_state[0], splitter.recovered_state[0]):
            state.temperature.fix(298.15)
            state.pressure.fix(101325)
        evaluate(splitter)
        fs.add_component(name + '_recombine', Mixer(property_package=fs.properties, inlet_list=['treated', 'wash'],
            energy_mixing_type=MixingType.none, momentum_mixing_type=MomentumMixingType.none))
        mix = getattr(fs, name + '_recombine')
        set_state(mix.treated_state[0], self.amounts(unit.separator.treated_state[0]), rho)
        set_state(mix.wash_state[0], self.amounts(splitter.recovered_state[0]), rho)
        mix.inherent_reaction_extent[0, 'H_OH'].fix(0.)
        mix.mixed_state[0].temperature.fix(298.15)
        mix.mixed_state[0].pressure.fix(101325)
        evaluate(mix)
        self.stages.append(unit)
        return unit, self.amounts(mix.mixed_state[0]), self.amounts(splitter.cake_state[0]), wash

    def floor(self, name, before, reagent, moisture, washes, *, density, carbonate_allowance=0.):
        """Ideal stoichiometry only; H supplies the liquid density, with no equilibrium or pH claim."""
        if not math.isfinite(density) or density <= 0:
            raise ValueError('Floor mode requires a positive supplied liquid density (kg/m3)')
        ps, aq = self.ps, before['aq']
        minerals = {'Brucite': aq['Mg']} if reagent == 'NaOH' else {
            p: aq[e] for p, e in (('Calcite', 'Ca'), ('Strontianite', 'Sr'), ('Witherite', 'Ba'))}
        if reagent == 'Na2CO3' and carbonate_allowance:
            total = sum(minerals.values())
            fraction = max(0., total - carbonate_allowance) / total if total else 0.
            minerals = {p: n * fraction for p, n in minerals.items()}
        dose = 2 * aq['Mg'] if reagent == 'NaOH' else sum(minerals.values())
        elements = {e: n + dose * ps.REAGENT_ELEMENTS[reagent].get(e, 0.) - sum(
            amount * ps.DB.species(p).elements().coefficient(e) for p, amount in minerals.items()) for e, n in aq.items()}
        liquid, _ = components(ps, elements)
        mass = sum(n * self.mass[j] for j, n in liquid.items())
        after = {'aq': elements, 'solids_mol': minerals, 'liquor_kg': mass, 'liquor_l': mass / density * 1000}
        result = self.stage(name, before, after, reagent, dose, moisture, washes, liquid['H2O'], floor_density=density)
        set_state(getattr(self.model.fs, name + '_recombine').mixed_state[0], result[1], density)
        return result

    def check(self):
        evaluate(self.model)
        assert_units_consistent(self.model)
        residuals = {}
        for unit in self.stages:
            for name in ('dissolution_reactor', 'precipitation_reactor'):
                if not hasattr(unit, name):
                    continue
                for key, c in getattr(unit, name).material_balances.items():
                    residuals[f'{unit.name}/{name}/{key}'] = value(u.convert(c.body - c.lower, to_units=u.mol / u.s)) * 3600
        if max(map(abs, residuals.values()), default=0.) > self.ps.CLOSURE_MOL:
            raise RuntimeError(f'Softening unit balances: {residuals}')
        return residuals
