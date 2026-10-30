"""Countercurrent PrOMMiS SolventExtraction stages on a chemistry owner's declared species and balances.

Flows are mol/h at 1 L/h of the owner's nominal aqueous feed: raw brine in extraction-only scope and
washed feed in full-process scope, so each flow equals today's mol per nominal litre. The Engine record
is affine in those amounts, so the basis is never rescaled inside the stages.
"""

from dataclasses import dataclass
from pathlib import Path
import shutil
import re
import time

import numpy as np
import idaes
from idaes.core import (Component, FlowDirection, FlowsheetBlock, MaterialBalanceType, MaterialFlowBasis, Phase,
                        PhysicalParameterBlock, ProcessBlockData, StateBlock, StateBlockData,
                        declare_process_block_class)
from idaes.core.base.property_meta import HasPropertyClassMetadata
from idaes.core.util.misc import add_object_reference
from prommis.solvent_extraction.solvent_extraction import SolventExtraction
from pyomo.common.config import ConfigValue
from pyomo.environ import (Block, ConcreteModel, Constraint, Expression, ExternalFunction, NonNegativeReals,
                           Objective, Param, RangeSet, Reference, Set, SolverFactory, Suffix, Var, units, value)
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.opt import check_optimal_termination
from pyomo.util.calc_var_value import calculate_variable_from_constraint


@dataclass
class Chemistry:
    species: tuple
    supports: tuple
    conservation: tuple
    charges: tuple
    control_species: tuple
    feed_offset: tuple
    feed_matrix: tuple
    record: str
    library: Path
    certify: object
    record_species: tuple = ()
    parameters: tuple = ()


UNITS = {'time': units.hour, 'length': units.m, 'mass': units.kg, 'amount': units.mol, 'temperature': units.K}


@declare_process_block_class('MolarFlowParameters')
class MolarFlowParametersData(PhysicalParameterBlock):
    """One phase whose state is the molar flow of each declared species."""

    CONFIG = PhysicalParameterBlock.CONFIG()
    CONFIG.declare('phase', ConfigValue(domain=str))
    CONFIG.declare('species', ConfigValue(domain=tuple))
    CONFIG.declare('molar_mass_kg_mol', ConfigValue(domain=dict, default=None))
    # Opt-in (#256): solution volume is a per-state flow the caller supplies, independent of the H2O amount.
    CONFIG.declare('declared_volume', ConfigValue(domain=bool, default=False))

    def build(self):
        super().build()
        self.add_component(self.config.phase, Phase())
        for s in self.config.species:
            self.add_component(s, Component())
        self._state_block_class = MolarFlowState  # noqa: F821 (declared by IDAES)

    @classmethod
    def define_metadata(cls, obj):
        obj.add_properties({'flow_mol_comp': {'method': None}, 'flow_vol': {'method': '_flow_vol'},
                            'conc_mass_comp': {'method': '_conc_mass_comp'}})
        obj.add_default_units(UNITS)


@declare_process_block_class('MolarFlowState', block_class=StateBlock)
class MolarFlowStateData(StateBlockData):
    """Component molar flows in mol/h at the 1 L/h basis, carried as the repository's dimensionless amounts."""

    def build(self):
        super().build()
        self.flow_mol_comp = Var(self.params.component_list, initialize=0.)

    def _flow_vol(self):
        if self.params.config.declared_volume:
            self.flow_vol = Var(domain=NonNegativeReals, initialize=1.)  # L/h of solution, fixed by the caller
            return
        # The #176 liquor accounting declares 1 kg water per litre.
        self.flow_vol = Expression(expr=self.flow_mol_comp['H2O'] * self.params.config.molar_mass_kg_mol['H2O'])

    def _conc_mass_comp(self):
        mass = self.params.config.molar_mass_kg_mol
        self.conc_mass_comp = Expression(self.params.component_list, rule=lambda b, j:
                                         b.flow_mol_comp[j] * mass[j] / b.flow_vol)

    def get_material_flow_terms(self, p, j):
        return self.flow_mol_comp[j] * units.mol / units.hour  # IDAES balances carry the units

    def get_material_flow_basis(self):
        return MaterialFlowBasis.molar

    def default_material_balance_type(self):
        return MaterialBalanceType.componentTotal

    def define_state_vars(self):
        return {'flow_mol_comp': self.flow_mol_comp}


@declare_process_block_class('ExtentReactions')
class ExtentReactionsData(ProcessBlockData, HasPropertyClassMetadata):
    """Heterogeneous reactions whose extents are algebraic transfer coordinates, keyed (reaction, phase, species).

    The empty element list leaves PrOMMiS SolventExtraction without distribution-coefficient equations.
    """

    CONFIG = ProcessBlockData.CONFIG()
    CONFIG.declare('stoichiometry', ConfigValue(domain=dict))

    def build(self):
        super().build()
        self.reaction_stoichiometry = dict(self.config.stoichiometry)
        self.reaction_idx = Set(initialize=list(dict.fromkeys(r for r, _, _ in self.reaction_stoichiometry)))
        self.element_list = Set(initialize=[])

    @classmethod
    def define_metadata(cls, obj):
        obj.add_default_units(UNITS)

    def build_reaction_block(self, *args, **kwargs):
        return ExtentReactionBlock(*args, parameters=self, **kwargs)  # noqa: F821 (declared by IDAES)


@declare_process_block_class('ExtentReactionBlock')
class ExtentReactionBlockData(ProcessBlockData):
    CONFIG = ProcessBlockData.CONFIG()
    CONFIG.declare('parameters', ConfigValue())

    def build(self):
        super().build()
        add_object_reference(self, 'params', self.config.parameters)


def build_stages(chemistry, aqueous_feed, organic_feed, stages):
    """Aqueous flows forward from stage 1, organic backward from stage N; Engine outlets close each stage."""
    if type(stages) is not int or stages not in range(1, 6):
        raise ValueError('stage count must be an integer from 1 through 5')
    chemistries = (chemistry,) * stages if isinstance(chemistry, Chemistry) else tuple(chemistry)
    if len(chemistries) != stages:
        raise ValueError('one fixed chemistry declaration is required per stage')
    chemistry = chemistries[0]
    species, basis = chemistry.species, np.asarray(chemistry.conservation)
    if any(c.species != species or c.supports != chemistry.supports or c.conservation != chemistry.conservation
           for c in chemistries):
        raise ValueError('outer species, phase allocation and conserved quantities must agree across stages')
    # Balance rows determine the pivot species; Engine equalities close the remaining degrees of freedom.
    pivots = []
    for j in range(len(species)):
        if np.linalg.matrix_rank(basis[:, [*pivots, j]]) > len(pivots):
            pivots.append(j)
    if len(pivots) != len(basis):
        raise ValueError('owner conservation rows must be independent')
    independent = [j for j in range(len(species)) if j not in pivots]
    # One extent per independent species, N = [-B_p^-1 B_i; I], so outlets conserve every owner row.
    extents = np.zeros((len(species), len(independent)))
    extents[pivots] = -np.linalg.solve(basis[:, pivots], basis[:, independent])
    extents[independent, range(len(independent))] = 1.
    if np.abs(basis @ extents).max() > 1e-12 or np.linalg.matrix_rank(extents) != len(species) - len(basis):
        raise ValueError('extent stoichiometry must span the null space of the owner conservation rows')
    phases = {name: p for p, support in enumerate(chemistry.supports) for name in support}
    if set(phases) != set(species) or sum(map(len, chemistry.supports)) != len(species):
        raise ValueError('this stage requires disjoint, complete owner phase supports')
    inlet = {s: float(aqueous_feed.get(s, 0.0) + organic_feed.get(s, 0.0)) for s in species}
    if any(not np.isfinite(n) or n < 0 for n in inlet.values()):
        raise ValueError('owner feed must be finite and nonnegative')
    model = ConcreteModel()
    model.fs = FlowsheetBlock(dynamic=False)
    fs = model.fs
    names = ('aqueous', 'organic')
    fs.aqueous_properties = MolarFlowParameters(phase='aqueous', species=chemistry.supports[0])  # noqa: F821 (declared by IDAES)
    fs.organic_properties = MolarFlowParameters(phase='organic', species=chemistry.supports[1])  # noqa: F821 (declared by IDAES)
    stoichiometry = {(species[j], names[phases[s]], s): float(extents[i, r])
                     for r, j in enumerate(independent) for i, s in enumerate(species) if extents[i, r]}
    fs.extents = ExtentReactions(stoichiometry=stoichiometry)  # noqa: F821 (declared by IDAES)
    fs.extraction = SolventExtraction(number_of_finite_elements=stages, heterogeneous_reaction_package=fs.extents,
        aqueous_stream={'property_package': fs.aqueous_properties, 'flow_direction': FlowDirection.forward},
        organic_stream={'property_package': fs.organic_properties, 'flow_direction': FlowDirection.backward})
    unit = fs.extraction.mscontactor
    fs.stages = RangeSet(stages)
    # Litres of nominal aqueous stage feed per raw litre: 1 for raw brine, v for washed feed (#211 item 5).
    fs.washed_feed_l_per_raw_l = Param(initialize=1., mutable=True)
    flows = {(k, s): getattr(unit, names[phases[s]])[0, k].flow_mol_comp[s] for k in fs.stages for s in species}
    feeds = {s: getattr(unit, names[phases[s]] + '_inlet_state')[0].flow_mol_comp[s] for s in species}
    for flow in (*flows.values(), *feeds.values()):
        flow.setlb(0.)
    for s, flow in feeds.items():
        flow.fix(inlet[s])
    fs.boundary = Reference(feeds)
    fs.stage = Block(fs.stages)
    for k in fs.stages:
        fs.stage[k].outlet = Reference({s: flows[k, s] for s in species})
    parameters = dict(chemistry.parameters)
    fs.parameters = Var(tuple(parameters), domain=NonNegativeReals, initialize=parameters)
    for s, n in parameters.items():
        fs.parameters[s].fix(n)
    fs.response = ExternalFunction(library=str(chemistry.library),
                                   function='epcsaft_equilibrium_phase_component_amount')
    model._chemistry = chemistry
    model._chemistries = chemistries
    model._feeds = (dict(aqueous_feed), dict(organic_feed))

    def incoming(k, s):
        if s in parameters:
            return fs.parameters[s]
        other = k - 1 if phases[s] == 0 else k + 1
        return flows[other, s] if other in fs.stages else feeds[s]

    model._incoming = incoming
    # Engine arguments are the entering flows themselves: a metal-free boundary feed is a fixed constant,
    # never a zero-bounded copy held by an equality, so the square model stays regular (version 1.0 fix).
    for k in fs.stages:
        chemistry = chemistries[k - 1]
        block = fs.stage[k]
        record_species = chemistry.record_species or species
        for s in set(species) - set(record_species):
            block.outlet[s].fix(0.0)
        block.equilibrium = Constraint([j for j in independent if species[j] in record_species],
            rule=lambda b, j, k=k, c=chemistry, names=record_species: b.outlet[species[j]] == fs.response(
                *(incoming(k, s) for s in c.control_species), c.record, str(phases[species[j]]),
                str(names.index(species[j]))))
    model.feasibility = Objective(expr=0.0)
    return model


def initialize(model, profiles):
    """Set owner-retained stage amounts and their extents; no surrogate, cascade sweep or equilibrium solve."""
    if len(profiles) != len(model.fs.stages):
        raise ValueError('one retained profile is required per stage')
    for k, amounts in zip(model.fs.stages, profiles, strict=True):
        if set(amounts) != set(model._chemistry.species):
            raise ValueError('retained profile must contain the complete declared species')
        for s, n in amounts.items():
            if not np.isfinite(n) or n < 0:
                raise ValueError('retained amounts must be finite and nonnegative')
            model.fs.stage[k].outlet[s].set_value(n)
    unit = model.fs.extraction.mscontactor
    for (_, k, r), extent in unit.heterogeneous_reaction_extent.items():
        extent.set_value(value(model.fs.stage[k].outlet[r] - model._incoming(k, r)))
    for name in ('aqueous', 'organic'):
        generation = getattr(unit, name + '_heterogeneous_reactions_generation')
        for index, constraint in getattr(unit, name + '_heterogeneous_reaction_constraint').items():
            calculate_variable_from_constraint(generation[index], constraint)


def _kkt(model, logfile=None):
    """Frozen version 1.0 raw KKT conventions and limits, applied to this square NL."""
    if logfile is not None:
        text = Path(logfile).read_text()
        def unscaled(name):
            matches = re.findall(r'^' + re.escape(name) + r'\.*:\s+\S+\s+(\S+)', text, re.MULTILINE)
            if not matches:
                raise RuntimeError(f'IPOPT omitted unscaled {name}')
            return float(matches[-1])
        bounds = [max(v.lb - value(v) if v.lb is not None else 0., 0., value(v) - v.ub if v.ub is not None else 0.)
                  for v in model.component_data_objects(Var) if v.lb is not None or v.ub is not None]
        multipliers = [0.] + [-n for n in model.ipopt_zL_out.values()] + list(model.ipopt_zU_out.values())
        for c in model.component_data_objects(Constraint, active=True):
            d = -model.dual.get(c, 0.)
            if not c.equality:
                multipliers += [d if c.upper is None else 0., -d if c.lower is None else 0.]
        return {'stationarity': unscaled('Dual infeasibility'),
                'complementarity': unscaled('Complementarity'),
                'feasibility': max(unscaled('Constraint violation'), max(bounds, default=0.)),
                'multiplier_sign': max(multipliers)}
    nlp = PyomoNLP(model)
    variables, constraints = nlp.get_pyomo_variables(), nlp.get_pyomo_constraints()
    x = np.array([value(v) for v in variables])
    dual = -np.array([model.dual.get(c, 0.0) for c in constraints])
    zl = np.array([model.ipopt_zL_out.get(v, 0.0) for v in variables])
    zu = -np.array([model.ipopt_zU_out.get(v, 0.0) for v in variables])
    nlp.set_primals(x)
    nlp.set_duals(dual)
    g = nlp.evaluate_constraints()
    lower, upper = nlp.primals_lb(), nlp.primals_ub()
    gl, gu = nlp.constraints_lb(), nlp.constraints_ub()
    lf, uf = np.isfinite(lower), np.isfinite(upper)
    rows = np.flatnonzero(gl < gu)
    up, down = np.maximum(dual[rows], 0.), np.maximum(-dual[rows], 0.)
    up_finite, down_finite = np.isfinite(gu[rows]), np.isfinite(gl[rows])
    stationarity = nlp.evaluate_grad_objective() + nlp.evaluate_jacobian().T @ dual - zl + zu
    return {
        'stationarity': float(np.max(abs(stationarity))),
        'complementarity': float(max(np.max(abs(zl[lf] * (x - lower)[lf]), initial=0),
                                     np.max(abs(zu[uf] * (upper - x)[uf]), initial=0),
                                     np.max(abs(up[up_finite] * (gu - g)[rows][up_finite]), initial=0),
                                     np.max(abs(down[down_finite] * (g - gl)[rows][down_finite]), initial=0))),
        'feasibility': float(max(np.max(np.maximum(gl - g, 0)), np.max(np.maximum(g - gu, 0)),
                                 np.max(np.maximum(lower - x, 0)), np.max(np.maximum(x - upper, 0)))),
        'multiplier_sign': float(max(np.max(-zl), np.max(-zu),
                                     np.max(up[~up_finite], initial=0), np.max(down[~down_finite], initial=0), 0)),
    }


def solve(model, directory, *, wall_seconds=290, hessian='exact'):
    """Exact Hessians by default; the caller's wall timer includes build, initialization and certificates."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    options = {'print_user_options': 'yes', 'hessian_approximation': hessian, 'bound_relax_factor': 0,
               'nlp_scaling_method': 'none', 'tol': 1e-11,
               'constr_viol_tol': 1e-12 / max(1., value(model.fs.washed_feed_l_per_raw_l)), 'dual_inf_tol': 1e-7,
               'compl_inf_tol': 1e-7, 'max_iter': 200, 'acceptable_iter': 0, 'bound_push': 1e-8,
               'bound_frac': 1e-8, 'mu_init': 1e-6, 'max_cpu_time': wall_seconds}
    path = directory / 'ipopt.opt'
    path.write_text(''.join(f'{key} {item}\n' for key, item in options.items()))
    for name in ('dual', 'ipopt_zL_out', 'ipopt_zU_out'):
        if not hasattr(model, name):
            setattr(model, name, Suffix(direction=Suffix.IMPORT))
    started = time.monotonic()
    result = SolverFactory('ipopt', executable=shutil.which('ipopt', path=idaes.bin_directory),  # ipopt.exe on Windows
                            options={'option_file_name': str(path)}).solve(
        model, logfile=str(directory / 'ipopt.log'))
    if not check_optimal_termination(result) or 'Optimal Solution Found' not in str(result.solver.message):
        raise RuntimeError(f'IPOPT unresolved: {result.solver.termination_condition}')
    diagnostics = _kkt(model, directory / 'ipopt.log')
    limits = {'stationarity': 1e-7, 'complementarity': 1e-7, 'feasibility': 1e-8, 'multiplier_sign': 1e-8}
    if any(not np.isfinite(diagnostics[k]) or diagnostics[k] > v for k, v in limits.items()):
        raise RuntimeError(f'KKT limits failed: {diagnostics}')
    model._solve = {'termination': str(result.solver.termination_condition), 'kkt': diagnostics,
                    'solve_and_kkt_s': time.monotonic() - started, 'options': options}
    return model._solve


def report(model):
    """Recompute conservation and owner strict certificates from the solved connected streams."""
    if not hasattr(model, '_solve'):
        raise ValueError('report requires a solved model')
    chemistry, rows, residuals = model._chemistry, [], {}
    model._feeds = tuple({s: value(model.fs.boundary[s]) for s in feed} for feed in model._feeds)
    for k in model.fs.stages:
        chemistry = model._chemistries[k - 1]
        inlet = {s: float(value(model._incoming(k, s))) for s in chemistry.species}
        inlet.update({s: value(model.fs.parameters[s]) for s, _ in chemistry.parameters})
        outlet = {s: float(value(model.fs.stage[k].outlet[s])) for s in chemistry.species}
        balances = {str(r): sum(c * (outlet[s] - inlet[s]) for s, c in zip(
            chemistry.species, row, strict=True)) for r, row in enumerate(chemistry.conservation)}
        residuals.update({f'stage{k}/{r}': n for r, n in balances.items()})
        certificate = chemistry.certify(inlet, outlet)
        terminal = {}
        if k == len(model.fs.stages):
            unit = model.fs.extraction.mscontactor
            for name, support in zip(('aqueous', 'organic'), chemistry.supports, strict=True):
                for s in support:
                    material = getattr(unit, name + '_material_balance')[0, k, s]
                    generation = getattr(unit, name + '_heterogeneous_reaction_constraint')[0, k, name, s]
                    g = getattr(unit, name + '_heterogeneous_reactions_generation')[0, k, name, s]
                    terminal[s] = {'material': -value(material.body - material.lower),
                                   'generation': value(generation.body - generation.lower),
                                   'extent_generation': value(g) - value(generation.body - generation.lower)}
            for j, constraint in model.fs.stage[k].equilibrium.items():
                terminal[chemistry.species[j]]['equilibrium'] = value(constraint.body - constraint.lower)
        charges = [sum(chemistry.charges[chemistry.species.index(s)] * outlet[s] for s in support)
                   for support in chemistry.supports]
        rows.append({'stage': k, 'inlet_mol_per_raw_l': inlet, 'outlet_mol_per_raw_l': outlet,
                     'balance_residuals_mol_per_raw_l': balances, 'phase_charge_eq_per_raw_l': charges,
                     'certificate': certificate, 'terminal_residuals_mol_per_nominal_l': terminal})
    final = {s: rows[-1 if s in chemistry.supports[0] else 0]['outlet_mol_per_raw_l'][s]
             for s in chemistry.species}
    inlet = {s: sum(feed.get(s, 0.0) for feed in model._feeds) for s in chemistry.species}
    residuals.update({f'global/{r}': sum(c * (final[s] - inlet[s]) for s, c in zip(
        chemistry.species, row, strict=True)) for r, row in enumerate(chemistry.conservation)})
    return {'stages': rows, 'outlet_mol_per_raw_l': final, 'solve': model._solve,
            'balance_residuals_mol_per_raw_l': residuals,
            'strict_certificate': all(r['certificate']['status'] == 'evaluated' and
                                      r['certificate']['outlet_max_difference_mol_per_raw_l'] <= 1e-9
                                      for r in rows)}
