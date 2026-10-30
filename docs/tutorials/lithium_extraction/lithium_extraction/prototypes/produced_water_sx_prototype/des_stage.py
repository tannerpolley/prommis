"""#216 DES-2: F/3 declarations for the shared countercurrent stage equations."""
import json
import math
from functools import lru_cache
from pathlib import Path

import epcsaft
from epcsaft import equilibrium as eq
import numpy as np
from pyomo.environ import Block, Constraint, Expression, Objective, Var, value

from analyses.prototypes.softened_ms2_epcsaft_sx.scripts import des_chemistry as d
from lithium_extraction.prototypes.hbta_topo_ms2_saponified.stage_chemistry import CONTINUATION
from . import brines, cascade, records, stage
from lithium_extraction.common.support.paths import ENGINE_ASL_LIBRARY

T = 298.15
PROMOTIONS = json.loads((d.STUDY / 'runs/des1/results/promotion.json').read_text())
M_DES, M_TOPO = (d.model(T, 'X').parameters.molar_masses[d.IDS.index(s)] for s in d.ORG[:2])


def feed():
    """Exact #216 synthetic feed; KCl stays in physical mass and charge accounting."""
    return records.extraction_feed(brines.raw_feed('MS2'), 'A', synthetic_softened=True)


@lru_cache(None)
def preparation(ph):
    f = feed()['physical_feed']
    inventory = dict(zip(d.AQ, (f['water'], f['Li'], f['Na'], f['Li']+f['Na']+f['H'], f['H'], 0., 0., 0.), strict=True))
    n, state, residual = d.prepare_inventory(T, ph, inventory)
    return {'pH': ph, 'total_N_mol': float(n[6]+n[7]), 'amounts': dict(zip(d.AQ, n.tolist(), strict=True)),
            'max_ln_residual': float(max(abs(residual)))}


def boundary(total_n, topo_fraction):
    f = feed()['physical_feed']
    aqueous = dict(zip(d.AQ, (f['water'], f['Li'], f['Na'], f['Li']+f['Na']+f['H'], f['H'], 0., total_n, 0.), strict=True))
    kcl_mass = f['K']*(brines.MOLAR_MASS['K']+brines.MOLAR_MASS['Cl'])/1000
    mass = sum(aqueous[s]*mw for s, mw in zip(d.AQ, d.MW, strict=True))+kcl_mass
    organic = dict(zip(d.ORG, (mass*(1-topo_fraction)/M_DES, mass*topo_fraction/M_TOPO, 0., 0.), strict=True))
    return aqueous, organic, mass


def direct(candidate, aqueous, organic, guess=None):
    """Declare the incoming amounts without resetting aqueous chemistry or organic loading."""
    promotion = next(p for p in PROMOTIONS if p['candidate'] == candidate)
    assert promotion['calibration_accepted']
    ln_k = promotion['promoted']['ln_K']
    offset, _ = d.references(T, candidate)
    amounts, potentials, check, compiled, result, problem, seed, elapsed = d.solve_inventory(
        candidate, T, ln_k, (aqueous, organic), guess)
    return {'status': 'evaluated', **check, 'phase_amounts_mol': amounts,
            'formula_checks': d.formula_checks(candidate, d.model(T, candidate), (aqueous, organic), amounts),
            'pH': -(potentials['hydrogen-cation']-offset['hydrogen-cation'])/math.log(10),
            'final_volumes_L': [result.amounts[p]/result.molar_densities[p]*1000 for p in range(2)],
            'Engine': {'requested_tolerance_met': result.requested_tolerance_met, 'topology_event': result.topology_event.name,
                       'jacobian_rank': result.jacobian_rank, 'unknown_count': result.unknown_count,
                       'floor': result.convergence.floor, 'classified_row_residual': result.convergence.residual}, 'seed': seed,
            'seconds': elapsed}, problem


def build(candidate, n, total_n, topo_fraction):
    aqueous, organic, mass = boundary(total_n, topo_fraction)
    initial, problem = direct(candidate, aqueous, organic)
    moieties = d.chemistry(candidate)[1]
    c = tuple(tuple(row.get(s, 0.) for s in d.IDS) for row in moieties.values())
    controls = tuple(s for s in d.IDS if s != 'chloride-anion')
    # ASL requires neutral feed directions. Actual Cl retains its material and C rows.
    matrix = tuple(tuple(float(s == q) if s != 'chloride-anion' else d.CHARGE.get(q, 0)
                         for q in controls) for s in d.IDS)
    anchor = tuple(aqueous.get(s, 0.)+organic.get(s, 0.) for s in controls)
    record = eq.asl_record(d.model(T, candidate), problem, CONTINUATION, (0.,)*len(d.IDS), matrix,
                           anchor, 0, .001)
    def certify(inlet, outlet):
        fresh, _ = direct(candidate, {s: inlet[s] for s in d.AQ}, {s: inlet[s] for s in d.ORG})
        # Physical metals and F are particle amounts / 3; all other amounts are unchanged.
        fresh['physical_outlet_max_difference_mol'] = max(abs(outlet[s]-sum(p[s] for p in fresh['phase_amounts_mol']))/
            (3 if s in ('des', *d.ORG[2:]) else 1) for s in d.IDS)
        fresh['particle_outlet_max_difference_mol_per_raw_l'] = max(abs(outlet[s]-sum(p[s] for p in fresh['phase_amounts_mol'])) for s in d.IDS)
        fresh['outlet_max_difference_mol_per_raw_l'] = fresh['physical_outlet_max_difference_mol']
        return fresh
    chemistry = stage.Chemistry(d.IDS, (d.AQ, d.ORG), c, tuple(d.CHARGE.get(s, 0) for s in d.IDS),
        controls, (0.,)*len(d.IDS), matrix, record, Path(epcsaft.__file__).resolve().parent/ENGINE_ASL_LIBRARY, certify)
    model = stage.build_stages(chemistry, aqueous, organic, n)
    model._candidate, model._initial = candidate, initial
    def solve_stage(k, aq, org, guess):
        try:
            solved, _ = direct(candidate, aq, org, guess)
        except RuntimeError as error:
            raise RuntimeError(f'DES-{candidate} initialization stage {k+1}, inlet Li {aq["lithium-cation"]}: {error}') from error
        amounts = solved['phase_amounts_mol']
        return {s: amounts[0][s] for s in d.AQ}, {s: amounts[1][s] for s in d.ORG}, solved, amounts
    model._initialization = cascade.sweep_stages(n, aqueous, organic, solve_stage)
    stage.initialize(model, [{s: sum(p[s] for p in amounts) for s in d.IDS}
                             for amounts in model._initialization['start']['guesses']])
    model.fs.des = Block()
    b = model.fs.des
    low, high = [preparation(ph)['total_N_mol'] for ph in (9.5, 10.5)]
    b.total_N = Var(initialize=total_n, bounds=(low, high))
    b.total_N.fix()
    b.ammonia_feed = Constraint(expr=model.fs.boundary['ammonia'] == b.total_N)
    model.fs.boundary['ammonia'].unfix()
    b.mass = Expression(expr=mass+d.MW[6]*(b.total_N-total_n))
    b.mass_equality = Constraint(expr=M_DES*model.fs.boundary['des']+M_TOPO*model.fs.boundary['topo'] == b.mass)
    b.topo_lower = Constraint(expr=M_TOPO*model.fs.boundary['topo'] >= .2*b.mass)
    b.topo_upper = Constraint(expr=M_TOPO*model.fs.boundary['topo'] <= .4*b.mass)
    model.fs.boundary['topo'].unfix()
    b.topo_fraction = Expression(expr=M_TOPO*model.fs.boundary['topo']/b.mass)
    b.extraction = Expression(expr=1-model.fs.stage[n].outlet['lithium-cation']/aqueous['lithium-cation'])
    return model


def release(model):
    model.fs.des.total_N.unfix()
    model.fs.boundary['des'].unfix()
    model.feasibility.deactivate()
    model.optimum = Objective(expr=-model.fs.des.extraction)


def charge_bound(model, result):
    """DES-0 F4 with the declared charge-neutral Engine map; actual flows decide F4."""
    c = np.array(model._chemistry.conservation)
    z = np.array(model._chemistry.charges)
    pivots = []
    for j in range(len(z)):
        if np.linalg.matrix_rank(c[:, [*pivots, j]]) > len(pivots):
            pivots.append(j)
    w = np.linalg.solve(c[:, pivots].T, z[pivots])
    a = w@c
    assert np.array_equal(a, z)
    amplification = float(2*np.linalg.norm(a, 1))
    assert amplification == 12
    n, u = len(result['stages']), 2**-53
    def roundoff(k, terms):
        return k*u/(1-k*u)*math.fsum(map(abs, terms))
    f = feed()['physical_feed']
    inlet = {s: value(model.fs.boundary[s]) for s in d.IDS}
    # Preparation adds neutral NH3; two additions form model Cl, three form physical Cl.
    fp = {'model_preparation': roundoff(2, [f[e] for e in ('Li', 'Na', 'H')]),
          'physical_preparation': roundoff(3, [f[e] for e in ('Li', 'Na', 'K', 'H')])}
    qfeed = abs(math.fsum(z[i]*inlet[s] for i, s in enumerate(d.IDS)))
    fp['feed_sum'] = roundoff(5, [z[i]*inlet[s] for i, s in enumerate(d.IDS)])
    unit, stoich = model.fs.extraction.mscontactor, model.fs.extents.reaction_stoichiometry
    residuals, mismatches = [], []
    for row in result['stages']:
        k, incoming, outgoing = row['stage'], row['inlet_mol_per_raw_l'], row['outlet_mol_per_raw_l']
        terms = [incoming[s]*d.CHARGE[s] for s in d.AQ if d.CHARGE[s] and s != 'chloride-anion']
        rebuilt = sum(terms)
        mismatch = abs(incoming['chloride-anion']-rebuilt)
        fp[f'rebuild_{k}'] = roundoff(4, terms)
        fp[f'mismatch_{k}'] = roundoff(5, [incoming['chloride-anion'], *terms])
        mismatches.append({'stage': k, 'actual_Cl': incoming['chloride-anion'], 'rebuilt_Cl': rebuilt, 'd_k': mismatch})
        for s in d.AQ:
            material = unit.aqueous_material_balance[0, k, s]
            generation = unit.aqueous_heterogeneous_reaction_constraint[0, k, 'aqueous', s]
            residuals += [abs(value(material.body-material.lower)), abs(value(generation.body-generation.lower))]
            if d.CHARGE[s]:
                g = value(unit.aqueous_heterogeneous_reactions_generation[0, k, 'aqueous', s])
                ext = [coef*value(unit.heterogeneous_reaction_extent[0, k, r])
                       for (r, phase, species), coef in stoich.items() if phase == 'aqueous' and species == s]
                fp[f'rows_{k}_{s}'] = roundoff(len(ext)+4, [incoming[s], outgoing[s], g, *ext])
        for r in model.fs.extents.reaction_idx:
            assert math.fsum(z[i]*stoich.get((r, 'aqueous' if s in d.AQ else 'organic', s), 0.)
                             for i, s in enumerate(d.IDS)) == 0
    out = result['outlet_mol_per_raw_l']
    physical_cl = f['Cl']+(out['chloride-anion']-inlet['chloride-anion'])
    physical_terms = [out[s]*d.CHARGE[s] for s in d.AQ if d.CHARGE[s] and s != 'chloride-anion']+[-physical_cl, f['K']]
    fp['physical_KCl_reconstruction'] = roundoff(8, [f['Cl'], out['chloride-anion'], inlet['chloride-anion'], *physical_terms])
    eta = math.fsum(fp.values())
    limits = [qfeed+amplification*j*1e-12+eta for j in range(1, 6)]
    physical_charge = math.fsum(physical_terms)
    injections = [{'N': j, 'Q_DES': limit, 'injected_actual_charge': physical_charge+1e-10,
                   'rejected': abs(physical_charge+1e-10) > limit} for j, limit in enumerate(limits, 1)]
    checks = {'material_generation': max(residuals) <= 1e-12, 'physical_charge': abs(physical_charge) <= limits[n-1],
              'inlet_mismatches': all(m['d_k'] <= limits[n-1] for m in mismatches),
              'ion_rejected': all(test['rejected'] for test in injections)}
    return {'passed': all(checks.values()), 'checks': checks, 'C': c.tolist(), 'pivots': [d.IDS[j] for j in pivots], 'w': w.tolist(), 'a': a.tolist(), 'c_I': [0.]*4,
            'A': amplification, 'v': 1., 'tau': 1e-12, 'q_feed': qfeed, 'eta_terms': fp, 'eta_fp': eta,
            'Q_DES_by_N': limits, 'physical_charge': physical_charge, 'max_material_generation_residual': max(residuals),
            'inlet_chloride_mismatch': mismatches, 'injected_actual_ion_checks': injections,
            'Engine_Cl_map': 'Li + Na + H - OH + NH4; actual stage Cl retained in material and conservation rows'}


def injected_ion_check(model, limit):
    """Perturb actual terminal-stage flow; never perturb the rebuilt Engine input."""
    k = len(model.fs.stages)
    actual = model.fs.stage[k].outlet['lithium-cation']
    before = value(actual)
    actual.set_value(before+1e-10)
    try:
        physical = feed()['physical_feed']
        out = model.fs.stage[k].outlet
        cl = physical['Cl']+(value(out['chloride-anion'])-value(model.fs.boundary['chloride-anion']))
        charge = math.fsum([d.CHARGE[s]*value(out[s]) for s in d.AQ if d.CHARGE[s] and s != 'chloride-anion']+
                           [-cl, physical['K']])
        assert abs(charge) > limit
        return {'N': k, 'actual_Li_before': before, 'actual_Li_after': value(actual),
                'injected_mol_per_raw_l': 1e-10, 'physical_charge': charge, 'Q_DES': limit, 'rejected': True}
    finally:
        actual.set_value(before)


def metrics(model, result):
    """Physical quantities per raw litre; source priors and prohibited-phase losses stay explicit."""
    from scipy.optimize import brentq
    aqueous, organic = model._feeds
    total_n = value(model.fs.des.total_N)
    ph = brentq(lambda ph: preparation(ph)['total_N_mol']-total_n, 9.5, 10.5, xtol=1e-12)
    prepared = preparation(ph)['amounts']
    initial_aq_volume = sum(prepared.values())/d.phase_state(d.model(T, model._candidate), T, prepared).molar_density*1000
    initial_org_volume = sum(organic.values())/d.phase_state(d.model(T, model._candidate), T, organic).molar_density*1000
    out = result['outlet_mol_per_raw_l']
    li, na = [out[s]/3 for s in d.ORG[2:]]
    net_li, net_na = [out[s]/3-organic[s]/3 for s in d.ORG[2:]]
    first = result['stages'][0]
    first_out = first['outlet_mol_per_raw_l']
    va, vo = first['certificate']['final_volumes_L']
    distribution = [(first_out[c]/3/vo)/(first_out[s]/va) for c, s in zip(d.ORG[2:], d.AQ[1:3], strict=True)]
    return {'E_Li': 1-out['lithium-cation']/aqueous['lithium-cation'],
            'stage1_D_Li': distribution[0], 'stage1_D_Na': distribution[1],
            'stage1_D_Li_over_D_Na': distribution[0]/distribution[1],
            'loaded_organic_Na_over_Li_Na': na/(li+na), 'net_Na_transfer_mol': net_na,
            'net_Li_transfer_mol': net_li, 'TOPO_wt_pct': 100*M_TOPO*organic['topo']/value(model.fs.des.mass),
            'preparation_pH': ph, 'initial_aqueous_N_mol': total_n, 'organic_preparation_ammonia_N_mol': 0., 'structural_DES_N_mol': organic['des']/3,
            'conditioned_aqueous_mass_kg': value(model.fs.des.mass), 'organic_mass_kg': value(model.fs.des.mass),
            'initial_model_volumes_L': [initial_aq_volume, initial_org_volume],
            'organic_circulation_F_mol': organic['des']/3, 'organic_circulation_TOPO_mol': organic['topo'],
            'formal_protons_regeneration_mol': net_li+net_na if model._candidate == 'X' else 0.,
            'formal_chloride_cotransfer_mol': net_li+net_na if model._candidate == 'S' else 0.,
            'modelled_losses_mol': {'DES_to_aqueous': 0., 'TOPO_to_aqueous': 0., 'ammonia_to_organic': 0.},
            'loss_basis': 'zero by prohibited-phase-support assumption',
            'stages': [{'stage': row['stage'], 'free_sites_F_mol': row['outlet_mol_per_raw_l']['des']/3,
                        'Li_occupied_sites_mol': row['outlet_mol_per_raw_l']['lithium-des']/3,
                        'Na_occupied_sites_mol': row['outlet_mol_per_raw_l']['sodium-des']/3,
                        'aqueous_N_mol': sum(row['outlet_mol_per_raw_l'][s] for s in ('ammonia', 'ammonium-cation')),
                        'equilibrium_pH': row['certificate']['pH'],
                        'final_model_volumes_L': row['certificate']['final_volumes_L']} for row in result['stages']]}
