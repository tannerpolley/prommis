"""Set C on generic stages, with the #166 inventory and stage support selected by its direct owner."""

import math
from pathlib import Path
import time

import epcsaft
import numpy as np
from epcsaft import equilibrium as eq
from pyomo.environ import Constraint

from analyses.prototypes.hbta_topo_ms2_prototype.scripts import countercurrent as cc
from analyses.shared.hbta_topo_configured_reactive_lle.scripts import mechanism_candidates as mc
from lithium_extraction.prototypes.hbta_topo_ms2_saponified.stage_chemistry import CONTINUATION
from . import records, references, stage
from lithium_extraction.common.support.paths import ENGINE_ASL_LIBRARY

SPECIES = tuple(s for s in mc.SPECIES if s != 'lithium-chloride-topo')
SUPPORTS = tuple(tuple(s for s in SPECIES if s in members) for members in (mc.AQUEOUS, mc.ORGANIC))
CONSERVED = ('Li', 'Na', 'H', 'BTA')  # #166 countercurrent.py's four outer closures.


def _charge_basis():
    conservation = np.array([[mc.MOIETIES[n].get(s, 0.) for s in SPECIES] for n in CONSERVED])
    z = np.array([mc.CHARGE.get(s, 0.) for s in SPECIES])
    pivots = []
    for j in range(len(SPECIES)):
        if np.linalg.matrix_rank(conservation[:, [*pivots, j]]) > len(pivots):
            pivots.append(j)
    a = np.linalg.solve(conservation[:, pivots].T, z[pivots]) @ conservation
    independent = [j for j in range(len(SPECIES)) if j not in pivots]
    return a, z - a, 2 * np.abs(a).sum() + np.abs((z - a)[independent]).sum()


def charge_limit(v):
    """#216 amendment 3: terminal set C charge amplification, eq per raw litre."""
    return float(v * _charge_basis()[2] * (1e-12 / max(1., v)) + 5e-13)


def state_from_inlet(inlet, ratio, extractant):
    return cc.stage_state('generic set C inlet', organic_to_aqueous=ratio, extractant=extractant,
        li=inlet['lithium-cation'], na=inlet['sodium-cation'],
        acidity=inlet['hydrogen-cation'] - inlet.get('hydroxide-anion', 0.),
        bta=inlet.get('bta-anion', 0.), li_org=inlet['lithium-carrier'] / ratio,
        na_org=inlet['sodium-carrier'] / ratio,
        hbta=sum(inlet[s] for s in ('hbta', 'lithium-carrier', 'sodium-carrier')) / ratio)


def washed_feed(washed):
    """Complete physical component inventory from Reaktoro; the original #176 model projection is unchanged."""
    v = washed['feed_l']
    state = cc.stage_state('washed set C boundary', organic_to_aqueous=1., extractant=0.,
                          li=washed['aqueous_mol']['Li'] / v, na=washed['aqueous_mol']['Na'] / v,
                          acidity=cc.acidity_at_ph(washed['pH']))
    aqueous, _ = mc.inventory(state)
    physical = dict(washed['physical_component_mol'])
    model = {records.SPECIES_LABELS[s]: n * v for s, n in aqueous.items()}
    adjustment = {s: model.get(s, 0.) - physical.get(s, 0.) for s in physical.keys() | model.keys()}
    return {'physical_feed': physical, 'model_feed': model, 'model_composition_adjustment': adjustment,
            'physical_component_charge': dict(washed['physical_component_charge'], BTA=-1, OH=-1),
            'aqueous_charge_tolerance_eq_per_raw_l': washed['aqueous_charge_tolerance_eq_per_raw_l'],
            'removed_at_no_cost': {}, 'washed_feed_l_per_raw_l': v,
            'basis': 'mol per raw litre; analytical ionic components, not free species concentrations',
            'labels': ['#158 set C extraction requires unsupported aqueous HBTA dissolution',
                       'softening physical-comparison failures remain: F1 6/8, F2 0/3',
                       'washed H component includes bound acid protons; no nominal acidity input added',
                       'fixed-design numerical verification, not physical validation; no full-process solve']}


def _chemistry(anchor, direct, constants, ratio, extractant):
    """Affine owner inventory on the #216 domain restricted by free-ligand capacity.

    The NaOH-first rule re-expresses net acidity through NaA.T while preserving every owner moiety.
    Water remains 55.508 mol per nominal aqueous litre, rather than carrying previous-stage water/OH.
    The independent direct certificate also checks equivalence to the owner's actual inventory.
    """
    with_water = 'RW' in direct['active_reactions']
    aqueous, organic = mc.inventory(anchor)
    names, active = mc.network(anchor, 'C', dict(aqueous, **organic, diluent=1.), with_water=with_water)
    mixture = mc.model(names)
    offset = mc.offsets(names, cc.TEMPERATURE_K)
    ln_engine = {n: mc.ln_k_at(n, constants.get(n, 0.), cc.TEMPERATURE_K) + math.fsum(
        c * offset[s] for s, c in nu.items()) for n, nu in active}
    diluent = mixture.state(cc.TEMPERATURE_K, P=mc.PRESSURE_PA,
        x=tuple(float(s == 'diluent') for s in names), phase='liquid').molar_density * .001
    controls = tuple(s for s in SPECIES if s not in ('water', 'chloride-anion', 'topo')) + ('extractant_mol_l', 'topo_total_mol')
    # Total TOPO is declared per candidate; R5 can change the BTA inventory.
    offsets, matrix = [], []
    for s in names:
        row = {c: float(s == c) for c in controls}
        constant = {'water': aqueous['water']}.get(s, 0.)
        if s == 'hydroxide-anion':
            row = dict.fromkeys(controls, 0.)
        elif s == 'chloride-anion':
            row = {c: mc.CHARGE.get(c, 0.) if c in mc.AQUEOUS else 0. for c in controls}
        elif s in ('sodium-cation', 'hbta'):
            row['hydroxide-anion'] = -1.
            constant = 0.
        elif s == 'sodium-carrier':
            row['hydroxide-anion'] = 1.
            constant = 0.
        elif s == 'topo':
            row = {c: -float(c in ('lithium-carrier', 'sodium-carrier')) for c in controls}
            row['hydroxide-anion'] = -1.
            row['topo_total_mol'] = 1.
            constant = 0.
        offsets.append(constant)
        matrix.append(tuple(row[c] for c in controls))
    aq, org = direct['phase_amounts_mol']
    inlet = dict(aqueous, **organic, diluent=ratio * diluent, extractant_mol_l=extractant,
                 topo_total_mol=ratio * extractant)
    q = tuple(inlet.get(s, 0.) for s in controls)
    mapped = {s: o + math.fsum(a * b for a, b in zip(row, q, strict=True))
              for s, o, row in zip(names, offsets, matrix, strict=True)}
    phases = [eq.Phase(label, kind='liquid', support=tuple(s for s in names if s in members),
              amount=eq.Free(),
              composition_guess=[max(amounts.get(s, 0.), 1e-30) / sum(amounts.get(t, 0.) for t in names if t in members)
                                 for s in names if s in members])
              for label, members, amounts in (('aqueous', mc.AQUEOUS, aq), ('organic', mc.ORGANIC, org))]
    problem = eq.Problem(T=cc.TEMPERATURE_K, P=mc.PRESSURE_PA, feed=eq.Amounts(mapped), phases=phases,
                        row_tolerance=1e-10,
                        reactions=[eq.Reaction(nu, equilibrium_constant=math.exp(ln_engine[n])) for n, nu in active])
    record = eq.asl_record(mixture, problem, CONTINUATION, tuple(offsets), tuple(matrix), q, 0, .001)

    def certify(inlet, outlet):
        current_ratio = inlet['diluent'] / diluent
        state = state_from_inlet(inlet, current_ratio, inlet['extractant_mol_l'])
        selected = mc.solve(state, 'C', constants)
        owner_aq, owner_org = mc.inventory(state)
        owner_feed = dict(owner_aq, **owner_org, diluent=inlet['diluent'])
        values = [inlet[s] for s in controls]
        feed = {s: o + sum(a * b for a, b in zip(row, values, strict=True))
                for s, o, row in zip(names, offsets, matrix, strict=True)}
        differences = {name: sum(c * (feed.get(s, 0.) - owner_feed.get(s, 0.)) for s, c in moiety.items())
                       for name, moiety in mc.MOIETIES.items()}
        selected['inventory_equivalence_mol_per_nominal_l'] = differences
        if max(map(abs, differences.values())) > 1e-9:
            selected['status'] = 'not_evaluable'
            selected['reason'] = 'affine inventory no longer equals the owner inventory; candidate unresolved'
        selected['support_matches'] = ('RW' in selected['active_reactions']) == with_water
        selected['outlet_max_difference_mol_per_raw_l'] = max(abs(outlet[s] - sum(
            p.get(s, 0.) for p in selected['phase_amounts_mol'])) for s in SPECIES)
        return selected

    return stage.Chemistry(SPECIES, SUPPORTS,
        tuple(tuple(mc.MOIETIES[n].get(s, 0.) for s in SPECIES) for n in CONSERVED),
        tuple(mc.CHARGE.get(s, 0.) for s in SPECIES), controls, tuple(offsets), tuple(matrix), record,
        Path(epcsaft.__file__).resolve().parent / ENGINE_ASL_LIBRARY, certify, names,
        (('extractant_mol_l', extractant), ('topo_total_mol', ratio * extractant)))


def build_flowsheet(washed, *, stages, scenario, direct_cascade, chemistry_by_support=None, physical_feed=None):
    ratio, c, r = (scenario[k] for k in ('organic_to_aqueous_ratio', 'extractant_mol_l', 'saponification_ratio'))
    # Bound-active optima return up to ~1e-12 outside 0.05–0.4 M; warm starts clip them onto the bound.
    if .05 - 1e-9 <= c <= .4 + 1e-9:
        c = min(max(c, .05), .4)
        scenario = scenario | {'extractant_mol_l': c}
    if not (.5 <= ratio <= 3 and 0 <= r <= 1 and .05 <= c <= .4):
        raise ValueError('set C outside #216 O/A, saponification or extractant bounds')
    cache = {} if chemistry_by_support is None else chemistry_by_support
    declarations = []
    for row, amounts in zip(direct_cascade['stages'], direct_cascade['start']['guesses'], strict=True):
        state = cc.stage_state('generic set C anchor', organic_to_aqueous=ratio, extractant=c,
            li=row['li_in_mol_l'], na=row['na_in_mol_l'], acidity=row['acidity_in_mol_l'],
            bta=row['bta_in_mol_l'], li_org=row['li_org_in_mol_l'], na_org=row['na_org_in_mol_l'],
            hbta=row['hbta_org_in_mol_l'])
        support = row['water_autoprotolysis_active']
        if support not in cache:
            direct = mc.solve(state, 'C', scenario['ln_k'], guess=amounts)
            cache[support] = _chemistry(state, direct, scenario['ln_k'], ratio, c)
        declarations.append(cache[support])
    boundary = cc.stage_state('generic set C boundary', organic_to_aqueous=ratio, extractant=c,
        li=washed['aqueous_mol']['Li'] / washed['feed_l'], na=washed['aqueous_mol']['Na'] / washed['feed_l'],
        acidity=washed.get('feed_acidity', cc.acidity_at_ph(washed['pH'])), na_org=r * c)
    aqueous, organic = mc.inventory(boundary)
    organic['diluent'] = ratio * mc.model(SPECIES).state(cc.TEMPERATURE_K, P=mc.PRESSURE_PA,
        x=tuple(float(s == 'diluent') for s in SPECIES), phase='liquid').molar_density * .001
    model = stage.build_stages(declarations, aqueous, organic, stages)
    model.fs.parameters['extractant_mol_l'].fix(c)
    model.fs.parameters['topo_total_mol'].unfix()
    model.fs.topo_quantity = Constraint(expr=model.fs.parameters['topo_total_mol'] ==
                                       ratio * model.fs.parameters['extractant_mol_l'])
    model.fs.washed_feed_l_per_raw_l.set_value(washed['feed_l'])
    model._design, model._washed, model._physical_feed = scenario, washed, washed_feed(washed) if physical_feed is None else physical_feed
    model._chemistry_by_support = cache
    profiles = [{s: sum(p.get(s, 0.) for p in amounts) for s in SPECIES}
                for amounts in direct_cascade['start']['guesses']]
    stage.initialize(model, profiles)
    def mapped_nonnegative(_, k, species):
        chemistry = model._chemistries[k - 1]
        j = chemistry.record_species.index(species)
        return chemistry.feed_offset[j] + sum(a * model._incoming(k, c) for a, c in
            zip(chemistry.feed_matrix[j], chemistry.control_species, strict=True)) >= 0
    model.fs.affine_capacity = Constraint(model.fs.stages, ('hbta', 'topo'), rule=mapped_nonnegative)

    return model


def solve(model, directory, *, started, wall_seconds=290, connect_units=None, hessian='exact',
          charge_limit_eq_per_raw_l=None):
    """Recheck the owner's selection outside IPOPT; rebuild the entire candidate at most three times."""
    attempts = []
    for rebuild in range(4):
        remaining = wall_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError('set C candidate exhausted its total wall budget')
        if connect_units is not None:
            connect_units(model)
        stage.solve(model, Path(directory) / str(rebuild), wall_seconds=remaining, hessian=hessian)
        result = report(model, stage.report(model), charge_limit_eq_per_raw_l=charge_limit_eq_per_raw_l)
        selected = [r['certificate'] for r in result['stages']]
        changed = [r['stage'] for r in result['stages'] if not r['certificate']['support_matches']]
        attempts.append({'rebuild': rebuild, 'changed_stages': changed,
                         'balance_residuals_mol_per_nominal_l': result['balance_residuals_mol_per_nominal_l'],
                         'R7': {k: v for k, v in result['R7'].items() if k != 'physical_raffinate'},
                         'strict_certificate': result['strict_certificate'], 'solve': result['solve']})
        if time.monotonic() - started > wall_seconds:
            raise TimeoutError('set C initialization/solve/checks exceeded total wall budget')
        if not changed:
            result['support_rebuilds'] = attempts
            return model, result
        if rebuild == 3:
            raise RuntimeError(f'set C support unresolved after three rebuilds: {changed}; {attempts}')
        rows, guesses = [], []
        for row, direct in zip(result['stages'], selected, strict=True):
            state = state_from_inlet(row['inlet_mol_per_nominal_l'], model._design['organic_to_aqueous_ratio'],
                                     model._design['extractant_mol_l'])
            rows.append({'li_in_mol_l': state.li, 'na_in_mol_l': row['inlet_mol_per_nominal_l']['sodium-cation'],
                         'acidity_in_mol_l': row['inlet_mol_per_nominal_l']['hydrogen-cation'] - row['inlet_mol_per_nominal_l']['hydroxide-anion'],
                         'bta_in_mol_l': state.bta, 'li_org_in_mol_l': state.li_org, 'na_org_in_mol_l': state.na_org,
                         'hbta_org_in_mol_l': state.hbta, 'water_autoprotolysis_active': 'RW' in direct['active_reactions']})
            guesses.append(direct['phase_amounts_mol'])
        model = build_flowsheet(model._washed, stages=len(rows), scenario=model._design,
            direct_cascade={'stages': rows, 'start': {'guesses': guesses}},
            chemistry_by_support=model._chemistry_by_support, physical_feed=model._physical_feed)
    raise AssertionError('unreachable support loop')


def report(model, result=None, *, charge_limit_eq_per_raw_l=None):
    if hasattr(model, '_decisions'):
        model._design.update({k: float(v.value) for k, v in model._decisions.items()})
    model._design['extractant_mol_l'] = float(model.fs.parameters['extractant_mol_l'].value)
    result = stage.report(model) if result is None else result
    v = model._washed['feed_l']
    final, feed = result['outlet_mol_per_raw_l'], model._physical_feed
    raffinate = {records.SPECIES_LABELS[s]: final[s] * v for s in SUPPORTS[0]}
    residuals = {k: n * v for k, n in result['balance_residuals_mol_per_raw_l'].items()}
    rows = []
    for original in result['stages']:
        row = dict(original)
        row['inlet_mol_per_nominal_l'] = row.pop('inlet_mol_per_raw_l')
        row['outlet_mol_per_nominal_l'] = row.pop('outlet_mol_per_raw_l')
        row['balance_residuals_mol_per_nominal_l'] = row['balance_residuals_mol_per_raw_l']
        row['balance_residuals_mol_per_raw_l'] = {k: n * v for k, n in row['balance_residuals_mol_per_raw_l'].items()}
        row['phase_charge_eq_per_nominal_l'] = row['phase_charge_eq_per_raw_l']
        row['phase_charge_eq_per_raw_l'] = [n * v for n in row['phase_charge_eq_per_raw_l']]
        row['certificate'] = dict(row['certificate'])
        row['certificate']['outlet_max_difference_mol_per_nominal_l'] = row['certificate']['outlet_max_difference_mol_per_raw_l']
        row['certificate']['outlet_max_difference_mol_per_raw_l'] *= v
        rows.append(row)
    strict = result['strict_certificate'] and all(r['certificate']['support_matches'] and
        r['certificate']['outlet_max_difference_mol_per_raw_l'] <= 1e-9 for r in rows)
    r7 = references.check_r7(feed, raffinate, balance_residuals_mol_per_raw_l=residuals, strict_certificate=strict,
                             charge_limit_eq_per_raw_l=charge_limit_eq_per_raw_l)
    a, c_charge, amplification = _charge_basis()
    terminal = result['stages'][-1]
    inlet = terminal['inlet_mol_per_raw_l']
    direct = {s: sum(p.get(s, 0.) for p in terminal['certificate']['phase_amounts_mol']) for s in SPECIES}
    residual = terminal['terminal_residuals_mol_per_nominal_l']
    def dot(coefficients, values):
        return math.fsum(t * values.get(s, 0.) for s, t in zip(SPECIES, coefficients, strict=True))
    charges = feed.get('physical_component_charge', references.CHARGE)
    physical = r7['physical_raffinate']
    model_charge = math.fsum(charges[e] * n for e, n in feed['model_feed'].items())
    beta = (0. if 'washed_feed_l_per_raw_l' in feed else r7['feed_charge_eq_per_raw_l']) - model_charge
    engine = {s: terminal['outlet_mol_per_raw_l'][s] - residual[s].get('equilibrium', 0.) for s in SPECIES}
    b = beta + v * (dot(np.array([mc.CHARGE.get(s, 0.) for s in SPECIES]), direct)
        - dot(a, {s: direct[s] - inlet[s] for s in SPECIES})
        + dot(c_charge, {s: engine[s] - direct[s] for s in SPECIES}))
    unit_roundoff = 2.**-53
    eta = 3 * unit_roundoff / (1 - 3 * unit_roundoff) * math.fsum(abs(charges[e]) *
        (abs(physical[e]) + abs(raffinate.get(e, 0.)) + abs(feed['model_feed'].get(e, 0.))) for e in physical)
    q = r7['extraction_charge_change_eq_per_raw_l'] if 'washed_feed_l_per_raw_l' in feed else r7['charge_eq_l']
    eta += unit_roundoff * abs(q)
    terms = {name: v * dot(c_charge if name == 'equilibrium' else a,
                         {s: residual[s].get(name, 0.) for s in SPECIES})
             for name in ('material', 'generation', 'equilibrium', 'extent_generation')}
    diagnostics = dict(beta_eq_per_raw_l=beta, b_eq_per_raw_l=b, eta_fp_eq_per_raw_l=eta,
        E0_eq_per_raw_l=abs(b) + abs(terms['extent_generation']) + eta,
        residual_terms_eq_per_raw_l=terms, amplification=float(amplification),
        q_test_eq_per_raw_l=b + math.fsum(terms.values()), physical_test_charge_eq_per_raw_l=q)
    return {k: value for k, value in result.items() if k != 'outlet_mol_per_raw_l'} | {
            'stages': rows, 'outlet_mol_per_nominal_l': final, 'strict_certificate': strict,
            'balance_residuals_mol_per_nominal_l': result['balance_residuals_mol_per_raw_l'],
            'balance_residuals_mol_per_raw_l': residuals,
            'feed': feed, 'R7': r7, 'charge_diagnostics': diagnostics, 'fixed_design': model._design, 'labels': feed['labels'],
            'conserved_quantity_names': CONSERVED,
            'li_extraction_fraction': 1 - final['lithium-cation'] / model._feeds[0]['lithium-cation'],
            'hbta_lost_fraction': final['bta-anion'] / (model._design['extractant_mol_l'] *
                                                      model._design['organic_to_aqueous_ratio'])}
