"""#216 F1 and F3: direct retained chemistry, frozen inputs and immutable comparisons."""

import csv
from dataclasses import replace
import json
import math
import sys

from analyses.prototypes.hbta_topo_ms2_prototype.scripts import countercurrent as cc
from analyses.prototypes.hbta_topo_ms2_prototype.scripts import li_na_exchange as lx
from analyses.prototypes.hbta_topo_ms2_prototype.scripts import set_c_exchange_band as sc
from .round0 import STUDY, compare, mc, save

H166 = sc.RESULTS
__all__ = ['cc', 'lx', 'mc', 'H166', 'rows', 'constants']
FIELDS = ('model_d_li model_d_na model_li_extraction_percent model_equilibrium_ph '
          'log10_model_over_observed_d_li aqueous_bta_mol_l hbta_lost_mg_l hbta_lost_percent '
          'na_loaded_organic_mol_l na_loaded_over_feed_na net_base_mol ph_error '
          'log10_imposed_over_forward_d_li').split()


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def checked(name, reference, result, fields, discrete=(), direct=None):
    result = {k: v for k, v in result.items() if reference.get(k, '') != ''}
    comparisons = compare(reference, result, {k: k for k in fields if k in reference})
    passed = all(v['scaled_difference'] <= 1e-6 for v in comparisons.values())
    meanings = {k: str(result.get(k, '')).replace('None', '') == reference[k] for k in discrete if k in reference}
    passed &= all(meanings.values())
    save(name, {'reference': reference, 'result': result, 'comparisons': comparisons,
                'discrete_comparisons': meanings, 'direct': direct, 'passed': passed})
    if not passed:
        raise RuntimeError(f'F1 failed {name}: {comparisons}; discrete {meanings}')


def constants(r):
    fit = json.loads((H166 / 'set_c_fit.json').read_text())
    end = fit['low'] if r['end'] == 'low' else fit['high']['r_0.70']
    kex, delta = end['ln_k_ex'], end['delta']
    delta += {'fitted': 0, 'delta_minus': -.5 * math.log(10), 'delta_plus': .5 * math.log(10)}[r['delta_case']]
    return kex, delta, sc.ln_k(kex, delta)


def hbta():
    contacts = {(c.dataset, c.identity): c for c in [*lx.contacts(), sc.CHA]}
    forward = {(r['end'], r['delta_case'], r['identity']): r for r in rows(H166 / 'set_c_contacts.csv')}
    seo = {r['point_id']: r for r in rows(lx.DATA / 'digitization/seo2025_figure2/points.csv')}
    ni = {r['point_id']: r for r in rows(lx.DATA / 'digitization/ni2024_figure10/points.csv')}
    ni_feed = {r['component']: float(r['content_g_l']) for r in rows(lx.DATA / 'tables/niSustainableEfficientRecovery2024_table1.csv')}
    for table, count in (('set_c_contacts', 94), ('set_c_holdout_scores', 259)):
        references = rows(H166 / f'{table}.csv')
        key = ('end', 'delta_case', 'dataset', 'identity') + (('route',) if 'holdout' in table else ())
        assert len(references) == len({tuple(r[k] for k in key) for r in references}) == count
        for i, r in enumerate(references):
            name = f'f1_{table}_{i}.json'
            if (STUDY / 'results' / name).exists():
                if json.loads((STUDY / 'results' / name).read_text())['passed']:
                    continue
                name = f'f1_{table}_{i}_corrected.json'
            print(table, i, r['identity'], r.get('route'), flush=True)
            if r.get('route') == 'not_scored':
                checked(name, r, {k: r[k] for k in ('route', 'status', 'error')}, [], ('route', 'status', 'error'))
                continue
            kex, delta, ln_k = constants(r)
            target = None
            if r['dataset'] == 'seo2025_figure2':
                p = seo[r['identity']]
                c = lx.Contact(r['dataset'], 'holdout', r['identity'], 1., .4,
                               float(p['saponification_degree_percent_inferred']) / 100,
                               1.85 / lx.LI_G_MOL, 31.2 / lx.NA_G_MOL, 8.52, sc._observed_d(p), 'source O/A 1')
                target = float(p['equilibrium_ph']) if r['route'] == 'imposed_ph' else None
            elif r['dataset'] == 'ni2024_figure10':
                identity, ph = r['identity'].split('_feed_ph_')
                p = ni[identity]
                c = lx.Contact(r['dataset'], r['role'], r['identity'], 1., .1,
                               float(p['saponification_degree_percent']) / 100,
                               ni_feed['Li+'] / lx.LI_G_MOL, ni_feed['Na+'] / lx.NA_G_MOL, float(ph),
                               sc._observed_d(p), 'source O/A 1')
            else:
                c = contacts[(r['dataset'], r['identity'])]
                if r.get('route') == 'imposed_ph_route_check':
                    target = sc._table2_ph()[1][c.identity]
            state = replace(c, saponification=min(c.saponification, 1.)).state()
            if c.saponification > 1:
                state = replace(state, base=state.base + (c.saponification - 1) * c.extractant)
            direct = target is not None or r.get('route') == 'forward_free_naoh'
            value = (mc.solve(state, 'C', ln_k) if target is None else
                     mc.solve_at_ph(replace(state, na=c.na), 'C', ln_k, target))
            if target is not None and not value['ph_unreachable']:
                assert abs(value['ph'] - target) <= 1e-7
            result = lx.record(c, value, kex, ln_k['R2'])
            if not direct:
                aq, org = value['phase_amounts_mol']
                result.update(aqueous_bta_mol_l=aq.get('bta-anion', 0),
                              hbta_lost_mg_l=aq.get('bta-anion', 0) * .21616 * 1e6,
                              hbta_lost_percent=100 * value['hbta_lost_to_aqueous_fraction'],
                              na_loaded_organic_mol_l=org.get('sodium-carrier', 0) / c.organic_to_aqueous,
                              na_loaded_over_feed_na=org.get('sodium-carrier', 0) / c.na if c.na else None)
            result.update(net_base_mol=value.get('net_base_mol'), ph_unreachable=value.get('ph_unreachable'))
            if r.get('ph_error'):
                result['ph_error'] = value['ph'] - float(seo[c.identity]['equilibrium_ph'])
                result['ph_pass'] = abs(result['ph_error']) <= .5
            result['d_pass'] = abs(result['log10_model_over_observed_d_li']) <= .30 if r.get('d_pass') else None
            if r.get('log10_imposed_over_forward_d_li'):
                result['log10_imposed_over_forward_d_li'] = math.log10(result['model_d_li'] /
                    float(forward[(r['end'], 'fitted', r['identity'])]['model_d_li']))
            if r.get('imposed_ph_outside_calibration'):
                bounds = sc._table2_ph()[1].values()
                result['imposed_ph_outside_calibration'] = not min(bounds) <= target <= max(bounds)
            checked(name, r, result, FIELDS, ('active_reactions', 'status', 'd_pass', 'ph_pass',
                    'ph_unreachable', 'imposed_ph_outside_calibration'), value)


def d2(basis):
    from analyses.prototypes.d2ehdtpa_buphen_ms2_prototype.scripts import reaction_coordinates as rc
    from analyses.prototypes.d2ehdtpa_buphen_ms2_prototype.scripts import calibrate_and_score as cs
    from analyses.prototypes.d2ehdtpa_buphen_ms2_prototype.scripts import conditional_figiel_packet as cf

    cf.select('figiel-2025-chloride-activity-refit/2')
    cf.use({'held': 'held_explicit_sr', 'figiel': 'packet'}[basis])
    folder = cs.ANALYSIS / 'results' / ('' if basis == 'held' else 'conditional_figiel_packet')
    q = json.loads((folder / 'fig7_exchange_start0.json').read_text())['ln_K_Li_Mg_Ca']
    for table, count in (('f2', 26), ('holdout_scores', 34)):
        reference = rows(folder / f'{table}.csv')
        keys = ('figure', 'series', 'point') + (('source_temperature_K',) if table != 'f2' else ())
        assert len(reference) == len({tuple(r[k] for k in keys) for r in reference}) == count
        for i, r in enumerate(reference):
            print('D2', basis, table, i, flush=True)
            name = f'f1_d2_{basis}_{table}_{i}.json'
            if (STUDY / 'results' / name).exists():
                if json.loads((STUDY / 'results' / name).read_text())['passed']:
                    continue
                name = f'f1_d2_{basis}_{table}_{i}_corrected.json'
            if r.get('scored') == 'False':
                checked(name, r, {k: r[k] for k in ('scored', 'reason')}, [], ('scored', 'reason'))
                continue
            source = next(s for s in rows(cs.REFERENCE / f'figure{r["figure"]}.csv')
                          if (s['series'], s['point']) == (r['series'], r['point']) and
                          (table == 'f2' or s['temperature_K'] == r['source_temperature_K']))
            if table == 'f2':
                s = cs.BASIS['states']['fig' + r['figure']]
                feed, b = s['feed_mol_L'].copy(), s['buphen_mol_L']
                if r['figure'] == '7':
                    b = float(source['x'])
                else:
                    feed['Na'] = (0 if r['point'] == '0' else float(source['x'])) / (1000 * cs.M_NACL)
                value = rc.solve(feed, s['sites_mol_L'], b, q)
            else:
                sites = float(source['d2ehdtpa_mol_L_organic'])
                value = rc.solve(*cs.conditions(source, 1e-6 if r['figure'] == '10' else 0), q,
                                 loading={'Li': sites} if r['figure'] == '10' else None)
            alpha = value['alpha'][r['series'][3:]]
            result = dict(alpha_model=alpha, error_decade=math.log10(alpha / float(source['alpha'])),
                          Na_eq_mol_kg_water=value['Na_mol_kg_water'], log10_Na_eq=math.log10(value['Na_mol_kg_water']))
            result['passed'] = abs(result['error_decade']) <= .30
            if table != 'f2':
                org = value['amounts_mol'][1]
                result.update(Li_site_fraction=org['lib2a'] / sites,
                              metal_site_fraction=(org['lib2a'] + 2 * org['mga2'] + 2 * org['caa2']) / sites,
                              below_ms2_design_site_range=sites < .020)
                if r['figure'] == '10':
                    other = rc.solve(*cs.conditions(source, 1e-7), q, loading={'Li': sites})
                    result['alpha_model_Na_1e_7'] = other['alpha'][r['series'][3:]]
                    result['Na_trace_falsifier_decade'] = math.log10(alpha / result['alpha_model_Na_1e_7'])
                    result['Na_trace_falsifier_passed'] = abs(result['Na_trace_falsifier_decade']) <= 1e-3
                    value['trace_solve'] = other
                elif sites == .05:
                    nine = [p for p in rows(cs.REFERENCE / 'figure9.csv') if p['series'] == r['series']]
                    observed = float(cs.np.interp(50, [float(p['x']) for p in nine], [float(p['alpha']) for p in nine]))
                    assert observed == float(r['fig9_alpha_50_g_L_NaCl'])
                    assert math.log10(float(source['alpha']) / observed) == float(r['source_scatter_fig8_vs_fig9_decade'])
            checked(name, r, result, ('alpha_model', 'error_decade', 'Na_eq_mol_kg_water', 'log10_Na_eq',
                    'alpha_model_Na_1e_7', 'Li_site_fraction', 'metal_site_fraction'),
                    ('passed', 'below_ms2_design_site_range', 'Na_trace_falsifier_passed'), value)


if __name__ == '__main__':
    hbta() if sys.argv[1] == 'hbta' else d2(sys.argv[1])
