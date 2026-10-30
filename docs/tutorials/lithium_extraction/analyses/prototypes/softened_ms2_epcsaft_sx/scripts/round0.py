"""#216 frozen-feed and retained-response comparisons; no retained result writes."""

import csv
import json
import math
from pathlib import Path

from analyses.shared.hbta_topo_configured_reactive_lle.scripts import mechanism_candidates as mc
from lithium_extraction.common.support.paths import REPO_ROOT
from lithium_extraction.prototypes.produced_water_sx_prototype import brines, records
from lithium_extraction.prototypes.produced_water_sx_prototype.flowsheet import installed_engine

STUDY = Path(__file__).resolve().parents[1] / 'runs/amendments_2_4'
H158 = REPO_ROOT / 'analyses/shared/hbta_topo_configured_reactive_lle/results'


def save(name, value, directory=None):
    with ((STUDY / 'results' if directory is None else directory) / name).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def compare(reference, result, names):
    return {k: {'retained': float(reference[k]), 'new': result[v],
                'scaled_difference': abs(float(reference[k]) - result[v]) /
                (1 / math.log(10) if k.startswith('log10') or k == 'error_decade' else
                 max(abs(float(reference[k])), abs(result[v]), 1e-6))}
            for k, v in names.items() if reference[k] != ''}


def run():
    engine = installed_engine()
    assert engine['sha256'] == '8897a0caab9f2508ed642f42c8b5226957fbe7035ed7bfdad71e4166798785b3'
    if not (STUDY / 'results/engine.json').exists():
        save('engine.json', engine)
    feed = {s: records.extraction_feed(brines.raw_feed('MS2'), s, synthetic_softened=True)
            for s in ('A', 'C', 'D2EHDTPA-Held', 'D2EHDTPA-Figiel')}
    for f in feed.values():
        assert abs(sum(brines.CHARGE.get(e, 0) * n for e, n in f['physical_feed'].items())) <= 1e-12
        for e, mg in (('Li', 168), ('Na', 64100), ('K', 2300)):
            assert abs(f['physical_feed'][e] * brines.MOLAR_MASS[e] * 1000 - mg) <= 1e-9
    if not (STUDY / 'results/feed.json').exists():
        save('feed.json', feed)
    reference = list(csv.DictReader((H158 / 'mechanism_candidates.csv').open()))
    summary = json.loads((H158 / 'mechanism_candidates.json').read_text())['summary']
    selected = [r for r in reference if r['reaction_set'] in ('A', 'C') and r['role'] == 'fit']
    assert len(selected) == len({(r['reaction_set'], r['case'], r['identity']) for r in selected}) == 8
    names = {'model_d_li': 'li_distribution', 'model_d_na': 'na_distribution',
             'model_li_extraction_percent': 'li_extraction_percent', 'model_ph': 'ph',
             'neutral_li_fraction': 'neutral_li_fraction', 'aqueous_bta_mol': 'aqueous_bta_mol',
             'hbta_lost_to_aqueous_fraction': 'hbta_lost_to_aqueous_fraction', 'net_base_mol': 'net_base_mol',
             'log10_model_over_observed_d_li': 'log10_model_over_observed_d_li'}
    for i, r in enumerate(selected):
        observation = next(o for o in mc.observations() if (o[0], o[1], o[2].identity) ==
                           (r['role'], r['case'], r['identity']))
        print('F1', r['reaction_set'], r['identity'], flush=True)
        try:
            prior = STUDY / f'results/f1_h158_{i}.json'
            value = (json.loads(prior.read_text())['result'] if prior.exists() else
                     mc.evaluate(observation, r['reaction_set'], summary[r['reaction_set']]['ln_k_298']))
            if value is None:
                assert r['status'] == 'not_representable' and float(r['model_d_li']) == 0
                save(f'f1_h158_{i}_amended.json', {'reference': r, 'result': None, 'passed': True})
                continue
            value['log10_model_over_observed_d_li'] = math.log10(value['li_distribution'] / observation[4])
            comparisons = compare(r, value, names)
            passed = all(c['scaled_difference'] <= 1e-6 for c in comparisons.values())
            passed &= value.get('ph_unreachable', False) == (r['ph_unreachable'] == 'True')
            if not value.get('ph_unreachable', False):
                passed &= abs(value['ph'] - observation[3]) <= 1e-7
            save(f'f1_h158_{i}_amended.json', {'reference': r, 'result': value, 'comparisons': comparisons, 'passed': passed})
        except Exception as error:
            save(f'f1_h158_{i}_failure.json', {'reference': r, 'error': repr(error)})
            raise
        if not passed:
            raise RuntimeError(f'F1 failed: {r["reaction_set"]} {r["identity"]}; Round 0 stops for diagnosis')


if __name__ == '__main__':
    run()


def fit_residuals():
    output = {}
    summary = json.loads((H158 / 'mechanism_candidates.json').read_text())['summary']
    for arm in ('A', 'C'):
        old, new = [], []
        for i in range(8):
            r = json.loads((STUDY / f'results/f1_h158_{i}_amended.json').read_text())
            reference, value = r['reference'], r['result']
            if reference['reaction_set'] != arm or value is None:
                continue
            for model, observed, result in (('model_d_li', 'observed_d_li', 'li_distribution'),
                                            ('model_d_na', 'observed_d_na', 'na_distribution')):
                if reference[observed]:
                    a = math.log10(float(reference[model]) / float(reference[observed]))
                    b = math.log10(value[result] / float(reference[observed]))
                    for vector, residual, ph in ((old, a, float(reference['model_ph'])), (new, b, value['ph'])):
                        miss = abs(ph - float(reference['target_ph'])) if reference['ph_unreachable'] == 'True' else 0
                        vector.append(residual + math.copysign(miss, residual))
        assert len(old) == len(new) == 4 and max(abs(a - b) for a, b in zip(old, new)) <= 1e-6 / math.log(10)
        output[arm] = {'csv_residuals': old, 'new_residuals': new,
                       'csv_cost': sum(v*v for v in old)/2, 'new_cost': sum(v*v for v in new)/2,
                       'fit_history_residuals': summary[arm]['fit_residuals_log10'],
                       'fit_history_cost': summary[arm]['fit_cost']}
    save('f1_h158_fit_residuals.json', output)
