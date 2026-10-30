"""Resumable units for #176; run each numerical unit separately under the shared CPU lock."""

import argparse
import itertools
import json
import math
import time
from pathlib import Path

from analyses.prototypes.hbta_topo_ms2_prototype.scripts.li_na_exchange import _write_csv
from lithium_extraction.prototypes.produced_water_sx_prototype import flowsheet as fs

ANALYSIS = Path(__file__).resolve().parents[1]
RESULTS = ANALYSIS / 'results'
PARAMETERS = json.loads((ANALYSIS / 'inputs/full_chain_parameters.json').read_text())


def write(name, value):
    path = RESULTS / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False) + '\n')
    tmp.replace(path)


def run(args):
    started = time.perf_counter()
    if args.step == 'soften':
        from analyses.prototypes.hbta_topo_ms2_prototype.scripts.precipitative_softening import MS2_PH, ms2_feed
        feed = ms2_feed() | {'pH': MS2_PH}
        values = {case: fs.washed_feed(feed, PARAMETERS[case]['moisture'], PARAMETERS[case]['wash_volumes'])
                  for case in ('base', 'optimistic')}
        values['wall_time_seconds'] = time.perf_counter() - started
        write('full_chain_softening.json', values)
    elif args.step == 'extract':
        summary = json.loads((RESULTS / 'set_c_summary.json').read_text())
        engine = fs.installed_engine()
        units = list(itertools.product(('base', 'optimistic'), ('low', 'high_r_0.70'), (.1, .2, .3, .4)))
        case, end, concentration = units[args.unit]
        path = RESULTS / 'full_chain_extraction.json'
        done = json.loads(path.read_text()) if path.exists() else {}
        if str(args.unit) in done:
            print(f'Unit {args.unit} already retained', flush=True)
            return
        feed = json.loads((RESULTS / 'full_chain_softening.json').read_text())[case]
        # The summary uses high_r_0.70 for the Zhang 2021 end.
        constants = summary['ends']['high' if end == 'high_r_0.70' else end]['ln_k']
        values = []
        for bracket, shift in (('fitted', 0.), ('minus', -.5 * math.log(10)), ('plus', .5 * math.log(10))):
            result = fs.extract(feed, {k: value + shift for k, value in constants.items()}, concentration)
            values.append({'case': case, 'end': end, 'delta_case': bracket, 'extractant_mol_l': concentration,
                           'feed_na_mol_l': feed['aqueous_mol']['Na'] / feed['feed_l'],
                           'feed_li_mg_l': feed['aqueous_mol']['Li'] * 6940 / feed['feed_l'], **result})
            print(f'Unit {args.unit} {case}/{end}/{concentration}/{bracket}: '
                  f"Li recovery {100 * result['li_extraction_fraction']:.5f}%, {result['sweeps']} sweeps", flush=True)
        done[str(args.unit)] = {'values': values, 'wall_time_seconds': time.perf_counter() - started, 'engine': engine}
        write('full_chain_extraction.json', done)
    else:
        soft = json.loads((RESULTS / 'full_chain_softening.json').read_text())
        extraction = json.loads((RESULTS / 'full_chain_extraction.json').read_text())
        if len(extraction) != 16:
            raise ValueError('All 16 extraction units (48 cascades) are required before costing')
        if json.loads((RESULTS / 'carbonation_summary.json').read_text())['falsifier_passed']:
            raise ValueError('Empirical fallback is authorized only after the failed Pitzer comparison')
        rows = [r for unit in extraction.values() for r in unit['values']]
        costs, sensitivity, candidates = {}, [], []
        for case in ('base', 'optimistic'):
            eligible = []
            for r in rows:
                if r['case'] != case or r['end'] != PARAMETERS[case]['band_end'] or r['delta_case'] != 'fitted':
                    continue
                assumptions = PARAMETERS['shared'] | PARAMETERS[case]
                assumptions['extractant_mol_l'] = r['extractant_mol_l']
                assumptions['diluent_usd_kg'] = (assumptions['diluent_usd_gal'] / 3.785411784) / (
                    r['diluent_mol'] * .17033484 / soft[case]['feed_l'])
                candidate = {'case': case, 'extractant_mol_l': r['extractant_mol_l']}
                try:
                    result = fs.costs_and_balances(soft[case], r, case, assumptions)
                except ValueError as error:
                    candidate['diagnostic'] = str(error)
                else:
                    candidate.update(operating_usd_kg_li=result['operating_usd_kg_li'],
                                     strip_na_fraction=result['strip_na_fraction'],
                                     mother_nacl_mol_kg=result['mother_nacl_mol_kg'])
                    eligible.append((result, r, assumptions))
                candidates.append(candidate)
            costs[case], r, assumptions = min(eligible, key=lambda item: item[0]['operating_usd_kg_li'])
            for efficiency in (.9, .95, .9867, .999):
                point = {'case': case, 'assumed_na_scrub_efficiency': efficiency}
                try:
                    result = fs.costs_and_balances(soft[case], r, case, assumptions, efficiency)
                except ValueError as error:
                    point['diagnostic'] = str(error)
                else:
                    point.update(strip_na_fraction=result['strip_na_fraction'],
                                 operating_usd_kg_li=result['operating_usd_kg_li'])
                sensitivity.append(point)
        costs['wall_time_seconds'] = time.perf_counter() - started
        write('full_chain_results.json', costs)
        _write_csv(RESULTS / 'full_chain_scrub_sensitivity.csv', sensitivity)
        _write_csv(RESULTS / 'full_chain_cost_candidates.csv', candidates)
        _write_csv(RESULTS / 'full_chain_extraction.csv', [{k: r[k] for k in ('case', 'end', 'delta_case',
                   'extractant_mol_l', 'feed_na_mol_l', 'feed_li_mg_l', 'li_extraction_fraction', 'hbta_lost_fraction',
                   'sweeps')} | {f'{e}_closure_mol': n for e, n in r['closure_mol'].items()} for r in rows])
    print(f'{args.step} unit {args.unit}: {time.perf_counter() - started:.3f} s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=('soften', 'extract', 'cost'))
    parser.add_argument('--unit', type=int, choices=range(16), default=0)
    run(parser.parse_args())
