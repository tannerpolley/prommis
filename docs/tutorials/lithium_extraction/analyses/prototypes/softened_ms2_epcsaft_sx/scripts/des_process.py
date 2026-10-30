"""#216 DES-2 fixed-decision verification, before the owner's optimization release."""
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from pyomo.environ import value
from lithium_extraction.prototypes.produced_water_sx_prototype import des_stage as adapter, stage
from . import des_chemistry as d

OUT = d.STUDY/'runs/des2/results'


def save(name, result):
    (OUT/name).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


def timeout(signum, frame):
    raise TimeoutError('DES-2 attempt exhausted its 290 s wall budget')


def fixed(candidate, n):
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(290)
    started = time.monotonic()
    print("fixed", candidate, n, "starting", flush=True)
    record = {'candidate': candidate, 'N': n, 'accepted': False,
              'label': 'conditional extrapolative predictions in composition and loading'}
    model = None
    try:
        total_n = adapter.preparation(10.)['total_N_mol']
        model = adapter.build(candidate, n, total_n, .3)
        record['initialization'] = model._initialization
        stage.solve(model, Path('build/issue216/des2')/f'{candidate}_fixed{n}')
        result = stage.report(model)
        record['result'] = result
        record['F4'] = adapter.charge_bound(model, result)
        record['actual_stream_injection'] = adapter.injected_ion_check(model, record['F4']['Q_DES_by_N'][n-1])
        assert record['F4']['passed'] and result['strict_certificate']
        assert max(map(abs, result['balance_residuals_mol_per_raw_l'].values())) <= 1e-9
        assert all(row['certificate']['physical_outlet_max_difference_mol'] <= 1e-9 for row in result['stages'])
        record['quantities'] = adapter.metrics(model, result)
        record.update(total_N_mol=total_n, topo_wt_pct=30., preparation_pH=10.,
                      E_Li=value(model.fs.des.extraction), fixed_direct=model._initial, accepted=True)
    except Exception as error:
        record['error'] = f'{type(error).__name__}: {error}'
        if model is not None:
            record['actual_stage_outlets'] = [{s: value(model.fs.stage[k].outlet[s]) for s in d.IDS} for k in model.fs.stages]
            record['actual_stage_inlets'] = [{s: value(model._incoming(k, s)) for s in d.IDS} for k in model.fs.stages]
            record['solve'] = getattr(model, '_solve', None)
    finally:
        signal.alarm(0)
        record['seconds'] = time.monotonic()-started
        save(f'fixed_{candidate}_N{n}.json', record)
        print(candidate, n, record['accepted'], record.get('E_Li', record.get('error')), record['seconds'], flush=True)
    return record


def summarize_pmax():
    directory = d.STUDY/'runs/des2/pmax'
    records = [(p.name, json.loads(p.read_text())) for p in sorted(directory.glob('DES-*_N*.json'))]
    groups = []
    for candidate in ('X', 'S'):
        previous = None
        for n in range(1, 6):
            starts = [(p, r) for p, r in records if r['arm'] == 'DES-'+candidate and r['n'] == n]
            accepted = sorted(((p, r) for p, r in starts if r['accepted']),
                              key=lambda pair: pair[1]['quantities']['E_Li'], reverse=True)
            distinct = []
            for path, row in accepted:
                extraction = row['quantities']['E_Li']
                match = next((g for g in distinct if abs(extraction-g['E_Li']) <= 1e-4*abs(g['E_Li'])), None)
                if match is None:
                    distinct.append(dict(E_Li=extraction, sources=[path]))
                else:
                    match['sources'].append(path)
            best = accepted[0][1]['quantities']['E_Li'] if accepted else None
            trend = None if best is None or previous is None else best >= previous-1e-6
            groups.append(dict(candidate=candidate, N=n, starts=[p for p, r in starts],
                               attempted_starts=len(starts), accepted_starts=len(accepted),
                               distinct_objectives=distinct, P_max_raw=best, F9_previous_N=trend))
            previous = best
    (directory/'summary.json').write_text(json.dumps(groups, indent=2, allow_nan=False)+'\n')
    return groups


def pmax():
    for candidate in ('X', 'S'):
        for n in range(1, 6):
            for start in range(3):
                subprocess.run([sys.executable, '-m', __package__+'.optimization', 'DES-'+candidate, str(n), 'max', str(start)], check=True)
    summarize_pmax()


if __name__ == '__main__':
    if sys.argv[1:] == ['pmax']:
        pmax()
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        save('preparation.json', [adapter.preparation(ph) for ph in (9.5, 10., 10.5)])
        for candidate in ('X', 'S'):
            for n in range(1, 6):
                fixed(candidate, n)
