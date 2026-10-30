"""#216 P-max and P-min, with independent fixed-design outlet verification."""

import json
import re
import signal
import sys
import time

from pyomo.environ import Objective, minimize, value
from scipy.optimize import brentq

from analyses.prototypes.d2ehdtpa_buphen_ms2_prototype.scripts import reaction_coordinates as rc
from lithium_extraction.common.support.paths import REPO_ROOT
from lithium_extraction.prototypes.produced_water_sx_prototype import brines, d2_stage, des_stage as des, records, references, set_a, set_c, stage
from .decisions import attach
from .des_process import timeout
from .replay import cc, constants, mc
from .round0 import H158, STUDY, save


def build(arm, n, start, fixed=None, cache=None, feed=None, initialization=None, initialization_log=None):
    if arm.startswith('DES'):
        design = dict(total_N_mol=des.preparation((9.5, 10., 10.5)[start])['total_N_mol'], topo_wt_fraction=(.2, .3, .4)[start])
        design.update(fixed or {})
        if initialization_log is not None:
            initialization_log.update(design)
        model = des.build(arm[-1], n, design['total_N_mol'], design['topo_wt_fraction'])
        if initialization_log is not None:
            initialization_log.update(cascade=model._initialization)
        model._design, model._physical_feed = design, des.feed()
        model._decisions = dict(total_N_mol=model.fs.des.total_N, topo_wt_fraction=model.fs.des.topo_fraction)
        return model
    d2 = arm.startswith('D2')
    feed = records.extraction_feed(brines.raw_feed('MS2'), 'D2EHDTPA-' + arm[2:] if d2 else 'A', synthetic_softened=True) if feed is None else feed
    if d2:
        folder = rc.INPUTS.parent / 'results' / ('' if arm == 'D2Held' else 'conditional_figiel_packet')
        design = dict(B=(.25, .625, 1.)[start], S=(.9, 1.2, 1.5)[start], O_A=1., Sr_decade=0.,
                      site_basis_li_mol_l=feed['model_feed']['Li'],
                      ln_K_Li_Mg_Ca=json.loads((folder / 'fig7_exchange_start0.json').read_text())['ln_K_Li_Mg_Ca'])
        design.update(fixed or {})
        model = d2_stage.build_flowsheet(feed, solvent='D2EHDTPA-' + arm[2:], stages=n, scenario=design)
        initial = rc.solve({e: feed['model_feed'].get(e, 0) for e in rc.ION if e != 'Cl'},
                           design['S'] * design['site_basis_li_mol_l'] / design['O_A'], design['B'],
                           design['ln_K_Li_Mg_Ca'], organic_to_aqueous=design['O_A'])
        profile = {s: sum(p.get(s, 0) for p in initial['amounts_mol']) for s in model._chemistry.species}
        stage.initialize(model, [profile] * n)
    else:
        if 'acid' in arm:
            ln_k = json.loads((H158 / 'mechanism_candidates.json').read_text())['summary']['A' if arm == 'acid_A' else 'C']['ln_k_298']
        else:
            _, _, ln_k = constants(dict(zip(('end', 'delta_case'), arm.split(':'))))
        design = dict(organic_to_aqueous_ratio=(.75, 1.5, 2.5)[start],
                      extractant_mol_l=((.05, .175, .30) if 'acid' in arm else (.1, .25, .4))[start],
                      saponification_ratio=0. if 'acid' in arm else (.5, .75, 1.)[start])
        design.update(fixed or {})
        ratio, c, sap = (design[k] for k in ('organic_to_aqueous_ratio', 'extractant_mol_l', 'saponification_ratio'))
        profile = None
        log = {} if initialization_log is None else initialization_log
        if initialization is not None:
            from pathlib import Path
            neighbour = json.loads(Path(initialization['source']).read_text())
            assert neighbour['arm'] == arm and neighbour['n'] == n and neighbour['accepted']
            log.update(initialization, steps=[])
            previous = neighbour['design']
            amounts = [row['certificate']['phase_amounts_mol'] for row in neighbour['result']['stages']]
            profile = {'guesses': amounts, 'organic': [dict(li=p[1]['lithium-carrier'] / previous['organic_to_aqueous_ratio'],
                na=p[1]['sodium-carrier'] / previous['organic_to_aqueous_ratio'],
                hbta=sum(p[1].get(s, 0.) for s in ('hbta', 'lithium-carrier', 'sodium-carrier')) /
                previous['organic_to_aqueous_ratio']) for p in amounts]}
            begun = time.monotonic()
            for step_number in range(1, 9):
                controls = [previous[k] + step_number / 8 * (target - previous[k]) for k, target in zip(
                    ('organic_to_aqueous_ratio', 'extractant_mol_l', 'saponification_ratio'),
                    (ratio, c, initialization['target_R']), strict=True)]
                step = dict(step=step_number, controls=controls)
                log['steps'].append(step)
                before = time.monotonic()
                try:
                    if before - begun >= 290:
                        raise TimeoutError('eight-step continuation exhausted its total 290 s budget')
                    continued = cc.countercurrent(ln_k, reaction_set='C', stage_count=n,
                        organic_to_aqueous=controls[0], extractant=controls[1], saponification_fraction=controls[2],
                        feed_li=feed['model_feed']['Li'], feed_na=feed['model_feed']['Na'],
                        feed_acidity=feed['model_feed']['H'], start=profile)
                    profile = continued['start']
                    step['cascade'] = continued
                except RuntimeError as error:
                    step['error'] = repr(error)
                    raise
                finally:
                    step['elapsed_s'] = time.monotonic() - before
                    log['elapsed_s'] = time.monotonic() - begun
                if time.monotonic() - begun > 290:
                    raise TimeoutError('eight-step continuation exceeded its total 290 s budget')
        direct = None
        def initialize(preparation):
            nonlocal direct
            direct = cc.countercurrent(ln_k, reaction_set='A' if arm == 'acid_A' else 'C', stage_count=n,
                organic_to_aqueous=ratio, extractant=c, saponification_fraction=preparation,
                feed_li=feed['model_feed']['Li'], feed_na=feed['model_feed']['Na'], feed_acidity=feed['model_feed']['H'], start=profile)
            return min(ratio * (min(row['hbta_org_in_mol_l'], c) - row['li_org_in_mol_l'] - row['na_org_in_mol_l'])
                - (direct['start']['guesses'][k-1][0].get('hydroxide-anion', 0.) if k else 0.)
                for k, row in enumerate(direct['stages']))
        if initialize(sap) < 0 and fixed is None:
            sap = brentq(initialize, .5, sap, xtol=1e-14) - 1e-12
            initialize(sap)
        design['saponification_ratio'], design['initial_R'] = sap, sap
        if arm == 'acid_A':
            model = set_a.build_flowsheet(feed, stages=n, scenario={k: v for k, v in design.items() if k != 'initial_R'})
            stage.initialize(model, [{s: sum(p.get(s, 0) for p in amounts) for s in model._chemistry.species}
                                     for amounts in direct['start']['guesses']])
        else:
            washed = dict(feed_l=1., aqueous_mol=feed['model_feed'], pH=5.77, feed_acidity=feed['model_feed']['H'])
            design['ln_k'] = ln_k
            model = set_c.build_flowsheet(washed, stages=n, scenario=design, direct_cascade=direct,
                                         physical_feed=feed, chemistry_by_support=cache)
    return model


def solve(model, arm, directory, target, release, hessian='exact'):
    if arm.startswith('DES'):
        assert target is None, 'DES supports P-max only'
        if release:
            des.release(model)
        stage.solve(model, directory, hessian=hessian)
        return model, stage.report(model)
    def connect(m):
        attach(m, arm, release=release, target=target if release else None)
        if release:
            m.feasibility.deactivate()
            ratio = next(iter(m._decisions.values()))
            m.optimum = Objective(expr=ratio if target is not None else -m.fs.paper1.extraction, sense=minimize)
    if arm != 'acid_A' and not arm.startswith('D2'):
        return set_c.solve(model, directory, started=time.monotonic(), connect_units=connect, hessian=hessian,
                           charge_limit_eq_per_raw_l=set_c.charge_limit(model._washed['feed_l']))
    connect(model)
    stage.solve(model, directory, hessian=hessian)
    return model, stage.report(model)


def physical_check(model, result, arm):
    if arm.startswith('DES'):
        bound = des.charge_bound(model, result)
        return dict(failures=[] if bound['passed'] else ['F4'], F4=bound,
                    actual_stream_injection=des.injected_ion_check(model, bound['Q_DES_by_N'][len(model.fs.stages)-1]))
    out = result.get('outlet_mol_per_nominal_l', result.get('outlet_mol_per_raw_l'))
    raffinate = model._physical_feed['model_feed'] | {records.SPECIES_LABELS[s]: out[s] for s in model._chemistry.supports[0]}
    return references.check_r7(model._physical_feed, raffinate,
        balance_residuals_mol_per_raw_l=result['balance_residuals_mol_per_raw_l'],
        strict_certificate=result['strict_certificate'],
        charge_limit_eq_per_raw_l=None if arm == 'acid_A' or arm.startswith('D2') else set_c.charge_limit(model._washed['feed_l']))


def run(arm, n, target, start, fixed=None, feed=None, release=True, case='', gate=None, initialization=None):
    is_des = arm.startswith('DES')
    directory = des.d.STUDY/'runs/des2/pmax' if is_des else STUDY/'results'
    directory.mkdir(parents=True, exist_ok=True)
    if is_des:
        assert target is None, 'DES supports P-max only'
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(290)
    for unit in ('0', '14', 'held'):
        assert json.loads((STUDY / f'results/f2_capacity/f2_{unit}.json').read_text())['passed']
    label = f'{case}{"capacity_" if "acid" not in arm and not arm.startswith(("D2", "DES")) else ""}{arm.replace(":", "_")}_N{n}_{("max" if release else "fixed") if target is None else target}_start{start}'
    started = time.monotonic()
    print('optimization', label, flush=True)
    try:
        initialization_log = {}
        model = build(arm, n, start, fixed=fixed, feed=feed, initialization=initialization, initialization_log=initialization_log)
        model, result = solve(model, arm, REPO_ROOT / 'build/issue216_engine_8298c602' / label, target, release)
        design = {k: value(v) for k, v in model._decisions.items()}
        if not is_des:
            ratio, concentration, preparation = design.values()
        out = result.get('outlet_mol_per_nominal_l', result.get('outlet_mol_per_raw_l'))
        d2 = arm.startswith('D2')
        complex_li, complex_na = ('lib2a', 'naa') if d2 else ('lithium-carrier', 'sodium-carrier')
        extraction = 1 - out['lithium-cation'] / model._feeds[0]['lithium-cation']
        residuals = result['balance_residuals_mol_per_raw_l']
        moieties = des.d.chemistry(arm[-1])[1] if is_des else rc.MOIETIES if d2 else {name: mc.MOIETIES[name] for name in
            (set_a.CONSERVED if arm == 'acid_A' else set_c.CONSERVED)}
        for row in result['stages']:
            entering = row.get('inlet_mol_per_nominal_l', row.get('inlet_mol_per_raw_l'))
            leaving = row.get('outlet_mol_per_nominal_l', row.get('outlet_mol_per_raw_l'))
            row['named_conservation_mol'] = {name: sum(w * (leaving.get(s, 0) - entering.get(s, 0))
                for s, w in coefficients.items()) for name, coefficients in moieties.items()}
        result['named_global_conservation_mol'] = {name: sum(w * (out.get(s, 0) - sum(p.get(s, 0) for p in model._feeds))
            for s, w in coefficients.items()) for name, coefficients in moieties.items()}
        assert max(abs(v) for row in result['stages'] for v in row['named_conservation_mol'].values()) <= 1e-9
        assert max(map(abs, result['named_global_conservation_mol'].values())) <= 1e-9
        r7 = physical_check(model, result, arm)
        result['R7'] = r7
        assert result['strict_certificate'] and not r7['failures'] and max(map(abs, residuals.values())) <= 1e-9
        fresh = build(arm, n, (start + 1) % 3, fixed=design, feed=feed)
        fresh, square = solve(fresh, arm, REPO_ROOT / 'build/issue216_engine_8298c602' / (label + '_square'), None, False)
        square['R7'] = physical_check(fresh, square, arm)
        square_out = square.get('outlet_mol_per_nominal_l', square.get('outlet_mol_per_raw_l'))
        difference = max(abs(out[s] - square_out[s]) / (3 if is_des and s in ('des', *des.d.ORG[2:]) else 1) for s in out)
        for a, b in zip(result['stages'], square['stages'], strict=True):
            a = a.get('outlet_mol_per_nominal_l', a.get('outlet_mol_per_raw_l'))
            b = b.get('outlet_mol_per_nominal_l', b.get('outlet_mol_per_raw_l'))
            difference = max(difference, max(abs(a[s] - b[s]) / (3 if is_des and s in ('des', *des.d.ORG[2:]) else 1) for s in a))
        assert difference <= 1e-9 and square['strict_certificate'] and not square['R7']['failures'], f'F7 difference {difference} mol'
        if is_des:
            quantities = des.metrics(model, result)
        else:
            inlet = model._feeds[1]
            loaded_li, loaded_na = out[complex_li], out[complex_na]
            first = result['stages'][0]
            first_out = first.get('outlet_mol_per_nominal_l', first.get('outlet_mol_per_raw_l'))
            selectivity = (first_out[complex_li] / first_out['lithium-cation']) / (first_out[complex_na] / first_out['sodium-cation'])
            quantities = dict(extraction_fraction=extraction, stage1_li_na_selectivity=selectivity,
                loaded_organic_na_fraction=loaded_na / (loaded_li + loaded_na), net_na_to_organic_mol=loaded_na - inlet[complex_na],
                extractant_circulation_mol_m3=1000 * (preparation * model._design['site_basis_li_mol_l'] if d2 else ratio * concentration),
                initial_naoh_mol_m3=1000 * (preparation * model._design['site_basis_li_mol_l'] if d2 else ratio * concentration * preparation),
                formal_hcl_mol_m3=1000 * (loaded_li + loaded_na),
                model_bta_loss_mol_m3=1000 * out.get('bta-anion', 0),
                buphen_circulation_mol_m3=1000 * ratio * concentration if d2 else None,
                model_buphen_dissolution_mol_m3=1000 * result['stages'][-1]['outlet_mol_per_raw_l']['buphen'] / ratio / d2_stage.owner.BUPHEN_ORGANIC_AQUEOUS_PARTITION if d2 else None,
                formal_naoh_mol_m3=1000 * (preparation * model._design['site_basis_li_mol_l'] if d2 else ratio * concentration * preparation),
                loading=loaded_li / (inlet['naa'] if d2 else ratio * concentration))
        multiplier = model.dual.get(model.fs.paper1.target) if target is not None else None
        save(label + '.json', dict(arm=arm, n=n, target=target, start=start, case=case, released=release, eligibility_gate=gate, design=design, quantities=quantities,
            initialization_correction=initialization_log,
            initialization_projection=None if is_des else {'prescribed_R': (fixed or {}).get('saponification_ratio', 0. if 'acid' in arm else (.5, .75, 1.)[start]) if not d2 else None, 'initial_R': model._design.get('initial_R'), 'metric': 'nearest R at fixed O/A and c; root xtol 1e-14, inward offset 1e-12'},
            multiplier=multiplier, multipliers={str(c): d for c, d in model.dual.items()},
            active_set=sorted([str(c) for c, d in model.dual.items() if abs(d) > 1e-7] +
                [str(v) + side for suffix, side in ((model.ipopt_zL_out, '/lower'), (model.ipopt_zU_out, '/upper'))
                 for v, d in suffix.items() if abs(d) > 1e-7]),
            bound_multipliers={str(v): [model.ipopt_zL_out.get(v, 0), model.ipopt_zU_out.get(v, 0)]
                               for v in model.component_data_objects(stage.Var)},
            result=result, fixed_replay=square, f7_difference_mol=difference,
            iterations={str(p.relative_to(REPO_ROOT)): int(re.search(r'Number of Iterations.*:\s+(\d+)', p.read_text())[1])
                        for p in (REPO_ROOT / 'build/issue216_engine_8298c602' / label).rglob('ipopt.log')},
            wall_seconds=time.monotonic() - started, accepted=True), directory=directory)
    except Exception as error:
        save(label + f'_failure_{time.time_ns()}.json', dict(arm=arm, n=n, target=target, start=start, eligibility_gate=gate, error=repr(error),
             result=locals().get('result'), initialization_correction=locals().get('initialization_log'),
             initial_design=getattr(locals().get('model'), '_design', None), wall_seconds=time.monotonic() - started, accepted=False), directory=directory)
        print(label, repr(error), flush=True)
    finally:
        if is_des:
            signal.alarm(0)


if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]), None if sys.argv[3] == 'max' else float(sys.argv[3]), int(sys.argv[4]))
