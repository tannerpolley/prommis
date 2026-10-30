"""Sequential #201 campaign with 290 s candidate budgets and resumable local checkpoints."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
STUDY = ROOT / 'analyses/prototypes/modular_process'
RESULTS = STUDY / 'results/campaign'


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def assumptions(site, case, front_end=None):
    result = json.loads((STUDY / f'results/full_process/{site}_{case}.json').read_text())
    if front_end is not None:
        result['softening'] = json.loads((STUDY / f'results/front_end/softening/{front_end}_{site}_{case}.json').read_text())
    return result


def jobs():
    rows = []
    for solvent, site, scope, case, n in itertools.product(
            ('fresh', 'A', 'C'), ('MS2', 'BR2', 'MarsHill', 'Westmoreland'),
            ('extraction_only', 'full_process'), ('base', 'optimistic'), range(1, 6)):
        for end in ('low', 'high') if solvent == 'C' else ('fitted',):
            identity = f'{solvent}_{site}_{scope}_{case}_n{n}_{end}'
            rows.append(dict(id=identity, solvent=solvent, site=site, scope=scope, case=case, stages=n, end=end))
    for basis, site in itertools.product(('Held', 'Figiel'), ('MS2', 'BR2', 'MarsHill', 'Westmoreland')):
        rows.append(dict(id=f'D2{basis}_{site}', solvent='D2EHDTPA-' + basis, site=site,
                         scope='extraction_only', case='base', stages=5, end='fitted'))
    for basis, site, case in itertools.product(('Held', 'Figiel'), ('MS2', 'BR2', 'MarsHill', 'Westmoreland'), ('base', 'optimistic')):
        rows.append(dict(id=f'D2{basis}_{site}_full_process_{case}', solvent='D2EHDTPA-' + basis,
                         site=site, scope='full_process', case=case, stages=5, end='fitted'))
    return rows


def build(job, fixed=None):
    from lithium_extraction.prototypes.produced_water_sx_prototype import brines, fresh, records, set_a, set_c, stage
    from analyses.prototypes.hbta_topo_ms2_prototype.scripts import countercurrent as cc

    solvent, site, scope = (job[k] for k in ('solvent', 'site', 'scope'))
    reference = assumptions(site, job['case'], job.get('front_end'))
    soft = reference['softening']
    washed = soft[job['case']]
    raw = brines.raw_feed(site)
    if scope == 'full_process':
        feed = set_c.washed_feed(washed)
        feed['raw_brine'] = raw
        feed['labels'] = [s for s in feed['labels'] if solvent == 'C' or '#158 set C' not in s]
        feed['labels'] += raw['labels'] + records.SOLVENTS[solvent]['labels']
    else:
        feed = records.extraction_feed(raw, solvent)
        feed['raw_brine'] = raw
        washed = {'feed_l': 1., 'aqueous_mol': {'Li': feed['model_feed']['Li'], 'Na': feed['model_feed']['Na']},
                  'pH': -math.log10(feed['model_feed']['H'])}
    if solvent == 'C':
        c = reference['fixed_design']['extractant_mol_l']
        if job['phase'] != 'fixed' and fixed and fixed.get('status') in ('grid', 'optimal'):
            c = fixed['fixed_design']['extractant_mol_l']
        constants = json.loads((ROOT / 'analyses/prototypes/hbta_topo_ms2_prototype/results/set_c_summary.json').read_text())['ends'][job['end']]['ln_k']
        design = dict(organic_to_aqueous_ratio=1., extractant_mol_l=c, saponification_ratio=.7, ln_k=constants)
        if fixed and 'stages' in fixed:
            direct = direct_from_result(fixed, job['stages'], front_end=job.get('front_end'))
        else:
            direct = cc.countercurrent(constants, reaction_set='C', stage_count=job['stages'], organic_to_aqueous=1.,
                extractant=c, saponification_fraction=.7, feed_li=washed['aqueous_mol']['Li'] / washed['feed_l'],
                feed_na=washed['aqueous_mol']['Na'] / washed['feed_l'], feed_acidity=cc.acidity_at_ph(washed['pH']))
        model = set_c.build_flowsheet(washed, stages=job['stages'], scenario=design, direct_cascade=direct, physical_feed=feed)
        if scope == 'extraction_only':
            model._physical_feed = feed
    else:
        owner = fresh if solvent == 'fresh' else set_a
        source = json.loads((STUDY / ('results/fresh_r1.json' if solvent == 'fresh' else 'results/set_a_r2.json')).read_text())
        design = dict(source['fixed_design'])
        if job['phase'] != 'fixed' and fixed and fixed.get('status') in ('grid', 'optimal'):
            design = dict(fixed['fixed_design'])
            if solvent == 'fresh':
                design.pop('saponification_ratio', None)
        if scope == 'full_process':
            v = washed['feed_l']
            nominal = dict(feed, model_feed={k: n / v for k, n in feed['model_feed'].items()})
            model = owner.build_flowsheet(nominal, stages=job['stages'], scenario=design)
            model._washed, model._physical_feed = washed, feed
        else:
            model = owner.build_flowsheet(feed, stages=job['stages'], scenario=design)
        profiles = []
        old_rows = fixed['stages'] if fixed and 'stages' in fixed else source['stages']
        for k in range(job['stages']):
            old = old_rows[round(k * (len(old_rows) - 1) / max(job['stages'] - 1, 1))]
            profile = dict(old.get('outlet_mol_per_nominal_l', old.get('outlet_mol_per_raw_l')))
            if not fixed:
                original = source['stages'][0]['inlet_mol_per_raw_l']
                for s in model._chemistry.supports[0]:
                    if original.get(s, 0):
                        profile[s] *= model._feeds[0].get(s, 0) / original[s]
            profiles.append(profile)
        stage.initialize(model, profiles)
    model._case = job['case']
    return model, soft, reference


def direct_from_result(result, stage_count=None, *, front_end=None):
    from lithium_extraction.prototypes.produced_water_sx_prototype import set_c
    rows, guesses = [], []
    old = result['stages']
    n = len(old) if stage_count is None else stage_count
    profiles = None
    if front_end and n > len(old):
        # Interpolate phase-stream boundaries, preserving every stage's conserved quantities.
        aqueous = [old[0]['inlet_mol_per_nominal_l']] + [r['outlet_mol_per_nominal_l'] for r in old]
        organic = [r['outlet_mol_per_nominal_l'] for r in old] + [old[-1]['inlet_mol_per_nominal_l']]
        profiles = []
        for k in range(n):
            profile = {}
            for phase, streams, offset in zip(set_c.SUPPORTS, (aqueous, organic), (1, 0), strict=True):
                position = (k + offset) * len(old) / n
                left = min(int(position), len(old)-1)
                fraction = position-left
                profile.update({s: (1-fraction)*streams[left][s] + fraction*streams[left+1][s] for s in phase})
            profiles.append(profile)
    for k in range(n):
        row = old[round(k * (len(old) - 1) / max(n - 1, 1))]
        inlet = row['inlet_mol_per_nominal_l']
        if profiles is not None:
            inlet = {s: (aqueous[0] if k == 0 else profiles[k-1])[s] if s in set_c.SUPPORTS[0]
                     else (organic[-1] if k == n-1 else profiles[k+1])[s] for s in set_c.SPECIES}
        state = set_c.state_from_inlet(inlet, 1., result['fixed_design']['extractant_mol_l'])
        certificate = row['certificate']
        rows.append(dict(li_in_mol_l=state.li, na_in_mol_l=inlet['sodium-cation'],
            acidity_in_mol_l=inlet['hydrogen-cation'] - inlet.get('hydroxide-anion', 0.), bta_in_mol_l=state.bta,
            li_org_in_mol_l=state.li_org, na_org_in_mol_l=state.na_org, hbta_org_in_mol_l=state.hbta,
            water_autoprotolysis_active='RW' in certificate['active_reactions']))
        guesses.append(certificate['phase_amounts_mol'] if profiles is None else
                       [{s: profiles[k][s] for s in phase} for phase in set_c.SUPPORTS])
    return {'stages': rows, 'start': {'guesses': guesses}}


def d2_simulation(job, directory, remaining, brine=None, start=None):
    """`brine` replaces the site's raw feed; the site still sets the design's site basis; `start` replaces the
    extraction-only stage profiles (#245)."""
    from lithium_extraction.prototypes.produced_water_sx_prototype import brines, d2_full_process, d2_stage as d2, stage
    raw = brines.raw_feed(job['site'])
    source, old = d2.reference_record(job['solvent'])
    ratio = 1.25 if job['site'] == 'MarsHill' else 1.
    design = dict(B=1., S=1.2, O_A=ratio, Sr_decade=0., site_basis_li_mol_l=raw['mol']['Li'],
                  ln_K_Li_Mg_Ca=old['ln_K_Li_Mg_Ca'])
    model = d2.build_flowsheet(raw if brine is None else brine, solvent=job['solvent'], scope=job['scope'], scenario=design)
    if job['scope'] == 'full_process':
        retained = json.loads((RESULTS / 'fixed' / f'D2{job["solvent"].split("-")[-1]}_{job["site"]}.json').read_text())
        profiles = [row['outlet_mol_per_raw_l'] for row in retained['stages']]
    elif start is not None:
        profiles = start
    else:
        profiles = []
        reference_sites = d2.owner.BASIS['site_basis_li_mol_L'] * old['case']['S']
        total_sites = design['S'] * design['site_basis_li_mol_l']
        reference_feed = d2.reference_feed(job['solvent'])['model_feed']
        for oldrow in old['extraction']['stages']:
            profile = {s: sum(p.get(s, 0.) for p in oldrow['amounts_mol']) for s in model._chemistry.species}
            for s in model._chemistry.supports[0]:
                name = d2.records.SPECIES_LABELS[s]
                if reference_feed.get(name, 0.):
                    profile[s] *= model._feeds[0][s] / reference_feed[name]
            for s in model._chemistry.supports[1]:
                if s not in ('diluent', 'buphen'):
                    profile[s] *= total_sites / reference_sites
            profile['diluent'] *= ratio
            profile['buphen'] = ratio * design['B'] - 2 * profile[d2.rc.COMPLEX['Li']]
            profiles.append(profile)
    stage.initialize(model, profiles)
    stage.solve(model, directory, wall_seconds=remaining())
    result = d2.report(model, case=job['case'])
    result['sites_mol_l_organic'] = d2.owner.site_supply(1., 1.2, ratio, 5, site_basis_li_mol_l=raw['mol']['Li'])
    result['strip_li_recovery_fraction'] = result['accounts']['R_rec']
    result['product_recovery_fraction'] = None
    result['cost_usd_kg_li'] = result['accounts']['priced_subtotal_usd_kg_li']
    result['cost_usd_m3_raw_brine'] = result['accounts']['priced_subtotal_usd_m3']
    result['cost_basis'] = result['accounts']['li_denominator']
    result['calibration_coverage'] = [
        {'quantity': name, 'value': result['feed']['model_feed'][name], 'unit': 'mol/L aqueous',
         'basis': 'nominal raw aqueous model feed', 'comparison': 'physical calibration coverage unverified',
         'source_owner': d2.records.SOLVENTS[job['solvent']]['feed_owner']} for name in ('Li', 'Na')] + [
        {'quantity': 'process pH', 'value': None, 'unit': 'dimensionless',
         'comparison': 'H/pH not represented by existing D2 feed chemistry',
         'source_owner': d2.records.SOLVENTS[job['solvent']]['feed_owner']}] 
    result['na_metric'] = {'name': 'loaded-organic Na/(Li+Na)', 'value':
        result['outlet_mol_per_raw_l']['naa'] / (result['outlet_mol_per_raw_l']['naa'] + result['outlet_mol_per_raw_l'][d2.rc.COMPLEX['Li']])}
    result['outlet_li_loading_fractions'] = []
    if job['scope'] == 'full_process':
        stopping_point = 'construction'
        try:
            a = assumptions(job['site'], job['case'])['full_process']['assumptions']
            try:
                initial = d2_full_process.accounts(d2.strip_accounts(result['accounts'], job['solvent']), a, raw['mol']['Li'])
            except ValueError as error:
                if 'outside Ge' not in str(error) and 'no Li2CO3 product' not in str(error):
                    raise
                return result | {'downstream_admissible': False, 'reason': str(error)}
            d2_full_process.connect_units(model, initial, a)
            stopping_point = 'solve'
            stage.solve(model, directory / 'full_process', wall_seconds=remaining())
            stopping_point = 'reconciliation'
            downstream_result = result | d2.report(model, case=job['case'])
            return d2_full_process.report(model, downstream_result)
        except Exception as error:
            return result | {'reason': f'{type(error).__name__}: {error}',
                             'stopping_point': stopping_point, 'exception_traceback': traceback.format_exc()}
    return result


def fixed_disposition(result, solvent):
    """An uncertified state cannot prove a fixed design physically infeasible."""
    from lithium_extraction.prototypes.produced_water_sx_prototype import optimization as opt
    if not result['strict_certificate']:
        reasons = ['owner strict certificate does not hold']
        reasons += [r['certificate']['reason'] for r in result['stages']
                    if r['certificate'].get('reason')]
        return dict(status='unresolved', reason='; '.join(dict.fromkeys(reasons)))
    if result.get('downstream_admissible') is False:
        return dict(status='infeasible',
                    reason='fixed design: carbonation outside the measured solubility data or acid bounds')
    failures = opt.target_failures(result, solvent)
    numerical = any('certificate' in f or 'balance' in f or 'charge' in f or 'inconsistent' in f for f in failures)
    numerical |= any(not math.isfinite(n) for n in result['R7']['physical_raffinate'].values())
    if solvent == 'C' and not numerical and result['R7']['physical_raffinate']['H'] < 0:
        return dict(status='infeasible', reason='; '.join([
            'Fixed design is infeasible under the declared nominal-acidity composition mapping: reconstructed hydrogen is negative. '
            'Buffered-brine behavior and additional acidity conditioning were not evaluated.',
            *(f for f in failures if 'physical raffinate' not in f)]))
    return dict(status='grid' if not failures else 'unresolved' if numerical else 'infeasible',
                reason='fixed design grid result' if not failures else '; '.join(failures))


def worker(job, *, persist=True, wall_seconds=290):
    started = time.monotonic()
    directory = (RESULTS / 'diagnostics' if 'front_end' in job else ROOT / 'build/modular_campaign') / job['id']
    directory.mkdir(parents=True, exist_ok=True)
    from pyomo.common.tempfiles import TempfileManager
    from lithium_extraction.prototypes.produced_water_sx_prototype import flowsheet, full_process, optimization as opt, set_c, stage
    TempfileManager.tempdir = str(directory)
    result = dict(type='prototype', candidate=job, status='unresolved', engine=flowsheet.installed_engine(),
                  claim='model prediction; no physical validation', candidate_wall_limit_s=wall_seconds,
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (ROOT/'lithium_extraction/prototypes/produced_water_sx_prototype').glob('*.py')})
    def remaining():
        left = wall_seconds - (time.monotonic() - started)
        if left <= 0:
            raise TimeoutError(f'{wall_seconds} s candidate wall budget exhausted')
        return left
    try:
        if job['solvent'].startswith('D2EHDTPA'):
            result.update(d2_simulation(job, directory / 'solver', remaining))
            extraction_failed = not result['strict_certificate'] or bool(result['R7']['failures'])
            result['status'] = ('unresolved' if extraction_failed or result.get('exception_traceback') else
                               'infeasible' if result.get('downstream_admissible') is False else 'grid')
            if extraction_failed:
                result['reason'] = 'uncertified extraction; ' + '; '.join(result['R7']['failures'])
            result.setdefault('reason', 'fixed campaign simulation; conditional priced operating subtotal, no optimization')
        else:
            fixed = None if job['phase'] == 'fixed' else json.loads((RESULTS / 'fixed' / (job['parent_id'] + '.json')).read_text())
            if fixed is None and job['solvent'] == 'C':
                neighbours = [r for f in (RESULTS / 'fixed').glob('*.json')
                    if (r := json.loads(f.read_text())).get('stages') and all(
                        r['candidate'][k] == job[k] for k in ('solvent', 'site', 'scope', 'case', 'end'))
                    and (not job.get('front_end') or r['candidate']['stages'] < job['stages'])]
                fixed = min(neighbours, key=lambda r: abs(r['candidate']['stages']-job['stages']), default=None)
            initial = fixed
            if job.get('front_end'):
                result['initialization_source'] = ('fixed/' + fixed['candidate']['id'] + '.json'
                                                   if fixed else 'direct cascade')
            if job['phase'] != 'fixed':
                paths = [RESULTS / 'selected' / (job['id'] + '.json')]
                paths += sorted((RESULTS / 'selected').glob('*.json'))
                paths += sorted((RESULTS / 'feasibility').glob('*.json'))
                warm = [(p, r) for p in paths if p.exists()
                    and (r := json.loads(p.read_text())).get('status') in ('optimal', 'grid')
                    and (not job.get('front_end') or r['candidate']['stages'] < job['stages'])
                    and all(r['candidate'][k] == job[k] for k in ('solvent', 'site', 'scope', 'case', 'end'))]
                # Keep a feasible same-stage fixed profile before interpolating another stage count.
                if fixed['status'] == 'grid':
                    result['initialization_source'] = 'fixed/' + job['parent_id'] + '.json'
                    warm = [(p, r) for p, r in warm if r['candidate']['stages'] == job['stages']]
                if warm:
                    p, initial = min(warm, key=lambda item: (
                        abs(item[1]['candidate']['stages'] - job['stages']), item[1]['cost_usd_kg_li']))
                    result['initialization_source'] = str(p.relative_to(RESULTS))
            model, soft, reference = build(job, initial)
            opt.decisions(model, job['solvent'])
            if job['phase'] == 'fixed':
                if job['solvent'] == 'C':
                    # A support rebuild returns a new model; its decisions and costing join the solve.
                    model, extracted = set_c.solve(model, directory / 'extraction', started=started, wall_seconds=wall_seconds,
                        connect_units=lambda m: hasattr(m, '_decisions') or opt.decisions(m, job['solvent']))
                else:
                    stage.solve(model, directory / 'extraction', wall_seconds=remaining())
                if job['scope'] == 'full_process':
                    if job['solvent'] == 'C':
                        result.update(extracted)
                    else:
                        result.update(opt.nominal_to_raw_litre(stage.report(model), model._washed['feed_l']),
                                      fixed_design=dict(model._design))
                    extracted = full_process.extraction(model)
                    enumeration = full_process.enumerate_units(model._washed, extracted['loaded_organic'],
                        reference['full_process']['assumptions'],
                        organic_to_aqueous_ratio=model._design['organic_to_aqueous_ratio'])
                    result['unit_enumeration'] = enumeration
                    candidate = next((r for r in enumeration if r['feasible']), None)
                    if candidate is None:
                        result['downstream_admissible'] = False
                    else:
                        full_process.connect_units(model, job['case'], reference['full_process']['assumptions'], candidate)
                        stage.solve(model, directory / 'full', wall_seconds=remaining())
                        result.update(opt.report(model, job['scope'], soft))
                else:
                    result.update(opt.report(model, job['scope']))
                result.update(fixed_disposition(result, job['solvent']))
            else:
                candidate = job.get('unit')
                if job['scope'] == 'full_process' and candidate is None:
                    enumeration = full_process.enumerate_units(model._washed,
                        full_process.extraction(model)['loaded_organic'], reference['full_process']['assumptions'],
                        organic_to_aqueous_ratio=model._design['organic_to_aqueous_ratio'])
                    candidate = next((r for r in enumeration if r['feasible']),
                                     reference['unit_enumeration_selected'])
                provisional = job['phase'] in ('selected', 'feasibility') and job['scope'] == 'full_process'
                selections = []
                def connect(m):
                    if not hasattr(m, '_decisions'):
                        opt.decisions(m, job['solvent'])
                    if job['scope'] == 'full_process' and not hasattr(m, '_full_process'):
                        full_process.connect_units(m, job['case'], reference['full_process']['assumptions'], candidate,
                                                   initial_accounts=None if job.get('front_end') == 'sorbent' else reference['full_process'])
                    if not hasattr(m, 'minimum_cost'):
                        opt.add_optimization(m, job['scope'])
                        if job['phase'] == 'feasibility':
                            m.minimum_cost.deactivate()
                            m.feasibility.activate()
                    if provisional:
                        m.fs.carbonation.interval.deactivate()
                        m.fs.scrub.other_bound.deactivate()
                while True:
                    if job['solvent'] == 'C':
                        model, _ = set_c.solve(model, directory / 'released', started=started, wall_seconds=wall_seconds, connect_units=connect)
                    else:
                        connect(model)
                        stage.solve(model, directory / 'released', wall_seconds=remaining())
                    opt.synchronize(model)
                    if not provisional:
                        break
                    enumeration = full_process.enumerate_units(model._washed,
                        full_process.extraction(model)['loaded_organic'], reference['full_process']['assumptions'],
                        organic_to_aqueous_ratio=model._design['organic_to_aqueous_ratio'])
                    selected = next((r for r in enumeration if r['feasible']), None)
                    if selected is None:
                        raise RuntimeError('converged extraction has no admissible downstream interval/regime')
                    selections.append(selected)
                    consistent = all(selected[k] == candidate[k] for k in
                                     ('scrub_regime', 'carbonation_interval_mol_kg'))
                    candidate = selected
                    model._full_process['candidate'] = candidate
                    full_process.select_units(model, candidate)
                    if consistent:
                        break
                    remaining()
                result['unit_selections'] = selections
                result.update(opt.report(model, job['scope'], soft))
                failures = opt.target_failures(result, job['solvent'])
                if job['phase'] != 'feasibility' and fixed['status'] == 'grid' and result['cost_usd_kg_li'] > fixed['cost_usd_kg_li'] * (1 + 1e-6):
                    failures.append('optimum worse than accepted fixed design at identical conditions')
                result.update(status='optimal' if not failures else 'unresolved', reason='; '.join(failures) or 'IPOPT optimum; strict certificates, balances and version 1.0 KKT limits pass')
                if job['phase'] == 'feasibility' and not failures:
                    result.update(status='grid', reason='feasibility-only solve; all original bounds and targets retained; cost not minimized')
                result['fixed_design_cost_usd_kg_li'] = fixed.get('cost_usd_kg_li') if fixed['status'] == 'grid' else None
    except Exception as error:
        result.update(status='unresolved', reason=f'{type(error).__name__}: {error}', stopping_point=job['phase'])
        result['exception_traceback'] = traceback.format_exc()
        if 'model' in locals():
            from pyomo.environ import Constraint, value
            result['last_stage_amounts'] = [{s: value(n) for s,n in model.fs.stage[k].outlet.items()} for k in model.fs.stages]
            result['last_decision_values'] = {k: value(v) for k, v in model._decisions.items()}
            result['last_li_extraction_fraction'] = value(model.fs.design.li_extraction)
            if hasattr(model.fs, 'carbonation'):
                result['last_carbonation_nacl_mol_kg'] = value(model.fs.carbonation.nacl_mol_kg)
            residuals = {}
            for c in model.component_data_objects(Constraint, active=True):
                # Engine responses are evaluated only in IPOPT's fresh process, not accumulated here.
                if '.equilibrium[' in c.name:
                    continue
                body = value(c.body)
                residuals[c.name] = dict(lower=None if c.lower is None else body-value(c.lower),
                                        upper=None if c.upper is None else value(c.upper)-body)
            result['last_algebraic_margins'] = residuals
            logs = sorted(directory.rglob('ipopt.log'), key=lambda p: p.stat().st_mtime)
            if logs:
                logfile = logs[-1]
                result['solver_log'] = str(logfile.relative_to(ROOT))
                result['solver_log_sha256'] = hashlib.sha256(logfile.read_bytes()).hexdigest()
                result['solver_exit'] = logfile.read_text().rsplit('EXIT:', 1)[-1].strip()
    finally:
        result['elapsed_wall_s'] = time.monotonic() - started
        if result['elapsed_wall_s'] > wall_seconds:
            result.update(status='unresolved', reason=f'{wall_seconds} s candidate wall limit, including initialization and checks')
        if persist:
            save(RESULTS / job['phase'] / (job['id'] + '.json'), result)
    print(job['id'], result['status'], result['elapsed_wall_s'], result.get('reason', ''), flush=True)
    return result


def valid_result(path, job):
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if data['candidate']['id'] != job['id'] or data['status'] not in ('grid', 'optimal', 'infeasible', 'unresolved'):
            raise ValueError('candidate or disposition missing')
        if data['status'] in ('grid', 'optimal'):
            if not data['stages'] or not data['strict_certificate'] or data['R7']['failures'] or not math.isfinite(data['cost_usd_kg_li']):
                raise ValueError('incomplete numerical result')
        if not math.isfinite(data['elapsed_wall_s']) or not data.get('reason'):
            raise ValueError('timing or reason missing')
        return data
    except (ValueError, KeyError, TypeError):
        path.unlink()
        return None


def launch(job, *, rerun=False):
    output = RESULTS / job['phase'] / (job['id'] + '.json')
    retained = valid_result(output, job)
    if retained is not None and not rerun:
        return retained
    if retained is None and job['phase'] == 'selected':
        previous = [r for f in (RESULTS / 'released').glob(job['id'] + '*.json')
            if (r := valid_result(f, dict(job, id=f.stem))) is not None and r['status'] == 'optimal'
            and r['candidate'].get('parent_id', r['candidate']['id']) == job['id']]
        if previous:
            best = min(previous, key=lambda r: r['cost_usd_kg_li'])
            best = dict(best, candidate=job, reused_from='released/' + best['candidate']['id'] + '.json')
            save(output, best)
            return best
    retain_previous(job)
    return worker(job)


def retain_solver_output(job):
    logs = []
    for p in (ROOT / 'build/modular_campaign' / job['id']).rglob('ipopt.log'):
        data = p.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        destination = RESULTS / 'diagnostics/solver_output' / (digest + '.txt')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        logs.append(dict(source=str(p.relative_to(ROOT)), retained=str(destination.relative_to(RESULTS)),
                         sha256=digest))
    return logs


def retain_previous(job):
    retained = valid_result(RESULTS / job['phase'] / (job['id'] + '.json'), job)
    logs = retain_solver_output(job)
    if retained is not None:
        retained['retained_solver_output'] = logs
        save(RESULTS / 'diagnostics/previous' / job['phase'] / (job['id'] + '.json'), retained)


def checkpoint():
    subprocess.run(['git','add',str(RESULTS.relative_to(ROOT))],cwd=ROOT,check=True)
    changed = subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode
    if changed:
        subprocess.run(['git','commit','-m','Retain partial modular process campaign results'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)


def released_jobs(fixed_jobs):
    from lithium_extraction.prototypes.produced_water_sx_prototype import optimization as opt, brines
    rows = []
    for job in fixed_jobs:
        if job['solvent'].startswith('D2EHDTPA'):
            continue
        row = dict(job, phase='selected', parent_id=job['id'])
        reference = assumptions(job['site'], job['case'], job.get('front_end'))
        inlet_li = (reference['softening'][job['case']]['aqueous_mol']['Li'] /
                    reference['softening'][job['case']]['feed_l'] if job['scope'] == 'full_process'
                    else brines.raw_feed(job['site'])['mol']['Li'])
        target = opt.TARGETS[job['solvent']]
        capacity = .72 * opt.BOUNDS[job['solvent']][0][1] * opt.BOUNDS[job['solvent']][1][1]
        if target is not None and capacity < target * inlet_li:
            save(RESULTS / 'selected' / (row['id'] + '.json'), dict(type='prototype', candidate=row,
                status='infeasible', reason='Li target exceeds maximum loaded-organic capacity throughout decision bounds',
                capacity_mol_per_nominal_l=capacity, required_li_mol_per_nominal_l=target*inlet_li, elapsed_wall_s=0.))
        elif job['scope'] == 'full_process':
            a = reference['full_process']['assumptions']
            ratio_lo = opt.BOUNDS[job['solvent']][0][0]
            li_max = min(inlet_li, capacity) * (1-a['scrub_li_loss']) * reference['full_process']['strip_li_fraction']
            salt_min = a['concentrate_li_mol_kg'] * (a['acid_recovery'] +
                (1-a['acid_recovery']) * .25 * ratio_lo / li_max)
            if salt_min > 3:
                save(RESULTS / 'selected' / (row['id'] + '.json'), dict(type='prototype', candidate=row,
                    status='infeasible', reason='carbonation outside the measured solubility data throughout decision bounds',
                    minimum_mother_nacl_mol_kg=salt_min, elapsed_wall_s=0.))
        rows.append(row)
    return rows


def front_worker(job):
    global RESULTS
    RESULTS = STUDY / 'results/front_end/campaign' / job['front_end']
    return worker(job)


def main():
    global RESULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--retry-unresolved', action='store_true')
    parser.add_argument('--front-end', choices=('none', 'sorbent', 'nf'))
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=8)
    args = parser.parse_args()
    started = time.monotonic()
    fixed_jobs = [dict(row, phase='fixed') for row in jobs()]
    if args.front_end is not None:
        if args.front_end == 'nf':
            parser.error('Connected nanofiltration requires the data gate and separate parameter-set review')
        RESULTS = STUDY / 'results/front_end/campaign' / args.front_end
        fixed_jobs = [dict(j, id=args.front_end+'_'+j['id'], front_end=args.front_end)
                      for j in fixed_jobs if j['solvent'] == 'C' and j['scope'] == 'full_process']
        for phase in ('fixed', 'selected'):
            phase_jobs = fixed_jobs if phase == 'fixed' else released_jobs(fixed_jobs)
            # Finish the nearest lower-stage profiles before starting the next stage count.
            for count in sorted({j['stages'] for j in phase_jobs}):
                work = [j for j in phase_jobs if j['stages'] == count and
                        valid_result(RESULTS/j['phase']/(j['id']+'.json'), j) is None]
                with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as pool:
                    for result in pool.map(front_worker, work):
                        print(phase, result['candidate']['id'], result['status'], flush=True)
        return
    if args.limit:
        fixed_jobs = fixed_jobs[:args.limit]
    print(f'campaign PID {os.getpid()}; {len(fixed_jobs)} fixed designs, sequential', flush=True)
    count = 0
    for phase in ('fixed', 'selected'):
        work = fixed_jobs if phase == 'fixed' else released_jobs(fixed_jobs)
        print(phase, len(work), 'candidates', flush=True)
        for job in work:
            retained = valid_result(RESULTS / job['phase'] / (job['id'] + '.json'), job)
            result = launch(job, rerun=args.retry_unresolved and retained is not None and retained['status'] == 'unresolved')
            count += 1
            print(count, result['candidate']['id'], result['status'], flush=True)
            if count % 20 == 0:
                checkpoint()
        checkpoint()
    save(RESULTS / 'run_timing.json', dict(elapsed_wall_s=time.monotonic()-started,
         execution='sequential Python; ordinary IPOPT calls', base_commit='0c07cdc2',
         issue='https://github.com/tannerpolley/Lithium_Extraction/issues/201'))
    checkpoint()


if __name__ == '__main__':
    main()
