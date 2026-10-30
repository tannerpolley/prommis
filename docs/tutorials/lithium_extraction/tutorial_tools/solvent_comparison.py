"""Helpers for the solvent-comparison tutorial: the MS-2 feed, the acid-form HBTA/TOPO design, the four-solvent
runs, figures, and the check against the study's retained results.

Every function that changes the flowsheet takes the model `m` as its first argument.
"""

import csv
import json
import time
from functools import cache

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pyomo.environ import Block, Constraint, Var, value
from pyomo.util.calc_var_value import calculate_variable_from_constraint

from . import ROOT, log_directory, number, plot_checks
from analyses.prototypes.hbta_topo_ms2_prototype.scripts import countercurrent
from analyses.prototypes.softened_ms2_epcsaft_sx.scripts import optimization
from lithium_extraction.prototypes.hbta_topo_ms2_saponified import stage_chemistry
from lithium_extraction.prototypes.produced_water_sx_prototype import brines, des_stage, records, set_a, stage

STUDY = ROOT / 'analyses/prototypes/softened_ms2_epcsaft_sx'
ROUND4 = STUDY / 'runs/round4/results'
REPLAY = json.loads((STUDY / 'runs/engine_c358de72/replay.json').read_text())
AMOUNT_LIMIT, SCALED_LIMIT = REPLAY['limits']['amount_mol_l'], REPLAY['limits']['owner_scaled']
SATURATED = 'indistinguishable from unity at the declared comparison scale'
STAGES = 3
SYSTEMS = {  # the study's four systems: label and the design columns of its retained results table
    'acid_A': ('Acid HBTA/TOPO', ('organic_to_aqueous_ratio', 'extractant_mol_l', 'saponification_ratio')),
    'low:fitted': ('Sodium-saponified HBTA/TOPO', ('organic_to_aqueous_ratio', 'extractant_mol_l', 'saponification_ratio')),
    'D2Figiel': ('D2EHDTPA/BuPhen', ('O_A', 'B', 'S')),
    'DES-S': ('TBAC:DA/TOPO DES', ('total_N_mol', 'topo_wt_fraction')),
}
COLORS = dict(zip(SYSTEMS, ['#0072B2', '#D55E00', '#009E73', '#CC79A7']))  # the manuscript's colors and markers
MARKERS = dict(zip(SYSTEMS, ['o', 's', 'D', '^']))


# ---------------------------------------------------------------- inputs

def draw_cascade():
    """Schematic of the three-stage countercurrent cascade."""
    fig, ax = plt.subplots(figsize=(11, 2.0))
    ax.set(xlim=(-2.7, 8.6), ylim=(-.9, .9))
    ax.axis('off')
    for k in range(STAGES):
        ax.add_patch(FancyBboxPatch((2.4 * k, -.55), 1.6, 1.1, boxstyle='round,pad=0.05', fc='#eef3f8', ec='#33557a'))
        ax.text(2.4 * k + .8, 0, f'stage {k + 1}', ha='center', va='center', fontsize=11)
    arrow = dict(arrowstyle='-|>', mutation_scale=14, lw=1.6)
    for k in range(-1, STAGES):  # brine runs forward along the top, organic backward along the bottom
        ax.add_patch(FancyArrowPatch((2.4 * k + 1.65, .3), (2.4 * k + 2.35, .3), color='#0072B2', **arrow))
        ax.add_patch(FancyArrowPatch((2.4 * k + 2.35, -.3), (2.4 * k + 1.65, -.3), color='#D55E00', **arrow))
    ax.text(-.85, .3, 'softened MS–2 brine', ha='right', va='center', color='#0072B2', fontsize=9)
    ax.text(7.25, .3, 'raffinate', ha='left', va='center', color='#0072B2', fontsize=9)
    ax.text(7.25, -.3, 'fresh organic', ha='left', va='center', color='#D55E00', fontsize=9)
    ax.text(-.85, -.3, 'loaded organic', ha='right', va='center', color='#D55E00', fontsize=9)
    plt.show()


def feed_table():
    """The synthetic softened MS-2 feed per litre, and the model feed each solvent family receives."""
    raw = brines.raw_feed('MS2')
    feeds = {'HBTA/TOPO and DES': records.extraction_feed(raw, 'A', synthetic_softened=True),
             'D2EHDTPA/BuPhen': records.extraction_feed(raw, 'D2EHDTPA-Figiel', synthetic_softened=True)}
    physical = feeds['HBTA/TOPO and DES']['physical_feed']
    table = pd.DataFrame({'mol per L feed': physical})
    table['mg/L'] = [physical[e] * brines.MOLAR_MASS[e] * 1000 if e in brines.MOLAR_MASS else None for e in table.index]
    for label, feed in feeds.items():
        table[f'model feed, {label} (mol/L)'] = pd.Series(feed['model_feed'])
    return table


def _rows(name):
    with (ROUND4 / name).open() as stream:
        return list(csv.DictReader(stream))


@cache
def retained():
    """Each system's retained three-stage maximum-recovery row: (optimum_quantities row, pmax_comparison row).

    The row whose recovery equals the maximum in pmax_comparison.csv; for a tie, the lowest optimizer start.
    """
    pmax = {r['arm']: r for r in _rows('pmax_comparison.csv')
            if r['comparison_set'] == 'Paper 1 row' and int(r['N']) == STAGES}
    rows = _rows('optimum_quantities.csv')
    return {arm: (min((r for r in rows if r['objective'] == 'P-max' and r['arm'] == arm and int(r['N']) == STAGES
                       and float(r['recovery_raw']) == float(pmax[arm]['recovery_raw'])), key=lambda r: int(r['start'])),
                  pmax[arm]) for arm in SYSTEMS}


def design(arm):
    """(optimizer start, retained controls) of one system."""
    row = retained()[arm][0]
    return int(row['start']), {k: float(row[k]) for k in SYSTEMS[arm][1]}


DESIGN_NAMES = {'organic_to_aqueous_ratio': 'O/A', 'O_A': 'O/A',
                'extractant_mol_l': 'c, HBTA and TOPO each (mol/L organic)',
                'saponification_ratio': 'r_Na, sodium-form fraction of HBTA', 'B': 'B, BuPhen (mol/L organic)',
                'S': 'S, sodium-form sites per mol feed Li', 'total_N_mol': 'ammonia nitrogen (mol per L feed)',
                'topo_wt_fraction': 'TOPO mass fraction of the organic'}


def designs():
    """The retained three-stage design of each system, one row per design variable."""
    return pd.DataFrame([{'system': SYSTEMS[a][0], 'design variable': DESIGN_NAMES[k], 'value': v}
                         for a in SYSTEMS for k, v in design(a)[1].items()]).set_index(['system', 'design variable'])


# ---------------------------------------------------------------- the acid-form HBTA/TOPO flowsheet built in the notebook

def acid_hbta_topo():
    """(chemistry, organic reference, brine feed by species) for acid-form HBTA/TOPO on softened MS-2.

    The chemistry is the compiled Engine record of reaction set A, anchored at the feed composition.
    """
    feed = records.extraction_feed(brines.raw_feed('MS2'), 'A', synthetic_softened=True)['model_feed']
    chemistry, reference = set_a.chemistry(stage_chemistry.anchor_state(feed['Li'], feed['Na'], feed['H']))
    return chemistry, reference, {s: feed[records.SPECIES_LABELS[s]] for s in chemistry.supports[0]}


def add_design(m, design, reference):
    """The design variables, held fixed inside the study's bounds, and the fresh-organic feed written in terms of them.

    Per litre of brine: free HBTA and free TOPO each O/A*c*(1 - r_Na), sodium-form HBTA O/A*c*r_Na, no lithium complex,
    and diluent filling O/A litres of organic. As in the study, the lithium complex leaving any stage may not exceed
    72 % of the extractant circulated, O/A*c.
    """
    m.fs.design = Block()
    d = m.fs.design
    bounds = {'organic_to_aqueous_ratio': (.5, 3.), 'extractant_mol_l': (.05, .30), 'saponification_ratio': (0., 0.)}
    for name, limits in bounds.items():  # the acid-form bounds of the study's decisions.attach
        d.add_component(name, Var(initialize=design[name], bounds=limits))
        getattr(d, name).fix()
    ratio, c, r_na = d.organic_to_aqueous_ratio, d.extractant_mol_l, d.saponification_ratio
    organic_in = m.fs.boundary
    d.organic_feed = Constraint(['diluent', 'hbta', 'topo', 'sodium-carrier'], rule=lambda b, s: organic_in[s] == {
        'diluent': ratio * reference.dodecane_mol_l_organic, 'hbta': ratio * c * (1 - r_na),
        'topo': ratio * c * (1 - r_na), 'sodium-carrier': ratio * c * r_na}[s])
    organic_in['lithium-carrier'].fix(0.)
    d.loading = Constraint(m.fs.stages, rule=lambda b, k: m.fs.stage[k].outlet['lithium-carrier'] <= .72 * ratio * c)
    return pd.Series({DESIGN_NAMES[name]: value(getattr(d, name)) for name in design}, name='fixed value').to_frame()


def initialize(m, brine):
    """Organic feed from the design, then the stage-by-stage countercurrent solution loaded into every stage.

    The extents and generation terms follow from the balances, so every equation except equilibrium starts satisfied.
    """
    for s, link in m.fs.design.organic_feed.items():
        calculate_variable_from_constraint(m.fs.boundary[s], link)
    ln_k = json.loads((ROOT / 'analyses/shared/hbta_topo_configured_reactive_lle/results/mechanism_candidates.json')
                      .read_text())['summary']['A']['ln_k_298']
    d = m.fs.design
    start = countercurrent.countercurrent(
        ln_k, reaction_set='A', stage_count=len(m.fs.stages), organic_to_aqueous=value(d.organic_to_aqueous_ratio),
        extractant=value(d.extractant_mol_l), saponification_fraction=value(d.saponification_ratio),
        feed_li=brine['lithium-cation'], feed_na=brine['sodium-cation'], feed_acidity=brine['hydrogen-cation'], start=None)
    species = m._chemistry.species
    stage.initialize(m, [{s: sum(p.get(s, 0.) for p in phases) for s in species} for phases in start['start']['guesses']])


def solve(m, name):
    """IPOPT with exact second derivatives and tight tolerances; its optimality conditions are checked from the log."""
    return stage.solve(m, log_directory(name))


def optimum_table(m, brine, starting):
    """The design and lithium recovery from the starting design, the optimizer, and the study's retained optimum."""
    d, row = m.fs.design, retained()['acid_A'][0]
    names = {'organic_to_aqueous_ratio': 'O/A', 'extractant_mol_l': 'c (mol/L organic)'}
    optimized = {names[k]: value(getattr(d, k)) for k in names}
    optimized['$E_{Li}$'] = 1 - value(m.fs.stage[len(m.fs.stages)].outlet['lithium-cation']) / brine['lithium-cation']
    table = pd.DataFrame({'starting design': starting, 'optimized here': optimized,
                          'retained optimum': {**{names[k]: float(row[k]) for k in names},
                                               '$E_{Li}$': float(row['recovery_raw'])}})
    table['relative difference'] = (table['optimized here'] / table['retained optimum'] - 1).abs()
    return table


def matches_repository_build(m, solved):
    """Largest difference (mol/L) between the notebook's stage outlets and the repository's own acid-form build."""
    repository = [outlet(row) for row in solved['acid_A']['result']['stages']]
    difference = max(abs(m.fs.stage[k].outlet[s].value - repository[k - 1][s]) for k in m.fs.stages for s in m._chemistry.species)
    assert difference <= AMOUNT_LIMIT, difference
    return difference


# ---------------------------------------------------------------- results

def outlet(row):
    return row.get('outlet_mol_per_nominal_l', row.get('outlet_mol_per_raw_l'))


def solve_all():
    """Build and solve every system at its retained controls; a failed design is reported, not raised."""
    solved, status = {}, []
    directory = log_directory('solvent_comparison')
    for arm, (label, _) in SYSTEMS.items():
        started, line = time.monotonic(), {'system': label}
        try:
            start, controls = design(arm)
            solved[arm] = _solve_design(arm, STAGES, start, controls, directory / arm.replace(':', '_'))
            result = solved[arm]['result']
            balance = max(map(abs, result['balance_residuals_mol_per_raw_l'].values()))
            line.update({'solved': True, 'largest balance error (mol/L)': balance,
                         'largest stage re-solve difference (mol/L)': max(
                             r['certificate']['outlet_max_difference_mol_per_raw_l'] for r in result['stages'])})
        except Exception as error:
            line.update(solved=False, failure=repr(error))
        status.append(line | {'wall time (s)': round(time.monotonic() - started, 1)})
    return solved, pd.DataFrame(status).set_index('system')


def _solve_design(arm, stages, start, controls, directory):
    """Build and solve one system at fixed design with the study's builder; raise unless every check passes."""
    model = optimization.build(arm, stages, start, fixed=controls)
    model, result = optimization.solve(model, arm, directory, None, False)
    failures = optimization.physical_check(model, result, arm)['failures']
    balance = max(map(abs, result['balance_residuals_mol_per_raw_l'].values()))
    assert result['strict_certificate'] and not failures and balance <= AMOUNT_LIMIT, (failures, balance)
    return dict(model=model, result=result)


def _quantities(arm, entry):
    """Reported quantities, as the study's driver (optimization.run) records them; per litre of feed."""
    model, result = entry['model'], entry['result']
    if arm.startswith('DES'):
        return des_stage.metrics(model, result)
    ratio, concentration, preparation = (value(v) for v in model._decisions.values())
    d2 = arm.startswith('D2')
    li_complex, na_complex = ('lib2a', 'naa') if d2 else ('lithium-carrier', 'sodium-carrier')
    out, first, inlet = outlet(result), outlet(result['stages'][0]), model._feeds[1]
    loaded_li, loaded_na = out[li_complex], out[na_complex]
    naoh = preparation * model._design['site_basis_li_mol_l'] if d2 else ratio * concentration * preparation
    return dict(
        extraction_fraction=1 - out['lithium-cation'] / model._feeds[0]['lithium-cation'],
        stage1_li_na_selectivity=(first[li_complex] / first['lithium-cation']) / (first[na_complex] / first['sodium-cation']),
        loaded_organic_na_fraction=loaded_na / (loaded_li + loaded_na), net_na_to_organic_mol=loaded_na - inlet[na_complex],
        extractant_circulation_mol_m3=1000 * (preparation * model._design['site_basis_li_mol_l'] if d2 else ratio * concentration),
        initial_naoh_mol_m3=1000 * naoh, formal_naoh_mol_m3=1000 * naoh, formal_hcl_mol_m3=1000 * (loaded_li + loaded_na),
        model_bta_loss_mol_m3=1000 * out.get('bta-anion', 0),
        buphen_circulation_mol_m3=1000 * ratio * concentration if d2 else None,
        model_buphen_dissolution_mol_m3=1000 * result['stages'][-1]['outlet_mol_per_raw_l']['buphen'] / ratio
            / optimization.d2_stage.owner.BUPHEN_ORGANIC_AQUEOUS_PARTITION if d2 else None,
        loading=loaded_li / (inlet['naa'] if d2 else ratio * concentration))


def profiles(solved):
    """Aqueous Li, Li/Na selectivity and organic Na fraction leaving each stage (DES organic counted ÷ 3)."""
    tables = {}
    for arm, entry in solved.items():
        li_org, na_org, scale = {'D2Figiel': ('lib2a', 'naa', 1), 'DES-S': ('lithium-des', 'sodium-des', 3)}.get(
            arm, ('lithium-carrier', 'sodium-carrier', 1))
        rows = []
        for row in entry['result']['stages']:
            out = outlet(row)
            li, na = out[li_org] / scale, out[na_org] / scale
            resolved = min(out['lithium-cation'], li) > AMOUNT_LIMIT  # a ratio below the comparison amount is blank
            rows.append({'stage': row['stage'], 'aqueous Li (mol/L)': out['lithium-cation'],
                         'Li/Na selectivity': (li / out['lithium-cation']) / (na / out['sodium-cation']) if resolved else np.nan,
                         'organic Na fraction': na / (li + na)})
        tables[SYSTEMS[arm][0]] = pd.DataFrame(rows).set_index('stage')
    return pd.concat(tables, names=['system'])


def plot_profiles(table):
    """The three stage profiles side by side; amounts below the comparison amount sit on the dashed line, open."""
    shift = dict(zip(SYSTEMS, (-.09, -.03, .03, .09)))  # coincident points stay visible
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), layout='constrained')
    for arm, (label, _) in SYSTEMS.items():
        if label not in table.index.get_level_values('system'):
            continue
        p = table.loc[label]
        x, below = p.index + shift[arm], p['aqueous Li (mol/L)'] <= AMOUNT_LIMIT
        style = dict(linestyle='none', marker=MARKERS[arm], color=COLORS[arm])
        axes[0].plot(x, p['aqueous Li (mol/L)'].mask(below), markersize=7, label=label, **style)
        axes[0].plot(x[below], [AMOUNT_LIMIT] * below.sum(), markersize=9, fillstyle='none', markeredgewidth=1.5, **style)
        axes[1].plot(x, p['Li/Na selectivity'], markersize=7, **style)
        axes[2].plot(x, p['organic Na fraction'], markersize=7, **style)
    axes[0].axhline(AMOUNT_LIMIT, color='0.4', linestyle='--', linewidth=1)
    axes[0].annotate('declared comparison amount', (3.1, AMOUNT_LIMIT), xytext=(0, 4), textcoords='offset points',
                     ha='right', va='bottom', fontsize=8, color='0.3')
    axes[0].set(yscale='log', ylabel='Aqueous Li leaving stage (mol/L feed)')
    axes[1].set(yscale='log', ylabel='Li/Na selectivity, $D_{Li}/D_{Na}$')
    axes[2].set(ylim=(0, 1.05), ylabel='Na / (Li + Na) in organic leaving stage')
    for ax in axes:
        ax.set(xlabel='Stage (brine enters stage 1)', xticks=[1, 2, 3], xlim=(0.6, 3.4))
    fig.legend(loc='outside lower center', ncols=4, frameon=False)
    fig.suptitle('Three-stage countercurrent profiles at the retained maximum-recovery designs (model predictions)',
                 fontsize=10)
    plt.show()


# ---------------------------------------------------------------- check against the retained results

TEXT = {'objective', 'evidence', 'source', 'arm', 'N', 'target', 'start', 'loading_basis', 'loss_assumption',
        'solvent', 'variant', 'comparison_set', 'meaning', 'objective_group_meaning', 'historical_diagnoses',
        'accepted_starts', 'attempted_starts', 'relative_objective_span', 'distinct_objectives'}


def _retained_columns(arm, entry):
    """Reproduced values under the retained results' column names (mapping restated from des_present.round4)."""
    q, model, result = _quantities(arm, entry), entry['model'], entry['result']
    des, d2 = arm.startswith('DES'), arm.startswith('D2')
    names = (('E_Li', 'stage1_D_Li_over_D_Na', 'loaded_organic_Na_over_Li_Na') if des else
             ('extraction_fraction', 'stage1_li_na_selectivity', 'loaded_organic_na_fraction'))
    row = dict(zip(('recovery_raw', 'selectivity', 'loaded_na_fraction'), (q[n] for n in names)))
    row['raffinate_li_mol_L'] = outlet(result)['lithium-cation']
    saturated = 0 <= row['raffinate_li_mol_L'] <= AMOUNT_LIMIT
    row['recovery_class'] = SATURATED if saturated else 'resolved below unity'
    row['recovery_pct'] = 100. if saturated else 100 * row['recovery_raw']
    row['recovery_display'] = SATURATED if saturated else row['recovery_raw']
    row.update({k: value(v) for k, v in model._decisions.items()})
    transfers = {('net_Na_transfer_mol' if des else 'net_na_to_organic_mol'): 'net_na_mol_m3'}
    for field, v in q.items():
        if field.endswith('_mol_m3') and v is not None and not (d2 and field == 'model_bta_loss_mol_m3'):
            row[field] = v
        if des and field.endswith(('_mol', '_kg')) and not isinstance(v, dict):
            transfers.setdefault(field, field + '_per_m3')
    row.update({column: 1000 * q[field] for field, column in transfers.items()})
    if des:
        row.update({f'{field}_mol_m3': 1000 * v for field, v in q['modelled_losses_mol'].items()})
        row['initial_volumes_L'] = q['initial_model_volumes_L']
        row['descriptive_volume_ratio'] = q['initial_model_volumes_L'][1] / q['initial_model_volumes_L'][0]
        row['fixed_mass_ratio'] = q['organic_mass_kg'] / q['conditioned_aqueous_mass_kg']
        row['Li_loading_per_F'] = q['net_Li_transfer_mol'] / q['organic_circulation_F_mol']
        row['total_metal_loading_per_F'] = (q['net_Li_transfer_mol'] + q['net_Na_transfer_mol']) / q['organic_circulation_F_mol']
    else:
        row['loading'] = q['loading']
        if not d2:
            row['topo_circulation_mol_m3'] = q['extractant_circulation_mol_m3']
    return row


def _compare(reference, reproduced, table):
    lines = []
    for column, text in reference.items():
        if column in TEXT or text == '':
            continue
        if column not in reproduced:
            lines.append(dict(table=table, field=column, retained=text, passed=False, note='not reproduced'))
            continue
        new = reproduced[column]
        if isinstance(new, str) or text == SATURATED:
            lines.append(dict(table=table, field=column, retained=text, reproduced=new, limit='equal text',
                              passed=str(new) == text))
            continue
        olds, news = (json.loads(text), new) if isinstance(new, list) else ([float(text)], [new])
        for old, new_value in zip(olds, news, strict=True):
            amount = column == 'raffinate_li_mol_L'
            difference = abs(old - new_value) / (1 if amount else max(abs(old), abs(new_value), 1e-6))
            limit = AMOUNT_LIMIT if amount else SCALED_LIMIT
            lines.append(dict(table=table, field=column, retained=old, reproduced=new_value, difference=difference,
                              limit=limit, passed=difference <= limit))
    for line in lines:
        line.setdefault('kind', 'reported amount' if line.get('limit') == AMOUNT_LIMIT else 'reported quantity')
    return lines


def _stage_amounts(arm, entry):
    """Every stage outlet against the retained record's optimizer solve ('result') and fixed-control replay."""
    row = retained()[arm][0]
    record = json.loads((ROOT / row['source']).read_text())
    assert record['design'] == {k: float(row[k]) for k in record['design']}
    divisor = dict.fromkeys(('des', *des_stage.d.ORG[2:]), 3) if arm.startswith('DES') else {}  # DES particles ÷ 3
    lines = []
    for name in ('result', 'fixed_replay'):
        differences = [abs(outlet(new)[s] - outlet(old)[s]) / divisor.get(s, 1)
                       for new, old in zip(entry['result']['stages'], record[name]['stages'], strict=True)
                       for s in outlet(old)]
        lines.append(dict(table=f"{row['source'].rsplit('/', 1)[-1]} [{name}]", kind=f'stage outlets ({name.replace("_", " ")})',
                          field=f'{len(differences)} stage outlet amounts', difference=max(differences),
                          limit=AMOUNT_LIMIT, passed=max(differences) <= AMOUNT_LIMIT))
    return lines


HEADLINE = {'recovery_raw': r'$E_{Li}$', 'selectivity': r'$\beta = D_{Li}/D_{Na}$',
            'loaded_na_fraction': r'$x_{Na,org}$', 'net_na_mol_m3': r'$\Delta n_{Na,org}$ (mol/m$^3$)',
            'raffinate_li_mol_L': r'$n_{Li,aq,N}$ (mol/L)'}


def headline(solved):
    """The study's headline quantities, reproduced for each system (LaTeX row names).

    A raffinate below the comparison amount is not resolved, so its recovery is shown as indistinguishable from 1.
    """
    rows = {}
    for arm, entry in solved.items():
        row = _retained_columns(arm, entry)
        cells = {HEADLINE[k]: number(row[k]) for k in HEADLINE}
        if row['raffinate_li_mol_L'] <= AMOUNT_LIMIT:
            cells[HEADLINE['recovery_raw']] = r'$\approx 1$'
            cells[HEADLINE['raffinate_li_mol_L']] = rf'$< 10^{{{np.log10(AMOUNT_LIMIT):.0f}}}$'
        rows[SYSTEMS[arm][0]] = cells
    return pd.DataFrame(rows)


def check_retained(solved):
    """Every retained field and stage outlet against the reproduction; stops on any failure.

    Draws every comparison against its tolerance and returns (summary per system, every comparison).
    """
    lines = []
    for arm, (label, _) in SYSTEMS.items():
        if arm not in solved:
            lines.append(dict(system=label, table='not reproduced', field='', passed=False))
            continue
        quantities, pmax = retained()[arm]
        reproduced = _retained_columns(arm, solved[arm])
        for line in (_compare(quantities, reproduced, 'optimum_quantities.csv') +
                     _compare(pmax, reproduced, 'pmax_comparison.csv') + _stage_amounts(arm, solved[arm])):
            lines.append(dict(system=label, **line))
    check = pd.DataFrame(lines)
    check.loc[check['limit'].eq('equal text'), 'kind'] = 'text field (equal)'
    failed = check[~check['passed']]
    assert failed.empty, f'reproduced values differ from the retained results:\n{failed}'
    summary = plot_checks(check, 'system', f'{len(check)} comparisons with the retained results')
    return summary, check.dropna(axis='columns', how='all')


# ---------------------------------------------------------------- exploring designs

BOUNDS = {  # the study's design bounds (decisions.attach and des_stage.build)
    'acid_A': {'organic_to_aqueous_ratio': (.5, 3.), 'extractant_mol_l': (.05, .30)},
    'low:fitted': {'organic_to_aqueous_ratio': (.5, 3.), 'extractant_mol_l': (.1, .4), 'saponification_ratio': (.5, 1.)},
    'D2Figiel': {'O_A': (.5, 2.), 'B': (.25, 1.), 'S': (.9, 1.5)},
    'DES-S': {'total_N_mol': (8.71e-5, 5.89e-3), 'topo_wt_fraction': (.2, .4)}}
SHORT_NAMES = {'organic_to_aqueous_ratio': 'O/A', 'extractant_mol_l': 'c', 'saponification_ratio': 'r_Na', 'O_A': 'O/A',
               'B': 'B', 'S': 'S', 'total_N_mol': 'N (mol/L)', 'topo_wt_fraction': 'w_TOPO'}
tried = []  # every design solved in the explorer, in order


class DesignOptions(widgets.VBox):
    """A solvent system, the number of stages and its design variables within the study's bounds.

    The sliders start at the system's retained three-stage maximum-recovery design.
    """

    def __init__(self):
        self.system = widgets.Dropdown(options=[(label, arm) for arm, (label, _) in SYSTEMS.items()],
                                       description='solvent system', style={'description_width': '200px'},
                                       layout=widgets.Layout(width='560px'))
        self.stages = widgets.IntSlider(value=STAGES, min=1, max=5, description='extraction stages',
                                        style={'description_width': '200px'}, layout=widgets.Layout(width='560px'))
        self.sliders = widgets.VBox()
        self.system.observe(lambda _: self._design_sliders(), names='value')
        self._design_sliders()
        super().__init__([self.system, self.stages, self.sliders])

    def _design_sliders(self):
        arm = self.system.value
        retained_design, sliders = design(arm)[1], []
        for name, (low, high) in BOUNDS[arm].items():
            kind = widgets.FloatLogSlider if name == 'total_N_mol' else widgets.FloatSlider
            limits = dict(min=np.log10(low), max=np.log10(high), step=.01) if name == 'total_N_mol' else dict(
                min=low, max=high, step=(high - low) / 100)
            slider = kind(value=min(max(retained_design[name], low), high), description=DESIGN_NAMES[name],
                          readout_format='.3g', continuous_update=False, style={'description_width': '200px'},
                          layout=widgets.Layout(width='560px'), **limits)
            slider.name = name
            sliders.append(slider)
        self.sliders.children = sliders

    @property
    def selection(self):
        return self.system.value, self.stages.value, {s.name: s.value for s in self.sliders.children}


def run_design(options):
    """Solve the selected system at the selected design with the study's builder, and add it to the comparison.

    Variables not on a slider (r_Na = 0 for the acid form) keep their retained values. A design that does not solve,
    or fails a check, is listed with the reason.
    """
    arm, stages, chosen = options.selection
    start, controls = design(arm)
    controls = controls | chosen
    row = {'system': SYSTEMS[arm][0], 'stages': stages,
           'design': ', '.join(f'{SHORT_NAMES[k]} = {number(v)}' for k, v in chosen.items())}
    try:
        entry = _solve_design(arm, stages, start, controls, log_directory('design') / arm.replace(':', '_'))
        quantities = _retained_columns(arm, entry)
        row |= {HEADLINE[k]: quantities[k] for k in HEADLINE}
    except Exception as error:
        row['failure'] = f'{type(error).__name__}: {str(error)[:200]}'
    tried.append(row)
    return pd.DataFrame([row])


def compare_designs():
    """Every design solved so far: lithium recovery against the sodium share of the loaded organic, and the table."""
    table = pd.DataFrame(tried)
    done = table.dropna(subset=[HEADLINE['recovery_raw']]) if HEADLINE['recovery_raw'] in table else table.iloc[:0]
    if len(done):
        fig, ax = plt.subplots(figsize=(7, 4))
        for arm, (label, _) in SYSTEMS.items():
            rows = done[done['system'] == label]
            if rows.empty:
                continue
            ax.scatter(rows[HEADLINE['loaded_na_fraction']], rows[HEADLINE['recovery_raw']], s=60, marker=MARKERS[arm],
                       color=COLORS[arm], label=label)
            for _, r in rows.iterrows():
                ax.annotate(f"{r['stages']}", (r[HEADLINE['loaded_na_fraction']], r[HEADLINE['recovery_raw']]),
                            xytext=(5, 3), textcoords='offset points', fontsize=8)
        ax.set(xlabel='Na / (Li + Na) in the loaded organic, $x_{Na,org}$', ylabel='lithium recovery, $E_{Li}$',
               xlim=(0, 1.02), ylim=(0, 1.05), title='Designs solved so far (labels: number of stages)')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
        plt.tight_layout()
        plt.show()
    return table
