"""Helpers for the produced-water process tutorial: waters, softening, the extraction feed and design, downstream
connections, figures, the second water and the check against the retained campaign.

Every function that changes the flowsheet takes the model `m` as its first argument.
"""

import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Patch
from pyomo.environ import Constraint, Expression, NonNegativeReals, Var, units as pyunits, value
from pyomo.network import Arc

from . import ROOT, log_directory, number, plot_checks
from analyses.prototypes.hbta_topo_ms2_prototype.scripts import countercurrent as cc
from analyses.prototypes.modular_process.scripts.run_campaign import fixed_disposition
from analyses.shared.hbta_topo_configured_reactive_lle.scripts import mechanism_candidates as mc
from lithium_extraction.prototypes.produced_water_sx_prototype import (
    brines, costing, downstream, flowsheet, full_process, optimization, set_c, stage)

# IDAES warns that some cost flows are expressions without a zero lower bound; optimization.report checks every
# cost flow is nonnegative after the solve.
logging.getLogger('idaes.core.base.costing_base').setLevel(logging.ERROR)

STUDY = ROOT / 'analyses/prototypes/modular_process'
BAND = json.loads((STUDY / 'results/targets/exchange_band.json').read_text())
WATERS = {'MS2': 'MS-2', 'MarsHill': 'Mars Hill'}
CASE, FIT = 'base', 'low'  # base-case prices; 'low' is the single-stage exchange fit
ACCEPTED = {site: next(g[FIT] for g in BAND['groups'] if g['site'] == site and g['case'] == CASE) for site in WATERS}
LN_K = json.loads((ROOT / 'analyses/prototypes/hbta_topo_ms2_prototype/results/set_c_summary.json')
                  .read_text())['ends'][FIT]['ln_k']
LCE_PER_LI = BAND['lce_to_li_mass_ratio']  # kg Li2CO3 per kg Li
LI_G_MOL = 6.94  # g/mol, the costing module's lithium molar mass
SOFTENING_PYTHON = Path(os.environ.get('LITHIUM_SOFTENING_PYTHON', sys.executable))  # Reaktoro is in this environment
SAPONIFIED = .7  # fraction of HBTA in sodium form, fixed in the campaign


# ---------------------------------------------------------------- inputs

def waters():
    """Measured analysis and accepted design of each water."""
    rows = {}
    for site, name in WATERS.items():
        raw, a = brines.raw_feed(site), ACCEPTED[site]
        rows[name] = {**{f'{ion} (mg/L)': raw['printed_mg_l'][ion] for ion in ('Li', 'Na', 'Ca', 'Mg', 'Cl')},
                      '(Ca+Mg)/Li (kg/kg)': raw['ca_mg_li_mass_ratio'], 'accepted stages': a['stages'],
                      'HBTA (mol/L organic)': a['extractant_mol_l'], 'O/A': a['organic_to_aqueous_ratio'],
                      'saponified fraction': a['saponification_ratio']}
    return pd.DataFrame(rows).T


def soften(site):
    """Run the Reaktoro softening script as a subprocess, as the campaign's whole-process runner does."""
    output = log_directory(f'{site}_softening') / 'softened.json'
    subprocess.run([str(SOFTENING_PYTHON), '-m', 'analyses.prototypes.modular_process.scripts.run_softening',
                    '--brine', site, '--case', CASE, '--output-file', str(output)], check=True, timeout=290, cwd=ROOT,
                   stdout=subprocess.DEVNULL)  # its progress log; errors still reach stderr
    return json.loads(output.read_text())


def softening_table(softened):
    kg = flowsheet.MASS_KG_PER_MOL
    rows = {}
    for site, soft in softened.items():
        w = soft[CASE]
        rows[WATERS[site]] = {'NaOH (kg/m3 raw)': 1000 * w['reagent_mol']['NaOH'] * kg['NaOH'],
                              'soda ash (kg/m3 raw)': 1000 * w['reagent_mol']['Na2CO3'] * kg['Na2CO3'],
                              'HCl, 31% (kg/m3 raw)': 1000 * w['reagent_mol']['HCl'] * kg['HCl31'],
                              'wet sludge (kg/m3 raw)': 1000 * w['wet_sludge_kg'],
                              'softened water (L per L raw)': w['feed_l'], 'Ca+Mg left (mg/L)': w['ca_mg_mg_l'],
                              'Li kept in softened water (%)': 100 * w['aqueous_mol']['Li'] / w['raw_mol']['Li']}
    return pd.DataFrame(rows).T


def design(site):
    """The accepted design: O/A 1, 70% saponified, the accepted HBTA concentration and the single-stage fit."""
    return dict(organic_to_aqueous_ratio=1., extractant_mol_l=ACCEPTED[site]['extractant_mol_l'],
                saponification_ratio=SAPONIFIED, ln_k=LN_K)


def starting_profile(site, softened):
    """Stage-by-stage countercurrent solution with the Engine alone: the flowsheet's starting point."""
    w, a = softened[site][CASE], ACCEPTED[site]
    return cc.countercurrent(LN_K, reaction_set='C', stage_count=a['stages'], organic_to_aqueous=1.,
                             extractant=a['extractant_mol_l'], saponification_fraction=SAPONIFIED,
                             feed_li=w['aqueous_mol']['Li'] / w['feed_l'], feed_na=w['aqueous_mol']['Na'] / w['feed_l'],
                             feed_acidity=cc.acidity_at_ph(w['pH']))


def stage_amounts(start):
    """Amount of each species leaving each stage in the starting profile (both phases summed per species)."""
    return [{s: sum(p.get(s, 0.) for p in phases) for s in set_c.SPECIES} for phases in start['start']['guesses']]


def stage_chemistries(site, start):
    """One compiled Engine record per extraction stage, anchored at that stage's starting composition.

    Set C switches water autoprotolysis on or off stage by stage; stages with the same active reactions share a record.
    """
    c, records, chemistries = ACCEPTED[site]['extractant_mol_l'], {}, []
    for row, guess in zip(start['stages'], start['start']['guesses'], strict=True):
        anchor = cc.stage_state('generic set C anchor', organic_to_aqueous=1., extractant=c,
            li=row['li_in_mol_l'], na=row['na_in_mol_l'], acidity=row['acidity_in_mol_l'], bta=row['bta_in_mol_l'],
            li_org=row['li_org_in_mol_l'], na_org=row['na_org_in_mol_l'], hbta=row['hbta_org_in_mol_l'])
        active = row['water_autoprotolysis_active']
        if active not in records:
            records[active] = set_c._chemistry(anchor, mc.solve(anchor, 'C', LN_K, guess=guess), LN_K, 1., c)
        chemistries.append(records[active])
    return chemistries


def feed_extraction(m, site, washed):
    """Fix the softened water and the fresh organic (70% of HBTA in sodium form) entering the stages.

    Amounts are mol per litre of softened water at O/A = 1; returns them in mmol/L.
    """
    c, v = ACCEPTED[site]['extractant_mol_l'], washed['feed_l']
    boundary = cc.stage_state('generic set C boundary', organic_to_aqueous=1., extractant=c,
        li=washed['aqueous_mol']['Li'] / v, na=washed['aqueous_mol']['Na'] / v,
        acidity=washed.get('feed_acidity', cc.acidity_at_ph(washed['pH'])), na_org=SAPONIFIED * c)
    aqueous, organic = mc.inventory(boundary)
    organic['diluent'] = mc.model(set_c.SPECIES).state(cc.TEMPERATURE_K, P=mc.PRESSURE_PA, phase='liquid',
        x=tuple(float(s == 'diluent') for s in set_c.SPECIES)).molar_density * .001  # mol diluent per L organic
    for s in set_c.SPECIES:
        m.fs.boundary[s].fix(aqueous.get(s, 0.) + organic.get(s, 0.))
    m._feeds = (dict(aqueous), dict(organic))
    return pd.DataFrame({'softened water': aqueous, 'fresh organic': organic}).fillna(0.).mul(1000).rename_axis(
        'mmol per L of softened water')


def add_extractant_inventory(m, site):
    """TOPO in the Engine record follows the extractant (O/A x c), and no stage may leave negative free HBTA or TOPO."""
    fs = m.fs
    fs.parameters['topo_total_mol'].unfix()
    fs.topo_quantity = Constraint(expr=fs.parameters['topo_total_mol'] == design(site)['organic_to_aqueous_ratio'] *
                                  fs.parameters['extractant_mol_l'])

    def free_ligand_nonnegative(_, k, s):
        chemistry = m._chemistries[k - 1]
        j = chemistry.record_species.index(s)
        return chemistry.feed_offset[j] + sum(a * m._incoming(k, q) for a, q in
                                              zip(chemistry.feed_matrix[j], chemistry.control_species, strict=True)) >= 0
    fs.affine_capacity = Constraint(fs.stages, ('hbta', 'topo'), rule=free_ligand_nonnegative)


def add_design(m, site, washed):
    """The design variables (O/A, c, saponified fraction), held fixed, and the fresh organic written in terms of them."""
    m._design, m._washed, m._physical_feed = design(site), washed, set_c.washed_feed(washed)
    optimization.decisions(m, 'C')
    return pd.Series({name: value(v) for name, v in m._decisions.items()}, name='value').to_frame()


def check_extraction(m):
    """Re-solve every stage directly with the Engine and confirm it keeps the same active reactions."""
    result = set_c.report(m, stage.report(m))
    assert result['strict_certificate'], 'a stage outlet differs from its direct Engine solution'
    assert all(r['certificate']['support_matches'] for r in result['stages']), 'a stage changed its active reactions'
    return result


def assumptions(site):
    """Downstream assumptions of the retained whole-process record (scrub, strip, organic loss, carbonation)."""
    return json.loads((STUDY / f'results/full_process/{site}_{CASE}.json').read_text())['full_process']['assumptions']


def downstream_alternatives(m, site, softened):
    """Every scrub-acid regime and carbonation-salt interval, with the first admissible one marked."""
    rows = full_process.enumerate_units(softened[site][CASE], full_process.extraction(m)['loaded_organic'],
                                        assumptions(site), organic_to_aqueous_ratio=1.)
    candidate = next(r for r in rows if r['feasible'])
    table = pd.DataFrame([{'scrub acid set by': r['scrub_regime'],
                           'NaCl interval (mol/kg)': '{:g} to {:g}'.format(*r['carbonation_interval_mol_kg']),
                           'mark': 'selected' if r is candidate else 'admissible' if r['feasible'] else ''} for r in rows])
    return candidate, table.pivot(index='NaCl interval (mol/kg)', columns='scrub acid set by', values='mark')


def downstream_accounts(m, site, candidate):
    """Closed-form material and cost accounts of the downstream steps, used as their starting point and prices."""
    a = dict(assumptions(site), extractant_mol_l=m._design['extractant_mol_l'])
    m._full_process = {'case': CASE, 'assumptions': a, 'candidate': candidate}
    return a, flowsheet.costs_and_balances(m._washed, full_process.extraction(m), CASE, a)


METALS = {'Li': 'lithium', 'Na': 'sodium'}


def add_scrub_and_strip(m, a, candidate, accounts):
    """Scrub and strip: one PrOMMiS SolventExtraction stage each, in acid exchange with the loaded organic.

    Each transfers a fixed fraction of each entering metal complex to the acid. The scrub takes the loaded organic
    leaving stage 1 and its acid is the larger of the acid supplied and the acid the scrubbed metals consume; the
    strip acid is 2.5 mol/L on a strip volume one tenth of the organic.
    """
    fs, v, ratio = m.fs, m._washed['feed_l'], m._decisions['organic_to_aqueous_ratio']
    fractions = {'scrub': {'Li': a['scrub_li_loss'], 'Na': a['scrub_na_efficiency']},
                 'strip': {'Li': accounts['strip_li_fraction'], 'Na': 1.}}
    for name, fraction in fractions.items():
        downstream.fixed_fraction_strip(fs, name, {e: f'{n}-carrier' for e, n in METALS.items()},
            {e: f'{n}-cation' for e, n in METALS.items()}, dict.fromkeys(METALS, 1.),
            dict.fromkeys(METALS, {'hbta': 1., 'topo': 1.}), fraction, spectators=('diluent',))
    scrub, strip, loaded = fs.scrub, fs.strip, fs.stage[1].outlet
    scrub.loaded_organic = Constraint(fs.loaded_properties.component_list, rule=lambda b, s:
                                      scrub.unit.mscontactor.organic_inlet_state[0].flow_mol_comp[s] == v * loaded[s])
    scrub.waste = Expression(tuple(METALS), rule=lambda b, e: b.aqueous_out[f'{METALS[e]}-cation'])
    scrub.acid_mol = Var(domain=NonNegativeReals, initialize=accounts['scrub_acid_mol'])
    scrub.acid = Constraint(('hydrogen-cation', 'chloride-anion'), rule=lambda b, s:
                            scrub.unit.mscontactor.aqueous_inlet_state[0].flow_mol_comp[s] == b.acid_mol)
    source, demand = a['scrub_hcl_mol_l'] * ratio * v / 10, sum(scrub.waste.values())
    active, other = (source, demand) if candidate['scrub_regime'] == 'source' else (demand, source)
    scrub.active_bound = Constraint(expr=scrub.acid_mol == active)
    scrub.other_bound = Constraint(expr=scrub.acid_mol >= other)
    strip.scrubbed_organic = Arc(source=scrub.unit.organic_outlet, destination=strip.unit.organic_inlet)
    strip.gross_acid_mol = Expression(expr=2.5 * ratio * v / 10)
    strip.acid = Constraint(('hydrogen-cation', 'chloride-anion'), rule=lambda b, s:
                            strip.unit.mscontactor.aqueous_inlet_state[0].flow_mol_comp[s] == b.gross_acid_mol)
    strip.metals = Expression(tuple(METALS), rule=lambda b, e: b.aqueous_out[f'{METALS[e]}-cation'])
    strip.acid_sufficiency = Constraint(expr=strip.aqueous_out['hydrogen-cation'] >= 0)


def initialize_downstream(m, a, accounts):
    """Scrub, strip and liquor units solved in sequence from the extraction; the recycle loop from its closed form."""
    full_process._initialize_units(m.fs, accounts, a['mother_recycle'])


def priced_flows(m, accounts):
    """Each priced item: (kg per hour at 1 m3/h raw water, price), and the lithium in the Li2CO3 product (kg/h)."""
    fs, kg = m.fs, flowsheet.MASS_KG_PER_MOL
    consumed = {'saponification_NaOH': fs.saponification.naoh_mol, 'scrub_HCl31': fs.scrub.acid_mol,
                'strip_HCl31': fs.strip.fresh_acid_mol, 'neutralization_NaOH': fs.neutralization.naoh_mol,
                'carbonation_Na2CO3': fs.carbonation.carbonate_mol, 'HBTA': fs.organic_loss.hbta_mol,
                'TOPO': fs.organic_loss.topo_mol, 'diluent': fs.organic_loss.diluent_mol}  # mol per L raw water
    reagent = {k: next((r for r in ('NaOH', 'Na2CO3', 'HCl31') if r in k), k) for k in consumed}
    flows = {k: ((consumed[k] * kg[reagent[k]] if k in consumed else mass) * costing.RAW_L_H * pyunits.kg,
                 accounts['prices_usd_kg'][k] * costing.usd_per(pyunits.kg))
             for k, mass in accounts['mass_kg_per_l'].items()}
    li_per_li2co3 = accounts['li_product_kg_m3'] / 1000 / accounts['product_li2co3_mol']  # kg Li per mol Li2CO3
    return flows, fs.carbonation.product_mol * li_per_li2co3 * costing.RAW_L_H * pyunits.kg


def solve(m, name):
    """IPOPT with exact second derivatives and tight tolerances; its optimality conditions are checked from the log."""
    return stage.solve(m, log_directory(name))


def whole_process_result(m, site, softened):
    """The repository's independent report of the solved process, and its check of every process requirement."""
    result = optimization.report(m, 'full_process', softened[site])
    assert fixed_disposition(result, 'C')['status'] == 'grid', 'design does not meet every process requirement'
    return result


def _decisions(model):
    return hasattr(model, '_decisions') or optimization.decisions(model, 'C')


def run_water(site, softened):
    """Every step of the lead case for another water, with the repository's own flowsheet builders."""
    w, a = softened[site][CASE], ACCEPTED[site]
    model = set_c.build_flowsheet(w, stages=a['stages'], scenario=design(site),
                                  direct_cascade=starting_profile(site, softened))
    optimization.decisions(model, 'C')
    model, _ = set_c.solve(model, log_directory(f'{site}_extraction'), started=time.monotonic(),
                           connect_units=_decisions)
    candidate, _ = downstream_alternatives(model, site, softened)
    full_process.connect_units(model, CASE, assumptions(site), candidate)
    stage.solve(model, log_directory(f'{site}_full'))
    return model, whole_process_result(model, site, softened)


# ---------------------------------------------------------------- results

def headline(model, result, softened, site):
    costing = model.fs.costing
    raw_li = softened[site][CASE]['raw_mol']['Li']
    return pd.Series({'Li extraction (%)': 100 * result['li_extraction_fraction'],
                      'Li recovery to Li2CO3 (%)': 100 * 2 * value(model.fs.carbonation.product_mol) / raw_li,
                      'Li2CO3 product (kg per m3 raw water)': value(costing.li_kg_h) * LCE_PER_LI,
                      'operating cost (USD per m3 raw water)': value(costing.operating_cost),
                      'operating cost (USD per kg Li)': value(costing.usd_kg_li),
                      'break-even price (USD per kg LCE)': value(costing.usd_kg_li) / LCE_PER_LI},
                     name=WATERS[site])


def stage_table(result):
    """mmol per litre of softened water leaving each extraction stage."""
    species = ('lithium-cation', 'sodium-cation', 'lithium-carrier', 'sodium-carrier', 'hbta', 'bta-anion')
    return pd.DataFrame({row['stage']: {s: 1000 * row['outlet_mol_per_nominal_l'][s] for s in species}
                         for row in result['stages']}).T.rename_axis('stage').style.format(number)


def stream_tables(result):
    """Every physical stream in mmol per litre of raw water: (aqueous and reagent streams, organic streams)."""
    streams = result['physical_streams_mol_per_raw_l']
    organic = {k: n for k, n in streams.items() if 'organic' in k}
    columns = ['Li', 'Na', 'K', 'Ca', 'Mg', 'Sr', 'Cl', 'H', 'OH', 'C', 'BTA', 'water']
    aqueous = pd.DataFrame({k: n for k, n in streams.items() if k not in organic}).T.reindex(columns=columns).mul(1000)
    return (aqueous.dropna(axis=1, how='all').style.format(number, na_rep=''),
            pd.DataFrame(organic).T.mul(1000).style.format(number, na_rep=''))


def lithium_path(result):
    """(step, stream kept, mol Li kept, (loss stream, mol Li lost)) per litre of raw water."""
    soft, p = result['softening'][CASE], result['physical_streams_mol_per_raw_l']
    scrubbed = p['loaded organic']['lithium-carrier'] - p['scrub waste']['Li']
    return [('Raw water', 'raw water', soft['raw_mol']['Li'], None),
            ('Softening', 'softened water', p['washed extraction feed']['Li'],
             ('softening cake', soft['cake_mol']['Li'] + soft['solid_mol']['Li'])),
            (f"Extraction\n{len(result['stages'])} stages", 'loaded organic', p['loaded organic']['lithium-carrier'],
             ('raffinate', p['physical raffinate']['Li'])),
            ('Scrub', 'scrubbed organic', scrubbed, ('scrub waste', p['scrub waste']['Li'])),
            ('Strip', 'strip liquor', result['full_process']['stripped_mol']['Li'],
             ('left on organic', p['regeneration lithium discharge']['Li'])),
            ('Carbonation', 'product', p['Li2CO3 product']['Li'], ('mother-liquor purge', p['mother liquor purge']['Li']))]


def lithium_diagram(result, site):
    """Lithium from raw water to product, with every loss; returns the closure (mol per L raw water)."""
    steps = lithium_path(result)
    raw = steps[0][2]
    grams = lambda n: 1000 * n * LI_G_MOL  # mol per L raw water -> g per m3 raw water  # noqa: E731
    fig, ax = plt.subplots(figsize=(13, 3.4))
    ax.set(xlim=(-.6, len(steps) + .4), ylim=(-1., 1.15))
    ax.axis('off')
    for i, (name, stream, kept, lost) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((i - .42, .05), .84, .6, boxstyle='round,pad=0.02', fc='#e8eef6', ec='#33557a'))
        ax.text(i, .35, name, ha='center', va='center', fontsize=10)
        ax.annotate('', xy=(i + .58, .35), xytext=(i + .42, .35), arrowprops=dict(arrowstyle='-|>', color='#33557a'))
        ax.text(i + .5, .72, f'{stream}\n{grams(kept):.1f} g ({100 * kept / raw:.1f}%)', ha='center', va='bottom',
                fontsize=8.5)
        if lost:
            ax.annotate('', xy=(i, -.45), xytext=(i, .05), arrowprops=dict(arrowstyle='-|>', color='#a23b2a'))
            ax.text(i, -.55, f'{lost[0]}\n{grams(lost[1]):.2f} g ({100 * lost[1] / raw:.1f}%)', ha='center',
                    va='top', fontsize=9, color='#a23b2a')
    ax.text(len(steps), .35, 'Li$_2$CO$_3$\nproduct', ha='center', va='center', fontsize=10, weight='bold')
    ax.set_title(f'{WATERS[site]}, base case: lithium per m$^3$ of raw water (g Li, share of raw-water lithium)',
                 fontsize=11)
    plt.show()
    return steps[0][2] - steps[-1][2] - sum(s[3][1] for s in steps if s[3])


GROUPS = {'Softening': {'softening_Na2CO3': 'soda ash (softening)', 'softening_NaOH': 'sodium hydroxide (softening)',
                        'softening_HCl31': 'hydrochloric acid (pH reset)', 'sludge': 'wet-sludge disposal'},
          'Extractant and diluent replacement': {'HBTA': 'HBTA', 'TOPO': 'TOPO', 'diluent': 'diluent'},
          'Extraction, scrub and strip chemicals': {'saponification_NaOH': 'sodium hydroxide (saponification)',
              'scrub_HCl31': 'hydrochloric acid (scrub)', 'strip_HCl31': 'hydrochloric acid (strip)',
              'neutralization_NaOH': 'sodium hydroxide (neutralization)'},
          'Carbonation': {'carbonation_Na2CO3': 'soda ash (carbonation)'}}


def costs(model):
    """Each priced item from the IDAES costing block: USD per m3 raw water, per kg Li, and its price."""
    c = model.fs.costing
    assert {k for g in GROUPS.values() for k in g} == set(c.used_flows)
    li = value(c.li_kg_h)
    return pd.DataFrame([{'step': step, 'item': label, 'USD/m3 raw water': c.aggregate_flow_costs[k].value,
                          'USD/kg Li': c.aggregate_flow_costs[k].value / li,
                          'price (USD/kg)': value(getattr(c, k + '_cost'))}
                         for step, items in GROUPS.items() for k, label in items.items()])


PRICE_ITEMS = {'softening reagents': ('softening_Na2CO3', 'softening_NaOH', 'softening_HCl31'),
               'sludge disposal': ('sludge',), 'HBTA': ('HBTA',), 'TOPO and diluent': ('TOPO', 'diluent'),
               'extraction, scrub and strip chemicals': ('saponification_NaOH', 'scrub_HCl31', 'strip_HCl31',
                                                         'neutralization_NaOH'),
               'carbonation soda ash': ('carbonation_Na2CO3',)}


def price_sensitivity(m, result):
    """Sliders that scale each group of prices; the operating cost and break-even price at the solved design.

    The flows stay at the solved design, so this is a price sensitivity, not a new optimization: with other prices
    the cheapest design could change.
    """
    c = m.fs.costing
    assert {k for items in PRICE_ITEMS.values() for k in items} == set(c.used_flows)
    usd_h = {k: c.aggregate_flow_costs[k].value for k in c.used_flows}  # USD/h at 1 m3/h raw water
    li, base = value(c.li_kg_h), min(p['lce_price_usd_kg'] for p in result['full_process']['price_break_evens'])
    sliders = {group: widgets.FloatSlider(value=1., min=.25, max=2., step=.05, description=group, readout_format='.2f',
                                          continuous_update=False, style={'description_width': '230px'},
                                          layout=widgets.Layout(width='600px')) for group in PRICE_ITEMS}

    def show(**scale):
        cost = {g: sum(usd_h[k] for k in items) * scale[g] / li for g, items in PRICE_ITEMS.items()}
        total = sum(cost.values())
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.barh(list(cost), list(cost.values()), color='#33557a')
        ax.invert_yaxis()
        ax.set(xlabel='operating cost (USD per kg Li)', xlim=(0, 2 * max(sum(usd_h[k] for k in items) / li
                                                                          for items in PRICE_ITEMS.values())),
               title=f'{number(total)} USD/kg Li; break-even {number(total / LCE_PER_LI)} USD/kg LCE '
                     f'({number(total / LCE_PER_LI / base)} times the {base:g} USD/kg LCE base price)')
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.show()
    return widgets.interactive(show, **sliders)


def plot_costs(table, site, total):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = dict(zip(GROUPS, ('#4c78a8', '#e45756', '#72b7b2', '#f58518')))
    order = table.sort_values('USD/kg Li')
    ax.barh(order['item'], order['USD/kg Li'], color=[colors[s] for s in order['step']])
    ax.legend(handles=[Patch(color=c, label=step) for step, c in colors.items()], fontsize=8, loc='lower right')
    ax.set_xlabel('operating cost (USD per kg Li in Li$_2$CO$_3$)')
    ax.set_title(f'{WATERS[site]}, base case: {total:.0f} USD/kg Li in total')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- comparing the waters

def summary(results):
    return pd.DataFrame({WATERS[s]: {
        'extraction stages': len(r['stages']), 'HBTA (mol/L organic)': r['fixed_design']['extractant_mol_l'],
        'Li extraction (%)': 100 * r['li_extraction_fraction'],
        'Li recovery to Li2CO3 (%)': 100 * r['product_recovery_fraction'],
        'Li2CO3 product (kg per m3 raw)': r['full_process']['li_product_kg_m3'] * LCE_PER_LI,
        'operating cost (USD/kg Li)': r['cost_usd_kg_li'], 'operating cost (USD/m3 raw water)': r['cost_usd_m3_raw_brine'],
        'break-even price (USD/kg LCE)': r['full_process']['break_even_lce_usd_kg']}
        for s, r in results.items()}).style.format(number)


def plot_profiles(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = {'MS2': '#33557a', 'MarsHill': '#e45756'}
    for site, r in results.items():
        k = [row['stage'] for row in r['stages']]
        out = [row['outlet_mol_per_nominal_l'] for row in r['stages']]
        axes[0].semilogy(k, [1000 * LI_G_MOL * o['lithium-cation'] for o in out], 'o', color=colors[site], ms=8,
                         label=WATERS[site])
        for s, face in (('lithium-carrier', colors[site]), ('sodium-carrier', 'none')):
            axes[1].plot(k, [1000 * o[s] for o in out], 'o', mec=colors[site], mfc=face, mew=1.5, ms=8,
                         label=f"{WATERS[site]}, {'Li' if s.startswith('lithium') else 'Na'} complex")
    axes[0].set(xlabel='extraction stage', ylabel='Li in water leaving stage (mg/L)', title='Aqueous lithium')
    axes[1].set(xlabel='extraction stage', ylabel='organic complex leaving stage (mmol/L)',
                title='Organic-phase complexes')
    for ax in axes:
        ax.set_xticks(range(1, max(a['stages'] for a in ACCEPTED.values()) + 1))
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8, ncol=2, loc='upper center', bbox_to_anchor=(.5, -.18))
    plt.tight_layout()
    plt.show()


def plot_break_even(results):
    """Break-even LCE price of each water against the base and optimistic lithium carbonate prices."""
    fp = results['MS2']['full_process']
    prices = {'base price': min(p['lce_price_usd_kg'] for p in fp['price_break_evens']),
              'optimistic price': json.loads((STUDY / 'results/targets/economic_targets.json').read_text())
              ['mars_hill']['lce_usd_kg']}
    names = [WATERS[s] for s in results]
    break_even = [r['full_process']['break_even_lce_usd_kg'] for r in results.values()]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(break_even, names, 'o', ms=10, color='#33557a', label='break-even price (base-case costs)')
    for (label, price), style in zip(prices.items(), ('--', ':')):
        ax.axvline(price, color='#a23b2a', ls=style, label=f'{label}: {price:g} USD/kg LCE')
    for x, n in zip(break_even, names):
        ax.annotate(f'{x:.0f}', (x, n), textcoords='offset points', xytext=(0, 10), ha='center')
    ax.set(xlim=(0, 1.15 * max(break_even)), ylim=(-.6, len(names) - .4), xlabel='lithium carbonate price (USD/kg LCE)')
    ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    plt.show()
    return pd.DataFrame({name: {'break-even (USD/kg LCE)': b, 'times base price': b / prices['base price'],
                                'times optimistic price': b / prices['optimistic price']}
                         for name, b in zip(names, break_even)}).T


# ---------------------------------------------------------------- check against the campaign record

def _compare(rows, name, actual, expected, *, rel=None, abs_=None, scale=None):
    error = abs(actual - expected)
    limit = abs_ if abs_ is not None else rel * abs(scale if scale is not None else expected)
    rows.append({'quantity': name, 'reproduced': actual, 'retained': expected, 'difference': error,
                 'limit': limit, 'within limit': error <= limit})


def _compare_softening(rows, fresh, retained):
    for name in ('reagent_mol', 'aqueous_mol', 'cake_mol', 'solid_mol'):
        assert set(fresh[name]) == set(retained[name]), f'softening {name} species differ from the retained record'
    for name in ('feed_l', 'wet_sludge_kg', 'dry_sludge_kg', 'pH', 'reagent_mol', 'aqueous_mol', 'cake_mol', 'solid_mol'):
        pairs = fresh[name].items() if isinstance(fresh[name], dict) else [(name, fresh[name])]
        for key, n in pairs:
            target = retained[name][key] if isinstance(fresh[name], dict) else retained[name]
            tolerance = 1e-9 if key == 'HCl' or name == 'pH' else 1e-12
            rows.append({'quantity': f'softening {name}' + (f' {key}' if key != name else ''), 'reproduced': n,
                         'retained': target, 'difference': abs(n - target), 'limit': max(tolerance, 1e-9 * abs(target)),
                         'within limit': math.isclose(n, target, rel_tol=1e-9, abs_tol=tolerance)})


KINDS = {'campaign': 'campaign result', 'exchange band': 'exchange-band result', 'cost item': 'cost item',
         'softening': 'softening', 'stage': 'stage outlets', 'physical': 'process streams', 'extractant': 'design',
         'ln K': 'design'}


def check_campaign(results, softened):
    """Every reproduced number against the campaign record and exchange band; stops on any failure.

    Draws every comparison against its tolerance and returns (Engine pins, summary per water, every comparison).
    """
    tol, checks, pins = BAND['reporting_relative_cost_tolerance'], [], {}
    engine = flowsheet.installed_engine()['source_commit'][:8]
    assert set(results) == set(softened) == set(WATERS), 'every water must be reproduced and checked'
    for site, r in results.items():
        a, rows = ACCEPTED[site], []
        record = json.loads((ROOT / a['source']).read_text())
        pins[WATERS[site]] = {'campaign Engine commit': record['engine']['source_commit'][:8], 'this run': engine}
        assert record['candidate']['stages'] == len(r['stages']) == a['stages'] and record['status'] == 'optimal'
        for key in ('cost_usd_kg_li', 'cost_usd_m3_raw_brine'):
            _compare(rows, f'campaign {key}', r[key], record[key], rel=tol)
            _compare(rows, f'exchange band {key}', r[key], a[key], rel=tol)
        b = r['full_process']['break_even_lce_usd_kg']
        _compare(rows, 'campaign break_even_lce_usd_kg', b, record['full_process']['break_even_lce_usd_kg'], rel=tol)
        _compare(rows, 'exchange band break_even_lce_usd_kg', b, a['break_even_lce_usd_kg'], rel=tol)
        for k, n in record['full_process']['cost_usd_kg_li'].items():
            _compare(rows, f'cost item {k} (USD/kg Li)', r['full_process']['cost_usd_kg_li'][k], n, rel=tol,
                     scale=record['full_process']['operating_usd_kg_li'])
        for key in ('li_extraction_fraction', 'product_recovery_fraction', 'hbta_lost_fraction'):
            _compare(rows, f'campaign {key}', r[key], record[key], abs_=1e-9)
        for key in ('li_extraction_fraction', 'product_recovery_fraction'):
            _compare(rows, f'exchange band {key}', r[key], a[key], abs_=1e-9)
        for reaction, n in record['fixed_design']['ln_k'].items():
            _compare(rows, f'ln K {reaction}', r['fixed_design']['ln_k'][reaction], n, abs_=0.)
        _compare(rows, 'extractant_mol_l', r['fixed_design']['extractant_mol_l'],
                 record['fixed_design']['extractant_mol_l'], abs_=1e-9)
        stage_error = max(abs(new['outlet_mol_per_nominal_l'][s] - old['outlet_mol_per_nominal_l'][s])
                          for new, old in zip(r['stages'], record['stages'], strict=True)
                          for s in old['outlet_mol_per_nominal_l'])
        _compare(rows, 'stage outlets, largest difference (mol/L)', stage_error, 0., abs_=1e-9)
        stream_error = max(abs(r['physical_streams_mol_per_raw_l'][k][s] - n)
                           for k, stream in record['physical_streams_mol_per_raw_l'].items() for s, n in stream.items())
        _compare(rows, 'physical streams, largest difference (mol/L raw)', stream_error, 0., abs_=1e-9)
        _compare_softening(rows, softened[site][CASE], record['softening'][CASE])
        checks.append(pd.DataFrame(rows).assign(water=WATERS[site]))
    checks = pd.concat(checks, ignore_index=True).rename(columns={'within limit': 'passed'})
    group = checks['quantity'].str.extract(r'^(campaign|exchange band|cost item|softening|stage|physical|extractant|ln K)')[0]
    checks['kind'] = group.map(KINDS)
    failed = checks[~checks['passed']]
    assert failed.empty, f'reproduced numbers outside their limits:\n{failed}'
    summary = plot_checks(checks, 'water', f'{len(checks)} comparisons with the retained campaign')
    return pd.DataFrame(pins).T, summary, checks


# ---------------------------------------------------------------- exploring the options, as the process explorer app does

ALL_WATERS = {'MS2': 'MS-2', 'BR2': 'BR-2', 'MarsHill': 'Mars Hill', 'Westmoreland': 'Westmoreland'}
FITS = {'low': 'single-stage fit', 'high': 'cascade fit'}
CALCULATIONS = {'retained': 'retained campaign result', 'fixed': 'solve the design', 'selected': 'optimize the design'}
STATUS = {'optimal': 'optimized', 'grid': 'solved at the fixed design', 'infeasible': 'no admissible design',
          'unresolved': 'not resolved'}
compared = []  # every selection shown so far, in order


class Options(widgets.VBox):
    """The whole-process options of the process explorer app, as widgets; `selection` reads their values."""

    def __init__(self):
        style, layout = {'description_width': '150px'}, widgets.Layout(width='620px')
        self.controls = {
            'site': widgets.Dropdown(options=[(n, k) for k, n in ALL_WATERS.items()], description='produced water'),
            'case': widgets.ToggleButtons(options=[('base prices', 'base'), ('optimistic prices', 'optimistic')],
                                          description='price case'),
            'end': widgets.ToggleButtons(options=[(n, k) for k, n in FITS.items()], description='Li/Na exchange'),
            'stages': widgets.IntSlider(value=5, min=1, max=5, description='extraction stages'),
            'calculation': widgets.ToggleButtons(options=[(n, k) for k, n in CALCULATIONS.items()],
                                                 description='calculation')}
        for control in self.controls.values():
            control.style, control.layout = style, layout
        super().__init__(list(self.controls.values()))

    @property
    def selection(self):
        return {k: c.value for k, c in self.controls.items()}


def _row(record, selection):
    """One comparison row from a campaign-format record."""
    row = {'produced water': ALL_WATERS[selection['site']], 'prices': selection['case'], 'Li/Na exchange': FITS[selection['end']],
           'stages': selection['stages'], 'calculation': CALCULATIONS[selection['calculation']],
           'result': STATUS.get(record['status'], record['status'])}
    if record['status'] in ('optimal', 'grid'):
        row |= {'HBTA (mol/L)': record['fixed_design']['extractant_mol_l'],
                'Li extraction (%)': 100 * record['li_extraction_fraction'],
                'Li to Li2CO3 (%)': 100 * record['product_recovery_fraction'],
                'USD/m3 raw water': record['cost_usd_m3_raw_brine'], 'USD/kg Li': record['cost_usd_kg_li'],
                'break-even (USD/kg LCE)': record['full_process']['break_even_lce_usd_kg']}
    else:
        row['reason'] = record.get('reason', '')
    return row


def run_selection(options):
    """The selected option: read from the retained campaign, or solved live by the app's own campaign worker.

    A live solve runs in a separate process, as in the app, and can take up to about five minutes.
    """
    selection = options.selection if isinstance(options, Options) else dict(options)
    if selection['calculation'] == 'retained':
        name = 'C_{site}_full_process_{case}_n{stages}_{end}.json'.format(**selection)
        record = json.loads((STUDY / 'results/campaign/selected' / name).read_text())
    else:
        from apps.process_explorer_web.local_service import solve_campaign
        record = solve_campaign({'solvent': 'set_c', 'scope': 'full', 'phase': selection['calculation'],
                                 **{k: selection[k] for k in ('site', 'case', 'stages', 'end')}})
    row = _row(record, selection)
    if row not in compared:  # the same selection is listed once
        compared.append(row)
    return pd.DataFrame([row])


def add_accepted_designs(case='base', end='low'):
    """The accepted design of every water at one price case and fit, from the retained campaign."""
    for group in BAND['groups']:
        if group['case'] == case:
            run_selection({'site': group['site'], 'case': case, 'end': end, 'stages': group[end]['stages'],
                           'calculation': 'retained'})
    return compare_selections()


def compare_selections():
    """Every selection shown so far: a table, and the cost per kg of lithium of those with a result."""
    table = pd.DataFrame(compared)
    solved = table.dropna(subset=['USD/kg Li']) if 'USD/kg Li' in table else table.iloc[:0]
    if len(solved):
        labels = [f"{r['produced water']}, {r['stages']} stages, {r['prices']}, {r['Li/Na exchange']}\n({r['calculation']})"
                  for _, r in solved.iterrows()]
        fig, ax = plt.subplots(figsize=(9, .55 * len(solved) + 1))
        ax.barh(labels, solved['USD/kg Li'], color='#33557a')
        for y, x in enumerate(solved['USD/kg Li']):
            ax.annotate(number(x), (x, y), xytext=(4, 0), textcoords='offset points', va='center', fontsize=8)
        ax.invert_yaxis()
        ax.set(xlabel='operating cost (USD per kg Li in Li$_2$CO$_3$)', xlim=(0, 1.15 * solved['USD/kg Li'].max()))
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.show()
    return table
