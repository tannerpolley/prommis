"""#188 unsoftened MS-2 option: strict reactive stages and one-cycle reagent/loss accounts."""

import csv
import json

from analyses.prototypes.d2ehdtpa_buphen_ms2_prototype.scripts import reaction_coordinates as rc
from lithium_extraction.prototypes.produced_water_sx_prototype.cascade import sweep_stages
from lithium_extraction.prototypes.produced_water_sx_prototype.flowsheet import _row
from lithium_extraction.common.support.paths import REPO_ROOT

ANALYSIS = REPO_ROOT / 'analyses/prototypes/d2ehdtpa_buphen_ms2_prototype'
BASIS = json.loads((ANALYSIS / 'inputs/stage2_basis.json').read_text())
MASSES = json.loads((ANALYSIS / 'inputs/reaction_basis.json').read_text())['molar_masses_kg_mol']
BUPHEN_ORGANIC_AQUEOUS_PARTITION = 1.4e4
LABELS = {'evidence_type': 'prediction', 'holdout_support': 'no holdout support',
          'activity_error_label': rc.ERROR_LABEL, 'transfer_label': 'Na and formulation transfer not supported (F2)',
          'temperature_K': rc.T, 'pressure_Pa': rc.P}


def ms2_feed():
    with (REPO_ROOT / 'data/reference/produced_water/smackover_selected_base_feed_ms2.csv').open() as stream:
        measured = {r['analyte']: float(r['value']) for r in csv.DictReader(stream) if r['analyte'] in MASSES}
    return {e: measured[e] / (1e6*MASSES[e]) for e in ('Li', 'Na', 'K', 'Mg', 'Ca', 'Sr')}


def site_supply(buphen, S, v, stage_count, *, site_basis_li_mol_l=None):
    """The #188 fixed design domain and site supply, also used by the generic simulation."""
    basis = BASIS['site_basis_li_mol_L'] if site_basis_li_mol_l is None else site_basis_li_mol_l
    sites = S*basis/v
    if not (.25 <= buphen <= 1 and .9 <= S <= 1.5 and .5 <= v <= 2 and .020 <= sites <= .060
            and stage_count in BASIS['stage_counts']):
        raise ValueError(f'Outside #188 domain: B={buphen}, S={S}, O/A={v}, N={stage_count}, A={sites}')
    return sites


def extract(constants, buphen, S, v, stage_count, *, sr_decade=0, start=None, progress=None, resume_sweeps=0):
    """Available NaA sites after H2S; unavailable HA/NaHS carried outside the Engine."""
    sites = site_supply(buphen, S, v, stage_count)
    feed = ms2_feed()
    fresh = {e: 0. for e in rc.COMPLEX}
    def solve_stage(k, aqueous, entering, guess):
        result = rc.solve(aqueous, sites, buphen, constants, organic_to_aqueous=v,
                          loading=entering, sr_decade=sr_decade)
        aq, org = result['amounts_mol']
        out_aq = {e: aq[rc.ION[e][0]] for e in feed}
        out_org = {e: org[c]/v for e, c in rc.COMPLEX.items()}
        row = {'stage': k + 1, 'D': result['D'], 'certificate': result['certificate'],
               'amounts_mol': result['amounts_mol'], 'free_B_mol_L': org['buphen']/v, **LABELS}
        return out_aq, out_org, row, None
    result = sweep_stages(stage_count, feed, fresh, solve_stage, start=start, progress=progress, resume_sweeps=resume_sweeps)
    aq, org = result['raffinate'], result['loaded_organic']
    closure = {e: feed[e] - aq[e] - v*org.get(e, 0) for e in feed if e != 'Na'}
    loaded_na = sites - sum(rc.ION[e][2]*org[e] for e in org)
    closure['Na'] = feed['Na'] + v*sites - aq['Na'] - v*loaded_na
    if max(map(abs, closure.values())) > 1e-9:
        raise RuntimeError(f'#188 cascade balance exceeds 1e-9 mol/L feed: {closure}')
    result.update(closure_mol=closure, sites_mol_L=sites, loaded_Na_mol_L=loaded_na,
                  li_extraction_fraction=1 - aq['Li']/feed['Li'], **LABELS)
    return result


def accounts(extraction, S, v, buphen, case, h2s, *, feed_li_mol_l=None, site_basis_li_mol_l=None,
             value_basis_li_mol_l=None):
    """USD/m3 feed and kg/m3 losses; missing prices remain symbolic, never zero-priced totals."""
    reg = 'data/reference/tea/hbta_topo_screening_cost_assumptions.csv'
    prices = {k: float(_row(reg, 'assumption_id', k + '_price')['base']) for k in ('naoh', 'hcl')}
    organic_reg = 'data/reference/tea/d2ehdtpa_buphen_price_register.csv'
    octanol_price = float(_row(organic_reg, 'component_id', '1_octanol')['delivered_price_usd_kg'])
    feed_li = ms2_feed()['Li'] if feed_li_mol_l is None else feed_li_mol_l
    site_basis = BASIS['site_basis_li_mol_L'] if site_basis_li_mol_l is None else site_basis_li_mol_l
    available = S*site_basis
    loaded = {e: v*n for e, n in extraction['loaded_organic'].items()}
    loaded['Na'] = v*extraction['loaded_Na_mol_L']
    acid = sum(rc.ION[e][2]*n for e, n in loaded.items())  # includes residual NaA; excludes unavailable HA.
    strip = {e: n*BASIS['strip_fractions'][case][e] for e, n in loaded.items()}
    retained = {e: loaded[e] - strip[e] for e in loaded}
    strip_closure = max(abs(loaded[e] - strip[e] - retained[e]) for e in loaded)
    if strip_closure > 1e-9 or abs(acid - available) > 1e-9:
        raise RuntimeError('Strip/site mass balance exceeds 1e-9 mol/L feed')
    rec = strip['Li']/feed_li
    li_kg = strip['Li']*MASSES['Li']*1000
    lce_kg = strip['Li']*.07389/2*1000
    f = BASIS['entrainment_fraction'][case]
    oct_diss = sum(BASIS[k]['pure_water_solubility_kg_m3'] for k in ('octanol_1', 'octanol_2'))
    oct_ent = f*v*1000*sum(BASIS[k]['vol_fraction']*BASIS[k]['density_kg_L'] for k in ('octanol_1', 'octanol_2'))
    b_diss = extraction['stages'][-1]['free_B_mol_L']/BUPHEN_ORGANIC_AQUEOUS_PARTITION*.29242*1000
    b_ent = f*v*buphen*.29242*1000
    a_ent = f*(available + h2s)*.35456*1000
    diluent = extraction['stages'][0]['amounts_mol'][1]['diluent']
    d_ent = f*.75*diluent*.17033484*1000  # explicit 5/20 vol% octanols replace modeled dodecane.
    costs = {'naoh': (available + h2s)*.039997*1000*prices['naoh'],
             'hcl31': acid*.036461/.31*1000*prices['hcl'],
             'octanol_dissolution': oct_diss*octanol_price, 'octanol_entrainment': oct_ent*octanol_price}
    subtotal = sum(costs.values())
    result = {'case': case, 'h2s_protonated_mol_L_feed': h2s, 'R_ext': extraction['li_extraction_fraction'],
              'strip_Li_fraction': BASIS['strip_fractions'][case]['Li'], 'R_rec': rec,
              'circulated_sites_mol_L_feed': available + h2s, 'H2S_fraction_of_available_sites': h2s/available,
              'strip_HCl_mol_L_feed': acid, 'strip_Cl_mol_L_feed': acid,
              'strip_unconsumed_HCl_mol_L_feed': acid - sum(rc.ION[e][2]*n for e, n in strip.items()),
              'NaOH_mol_L_feed': available + h2s, 'li_recovered_kg_m3': li_kg, 'lce_recovered_kg_m3': lce_kg,
              'strip_na_fraction': strip['Na']/(strip['Li'] + strip['Na']), 'strip_closure_mol': strip_closure,
              'BuPhen_dissolved_kg_m3': b_diss, 'BuPhen_entrained_kg_m3': b_ent,
              'BuPhen_loss_kg_m3': b_diss + b_ent, 'D2EHDTPA_loss_kg_m3': a_ent, 'dodecane_loss_kg_m3': d_ent,
              'octanol_dissolved_kg_m3': oct_diss, 'octanol_entrained_kg_m3': oct_ent,
              'priced_subtotal_usd_m3': subtotal, 'priced_subtotal_usd_kg_li': subtotal/li_kg,
              'priced_subtotal_usd_kg_lce': subtotal/lce_kg, 'missing_accounts': BASIS['missing_accounts'],
              'cost_expression': 'priced subtotal + BuPhen_loss*p_BuPhen + D2EHDTPA_loss*p_D2EHDTPA + dodecane_loss*p_dodecane; prices USD/kg, losses kg/m3',
              'break_even_label': 'BuPhen-only conditional upper bound, before D2EHDTPA/dodecane prices and excluded accounts',
              'octanol_dissolution_label': BASIS['octanol_dissolution_label'],
              'H2S_scope': 'HA unavailable outside Engine; equal Na and HS leave raffinate; no sulfide activity or residual-loading reuse model',
              'strip_assumptions': 'Na=Li strip fraction and Sr=alkaline-earth strip fraction; no independent Na or Sr strip measurement', **LABELS}
    for e in loaded:
        result.update({f'loaded_{e}_mol_L_feed': loaded[e], f'strip_{e}_mol_L_feed': strip[e],
                       f'retained_{e}_mol_L_feed': retained[e]})
        if e in ('Mg', 'Ca', 'Sr'):
            result[f'strip_{e}_per_Li_mol_mol'] = strip[e]/strip['Li']
    for k, cost in costs.items():
        result[f'{k}_usd_m3'] = cost
    value_basis = ms2_feed()['Li'] if value_basis_li_mol_l is None else value_basis_li_mol_l
    value_scale = feed_li/value_basis
    for label, value in (('low', 15.65*value_scale), ('high', 18.69*value_scale),
                         ('screening_25', feed_li*.07389/2*1000*25)):
        result.update({f'revenue_{label}_usd_m3': value*rec,
                       f'headroom_before_missing_{label}_usd_m3': value*rec - subtotal,
                       f'BuPhen_break_even_upper_{label}_usd_kg': (value*rec - subtotal)/(b_diss + b_ent)})
    return result
