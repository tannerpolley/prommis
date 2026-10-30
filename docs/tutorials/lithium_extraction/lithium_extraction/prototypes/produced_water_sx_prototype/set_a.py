"""Set A adapter on the generic connected stages, retaining the #163 chemistry and R2 basis."""

import csv
import math
from pathlib import Path

from analyses.shared.hbta_topo_configured_reactive_lle.scripts import mechanism_candidates as mc
from lithium_extraction.common.support.paths import REPO_ROOT
from lithium_extraction.prototypes.hbta_topo_ms2_saponified import stage_chemistry as owner
from . import fresh, records, references, stage

R2_CASE = 'ms2_divalents_omitted_n4_ropt'
R2_DIRECTORY = Path(references.SOURCES['R2']).parent / R2_CASE
CONSERVED = ('Li', 'Na', 'H', 'BTA', 'TOPO')


def reference_feed():
    """Original nominal Li/Na conversions, with coherent physical chloride-closure metadata."""
    return fresh.reference_feed('A', 'R2')


def chemistry(anchor):
    """Transport the owner's species, affine feed, continuation, balances and strict direct solve."""
    reference = owner.build_live_reference(anchor)
    species = owner.LIVE_SPECIES
    supports = tuple(tuple(s for s in species if s in members) for members in (mc.AQUEOUS, mc.ORGANIC))
    conservation = tuple(tuple(mc.MOIETIES[name].get(s, 0) for s in species) for name in CONSERVED)
    # The equal free HBTA/TOPO control is the owner's map, distinct from the five conserved totals.
    controls = ('diluent', 'lithium-cation', 'sodium-cation', 'hydrogen-cation',
                'hbta', 'sodium-carrier', 'lithium-carrier')

    def certify(inlet, outlet):
        ratio = inlet['diluent'] / reference.dodecane_mol_l_organic
        li, na = inlet['lithium-carrier'], inlet['sodium-carrier']
        total_hbta = inlet['hbta'] + li + na
        total_topo = inlet['topo'] + li + na
        state = owner.stage_state('modular set A converged inlet', organic_to_aqueous=ratio,
                                  li=inlet['lithium-cation'], na=inlet['sodium-cation'],
                                  hydrogen=inlet['hydrogen-cation'], carrier=total_hbta / ratio,
                                  incoming_li=li / total_hbta, incoming_na=na / total_hbta,
                                  hbta=total_hbta / ratio, topo=total_topo / ratio)
        direct = owner.solve_stage(state)
        if direct['active_reactions'] != list(mc.SETS['A']):
            return {**direct, 'status': 'not_evaluable',
                    'reason': f"state needs {direct['active_reactions']}; the live species map carries no OH-",
                    'outlet_max_difference_mol_per_raw_l': None}
        direct['outlet_max_difference_mol_per_raw_l'] = max(abs(outlet[s] - sum(
            phase.get(s, 0.0) for phase in direct['phase_amounts_mol'])) for s in species)
        return direct

    declaration = stage.Chemistry(species, supports, conservation,
                                  tuple(mc.CHARGE.get(s, 0) for s in species), controls,
                                  owner.FEED_OFFSET, owner.FEED_MATRIX, reference.record,
                                  reference.library, certify)
    return declaration, reference


def build_flowsheet(brine, solvent='A', scope='extraction_only', stages=4, scenario=None):
    """Fixed set A design; no old flowsheet wrapper, surrogate or released decisions."""
    if solvent != 'A' or scope != 'extraction_only':
        raise ValueError('set A adapter supports extraction only')
    if not scenario or set(scenario) != {'organic_to_aqueous_ratio', 'extractant_mol_l', 'saponification_ratio'}:
        raise ValueError('fixed design requires O/A, extractant molarity and saponification ratio')
    ratio, c, r = (scenario[k] for k in ('organic_to_aqueous_ratio', 'extractant_mol_l', 'saponification_ratio'))
    if any(not math.isfinite(n) or n <= 0 for n in (ratio, c)) or not math.isfinite(r) or not 0 <= r <= 1:
        raise ValueError('O/A and extractant must be positive; saponification must lie in [0, 1]')
    feed = dict(brine) if 'physical_feed' in brine else records.extraction_feed(brine, solvent, scope)
    if 'physical_feed' not in brine:
        feed['raw_brine'] = brine
    if 'unavailable' in feed:
        raise ValueError(feed['unavailable'])
    aqueous_support = tuple(s for s in owner.LIVE_SPECIES if s in mc.AQUEOUS)
    carried = {records.SPECIES_LABELS[s] for s in aqueous_support}
    if any(n != 0 for s, n in feed['model_feed'].items() if s not in carried):
        raise ValueError('set A live species map carries no OH- or other nonzero species outside its support')
    anchor = owner.anchor_state(feed['model_feed']['Li'], feed['model_feed']['Na'], feed['model_feed']['H'])
    declared, reference = chemistry(anchor)
    state = owner.stage_state('modular set A boundary', organic_to_aqueous=ratio, li=feed['model_feed']['Li'],
                              na=feed['model_feed']['Na'], hydrogen=feed['model_feed']['H'], carrier=c,
                              incoming_na=r)
    aqueous, organic = mc.inventory(state)
    aqueous = {s: aqueous[s] for s in aqueous_support}
    organic = {'diluent': ratio * reference.dodecane_mol_l_organic, **organic}
    model = stage.build_stages(declared, aqueous, organic, stages)
    model._physical_feed, model._design = feed, dict(scenario)
    return model


def reference_profiles(model):
    """Use complete amounts from the unrounded fixed R2 stage profiles."""
    with (REPO_ROOT / R2_DIRECTORY / 'stage_rows.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    ratio, c, r = (model._design[k] for k in ('organic_to_aqueous_ratio', 'extractant_mol_l', 'saponification_ratio'))
    if len(rows) != len(model.fs.stages) or any(float(rows[0][key]) != n for key, n in (
            ('organic_to_aqueous_ratio', ratio), ('carrier_mol_l_organic', c), ('saponification_ratio', r))):
        raise ValueError('R2 profiles apply only to their retained four-stage fixed design')
    profiles = []
    for row in rows:
        aqueous, organic = model._feeds
        amounts = dict(aqueous, **organic)
        for s, name in (('lithium-cation', 'li'), ('sodium-cation', 'na'), ('hydrogen-cation', 'hydrogen')):
            amounts[s] = float(row[f'{name}_out_mol_l'])
        amounts['lithium-carrier'] = ratio * c * float(row['equilibrium_li_loading_fraction'])
        amounts['sodium-carrier'] = ratio * c * float(row['equilibrium_na_loading_fraction'])
        amounts['hbta'] = amounts['topo'] = ratio * c - amounts['lithium-carrier'] - amounts['sodium-carrier']
        profiles.append(amounts)
    return profiles


def report(model, result=None):
    """Complete sodium, physical-stream and extraction-plus-saponification accounting."""
    result = stage.report(model) if result is None else result
    aqueous, organic = model._feeds
    final = result['outlet_mol_per_raw_l']
    extraction = (final['lithium-carrier'] - organic['lithium-carrier']) / aqueous['lithium-cation']
    net_na = final['sodium-carrier'] - organic['sodium-carrier']
    raffinate = dict(model._physical_feed['model_feed'])
    raffinate.update({records.SPECIES_LABELS[s]: final[s] for s in model._chemistry.supports[0]})
    r7 = references.check_r7(model._physical_feed, raffinate,
                             balance_residuals_mol_per_raw_l=result['balance_residuals_mol_per_raw_l'],
                             strict_certificate=result['strict_certificate'])
    ratio, c, r = (model._design[k] for k in ('organic_to_aqueous_ratio', 'extractant_mol_l', 'saponification_ratio'))
    loadings = [row['outlet_mol_per_raw_l']['lithium-carrier'] / (ratio * c) for row in result['stages']]
    labels = model._physical_feed['labels'] + [
        f'selected HBTA and TOPO = {c:.17g} mol/L organic each: physical calibration coverage unverified; '
        'existing #163 design interval [0.015, 0.030] mol/L organic, not physical calibration coverage',
        f'selected saponification = {r:.17g}: existing #163 design interval [0.70, 0.95], not physical calibration coverage',
        '#158 set A held-out fit misses the 0.30-decade criterion; worst error 0.86 decades',
        'saponification above 0.7 is extrapolated; Zhang 2017 Table 4 extraction decline is not represented',
        'unsupported divalents and equivalent chloride removed at no cost',
        'fixed-design numerical replay; not a new optimum; numerical verification, not physical validation',
        'extraction plus saponification screening cost; no full-process or commercial cost claim',
    ]
    return {**result, 'fixed_design': model._design, 'feed': model._physical_feed, 'R7': r7,
            'aqueous_inlet_mol_per_raw_l': aqueous, 'organic_inlet_mol_per_raw_l': organic,
            'conserved_quantity_names': CONSERVED, 'labels': labels,
            'li_extraction_fraction': extraction, 'net_na_extraction_fraction': net_na / aqueous['sodium-cation'],
            'loaded_organic_na_over_li_plus_na': final['sodium-carrier'] / (
                final['lithium-carrier'] + final['sodium-carrier']),
            'sodium_account_mol_per_raw_l': {'brine_in': aqueous['sodium-cation'],
                'saponified_organic_in': organic['sodium-carrier'], 'raffinate_out': final['sodium-cation'],
                'loaded_organic_out': final['sodium-carrier'], 'net_to_organic': net_na,
                'returned_to_aqueous': final['sodium-cation'] - aqueous['sodium-cation']},
            'saponification_NaOH_mol_per_raw_l': organic['sodium-carrier'],
            'outlet_li_loading_fractions': loadings, 'li_loading_margin': .72 - max(loadings),
            'minus_log10_outlet_h_mol_l': [-math.log10(row['outlet_mol_per_raw_l']['hydrogen-cation'])
                                          for row in result['stages']],
            'costs': fresh.extraction_costs(model, extraction, organic['sodium-carrier'])}
