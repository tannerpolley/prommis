"""#201 records and feed quantities only; no equilibrium or process model is built."""

import json
import math

from lithium_extraction.common.support.paths import REPO_ROOT
from . import brines

SHARED = 'analyses.shared.hbta_topo_configured_reactive_lle.scripts'
D2 = 'analyses.prototypes.d2ehdtpa_buphen_ms2_prototype.scripts'
SAPONIFIED = 'lithium_extraction.prototypes.hbta_topo_ms2_saponified'
FRESH = 'lithium_extraction.v3.hbta_topo_ms2_extraction'
SOLVENTS = {
    'fresh': {
        'owner': FRESH, 'feed_owner': f'{FRESH}.live_equilibrium:FEED_MATRIX/FEED_OFFSET',
        'chemistry_owner': f'{FRESH}.live_equilibrium',
        'conservation_owner': f'{SHARED}.independent_network:_basis_matrix',
        'certificate_owner': f'{SHARED}.independent_network',
        'domain_owner': 'data/processed/hbta_topo_ms2_engine_comparison/greenfield_current_v3/loaded_state_surrogate.json',
        'domain_basis': 'current loaded-organic surrogate numerical interpolation; not physical calibration',
        'labels': ['#158 fresh-solvent recovery discrepancy'],
    },
    'A': {
        'owner': SAPONIFIED, 'feed_owner': f'{SHARED}.mechanism_candidates:inventory',
        'chemistry_owner': f'{SAPONIFIED}.stage_chemistry',
        'conservation_owner': f'{SHARED}.mechanism_candidates:MOIETIES',
        'certificate_owner': f'{SHARED}.mechanism_candidates:is_strict',
        'domain_owner': 'analyses/prototypes/hbta_topo_ms2_saponified/README.md',
        'labels': ['set A cannot evaluate a state requiring hydroxide'],
    },
    'C': {
        'owner': SAPONIFIED, 'feed_owner': f'{SHARED}.mechanism_candidates:inventory',
        'chemistry_owner': f'{SHARED}.mechanism_candidates:network/solve',
        'conservation_owner': 'analyses.prototypes.hbta_topo_ms2_prototype.scripts.countercurrent',
        'certificate_owner': f'{SHARED}.mechanism_candidates:is_strict',
        'domain_owner': 'analyses/prototypes/hbta_topo_ms2_prototype/README.md',
        'labels': ['support selected per stage by existing owner', 'several fixed stage supports share Engine #452 records',
                   '#158 recovery requires unsupported aqueous HBTA dissolution'],
    },
    'D2EHDTPA-Held': {
        'owner': 'lithium_extraction.prototypes.produced_water_sx_prototype.d2ehdtpa_buphen',
        'feed_owner': f'{D2}.reaction_coordinates:aqueous_feed',
        'chemistry_owner': f'{D2}.reaction_coordinates',
        'conservation_owner': f'{D2}.reaction_coordinates:MOIETIES',
        'certificate_owner': f'{SHARED}.mechanism_candidates:is_strict',
        'domain_owner': f'{D2}.reaction_coordinates:payload',
        'labels': ['Held mixed-Mg discrepancy remains unresolved',
                   'Na and formulation transfer not supported (F2)'],
    },
    'D2EHDTPA-Figiel': {
        'owner': f'{D2}.conditional_figiel_packet',
        'feed_owner': f'{D2}.conditional_figiel_packet:lumped_feed',
        'chemistry_owner': f'{D2}.conditional_figiel_packet',
        'conservation_owner': f'{D2}.reaction_coordinates:MOIETIES (Sr lumped into Ca)',
        'certificate_owner': f'{SHARED}.mechanism_candidates:is_strict',
        'domain_owner': f'{D2}.conditional_figiel_packet:packet_payload',
        'labels': ['provisional conditional basis (#197)',
                   'binary salts only; Sr counted as Ca', 'Na and formulation transfer not supported (F2)'],
    },
    'TOP-FeCl3': {'owner': None, 'unavailable': 'not modeled (#193)'},
}

SCOPES = {
    'extraction_only': {
        'steps': ('saponification if used', 'extraction'),
        'cost_items': ('organic_makeup', 'stage_operation', 'electricity', 'saponification_NaOH'),
        'li_denominator': 'Li transferred to loaded organic (kg/y)',
        'feed_basis': 'raw brine after unsupported divalent/equal-Cl removal; fresh/A nominal acidity input',
        'excluded': ('capital', 'labour', 'unquantified nominal acidity-conditioning HCl and buffering'),
    },
    'full_process': {
        'steps': ('softening if required', 'saponification', 'extraction', 'scrub', 'strip',
                  'concentration', 'carbonation'),
        'cost_items': ('softening_NaOH', 'softening_Na2CO3', 'softening_HCl', 'sludge',
                       'saponification_NaOH', 'scrub_HCl', 'strip_HCl', 'neutralization_NaOH',
                       'carbonation_Na2CO3', 'HBTA_loss', 'TOPO_loss', 'diluent_loss'),
        'li_denominator': 'Li in Li2CO3 product (kg/y)',
        'feed_basis': 'extended softened, washed outlet',
        'excluded': ('capital', 'labour', 'energy'),
    },
}

# Species names are transport labels, not a new reaction or element-conservation model.
SPECIES_LABELS = {'water': 'water', 'lithium-cation': 'Li', 'sodium-cation': 'Na',
                  'hydrogen-cation': 'H', 'hydroxide-anion': 'OH', 'bta-anion': 'BTA',
                  'chloride-anion': 'Cl', 'potassium-cation': 'K', 'magnesium-cation': 'Mg',
                  'calcium-cation': 'Ca', 'strontium-cation': 'Sr'}


def extraction_feed(brine, solvent, scope='extraction_only', *, process_pH=brines.PROCESS_PH,
                    synthetic_softened=False):
    """Physical feed and signed model substitution, mol per litre of raw brine.

    Fresh and saponified HBTA share the same aqueous inventory at zero incoming organic
    loading/base/BTA; the shared owner supplies it without importing the v3 bucket.
    """
    record = SOLVENTS[solvent]
    SCOPES[scope]
    if 'unavailable' in record:
        return {'unavailable': record['unavailable']}
    if scope == 'full_process' and not solvent.startswith('D2EHDTPA'):
        return {'unavailable': 'full process requires the complete washed outlet; set C uses set_c.washed_feed'}
    if not math.isfinite(process_pH):
        raise ValueError('process pH must be finite')
    if brine['water_kg'] is None and not synthetic_softened:
        return {'unavailable': 'physical water inventory unavailable: SG or TDS not reported'}
    if synthetic_softened:
        physical = {e: brine['mol'][e] for e in ('Li', 'Na', 'K')}
        physical.update(water=55.508, H=10 ** -process_pH)
        physical['Cl'] = sum(physical[e] for e in ('Li', 'Na', 'K', 'H'))
    else:
        physical = dict(brine['mol'], water=brine['water_kg'] / brines.WATER_MOLAR_MASS_KG)
    if any(not math.isfinite(n) or n < 0 for n in physical.values()):
        raise ValueError('physical feed must be finite and nonnegative')
    removed = {}
    if solvent in ('fresh', 'A', 'C'):
        removed = {e: physical.get(e, 0.0) for e in ('Mg', 'Ca', 'Sr', 'Ba')}
        removed['Cl'] = sum(2 * n for n in removed.values())
        if physical['Cl'] < removed['Cl']:
            raise ValueError('chloride cannot cover the divalent charge removed at no cost')
        for e, n in removed.items():
            physical[e] = physical.get(e, 0.0) - n
        from analyses.shared.hbta_topo_configured_reactive_lle.scripts import mechanism_candidates as mc

        state = mc.State(identity='modular extraction inlet', temperature_k=298.15, organic_to_aqueous=1.0,
                         li=physical['Li'], na=physical['Na'],
                         hydrogen=10 ** -process_pH, hbta=0.0, topo=0.0)
        aqueous, _ = mc.inventory(state)
    else:
        from analyses.prototypes.d2ehdtpa_buphen_ms2_prototype.scripts import reaction_coordinates as rc

        cations = {e: physical.get(e, 0.0) for e in rc.ION if e != 'Cl'}
        if synthetic_softened:
            cations['K'] = 0.0  # #216: unchanged KCl remains in physical accounting.
        if solvent == 'D2EHDTPA-Figiel':
            from analyses.prototypes.d2ehdtpa_buphen_ms2_prototype.scripts.conditional_figiel_packet import lumped_feed

            cations = lumped_feed(cations)
        aqueous = rc.aqueous_feed(cations)
    model = {SPECIES_LABELS[e]: n for e, n in aqueous.items()}
    if synthetic_softened:
        removed = {e: brine['mol'].get(e, 0) for e in ('Mg', 'Ca', 'Sr', 'Ba')}
        removed['Cl'] = 2 * sum(removed.values())
    conditioning = {'H': physical['H'], 'Cl': physical['H']} if synthetic_softened else {}
    if solvent in ('fresh', 'A') and not synthetic_softened:
        # #201 decision 11 supplies the acidity inventory already used by the owner, without changing its feed.
        conditioning = {'H': model['H'], 'Cl': model['H']}
        for e, n in conditioning.items():
            physical[e] = physical.get(e, 0.0) + n
    adjustment = {e: model.get(e, 0.0) - physical.get(e, 0.0) for e in physical.keys() | model.keys()}
    labels = brine['labels'] + record['labels']
    quantities = (('projected Li', model['Li'], 'mol/L aqueous', 'li_mol_l_aqueous'),
                  ('projected Na', model['Na'], 'mol/L aqueous', 'na_mol_l_aqueous'),
                  ('process pH', process_pH, 'dimensionless', 'hydrogen_mol_l_aqueous'))
    for quantity, value, unit, _ in quantities:
        labels.append(f'{quantity} = {value:.17g} {unit}: physical calibration coverage unverified for arbitrary brine; '
                      f'owner {record["chemistry_owner"]}')
    if solvent == 'fresh':
        bounds = json.loads((REPO_ROOT / record['domain_owner']).read_text())['input_bounds']
        for quantity, value, unit, key in quantities:
            compared = model['H'] if quantity == 'process pH' else value
            low, high = bounds[key]
            position = 'inside' if low <= compared <= high else 'outside'
            hydrogen = f' (model H = {compared:.17g} mol/L aqueous)' if quantity == 'process pH' else ''
            labels.append(f'{quantity} = {value:.17g} {unit}{hydrogen}: {position} current loaded-organic surrogate '
                          f'numerical interpolation interval [{low:.17g}, {high:.17g}] mol/L aqueous; '
                          f'owner {record["domain_owner"]}; not physical calibration coverage')
    elif solvent.startswith('D2EHDTPA'):
        labels.append(f'process pH = {process_pH:.17g} dimensionless: H/pH not represented by '
                      f'existing feed chemistry; owner {record["feed_owner"]}')
    if solvent in ('fresh', 'A'):
        labels += ['record feed uses raw-brine mass basis; R1/R2 retain their original nominal-feed conversions',
                   'nominal H and equal-Cl conditioning input: existing model H amount on one nominal '
                   'aqueous litre per raw litre; not measured inventory, activity-based pH or titrated acid demand',
                   'nominal acidity-conditioning HCl cost and buffering unquantified and excluded; '
                   'cost completeness covers only listed extraction and saponification accounts']
    if solvent.startswith('D2EHDTPA'):
        labels += ['changed model input: raw-brine source/mass basis; R6 keeps the original #188 feed and site basis']
    return {'basis': 'mol per raw litre', 'physical_feed': physical, 'model_feed': model,
            'model_composition_adjustment': adjustment, 'removed_at_no_cost': removed,
            'assumed_conditioning_input_mol_per_raw_l': conditioning,
            'synthetic_softened': synthetic_softened,
            'dropped_components': ['Br', 'SO4', 'B', 'H2S', 'Fe', 'Mn', 'Zn', 'NPOC'] if synthetic_softened else [],
            'water_mass_basis': '55.508 mol water per nominal litre (#216)' if synthetic_softened else
                'Reaktoro pitzer.dat H2O molar mass 0.018016 kg/mol',
            'process_pH': process_pH,
            'pH_basis': brine['pH_basis'] if solvent in ('fresh', 'A', 'C') else
                'process assumption; H/pH is not represented by the existing D2EHDTPA feed owner',
            'labels': labels}


def scope_costs(scope, lines, *, annual_li_kg, annual_raw_brine_m3):
    """Account already-priced USD/y lines; never present a partial sum as a total."""
    record = SCOPES[scope]
    if any(not math.isfinite(n) or n <= 0 for n in (annual_li_kg, annual_raw_brine_m3)):
        raise ValueError('annual Li and raw-brine denominators must be finite and positive')
    unexpected = lines.keys() - set(record['cost_items'])
    if unexpected:
        raise ValueError(f'cost items outside {scope}: {sorted(unexpected)}')
    missing, subtotal = [], 0.0
    for item in record['cost_items']:
        line = lines.get(item, {})
        amount = line.get('usd_y')
        if amount is None or not line.get('price_source'):
            missing.append(item)
        else:
            if not math.isfinite(amount) or amount < 0:
                raise ValueError(f'{item}: annual cost must be finite and nonnegative')
            subtotal += amount
    return {'lines': lines, 'li_denominator': record['li_denominator'], 'excluded': record['excluded'],
            'missing_items': missing, 'completeness': 'partial: ' + ', '.join(missing) if missing else 'complete',
            'completeness_basis': 'listed scope accounts only; exclusions remain unpriced',
            'priced_subtotal_usd_y': subtotal, 'total_usd_y': None if missing else subtotal,
            'priced_subtotal_usd_kg_li': subtotal / annual_li_kg,
            'priced_subtotal_usd_m3_raw_brine': subtotal / annual_raw_brine_m3,
            'can_minimize_total': not missing}
