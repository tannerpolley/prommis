"""#201 R1–R8: read exact retained rows and compare named replay quantities, without solves."""

import csv
import json
import math

from lithium_extraction.common.support.paths import REPO_ROOT
from .brines import CHARGE

MS2 = 'analyses/prototypes/hbta_topo_ms2_prototype/results'
BRINES = 'analyses/prototypes/produced_water_sx_brines/results'
D2 = 'analyses/prototypes/d2ehdtpa_buphen_ms2_prototype/results'
SOURCES = {
    'R1': 'analyses/v3/hbta_topo_ms2_idaes_optimization/results/live/current/design3/result.json',
    'R2': 'analyses/prototypes/hbta_topo_ms2_saponified/results/design_table.csv',
    'R3': f'{MS2}/full_chain_extraction.json',
    'R4': f'{MS2}/full_chain_results.json',
    'R5': f'{BRINES}/summary.csv',
    'R6': f'{D2}/ms2_costs.csv',
}


def _selected_csv(path, selector):
    with (REPO_ROOT / path).open() as stream:
        matches = [(number, row) for number, row in enumerate(csv.DictReader(stream), 2)
                   if all(row[key] == value for key, value in selector.items())]
    if len(matches) != 1:
        raise ValueError(f'{path}: expected one retained row for {selector}, found {len(matches)}')
    number, row = matches[0]
    return row, f'{path}:{number}; {selector}'


def load_reference(reference):
    """Return source rows plus (value, unit, absolute/relative tolerance) for later replay.

    Quantities have the retained field names, prefixed by case/stage where needed.
    R6 retains its extraction-plus-strip priced subtotal and original feed/site basis.
    R7 is a criterion, not a retained physical-raffinate result; R8 is unavailable.
    """
    if reference == 'R7':
        return {'reference': reference, 'balance_mol_per_raw_l_limit': 1e-9,
                'physical_charge_eq_l_limit': 1e-12, 'washed_feed_charge_eq_per_raw_l_limit': 1e-9,
                'extraction_charge_change_eq_per_raw_l_limit': 1e-12, 'strict_certificate_required': True,
                'physical_raffinate_required': True, 'model_composition_adjustment_required': True,
                'washed_physical_feed': 'extended washed outlet; never reconstruct from raw'}
    if reference == 'R8':
        return {'reference': reference, 'unavailable': 'not modeled (#193)'}
    path = SOURCES[reference]
    rows, locators, quantities = {}, {}, {}

    def retain(case, row, locator, fields):
        rows[case], locators[case] = row, locator
        for field, unit, mode, tolerance in fields:
            quantities[f'{case}/{field}'] = (float(row[field]), unit, mode, tolerance)

    if reference in ('R1', 'R3', 'R4'):
        document = json.loads((REPO_ROOT / path).read_text())
    if reference == 'R1':
        retain('live_optimization', document['live_optimization'], path + '#live_optimization',
               [('organic_to_aqueous_ratio', 'L organic/L aqueous', 'relative', 1e-6),
                ('extraction_screening_cost_usd_kg_li', 'USD/kg extracted Li', 'relative', 1e-6)])
    elif reference == 'R2':
        selector = {'unit': 'ms2_divalents_omitted_n4_ropt'}
        row, locator = _selected_csv(path, selector)
        retain(selector['unit'], row, locator,
               [('a_li_extraction', 'fraction', 'absolute', 1e-6),
                ('organic_to_aqueous_ratio', 'L organic/L aqueous', 'relative', 1e-6),
                ('a_extraction_plus_naoh_usd_kg_li', 'USD/kg extracted Li', 'relative', 1e-6)])
    elif reference == 'R3':
        for unit, case, end, concentration in (('0', 'base', 'low', .1),
                                               ('14', 'optimistic', 'high_r_0.70', .3)):
            row = document[unit]['values'][0]
            if (row['case'], row['end'], row['delta_case'], row['extractant_mol_l']) != (
                    case, end, 'fitted', concentration):
                raise ValueError(f'{path}: unit {unit} no longer identifies the retained R3 case')
            retain(case, row, f'{path}#{unit}/values/0',
                   [('li_extraction_fraction', 'fraction', 'absolute', 1e-6),
                    ('hbta_lost_fraction', 'fraction per pass', 'absolute', 1e-6)])
            for stage in row['stages']:
                prefix = f'{case}/stage{stage["stage"]}'
                for key, value in stage.items():
                    if '_out_mol_l' in key:
                        basis = 'mol/L organic' if '_org_' in key else 'mol/L aqueous'
                        quantities[f'{prefix}/{key}'] = (value, basis, 'absolute', 1e-9)
                quantities[f'{prefix}/water_autoprotolysis_active'] = (
                    stage['water_autoprotolysis_active'], 'support', 'exact', 0)
    elif reference == 'R4':
        for case in ('base', 'optimistic'):
            retain(case, document[case], f'{path}#{case}',
                   [('operating_usd_kg_li', 'USD/kg product Li', 'relative', 1e-6),
                    ('raw_li_recovery', 'fraction', 'absolute', 1e-6)])
    elif reference == 'R5':
        for brine in ('BR2', 'MarsHill', 'Westmoreland'):
            for case in ('base', 'optimistic'):
                row, locator = _selected_csv(path, {'feed': brine, 'case': case,
                                                   'scale': '1.0', 'lce_usd_kg': '20.9'})
                retain(f'{brine}/{case}', row, locator,
                       [('operating_usd_kg_li', 'USD/kg product Li', 'relative', 1e-6),
                        ('recovery_percent', 'percent', 'absolute', 1e-4)])
    elif reference == 'R6':
        row, locator = _selected_csv(path, {'calibration': 'fig7_exchange', 'B': '1.0', 'S': '1.2',
                                           'O_A': '1.0', 'N': '5', 'Sr_decade': '0', 'case': 'base',
                                           'h2s_protonated_mol_L_feed': '0.0'})
        retain('Held/MS2', row, locator,
               [('R_ext', 'fraction', 'absolute', 1e-6), ('R_rec', 'fraction', 'absolute', 1e-6),
                ('priced_subtotal_usd_kg_li', 'USD/kg strip-recovered Li; intermediate subtotal', 'relative', 1e-6)])
    result = {'reference': reference, 'rows': rows, 'source_locators': locators, 'quantities': quantities,
              'evidence': 'retained numerical replay values; not physical uncertainty or validation'}
    if reference == 'R1':
        result['input_owner'] = 'lithium_extraction.v3.hbta_topo_ms2_extraction.multistage:nominal_ms2_case'
    if reference == 'R2':
        identity = 'analyses/prototypes/hbta_topo_ms2_saponified/results/identity_ms2_divalents_omitted.json'
        result['input_locator'] = identity + '#feed'
        result['feed'] = json.loads((REPO_ROOT / identity).read_text())['feed']
    if reference == 'R6':
        result['feed_owner'] = 'lithium_extraction.prototypes.produced_water_sx_prototype.d2ehdtpa_buphen:ms2_feed'
        result['site_basis'] = json.loads((REPO_ROOT / D2).parent.joinpath('inputs/stage2_basis.json').read_text())
        result['accounting'] = 'historical #188 extraction-plus-strip priced subtotal; neither user scope'
    return result


def compare_quantities(reference, actual):
    """List absent/nonfinite/mismatched quantities; no tolerance relaxation or solve."""
    expected = load_reference(reference)
    if 'quantities' not in expected:
        raise ValueError(f'{reference} uses criteria or an unavailable label, not numerical replay quantities')
    failures = []
    for name, (value, unit, mode, tolerance) in expected['quantities'].items():
        candidate = actual.get(name)
        if mode == 'exact':
            passed = type(candidate) is type(value) and candidate == value
        else:
            passed = (isinstance(candidate, (int, float)) and math.isfinite(candidate)
                      and abs(candidate - value) <= (tolerance * abs(value) if mode == 'relative' else tolerance))
        if not passed:
            failures.append(f'{name}: {candidate!r} versus {value!r} {unit} ({mode} {tolerance})')
    return failures


def physical_raffinate(feed, model_raffinate):
    """R_phys = F_phys + (R_model - F_model), including unchanged uncarried ions."""
    physical, model = feed['physical_feed'], feed['model_feed']
    return {e: physical.get(e, 0.0) + (model_raffinate.get(e, 0.0) - model.get(e, 0.0))
            for e in physical.keys() | model.keys() | model_raffinate.keys()}


def check_r7(feed, model_raffinate, *, balance_residuals_mol_per_raw_l, strict_certificate,
             charge_limit_eq_per_raw_l=None):
    """Check supplied owner-certified quantities; cannot certify an equilibrium state itself."""
    physical, model = feed['physical_feed'], feed['model_feed']
    charge_limit = (load_reference('R7')['physical_charge_eq_l_limit']
                    if charge_limit_eq_per_raw_l is None else charge_limit_eq_per_raw_l)
    required = set(model)
    if required - model_raffinate.keys():
        raise ValueError(f'model raffinate is incomplete: {sorted(required - model_raffinate.keys())}')
    if not balance_residuals_mol_per_raw_l:
        raise ValueError('R7 requires owner balance residuals on the raw-litre basis')
    errors = []
    if strict_certificate is not True:
        errors.append('owner strict certificate does not hold')
    if any(not math.isfinite(n) or abs(n) > 1e-9 for n in balance_residuals_mol_per_raw_l.values()):
        errors.append('balance exceeds 1e-9 mol per raw litre or is nonfinite')
    adjustment = {e: model.get(e, 0.0) - physical.get(e, 0.0) for e in physical.keys() | model.keys()}
    if feed.get('model_composition_adjustment') != adjustment:
        errors.append('signed model composition adjustment is absent or inconsistent')
    raffinate = physical_raffinate(feed, model_raffinate)
    if any(not math.isfinite(n) or n < 0 for n in raffinate.values()):
        errors.append('physical raffinate is negative or nonfinite')
    charges = feed.get('physical_component_charge', CHARGE)
    charge = math.fsum(charges[e] * n for e, n in raffinate.items())
    feed_charge = math.fsum(charges[e] * n for e, n in physical.items())
    difference = charge - feed_charge
    if 'washed_feed_l_per_raw_l' in feed:
        limit = load_reference('R7')['washed_feed_charge_eq_per_raw_l_limit']
        if feed['aqueous_charge_tolerance_eq_per_raw_l'] != limit:
            errors.append('washed feed charge tolerance disagrees with the softening owner limit')
        if not math.isfinite(feed_charge) or abs(feed_charge) > limit:
            errors.append('washed feed charge exceeds the softening owner limit or is nonfinite')
        if not math.isfinite(difference) or abs(difference) > charge_limit:
            errors.append(f'extraction charge change exceeds {charge_limit:g} eq per raw litre or is nonfinite')
    elif not math.isfinite(charge) or abs(charge) > charge_limit:
        errors.append(f'physical raffinate charge exceeds {charge_limit:g} eq/L or is nonfinite')
    return {'physical_raffinate': raffinate, 'charge_eq_l': charge, 'failures': errors,
            'feed_charge_eq_per_raw_l': feed_charge, 'extraction_charge_change_eq_per_raw_l': difference,
            'charge_limit_eq_per_raw_l': charge_limit}
