"""Connected #176 scrub, strip and Ge carbonation units on the extraction stages.

Scrub and strip are fixed-fraction PrOMMiS SolventExtraction stages; carbonation is an IDAES Mixer,
PrOMMiS Precipitator and IDAES Separator recycle loop. The historical owner remains unchanged and
independently evaluates every solved quantity and cost account. Downstream flows are mol/h at 1 L/h
raw brine (mol per raw litre); water is kg/raw L.
"""

import csv
import math

from pyomo.common.collections import ComponentSet
from pyomo.contrib.incidence_analysis import solve_strongly_connected_components
from pyomo.environ import (Constraint, Expression, NonNegativeReals, Var,
                           units as pyunits, value)
from pyomo.network import Arc
from pyomo.util.check_units import assert_units_consistent
from pyomo.util.subsystems import TemporarySubsystemManager, create_subsystem_block

from . import brines, carbonation, chain, costing, downstream, flowsheet, records, references

def extraction(model):
    """Transport the generic boundary amounts to the historical cost owner's named inputs."""
    v = model._washed['feed_l']
    aq = model.fs.stage[len(model.fs.stages)].outlet
    org = model.fs.stage[1].outlet
    free = 'free-carrier' if 'free-carrier' in org else 'hbta'
    return {'raffinate': {'li': value(aq['lithium-cation']), 'na': value(aq['sodium-cation']),
                         'bta': value(aq['bta-anion']) if 'bta-anion' in aq else 0.},
            'loaded_organic': {'li': value(org['lithium-carrier']), 'na': value(org['sodium-carrier']),
                              'hbta': sum(value(org[s]) for s in (free, 'lithium-carrier', 'sodium-carrier'))},
            'diluent_mol': value(org['diluent']) * v,
            'saponification_mol_per_raw_l': model._feeds[1].get('sodium-carrier', 0.) * v,
            'organic_volume_l_per_raw_l': model._design['organic_to_aqueous_ratio'] * v}


def enumerate_units(washed, loaded, assumptions, *, organic_to_aqueous_ratio=1.):
    """Enumerate both scrub bounds and every measured Ge interval before calling IPOPT.

    Fixed-design screening discards only analytically inadmissible alternatives;
    released extraction decisions require solving those alternatives in step 7.
    """
    v = washed['feed_l']
    li, na = (v * loaded[k] for k in ('li', 'na'))
    source = assumptions['scrub_hcl_mol_l'] * organic_to_aqueous_ratio * v / 10
    demand = assumptions['scrub_li_loss'] * li + assumptions['scrub_na_efficiency'] * na
    strip_eff = float(flowsheet._row('data/reference/extraction_models/hbta_topo_li_na/observations.csv',
        'observation_id', 'zhang2017_fig6_li_strip_2p5m_hcl')['value']) / 100
    stripped_li = (1 - assumptions['scrub_li_loss']) * li * strip_eff
    stripped_na = (1 - assumptions['scrub_na_efficiency']) * na
    gross = 2.5 * organic_to_aqueous_ratio * v / 10
    fresh = stripped_li + stripped_na + (1 - assumptions['acid_recovery']) * (gross - stripped_li - stripped_na)
    salt = fresh / (stripped_li / assumptions['concentrate_li_mol_kg']) if stripped_li > 0 else None
    with carbonation.GE_TABLE.open() as stream:
        knots = [float(r['nacl_mol_kg']) for r in csv.DictReader(stream) if r['series'] == 'NaCl']
    candidates = []
    for regime, acid, other in (('source', source, demand), ('stoichiometric', demand, source)):
        for lo, hi in zip(knots, knots[1:]):
            reasons = []
            if acid < other:
                reasons.append('scrub acid below the other bound')
            if gross < stripped_li + stripped_na:
                reasons.append('strip acid inventory below metal proton demand')
            if salt is None or not math.isfinite(salt) or not 0 <= salt <= 3:
                reasons.append('carbonation outside the measured solubility data')
            elif not lo <= salt <= hi:
                reasons.append('NaCl outside this interpolation interval')
            else:
                s = carbonation.empirical_solubility(assumptions['carbonation_temperature_k'], salt)
                if stripped_li - 2 * s * stripped_li / assumptions['concentrate_li_mol_kg'] <= 0:
                    reasons.append('concentration input leaves no Li2CO3 product')
            candidates.append({'scrub_regime': regime, 'carbonation_interval_mol_kg': [lo, hi],
                               'mother_nacl_mol_kg': salt, 'scrub_acid_mol': acid,
                               'scrub_other_bound_mol': other,
                               'feasible': not reasons, 'reasons': reasons})
    return candidates


def connect_units(model, case, assumptions, candidate, *, initial_accounts=None):
    """Write the owner's algebra with an explicit active scrub bound and fixed Ge interval."""
    washed, v = model._washed, model._washed['feed_l']
    initial = extraction(model)
    assumptions = dict(assumptions, extractant_mol_l=model._design['extractant_mol_l'])
    owner = (flowsheet.costs_and_balances(washed, initial, case, assumptions)
             if initial_accounts is None else initial_accounts)
    model._full_process = {'case': case, 'assumptions': assumptions, 'candidate': candidate}
    fs = model.fs
    fs.washed_feed_l_per_raw_l.set_value(v)
    org = fs.stage[1].outlet
    free = 'free-carrier' if 'free-carrier' in org else 'hbta'
    ratio = model._decisions['organic_to_aqueous_ratio'] if hasattr(model, '_decisions') else model._design['organic_to_aqueous_ratio']
    # Scrub and strip: one PrOMMiS stage each; the source fraction of entering metal-loaded organic is the only transfer closure.
    metals = {'Li': 'lithium', 'Na': 'sodium'}
    fractions = {'scrub': {'Li': assumptions['scrub_li_loss'], 'Na': assumptions['scrub_na_efficiency']},
                 'strip': {'Li': owner['strip_li_fraction'], 'Na': 1.}}
    for name, fraction in fractions.items():
        downstream.fixed_fraction_strip(fs, name, {e: f'{m}-carrier' for e, m in metals.items()},
            {e: f'{m}-cation' for e, m in metals.items()}, dict.fromkeys(metals, 1.),
            {e: {free: 1.} | ({'topo': 1.} if free == 'hbta' else {}) for e in metals}, fraction, spectators=('diluent',))
    scrub, strip = fs.scrub, fs.strip
    scrub.loaded_organic = Constraint(fs.loaded_properties.component_list, rule=lambda b, s:
                                      scrub.unit.mscontactor.organic_inlet_state[0].flow_mol_comp[s] == v * org[s])
    scrub.waste = Expression(tuple(metals), rule=lambda b, e: b.aqueous_out[f'{metals[e]}-cation'])
    source = assumptions['scrub_hcl_mol_l'] * ratio * v / 10
    demand = sum(scrub.waste.values())
    scrub.acid_mol = Var(domain=NonNegativeReals, initialize=owner['scrub_acid_mol'])
    scrub.acid = Constraint(('hydrogen-cation', 'chloride-anion'), rule=lambda b, s:
                            scrub.unit.mscontactor.aqueous_inlet_state[0].flow_mol_comp[s] == b.acid_mol)
    active, other = (source, demand) if candidate['scrub_regime'] == 'source' else (demand, source)
    scrub.active_bound = Constraint(expr=scrub.acid_mol == active)
    scrub.other_bound = Constraint(expr=scrub.acid_mol >= other)
    strip.scrubbed_organic = Arc(source=scrub.unit.organic_outlet, destination=strip.unit.organic_inlet)
    strip.gross_acid_mol = Expression(expr=2.5 * ratio * v / 10)
    strip.acid = Constraint(('hydrogen-cation', 'chloride-anion'), rule=lambda b, s:
                            strip.unit.mscontactor.aqueous_inlet_state[0].flow_mol_comp[s] == b.gross_acid_mol)
    strip.metals = Expression(tuple(metals), rule=lambda b, e: b.aqueous_out[f'{metals[e]}-cation'])
    strip.acid_sufficiency = Constraint(expr=strip.aqueous_out['hydrogen-cation'] >= 0)
    downstream.liquor_properties(fs)
    concentrate = chain.aqueous(fs, assumptions, ratio * v, strip.metals['Li'])
    chain.organic(fs, model, assumptions, v, free)
    downstream.connect(fs, concentrate, assumptions, candidate, owner)
    carbon = fs.carbonation
    _initialize_units(fs, owner, assumptions['mother_recycle'])
    # The original owner supplies all prices and kg/mol conversions, including 31% HCl.
    quantities = {'saponification_NaOH': fs.saponification.naoh_mol,
        'scrub_HCl31': scrub.acid_mol, 'strip_HCl31': strip.fresh_acid_mol,
        'neutralization_NaOH': fs.neutralization.naoh_mol, 'carbonation_Na2CO3': carbon.carbonate_mol,
        'HBTA': fs.organic_loss.hbta_mol, 'TOPO': fs.organic_loss.topo_mol, 'diluent': fs.organic_loss.diluent_mol}
    coefficients = {k: flowsheet.MASS_KG_PER_MOL[
        'NaOH' if 'NaOH' in k else 'Na2CO3' if 'Na2CO3' in k else 'HCl31' if 'HCl31' in k else k]
        for k in quantities}
    costing.attach(fs, {k: ((quantities[k] * coefficients[k] if k in quantities else mass) * costing.RAW_L_H * pyunits.kg,
                            owner['prices_usd_kg'][k] * costing.usd_per(pyunits.kg))
                        for k, mass in owner['mass_kg_per_l'].items()},
                   carbon.product_mol * (owner['li_product_kg_m3'] / 1000 / owner['product_li2co3_mol']) *
                   costing.RAW_L_H * pyunits.kg)
    assert_units_consistent(model)


def _initialize_units(fs, owner, recycle):
    """Scrub, strip and concentrate follow the live extraction; the recycle loop starts at the owner's closed form."""
    # References repeat unit variables, so the set keeps each once.
    units = tuple(getattr(fs, name) for name in ('scrub', 'strip', 'acid_recovery', 'strip_liquid', 'neutralization',
        'neutral_projection', 'concentration', 'regeneration', 'regenerated', 'organic_loss', 'organic_makeup',
        'unloaded_organic', 'saponification'))
    variables = ComponentSet(v for b in (*units, fs.carbonation.mixer.concentrate_state[0])
                             for v in b.component_data_objects(Var) if not v.fixed)
    equalities = [c for b in (*units, fs.carbonation.concentrate_feed.expanded_block)
                  for c in b.component_data_objects(Constraint, active=True) if c.equality]
    block = create_subsystem_block(equalities, list(variables))
    with TemporarySubsystemManager(to_fix=list(block.input_vars.values())):
        solve_strongly_connected_components(block)
    downstream.initialize(fs, owner, recycle)


def select_units(model, candidate):
    """Set the active bound and measured segment after inspecting a converged extraction."""
    fs, a = model.fs, model._full_process['assumptions']
    source = a['scrub_hcl_mol_l'] * model._decisions['organic_to_aqueous_ratio'] * model._washed['feed_l'] / 10
    demand = sum(fs.scrub.waste.values())
    active, other = (source, demand) if candidate['scrub_regime'] == 'source' else (demand, source)
    fs.scrub.active_bound.set_value(fs.scrub.acid_mol == active)
    fs.scrub.other_bound.set_value(fs.scrub.acid_mol >= other)
    fs.scrub.other_bound.activate()
    lo, hi = candidate['carbonation_interval_mol_kg']
    fs.carbonation.interval.set_value((lo, fs.carbonation.nacl_mol_kg, hi))
    fs.carbonation.interval.activate()
    s0, s1 = (carbonation.empirical_solubility(a['carbonation_temperature_k'], salt) for salt in (lo, hi))
    fs.carbonation.interpolation.set_value(fs.carbonation.solubility_mol_kg ==
        s0 + (s1 - s0) * (fs.carbonation.nacl_mol_kg - lo) / (hi - lo))


def report(model, result, softening):
    """Retain solved unit inventories, historical comparisons, complete physical R7 and costs."""
    data, fs = model._full_process, model.fs
    washed, v = model._washed, model._washed['feed_l']
    assumptions = dict(data['assumptions'], extractant_mol_l=model._design['extractant_mol_l'])
    extracted = extraction(model)
    owner = flowsheet.costs_and_balances(washed, extracted, data['case'], assumptions)
    owner['organic_volume_l_per_raw_l'] = extracted['organic_volume_l_per_raw_l']
    owner['strip_water_recycle_fraction'] = 1 - owner['fresh_water_kg'] / (owner['organic_volume_l_per_raw_l'] / 10)
    if model._design['organic_to_aqueous_ratio'] != 1.:
        owner['evaporation_alternative_mj_m3_latent_only'] = None
    quantities = {'scrub_acid_mol': fs.scrub.acid_mol, 'gross_strip_acid_mol': fs.strip.gross_acid_mol,
        'fresh_strip_acid_mol': fs.strip.fresh_acid_mol, 'neutralization_naoh_mol': fs.neutralization.naoh_mol,
        'carbonation_na2co3_mol': fs.carbonation.carbonate_mol,
        'product_li2co3_mol': fs.carbonation.product_mol, 'ge_li2co3_mol_kg': fs.carbonation.solubility_mol_kg,
        'mother_nacl_mol_kg': fs.carbonation.nacl_mol_kg, 'reactor_water_kg': fs.carbonation.water_kg,
        'fresh_water_kg': fs.concentration.net_water_kg, 'operating_usd_kg_li': fs.costing.usd_kg_li,
        'raw_li_recovery': 2 * fs.carbonation.product_mol / (
            washed['front_end']['original_raw_elements_mol']['Li'] if washed.get('front_end') else washed['raw_mol']['Li']),
        'li_product_kg_m3': fs.costing.li_kg_h}
    actual = {k: value(n) for k, n in quantities.items()}
    if max(abs(value(n)) for n in fs.saponification.prepared_difference.values()) > 1e-9:
        raise RuntimeError('Prepared organic differs from the extraction inlet')
    differences = {k: actual[k] - owner[k] for k in actual}
    if any(not math.isfinite(n) or abs(n) > 1e-9 for k, n in differences.items() if k != 'operating_usd_kg_li'):
        raise RuntimeError(f'Connected units differ from #176 owner: {differences}')
    reconciliation = costing.reconcile(fs.costing, owner['cost_usd_m3'], owner['operating_usd_kg_li'],
                                       owner['li_product_kg_m3'], owner['mass_kg_per_l'])
    actual['cost_usd_m3'] = reconciliation['usd_m3']
    balances = dict(result['balance_residuals_mol_per_raw_l'])
    balances.update({'softening/' + k: n for k, n in washed['closure_mol'].items()})
    balances.update({'chain/' + k: n for k, n in owner['closure_mol'].items()})
    for b in (fs.scrub, fs.strip, fs.carbonation):
        for c in b.component_data_objects(Constraint, active=True):
            if c.equality:
                balances[c.name] = value(c.body - c.lower)
    final = result['outlet_mol_per_nominal_l']
    raffinate = dict(result['feed']['model_feed'])
    raffinate.update({records.SPECIES_LABELS[s]: final[s] * v for s in model._chemistry.supports[0]})
    r7 = references.check_r7(result['feed'], raffinate,
        balance_residuals_mol_per_raw_l=balances, strict_certificate=result['strict_certificate'])
    stripped = {e: value(fs.strip.metals[e]) for e in ('Li', 'Na')}
    physical = {'washed extraction feed': result['feed']['physical_feed'], 'physical raffinate': r7['physical_raffinate'],
        'saponified organic input': {s: n * v for s, n in model._feeds[1].items()},
        'loaded organic': {s: final[s] * v for s in model._chemistry.supports[1]},
        'saponification NaOH input': {'Na': value(fs.saponification.naoh_mol), 'OH': value(fs.saponification.naoh_mol)},
        'scrub HCl input': {'H': actual['scrub_acid_mol'], 'Cl': actual['scrub_acid_mol']},
        'scrub waste': {e: value(fs.scrub.waste[e]) for e in ('Li', 'Na')} |
            {'H': (0. if data['candidate']['scrub_regime'] == 'stoichiometric' else
                  actual['scrub_acid_mol'] - sum(value(n) for n in fs.scrub.waste.values())), 'Cl': actual['scrub_acid_mol']},
        'gross strip HCl input': {'H': actual['gross_strip_acid_mol'], 'Cl': actual['gross_strip_acid_mol']},
        'recovered unused strip HCl': {'H': actual['gross_strip_acid_mol'] - actual['fresh_strip_acid_mol'],
                                     'Cl': actual['gross_strip_acid_mol'] - actual['fresh_strip_acid_mol']},
        'strip liquor before neutralization': stripped | {'H': actual['neutralization_naoh_mol'], 'Cl': actual['fresh_strip_acid_mol']},
        'regeneration lithium discharge': {'Li': value(fs.strip.regeneration_li)},
        'neutralization NaOH input': {'Na': actual['neutralization_naoh_mol'], 'OH': actual['neutralization_naoh_mol']},
        'carbonation soda ash input': {'Na': 2 * value(fs.carbonation.carbonate_mol), 'C': value(fs.carbonation.carbonate_mol)},
        'Li2CO3 product': {'Li': 2 * actual['product_li2co3_mol'], 'C': actual['product_li2co3_mol']},
        'mother liquor': {e: value(fs.carbonation.mother[e]) for e in ('Li', 'Na', 'C')},
        'mother liquor purge': {e: value(fs.carbonation.purge[e]) for e in ('Li', 'Na', 'C')},
        'mother liquor recycle': {e: value(fs.carbonation.mother[e] - fs.carbonation.purge[e]) for e in ('Li', 'Na', 'C')}}
    if washed.get('front_end'):
        physical.update({'front-end ' + ('internal wash return' if s == 'wash' else s): a
                         for s, a in washed['front_end']['streams_mol_per_raw_l'].items()})
    for name, state in (('organic loss', fs.organic_loss.loss_state[0]),
                        ('regenerated organic', fs.organic_loss.retained_state[0])):
        physical[name] = {j: value(state.flow_mol_comp[j]) for j in ('BTA_total', 'TOPO_total', 'diluent')}
    physical['organic make-up input'] = {label: value(fs.organic_makeup.fresh_state[0].flow_mol_comp[j])
        for label, j in (('HBTA', 'BTA_total'), ('TOPO', 'TOPO_total'), ('diluent', 'diluent'))}
    margins = {c.name: {'lower': None if c.lower is None else value(c.body - c.lower),
                       'upper': None if c.upper is None else value(c.upper - c.body)}
               for c in model.component_data_objects(Constraint, active=True) if not c.equality}
    if any(n < -1e-9 for row in margins.values() for n in row.values() if n is not None):
        raise RuntimeError(f'Full-process inequality violated: {margins}')
    return complete_reporting(result | {'softening': softening, 'case': data['case'],
        'full_process': owner | actual, 'R7': r7, 'balance_residuals_mol_per_raw_l': balances,
        'physical_streams_mol_per_raw_l': physical,
        'softening_streams_elements_mol_per_raw_l': {k: washed[k] for k in ('raw_mol', 'cake_mol', 'solid_mol', 'wash_elements_mol')} |
            {'reagent_elements_mol': washed['reagent_elements_mol']},
        'owner_differences': differences, 'idaes_costing_reconciliation': reconciliation, 'inequality_margins': margins, 'unit_enumeration_selected': data['candidate'],
        'labels': [s for s in result['labels'] if 'no full-process solve' not in s] + [
            'fixed-design model prediction and numerical replay; no physical validation',
            '#176 accounts per kg Li in Li2CO3; capital, labour and energy excluded',
            'scrub/strip laboratory fractions transferred to produced water; no finished-product assay',
            'downstream H/O and regeneration counterions unquantified in the #176 accounting',
            data['assumptions']['organic_loss_provenance']]}, fs)


def complete_reporting(result, fs):
    """Expose known #176 inventories and washed-feed coverage without rerunning chemistry."""
    washed = result['softening'][result['case']]
    unit = result['full_process']
    physical = result['physical_streams_mol_per_raw_l']
    purge_cl = unit['fresh_strip_acid_mol']
    mother_cl = unit['mother_nacl_mol_kg'] * unit['reactor_water_kg']
    for name, chloride in (('mother liquor', mother_cl), ('mother liquor purge', purge_cl),
                          ('mother liquor recycle', mother_cl - purge_cl)):
        physical[name]['Cl'] = chloride
    negative = {f'{name}/{s}': n for name, stream in physical.items() for s, n in stream.items()
                if not math.isfinite(n) or n < 0}
    if negative:
        raise RuntimeError(f'Negative or nonfinite physical full-process stream: {negative}')
    net, circulating = unit['fresh_water_kg'], unit['reactor_water_kg']
    result['water_streams'] = {
        'raw feed': {'kg_per_raw_l': result['softening']['raw_feed']['water_kg'],
                     'basis': 'source-record water mass; brines.raw_feed'},
        'softening wash input': {'kg_per_raw_l': washed['wash_water_mol'] * brines.WATER_MOLAR_MASS_KG,
                                 'basis': 'pure wash water, 1 kg/L; Reaktoro H2O molar mass'},
        'washed aqueous H2O species': {
            'kg_per_raw_l': washed['aqueous_species_mol']['H2O'] * brines.WATER_MOLAR_MASS_KG,
            'basis': 'retained Reaktoro H2O species; distinct from analytical water component'},
        'gross strip liquor': {'kg_per_raw_l': value(fs.strip_liquid.properties_out[0].flow_mol_phase_comp['Liq', 'H2O']) * 3600 * brines.WATER_MOLAR_MASS_KG,
                               'basis': '#176 nominal strip-water accounting, 1 kg/L at organic/aqueous 10'},
        'strip water recycle': {'kg_per_raw_l': value(fs.concentration.recycle_state[0].flow_mol_comp['H2O']) * brines.WATER_MOLAR_MASS_KG,
                                'basis': '#176 declared strip-water recycle; energy/equipment excluded'},
        'net water to carbonation': {'kg_per_raw_l': net,
                                     'basis': '#176 fresh stripped Li / concentrate_li_mol_kg; not external fresh water'},
        'mother liquor': {'kg_per_raw_l': circulating, 'basis': '#176 reactor water at fixed mother recycle'},
        'mother liquor purge': {'kg_per_raw_l': net, 'basis': '#176 net water leaves in the mother purge'},
        'mother liquor recycle': {'kg_per_raw_l': circulating - net,
                                  'basis': '#176 circulating water minus purge water'},
        **{name: {'kg_per_raw_l': None, 'basis': 'not quantified by #176'} for name in (
            'scrub water', 'recovered acid water', 'cake liquid H2O species', 'solid H2O species',
            'Li2CO3 product moisture', 'organic water')}}
    feed = result['feed']
    if washed.get('front_end'):
        result['water_streams']['front-end fresh water'] = dict(
            kg_per_raw_l=washed['front_end']['water_input_kg_per_raw_l'], basis='H-D0 V1/V7; wash return is internal')
    record = records.SOLVENTS[result.get('solvent', 'C')]
    result['calibration_coverage'] = [
        {'quantity': quantity, 'value': number, 'unit': unit_name,
         'basis': 'softened, washed aqueous inlet; ' + basis,
         'source_owner': record['feed_owner'] if quantity != 'process pH' else
                         'analyses.prototypes.hbta_topo_ms2_prototype.scripts.countercurrent:acidity_at_ph',
         'chemistry_owner': record['chemistry_owner'], 'coverage_owner': record['domain_owner'],
         'comparison': 'physical calibration coverage unverified',
         'comparison_basis': 'existing owner establishes no quantitative physical-calibration bounds for arbitrary brine; '
                             'source-data extrema and numerical interpolation are separate'}
        for quantity, number, unit_name, basis in (
            ('projected Li', feed['model_feed']['Li'] / washed['feed_l'], 'mol/L washed aqueous',
             'owner model projection divided by retained washed volume'),
            ('projected Na', feed['model_feed']['Na'] / washed['feed_l'], 'mol/L washed aqueous',
             'owner model projection divided by retained washed volume'),
            ('process pH', washed['pH'], 'dimensionless',
             'retained softening pH conditioning; separate from raw measured pH'))]
    return result
