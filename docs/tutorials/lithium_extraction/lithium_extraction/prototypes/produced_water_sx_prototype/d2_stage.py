"""D2EHDTPA/BuPhen fixed simulation on #188 Held or #197 conditional Figiel chemistry."""

import json
import math
from pathlib import Path

import epcsaft
from epcsaft import equilibrium as eq
import numpy as np

from analyses.prototypes.d2ehdtpa_buphen_ms2_prototype.scripts import reaction_coordinates as rc
from . import brines, d2ehdtpa_buphen as owner, records, references, stage
from analyses.prototypes.d2ehdtpa_buphen_ms2_prototype.scripts import conditional_figiel_packet as ion_basis
from lithium_extraction.common.support.paths import ENGINE_ASL_LIBRARY

CONTINUATION = eq.Continuation(initial_step=.25, minimum_step=1/64, max_steps=64, max_retries_per_step=4)
REFERENCE = owner.ANALYSIS / 'results/ms2_0908.json'


def reference_feed(solvent='D2EHDTPA-Held'):
    """Original #188 cation conversions, preserved source fields and coherent charge closure."""
    original = ion_basis.MS2_FEED()
    raw = brines.raw_feed('MS2', reference_cations=original)
    raw['reference_conversion_basis'] = {'owner': 'd2ehdtpa_buphen.ms2_feed',
        'molar_masses_kg_mol': {e: owner.MASSES[e] for e in original},
        'site_basis_li_mol_l': owner.BASIS['site_basis_li_mol_L']}
    raw['labels'].append('R6 original nominal #188 cation conversions; separate 0.024204 M site basis')
    feed = records.extraction_feed(raw, solvent)
    feed['raw_brine'] = raw
    feed['labels'] = [label for label in feed['labels'] if not label.startswith('changed model input:')]
    return feed


def build_flowsheet(brine, solvent='D2EHDTPA-Held', scope='extraction_only', stages=5, scenario=None):
    """One basis per process; chemistry assembly and direct certificates remain source-owned."""
    if solvent not in ('D2EHDTPA-Held', 'D2EHDTPA-Figiel') or scope not in records.SCOPES:
        raise ValueError('Unknown D2EHDTPA solvent or process scope')
    if not scenario or set(scenario) != {'B', 'S', 'O_A', 'Sr_decade', 'site_basis_li_mol_l', 'ln_K_Li_Mg_Ca'}:
        raise ValueError('fixed simulation requires B, S, O/A, Sr coordinate, site basis and source ln K')
    b, supply, ratio, site_basis = (scenario[k] for k in ('B', 'S', 'O_A', 'site_basis_li_mol_l'))
    sites = owner.site_supply(b, supply, ratio, stages, site_basis_li_mol_l=site_basis)
    if not all(math.isfinite(n) for n in (b, supply, ratio, sites, scenario['Sr_decade'], *scenario['ln_K_Li_Mg_Ca'])):
        raise ValueError('fixed design quantities must be finite')
    feed = dict(brine) if 'physical_feed' in brine else records.extraction_feed(brine, solvent, scope)
    if 'physical_feed' not in brine:
        feed['raw_brine'] = brine
    if 'unavailable' in feed:
        raise ValueError(feed['unavailable'])
    ion_basis.use('packet' if solvent == 'D2EHDTPA-Figiel' else 'held_explicit_sr')
    cations = {e: feed['model_feed'].get(e, 0) for e in rc.ION if e != 'Cl'}
    constants = scenario['ln_K_Li_Mg_Ca']
    mixture, problem, aqueous, organic, species, _, _, _ = rc.build_problem(
        cations, sites, b, constants, organic_to_aqueous=ratio, sr_decade=scenario['Sr_decade'])
    totals = {'sites_total_mol': sites * ratio, 'buphen_total_mol': b * ratio}
    controls = tuple(s for s in species if s not in (rc.ION['Cl'][0], 'water', 'naa', 'buphen')) + tuple(totals)
    charges = tuple(next((z for name, _, z in rc.ION.values() if name == s), 0) for s in species)
    inventory = problem.feed.values
    matrix, offset = [], []
    for s in species:
        row = {q: float(s == q) for q in controls}
        constant = 0.
        if s == 'water':
            constant, row = inventory[s], dict.fromkeys(controls, 0.)
        elif s in ('naa', 'buphen'):
            moiety = rc.MOIETIES['A' if s == 'naa' else 'B']
            row = {q: -moiety.get(q, 0) for q in controls}
            row['sites_total_mol' if s == 'naa' else 'buphen_total_mol'] = 1.
        elif s == rc.ION['Cl'][0]:
            row = {q: charges[species.index(q)] if q in species else 0. for q in controls}
        offset.append(constant)
        matrix.append(tuple(row[q] for q in controls))
    anchor = tuple((inventory | totals)[s] for s in controls)
    if max(abs(a + sum(x * y for x, y in zip(row, anchor, strict=True)) - inventory[s])
           for s, a, row in zip(species, offset, matrix, strict=True)) > 1e-12:
        raise ValueError('reduced controls do not reproduce the source inventory')
    offset, matrix = tuple(offset), tuple(matrix)
    record = eq.asl_record(mixture, problem, CONTINUATION, offset, matrix,
                           anchor, species.index('water'), .001)
    conserved, rows, omitted = [], [], {}
    for name, row in rc.MOIETIES.items():
        values = tuple(row.get(s, 0) for s in species)
        if not any(values):
            omitted[name] = 'structurally absent from source-selected species support'
        elif np.linalg.matrix_rank(np.asarray([*rows, values])) > len(rows):
            rows.append(values)
            conserved.append(name)
        else:
            omitted[name] = 'dependent on preceding source moiety rows on this support'

    def certify(inlet, outlet):
        v = inlet['diluent'] / (organic['diluent'] / ratio)
        total_sites = sum(rc.MOIETIES['A'].get(s, 0) * n for s, n in inlet.items()) / v
        total_b = sum(rc.MOIETIES['B'].get(s, 0) * n for s, n in inlet.items()) / v
        loaded = {e: inlet[c] / v for e, c in rc.COMPLEX.items() if c in inlet}
        carried = {e: inlet[rc.ION[e][0]] for e in cations if rc.ION[e][0] in inlet}
        direct = rc.solve(carried, total_sites, total_b, constants, organic_to_aqueous=v,
                          loading=loaded, sr_decade=scenario['Sr_decade'])
        difference = max(abs(outlet[s] - sum(p.get(s, 0) for p in direct['amounts_mol'])) for s in species)
        return {'status': 'evaluated', **direct['certificate'], 'direct_amounts_mol': direct['amounts_mol'],
                'D': direct['D'], 'outlet_max_difference_mol_per_raw_l': difference}

    chemistry = stage.Chemistry(species, (tuple(aqueous), tuple(organic)), tuple(rows), charges, controls,
                                offset, matrix, record,
                                Path(epcsaft.__file__).resolve().parent / ENGINE_ASL_LIBRARY, certify,
                                parameters=tuple(totals.items()))
    model = stage.build_stages(chemistry, aqueous, organic, stages)
    model._physical_feed, model._design, model._solvent = feed, dict(scenario), solvent
    model._conserved_names, model._omitted_moieties = conserved, omitted
    return model


def reference_record(solvent='D2EHDTPA-Held'):
    path = REFERENCE if solvent == 'D2EHDTPA-Held' else REFERENCE.parent / 'conditional_figiel_packet/reference_packet_fig7_exchange.json'
    return path, json.loads(path.read_text())


def strip_accounts(accounts, solvent):
    """Figiel's selected ion basis lumps Sr into Ca; every other strip amount is required."""
    return accounts | {'strip_Sr_mol_L_feed': 0.} if solvent == 'D2EHDTPA-Figiel' else accounts


def reference_profiles(model):
    """Complete phase amounts from the exact accepted fixed-design cascade, without refitting."""
    _, retained = reference_record(model._solvent)
    case = retained['case']
    if len(model.fs.stages) != case['N'] or any(model._design[k] != case[k] for k in ('B', 'S', 'O_A', 'Sr_decade')):
        raise ValueError('retained profiles apply only to the original fixed design')
    if model._design['ln_K_Li_Mg_Ca'] != retained['ln_K_Li_Mg_Ca']:
        raise ValueError('retained profiles require their original source coordinates')
    return [{s: sum(p.get(s, 0) for p in row['amounts_mol']) for s in model._chemistry.species}
            for row in retained['extraction']['stages']]


def report(model, result=None, *, case='base'):
    """Source moiety, physical raffinate and #188 strip/loss priced-subtotal evidence."""
    result = stage.report(model) if result is None else result
    species = model._chemistry.species
    residuals = dict(result['balance_residuals_mol_per_raw_l'])
    for row in result['stages']:
        inlet, outlet = row['inlet_mol_per_raw_l'], row['outlet_mol_per_raw_l']
        for name, moiety in rc.MOIETIES.items():
            residuals[f'stage{row["stage"]}/{name}'] = sum(moiety.get(s, 0) * (outlet[s] - inlet[s]) for s in species)
    final = result['outlet_mol_per_raw_l']
    for name, moiety in rc.MOIETIES.items():
        residuals[f'global/{name}'] = sum(moiety.get(s, 0) * (final[s] - sum(p.get(s, 0) for p in model._feeds)) for s in species)
    ratio = model._design['O_A']
    li_feed = model._feeds[0][rc.ION['Li'][0]]
    raffinate = {records.SPECIES_LABELS[s]: final[s] for s in model._chemistry.supports[0]}
    r7 = references.check_r7(model._physical_feed, raffinate, balance_residuals_mol_per_raw_l=residuals,
                             strict_certificate=result['strict_certificate'])
    extraction = {'loaded_organic': {e: final[c] / ratio for e, c in rc.COMPLEX.items() if c in final},
        'loaded_Na_mol_L': final['naa'] / ratio, 'li_extraction_fraction': 1 - raffinate['Li'] / li_feed,
        'stages': [{'free_B_mol_L': row['outlet_mol_per_raw_l']['buphen'] / ratio,
                    'amounts_mol': [{s: row['outlet_mol_per_raw_l'][s] for s in support}
                                    for support in model._chemistry.supports]} for row in result['stages']]}
    # Translate the #188 MS-2 value basis using this feed's nominal or raw-source Li conversion.
    value_basis = li_feed * (brines.raw_feed('MS2')['printed_mg_l']['Li'] /
                             model._physical_feed['raw_brine']['printed_mg_l']['Li'])
    accounts = owner.accounts(extraction, model._design['S'], ratio, model._design['B'], case, 0.,
                              feed_li_mol_l=li_feed, site_basis_li_mol_l=model._design['site_basis_li_mol_l'],
                              value_basis_li_mol_l=value_basis)
    accounts.update(accounting_basis='historical #188 extraction-plus-strip intermediate; neither user scope',
                    li_denominator='kg strip-recovered Li per m3 raw brine', total_usd_kg_li=None,
                    can_minimize_total=False, completeness='partial: ' + accounts['missing_accounts'])
    return {**result, 'balance_residuals_mol_per_raw_l': residuals, 'R7': r7, 'feed': model._physical_feed,
            'fixed_design': model._design, 'solvent': model._solvent,
            'conserved_quantity_names': model._conserved_names, 'omitted_moieties': model._omitted_moieties,
            'aqueous_inlet_mol_per_raw_l': model._feeds[0], 'organic_inlet_mol_per_raw_l': model._feeds[1],
            'li_extraction_fraction': extraction['li_extraction_fraction'], 'accounts': accounts,
            'sodium_account_mol_per_raw_l': {'brine_in': model._feeds[0][rc.ION['Na'][0]],
                'saponified_organic_in': model._feeds[1]['naa'], 'raffinate_out': final[rc.ION['Na'][0]],
                'loaded_organic_out': final['naa']},
            'labels': model._physical_feed['labels'] + [label for label in owner.LABELS.values() if isinstance(label, str)] +
                ['fixed-design numerical verification, not physical MS-2 validation; no optimizer or full process']}
