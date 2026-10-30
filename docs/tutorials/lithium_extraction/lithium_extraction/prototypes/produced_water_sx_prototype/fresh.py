"""Fresh HBTA/TOPO adapter: shared H1 chemistry, frozen feed convention and R1 accounting."""

import csv
from dataclasses import replace
import math
import json
from pathlib import Path

import epcsaft
from epcsaft import equilibrium as eq

from analyses.shared.hbta_topo_configured_reactive_lle.scripts import independent_network as network
from analyses.shared.hbta_topo_configured_reactive_lle.scripts.replay_current_engine import conditions, fit_coordinates
from lithium_extraction.common.economics.hbta_topo import load_screening_cost_assumptions
from lithium_extraction.common.pretreatment import NAOH_KG_MOL
from lithium_extraction.common.support.paths import ENGINE_ASL_LIBRARY, REPO_ROOT
from lithium_extraction.common.thermodynamics.hbta_topo.constants import LI_MOLAR_MASS_G_MOL, NA_MOLAR_MASS_G_MOL
from . import brines, costing, records, references, stage


R1_PROFILES = Path(references.SOURCES['R1']).with_name('optimized_stage_rows.csv')


def reference_feed(solvent="fresh", reference="R1"):
    """R1/R2 nominal 6.94/22.98976928 conversions, separate from the new raw-source basis."""
    raw = brines.raw_feed('MS2')
    source_chloride, source_closure = raw['mol']['Cl'], raw['charge_closure_mol_l']
    for name, mass in (('Li', LI_MOLAR_MASS_G_MOL), ('Na', NA_MOLAR_MASS_G_MOL)):
        nominal = raw['printed_mg_l'][name] / 1000 / mass
        raw['mol']['Cl'] += nominal - raw['mol'][name]
        raw['mol'][name] = nominal
    raw['charge_closure_mol_l'] += raw['mol']['Cl'] - source_chloride
    # Keep the source owner's reported-anion denominator while updating the nominal chloride adjustment.
    raw['charge_closure_percent_anion_eq'] *= raw['charge_closure_mol_l'] / source_closure
    raw['labels'].append(f'{reference} nominal MS-2 conversion basis; original cation masses and charge closure retained')
    feed = records.extraction_feed(raw, solvent)
    feed['raw_brine'] = raw
    return feed


def chemistry():
    """Derive the frozen affine map from the independent owner's inventory, support and charge rows."""
    condition = conditions()[2]
    topology = network.TOPOLOGIES['t1_mbta_topo']
    mixture = network.build_model('t1_mbta_topo', 'dodecane')[1]
    reactions = network._reactions(condition, *fit_coordinates())
    seeded = network._problem(condition, topology, mixture, reactions, network.INITIALIZATION_SEEDS[0])
    amounts = network._initial_amounts(condition, topology, mixture, (0.0, 0.0, 0.0))
    feed = tuple(sum(row[j] for row in amounts) for j in range(len(network.SPECIES)))
    controls = tuple(s for s in network.SPECIES if s not in ('water', 'chloride-anion'))
    # The owner carries only one anion: its charge row uniquely supplies the eliminated chloride coordinate.
    charges = dict(zip(network.SPECIES, network.CHARGES, strict=True))
    matrix = tuple(tuple(float(s == c) if s != 'chloride-anion' else
                         -charges[c] / charges[s] for c in controls) for s in network.SPECIES)
    offset = tuple(feed[j] if s == 'water' else 0.0 for j, s in enumerate(network.SPECIES))
    anchor = tuple(feed[network.SPECIES.index(s)] for s in controls)
    problem = eq.Problem(T=seeded.T, P=seeded.P, phases=seeded.phases, reactions=seeded.reactions,
                         feed=eq.Amounts(dict(zip(network.SPECIES, feed, strict=True))), row_tolerance=1e-10)
    continuation = eq.Continuation(initial_step=.25, minimum_step=.25, max_steps=4, max_retries_per_step=1)
    record = eq.asl_record(mixture, problem, continuation, offset, matrix, anchor, 0, .001)

    def certify(inlet, outlet):
        inventory = sum(inlet[s] for s in network.ORGANIC_SUPPORT if s != 'diluent')
        ratio = inlet['diluent'] / amounts[1][network.SPECIES.index('diluent')]
        state = replace(condition, identity='modular fresh converged inlet', organic_volume_l=ratio,
                        li_mol_l_aqueous=inlet['lithium-cation'], na_mol_l_aqueous=inlet['sodium-cation'],
                        hydrogen_mol_l_aqueous=inlet['hydrogen-cation'],
                        chloride_mol_l_aqueous=inlet['chloride-anion'],
                        initial_ph=-math.log10(inlet['hydrogen-cation']),
                        hbta_mol_l_organic=inventory / ratio, topo_mol_l_organic=inventory / ratio,
                        li_mol_l_organic=inlet['lithium-carrier'] / ratio,
                        na_mol_l_organic=inlet['sodium-carrier'] / ratio, saponification_fraction=0.0)
        direct = network.solve_condition(state, 't1_mbta_topo', 'dodecane', *fit_coordinates(),
                                         initialization_seeds=network.INITIALIZATION_SEEDS[:1])
        if 'phase_amounts_mol' in direct:
            direct['outlet_max_difference_mol_per_raw_l'] = max(abs(outlet[s] - sum(
                p.get(s, 0.0) for p in direct['phase_amounts_mol'])) for s in network.SPECIES)
        else:
            direct['outlet_max_difference_mol_per_raw_l'] = None
        return direct

    declaration = stage.Chemistry(network.SPECIES, network.phase_supports(condition), network._basis_matrix(),
                                  network.CHARGES, controls, offset, matrix, record,
                                  Path(epcsaft.__file__).resolve().parent / ENGINE_ASL_LIBRARY, certify)
    return declaration, feed[network.SPECIES.index('diluent')]


def build_flowsheet(brine, solvent='fresh', scope='extraction_only', stages=3, scenario=None):
    """Build fresh fixed-design extraction only; later issue stages add other solvent/scope owners."""
    if solvent != 'fresh' or scope != 'extraction_only':
        raise ValueError('PR2 supports fresh HBTA/TOPO extraction only')
    if not scenario or set(scenario) != {'organic_to_aqueous_ratio', 'extractant_mol_l'}:
        raise ValueError('fixed design requires O/A and extractant molarity')
    ratio, concentration = scenario['organic_to_aqueous_ratio'], scenario['extractant_mol_l']
    if any(not math.isfinite(n) or n <= 0 for n in (ratio, concentration)):
        raise ValueError('O/A and extractant molarity must be finite and positive')
    feed = dict(brine) if 'physical_feed' in brine else records.extraction_feed(brine, solvent, scope)
    if 'physical_feed' not in brine:
        feed['raw_brine'] = brine
    if 'unavailable' in feed:
        raise ValueError(feed['unavailable'])
    declared, diluent = chemistry()
    carried = {records.SPECIES_LABELS[s] for s in network.AQUEOUS_SUPPORT}
    if any(n != 0 for s, n in feed['model_feed'].items() if s not in carried):
        raise ValueError('fresh owner cannot represent nonzero species outside its declared support')
    aqueous = {s: feed['model_feed'][records.SPECIES_LABELS[s]] for s in network.AQUEOUS_SUPPORT}
    organic = {s: 0.0 for s in network.ORGANIC_SUPPORT}
    organic.update({'diluent': ratio * diluent, 'free-carrier': ratio * concentration})
    model = stage.build_stages(declared, aqueous, organic, stages)
    model._physical_feed = feed
    model._design = dict(scenario)
    return model


def reference_profiles(model):
    """Translate retained concentrations to the owner's stage amounts; only the fixed R1 profile is supplied."""
    with (REPO_ROOT / R1_PROFILES).open() as stream:
        rows = list(csv.DictReader(stream))
    ratio, c = model._design['organic_to_aqueous_ratio'], model._design['extractant_mol_l']
    if len(rows) != len(model.fs.stages) or ratio != float(rows[0]['organic_to_aqueous_ratio']):
        raise ValueError('R1 profiles apply only to their retained three-stage O/A')
    profiles = []
    for row in rows:
        aq, org = model._feeds
        amounts = dict(aq, **org)
        for name, field in (('lithium-cation', 'li'), ('sodium-cation', 'na'), ('hydrogen-cation', 'hydrogen')):
            amounts[name] = float(row[f'{field}_out_mol_l'])
        amounts['lithium-carrier'] = ratio * c * float(row['equilibrium_li_loading_fraction'])
        amounts['sodium-carrier'] = ratio * c * float(row['equilibrium_na_loading_fraction'])
        amounts['free-carrier'] = ratio * c - amounts['lithium-carrier'] - amounts['sodium-carrier']
        profiles.append(amounts)
    return profiles


def report(model, result=None):
    result = stage.report(model) if result is None else result
    aqueous, organic = model._feeds
    final = result['outlet_mol_per_raw_l']
    extraction = (final['lithium-carrier'] - organic['lithium-carrier']) / aqueous['lithium-cation']
    sodium = (final['sodium-carrier'] - organic['sodium-carrier']) / aqueous['sodium-cation']
    raffinate = dict(model._physical_feed['model_feed'])
    raffinate.update({records.SPECIES_LABELS[s]: final[s] for s in network.AQUEOUS_SUPPORT})
    r7 = references.check_r7(model._physical_feed, raffinate,
                             balance_residuals_mol_per_raw_l=result['balance_residuals_mol_per_raw_l'],
                             strict_certificate=result['strict_certificate'])
    costs = extraction_costs(model, extraction)
    ratio, c = model._design['organic_to_aqueous_ratio'], model._design['extractant_mol_l']
    loadings = [r['outlet_mol_per_raw_l']['lithium-carrier'] / (ratio * c) for r in result['stages']]
    domain = records.SOLVENTS['fresh']['domain_owner']
    low, high = json.loads((REPO_ROOT / domain).read_text())['input_bounds']['carrier_mol_l_organic']
    position = 'inside' if low <= c <= high else 'outside'
    concentration_labels = [
        f'selected HBTA and TOPO = {c:.17g} mol/L organic each: physical calibration coverage unverified; '
        f'owner {records.SOLVENTS["fresh"]["chemistry_owner"]}',
        f'selected HBTA and TOPO = {c:.17g} mol/L organic each: {position} current loaded-organic surrogate '
        f'numerical interpolation interval [{low:.17g}, {high:.17g}] mol/L organic; owner {domain}; '
        'not physical calibration coverage',
    ]
    return {**result, 'fixed_design': model._design, 'feed': model._physical_feed, 'R7': r7, 'costs': costs,
            'li_extraction_fraction': extraction, 'na_extraction_fraction': sodium,
            'outlet_li_loading_fractions': loadings,
            'target_margins': {'Li': extraction - .95, 'Na': .02 - sodium, 'Li_loading': .72 - max(loadings)},
            'labels': model._physical_feed['labels'] + concentration_labels + ['fixed-design numerical replay; not a new optimum',
                       'numerical verification; not physical validation; #158 fresh recovery discrepancy remains']}


def extraction_costs(model, extraction, naoh_mol_per_raw_l=0.0):
    """R1/R2 extraction and saponification costs at the original 1000 L/h basis; an independent formula
    from `costing.add_extraction`, reading the same named constants."""
    aqueous, _ = model._feeds
    # Source-period nominal accounting shared by R1/R2, from the common register.
    assumptions = load_screening_cost_assumptions()
    ratio, c = model._design['organic_to_aqueous_ratio'], model._design['extractant_mol_l']
    annual_feed_m3 = assumptions.annual_operating_hours  # frozen 1000 L/h aqueous feed
    makeup_l = annual_feed_m3 * 1000 * ratio * (assumptions.organic_entrainment_loss_fraction +
               assumptions.organic_degradation_loss_fraction + assumptions.organic_purge_fraction)
    formulation = assumptions.formulated_organic_density * assumptions.formulated_organic_price + (
        c - costing.FORMULATION_ANCHOR_MOL_L) * sum(m * costing.MARGINAL_USD_KG for m in costing.MARGINAL_KG_MOL.values())
    count = len(model.fs.stages)
    total_flow = 1 + ratio  # m3/h at the R1 1000 L/h feed
    electricity_kwh = (assumptions.mixer_power_density * total_flow * costing.mixing_time_h() * count +
                       1000 * costing.GRAVITY_M_S2 * assumptions.pump_head * total_flow / 3600 /
                       assumptions.pump_efficiency / 1000 * count) * assumptions.annual_operating_hours
    source = 'data/reference/tea/hbta_topo_screening_cost_assumptions.csv; base; R1/R2 source-period nominal USD'
    lines = {
        'organic_makeup': {'usd_y': makeup_l * formulation, 'price_source': source +
                          f'; {costing.FORMULATION_ANCHOR_MOL_L} M formulation anchor, marginal HBTA/TOPO '
                          f'{costing.MARGINAL_USD_KG:g} USD/kg each (R1 screening)'},
        'stage_operation': {'usd_y': annual_feed_m3 * assumptions.extraction_stage_operation_auxiliary * count,
                            'price_source': source},
        'electricity': {'usd_y': electricity_kwh * assumptions.electricity_price, 'price_source': source},
        'saponification_NaOH': {'usd_y': annual_feed_m3 * 1000 * naoh_mol_per_raw_l *
                               NAOH_KG_MOL * assumptions.naoh_price,
                               'price_source': source + '; naoh_price (Zhang 2026 Table 8)' if naoh_mol_per_raw_l
                               else 'not applicable: fresh organic'},
    }
    return records.scope_costs('extraction_only', lines,
       annual_li_kg=annual_feed_m3 * 1000 * aqueous['lithium-cation'] * extraction * LI_MOLAR_MASS_G_MOL / 1000,
       annual_raw_brine_m3=annual_feed_m3)
