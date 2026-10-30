"""Lithium #201 scope objectives and released extraction decisions; chemistry stays source-owned."""

from pyomo.environ import Block, Constraint, Expression, Objective, Var, value

from lithium_extraction.common.economics.hbta_topo import load_screening_cost_assumptions

from . import costing, fresh, full_process, records, set_a, set_c

BOUNDS = {'fresh': ((.5, 1.5), (.0075, .030), (0., 0.)),
          'A': ((.5, 1.5), (.015, .030), (.70, .95)),
          'C': ((1., 1.), (.1, .4), (.70, .70))}
TARGETS = {'fresh': .95, 'A': None, 'C': .95}


def decisions(model, solvent):
    """Fixed design first; boundaries and downstream accounts share these decision quantities."""
    fs = model.fs
    fs.design = Block()
    b = fs.design
    values = dict(model._design, saponification_ratio=model._design.get('saponification_ratio', 0.))
    names = ('organic_to_aqueous_ratio', 'extractant_mol_l', 'saponification_ratio')
    model._decisions = {}
    for name, bounds in zip(names, BOUNDS[solvent], strict=True):
        variable = Var(bounds=bounds, initialize=values[name])
        b.add_component(name, variable)
        variable.fix(values[name])
        model._decisions[name] = variable
    ratio, c, sap = (model._decisions[n] for n in names)
    inlet = fs.boundary
    initial_ratio = model._design['organic_to_aqueous_ratio']
    inlet['diluent'].unfix()
    b.diluent = Constraint(expr=inlet['diluent'] == ratio * model._feeds[1]['diluent'] / initial_ratio)
    free = 'free-carrier' if solvent == 'fresh' else 'hbta'
    inlet[free].unfix()
    b.free_extractant = Constraint(expr=inlet[free] == ratio * c * (1 - sap))
    if solvent != 'fresh':
        inlet['sodium-carrier'].unfix()
        b.saponified_extractant = Constraint(expr=inlet['sodium-carrier'] == ratio * c * sap)
    if solvent != 'fresh':
        inlet['topo'].unfix()
        b.free_topo = Constraint(expr=inlet['topo'] == inlet[free])
    if solvent == 'C':
        fs.parameters['extractant_mol_l'].unfix()
        b.engine_extractant = Constraint(expr=fs.parameters['extractant_mol_l'] == c)
    b.li_extraction = Expression(expr=1 - fs.stage[len(fs.stages)].outlet['lithium-cation'] /
                               model._feeds[0]['lithium-cation'])
    model._solvent = solvent
    costing.add_extraction(model)  # full_process.connect_units replaces it with the full-process accounts


def add_optimization(model, scope, *, minimum_li_extraction=None, sodium_limit=None):
    """The extraction target remains on extraction, including in the full-process scope."""
    solvent, b, fs = model._solvent, model.fs.design, model.fs
    for variable in model._decisions.values():
        if variable.lb != variable.ub:
            variable.unfix()
    target = TARGETS[solvent] if minimum_li_extraction is None else minimum_li_extraction
    if target is not None:
        b.li_target = Constraint(expr=b.li_extraction >= target)
    ratio, c = (model._decisions[n] for n in ('organic_to_aqueous_ratio', 'extractant_mol_l'))
    b.loading = Constraint(fs.stages, rule=lambda _, k:
                           fs.stage[k].outlet['lithium-carrier'] <= .72 * ratio * c)
    if solvent == 'fresh':
        limit = .02 if sodium_limit is None else sodium_limit
        b.sodium = Constraint(expr=fs.stage[1].outlet['sodium-carrier'] <=
                              limit * model._feeds[0]['sodium-cation'])
    elif sodium_limit is not None:
        li, na = (fs.stage[1].outlet[s] for s in ('lithium-carrier', 'sodium-carrier'))
        if scope == 'full_process':
            li, na = (fs.strip.metals[s] for s in ('Li', 'Na'))
        b.sodium = Constraint(expr=na <= sodium_limit * (li + na))
    feed = model._physical_feed
    volume = model._washed['feed_l'] if scope == 'full_process' else 1.
    deficits = {s: feed['model_feed'][records.SPECIES_LABELS[s]] -
                feed['physical_feed'].get(records.SPECIES_LABELS[s], 0.)
                for s in model._chemistry.supports[0]}
    b.physical_raffinate = Constraint(tuple(s for s, n in deficits.items() if n > 0),
        rule=lambda _, s: volume * fs.stage[len(fs.stages)].outlet[s] >= deficits[s])
    model.feasibility.deactivate()
    model.minimum_cost = Objective(expr=fs.costing.usd_kg_li)


def synchronize(model):
    """Report the solved boundaries and decision values, never the initialization design."""
    model._design.update({k: value(v) for k, v in model._decisions.items()})
    model._feeds = tuple({s: value(model.fs.boundary[s]) for s in feed} for feed in model._feeds)


def nominal_to_raw_litre(result, v):
    """Label nominal washed-feed amounts and scale conserved residuals to the raw litre."""
    result['outlet_mol_per_nominal_l'] = result.pop('outlet_mol_per_raw_l')
    residuals = result['balance_residuals_mol_per_raw_l']
    result['balance_residuals_mol_per_nominal_l'] = residuals
    result['balance_residuals_mol_per_raw_l'] = {k: n * v for k, n in residuals.items()}
    for row in result['stages']:
        for name in ('inlet', 'outlet', 'balance_residuals'):
            row[name + '_mol_per_nominal_l'] = row.pop(name + '_mol_per_raw_l')
        row['balance_residuals_mol_per_raw_l'] = {
            k: n * v for k, n in row['balance_residuals_mol_per_nominal_l'].items()}
        row['phase_charge_eq_per_nominal_l'] = row['phase_charge_eq_per_raw_l']
        row['phase_charge_eq_per_raw_l'] = [n * v for n in row['phase_charge_eq_per_nominal_l']]
        difference = row['certificate']['outlet_max_difference_mol_per_raw_l']
        row['certificate']['outlet_max_difference_mol_per_nominal_l'] = difference
        row['certificate']['outlet_max_difference_mol_per_raw_l'] = None if difference is None else difference * v
    return result


def report(model, scope, softening=None):
    synchronize(model)
    solvent = model._solvent
    if solvent == 'C':
        result = set_c.report(model)
    else:
        result = (fresh if solvent == 'fresh' else set_a).report(model)
        if scope == 'full_process':
            nominal_to_raw_litre(result, model._washed['feed_l'])
    result['solvent'] = solvent
    if scope == 'full_process':
        result = full_process.report(model, result, softening)
    else:
        if solvent == 'C':
            result['costs'] = fresh.extraction_costs(model, result['li_extraction_fraction'],
                                                   model._feeds[1]['sodium-carrier'])
        owner = result['costs']
        hours = load_screening_cost_assumptions().annual_operating_hours  # owner's annual m3 at 1 m3/h
        result['idaes_costing_reconciliation'] = costing.reconcile(model.fs.costing,
            {k: line['usd_y'] / hours for k, line in owner['lines'].items()}, owner['priced_subtotal_usd_kg_li'],
            owner['priced_subtotal_usd_m3_raw_brine'] / owner['priced_subtotal_usd_kg_li'])
        feed = result['feed']
        result['calibration_coverage'] = [
            {'quantity': name, 'value': feed['model_feed'][key], 'unit': unit,
             'basis': 'projected nominal aqueous inlet', 'comparison': 'physical calibration coverage unverified',
             'source_owner': records.SOLVENTS[solvent]['feed_owner']}
            for name, key, unit in (('projected Li', 'Li', 'mol/L aqueous'), ('projected Na', 'Na', 'mol/L aqueous'))]
    ratio, c = (model._design[n] for n in ('organic_to_aqueous_ratio', 'extractant_mol_l'))
    result['outlet_li_loading_fractions'] = [value(model.fs.stage[k].outlet['lithium-carrier']) / (ratio * c)
                                           for k in model.fs.stages]
    li, na = (value(model.fs.stage[1].outlet[s]) for s in ('lithium-carrier', 'sodium-carrier'))
    result['na_metric'] = {'name': 'net Na extracted / feed Na' if solvent == 'fresh' else
                          'strip-liquor Na/(Li+Na)' if scope == 'full_process' else 'loaded-organic Na/(Li+Na)',
                          'value': na / model._feeds[0]['sodium-cation'] if solvent == 'fresh' else
                          result['full_process']['strip_na_fraction'] if scope == 'full_process' else na / (li + na)}
    result['cost_usd_kg_li'] = (result['full_process']['operating_usd_kg_li'] if scope == 'full_process' else
                               value(model.fs.costing.usd_kg_li))
    result['cost_usd_m3_raw_brine'] = (sum(result['full_process']['cost_usd_m3'].values()) if scope == 'full_process'
                                    else value(model.fs.costing.operating_cost))
    result['cost_basis'] = records.SCOPES[scope]['li_denominator']
    result['hydraulics'] = costing.hydraulic_report(value(model.fs.washed_feed_l_per_raw_l),
                                                    ratio, len(model.fs.stages), scope)
    result['product_recovery_fraction'] = (result['full_process']['raw_li_recovery'] if scope == 'full_process' else None)
    result['labels'] = [s for s in result['labels'] if 'fixed-design' not in s and 'no full-process' not in s]
    result['labels'] += ['model prediction; no physical validation; local solution, no global optimum claim']
    return result


def target_failures(result, solvent):
    failures = list(result['R7']['failures'])
    target = TARGETS[solvent]
    if target is not None and result['li_extraction_fraction'] < target - 1e-9:
        failures.append('Li extraction target not met')
    if max(result['outlet_li_loading_fractions']) > .72 + 1e-9:
        failures.append('Li loading exceeds 0.72 per pair')
    if solvent == 'fresh' and result['na_metric']['value'] > .02 + 1e-9:
        failures.append('Na extraction exceeds 0.02 feed Na')
    return failures
