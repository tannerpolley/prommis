"""IDAES operating-cost block for the extraction-only and full-process scopes (Lithium #211 stage 1).

Every flow is per hour at 1 m3/h raw brine, so an aggregate flow cost in USD/h equals the retained
USD/m3 line and IDAES aggregate equalities stay order one under IPOPT's 1e-12 constraint limit.
Register prices are carried at face value: source-period nominal USD without escalation.
`fresh.extraction_costs` and `flowsheet.costs_and_balances` remain the independent historical owners.
"""

import math

from idaes.core import FlowsheetCostingBlockData, declare_process_block_class, register_idaes_currency_units
from pyomo.environ import Block, Constraint, Expression, units as pyunits, value

from lithium_extraction.common.economics.hbta_topo import load_screening_cost_assumptions
from lithium_extraction.common.pretreatment import NAOH_KG_MOL
from lithium_extraction.common.thermodynamics.hbta_topo.constants import LI_MOLAR_MASS_G_MOL
from . import flowsheet

# Prices are declared before any costing block exists, so the currency is registered on import.
register_idaes_currency_units()
if not hasattr(pyunits, 'source_period_nominal_USD'):
    pyunits.load_definitions_from_strings(['source_period_nominal_USD = USD_2023'])
RAW_L_H = 1000 / pyunits.hr  # 1 m3/h raw brine; model amounts are per raw (or nominal aqueous) litre
GRAVITY_M_S2 = 9.80665
# R1 screening organic: 0.015 M formulation anchor; marginal HBTA and TOPO above it at 100 USD/kg each.
FORMULATION_ANCHOR_MOL_L = .015
MARGINAL_KG_MOL = {'HBTA': .21616, 'TOPO': .3866349}
MARGINAL_USD_KG = 100.


def mixer_settler():
    """Contactor register row: 6 min mixing time, 2.443 m3/h per m2 settler loading."""
    return flowsheet._row('data/reference/tea/hbta_topo_contactor_technology_register.csv', 'technology_id', 'mixer_settler')


def mixing_time_h():
    return float(mixer_settler()['mixing_time_min']) / 60


@declare_process_block_class('OperatingCosting')
class OperatingCostingData(FlowsheetCostingBlockData):
    def build_global_params(self):
        self.base_currency = pyunits.source_period_nominal_USD
        self.base_period = pyunits.hr

    def build_process_costs(self):
        self.operating_cost = Expression(expr=sum(self.aggregate_flow_costs.values()))

    def initialize_build(self):
        pass


def usd_per(unit):
    return pyunits.source_period_nominal_USD / unit


def attach(fs, flows, li_kg_h):
    """`flows` maps flow type to (flow per hour at 1 m3/h raw brine, price); `li_kg_h` is the scope's Li basis.

    A later scope replaces an earlier block, so each account is costed exactly once.
    """
    if hasattr(fs, 'costing'):
        fs.del_component(fs.costing)
    fs.costing = OperatingCosting()  # noqa: F821 (declared by the decorator above)
    for name, (flow, price) in flows.items():
        fs.costing.register_flow_type(name, price)
        fs.costing.cost_flow(flow, name)
    fs.costing.cost_process()
    fs.costing.li_kg_h = Expression(expr=li_kg_h)
    fs.costing.usd_kg_li = Expression(expr=fs.costing.operating_cost / fs.costing.li_kg_h)
    fs.costing.initialize()
    return fs.costing


def reconcile(costing, owner_usd_m3, owner_usd_kg_li, owner_li_kg_m3, owner_kg_l=None):
    """Reconcile primitive accounts and Li mass, propagating their bounds into USD/kg Li (#219).

    Solved aggregate residuals and values are read first, with nothing recalculated. The arithmetic
    check evaluates the same flow and price expressions at the identical physical quantities.
    Accounts may span several flow types named `<account>_<part>`; USD/m3 equals USD/h at 1 m3/h.
    """
    residuals = {c.name: value(c.body) - value(c.upper)
                 for c in costing.component_data_objects(Constraint, active=True)}
    flow = {}
    for f in costing.used_flows:
        flow[f], = costing._registered_flows[f]  # costed exactly once
    rate = costing.base_currency / costing.base_period
    exact = {f: value(pyunits.convert(flow[f] * getattr(costing, f + '_cost'), to_units=rate)) for f in flow}
    solved = {f: costing.aggregate_flow_costs[f].value for f in flow}

    def account(numbers):
        return {k: sum(n for f, n in numbers.items() if f == k or f.startswith(k + '_')) for k in owner_usd_m3}

    li = value(costing.li_kg_h)
    if not all(math.isfinite(n) for n in (li, owner_li_kg_m3, owner_usd_kg_li, *owner_usd_m3.values())) or li <= 0 or owner_li_kg_m3 <= 0:
        raise RuntimeError('Cost reconciliation requires finite values and positive product Li mass')
    limits = {name: dict.fromkeys(owner_usd_m3, 1e-9) for name in ('arithmetic', 'solved')}
    material_differences = {}
    gamma6 = 6 * 2.**-53 / (1 - 6 * 2.**-53)
    for k, reagent in (('neutralization_NaOH', 'NaOH'), ('carbonation_Na2CO3', 'Na2CO3')):
        if k not in flow:
            continue
        mass, price = flowsheet.MASS_KG_PER_MOL[reagent], value(getattr(costing, k + '_cost'))
        material_differences[k] = (value(flow[k]) / 1000 - owner_kg_l[k]) / mass
        if not math.isfinite(material_differences[k]) or abs(material_differences[k]) > 1e-9:
            raise RuntimeError(f'Molar reagent reconciliation fails: {material_differences}')
        # #224: mol/raw L -> kg/raw L -> kg/h -> USD/h, at 1 m3/h raw feed.
        leading = 1000 * abs(price) * mass * 1e-9
        rounding = gamma6 * (abs(exact[k]) + abs(owner_usd_m3[k]))
        tau = 1e-12 / max(1., value(costing.parent_block().washed_feed_l_per_raw_l))
        limits['arithmetic'][k] = leading + rounding
        limits['solved'][k] = leading + rounding + abs(price) * tau + tau
    quotient_limits = {name: (sum(bounds.values()) + abs(owner_usd_kg_li) * 1e-9) / li
                       for name, bounds in limits.items()}
    checks = {}
    for name, numbers, usd_kg_li in (('solved', solved, value(costing.usd_kg_li)),
                                     ('arithmetic', exact, sum(exact.values()) / li)):
        checks[name] = {'usd_m3/' + k: n - owner_usd_m3[k] for k, n in account(numbers).items()}
        checks[name]['usd_kg_li'] = usd_kg_li - owner_usd_kg_li
        if owner_kg_l is not None:
            kg_h = {f: (getattr(costing, 'aggregate_flow_' + f).value if name == 'solved' else value(flow[f]))
                    for f in flow}
            checks[name].update({'kg_l/' + k: n / 1000 - owner_kg_l[k] for k, n in account(kg_h).items()})
    record = {'limit_absolute': 1e-9, 'aggregate_residuals_after_solve': residuals,
              'solved_minus_owner': checks['solved'], 'arithmetic_minus_owner': checks['arithmetic'],
              'usd_m3': account(solved), 'product_li_kg_m3': li,
              'product_li_kg_m3_difference': li - owner_li_kg_m3,
              'usd_kg_li_limit_absolute': quotient_limits['solved'],
              'usd_kg_li_limits': quotient_limits, 'usd_m3_limits': limits,
              'reagent_difference_mol_raw_l': material_differences}
    numbers = [*residuals.values(), li - owner_li_kg_m3,
               *(n for check in checks.values() for k, n in check.items() if k.startswith('kg_l/'))]
    record['passed'] = (all(math.isfinite(n) and abs(n) <= 1e-9 for n in numbers) and
                        all(math.isfinite(c['usd_m3/' + k]) and abs(c['usd_m3/' + k]) <= limits[name][k]
                            for name, c in checks.items() for k in owner_usd_m3) and
                        all(math.isfinite(c['usd_kg_li']) and abs(c['usd_kg_li']) <= quotient_limits[name]
                            for name, c in checks.items()))
    if not record['passed']:
        raise RuntimeError(f'IDAES operating costing differs from the historical owner: {record}')
    return record


def add_extraction(model):
    """The extraction and saponification accounts of `fresh.extraction_costs` on the released decisions."""
    a = load_screening_cost_assumptions()
    fs, d = model.fs, model._decisions
    ratio, c = d['organic_to_aqueous_ratio'], d['extractant_mol_l']
    count, total_m3_h = len(fs.stages), 1 + ratio
    makeup_l_h = 1000 * ratio * (a.organic_entrainment_loss_fraction + a.organic_degradation_loss_fraction +
                                 a.organic_purge_fraction)
    kw = (a.mixer_power_density * total_m3_h * mixing_time_h() * count +
          1000 * GRAVITY_M_S2 * a.pump_head * total_m3_h / 3600 / a.pump_efficiency / 1000 * count)
    naoh = 0. if model._solvent == 'fresh' else fs.boundary['sodium-carrier']
    kg_h = pyunits.kg / pyunits.hr
    flows = {
        'organic_makeup_formulation': (makeup_l_h * a.formulated_organic_density * kg_h,
                                       a.formulated_organic_price * usd_per(pyunits.kg)),
        **{'organic_makeup_' + k: (makeup_l_h * (c - FORMULATION_ANCHOR_MOL_L) * m * kg_h,
                                   MARGINAL_USD_KG * usd_per(pyunits.kg)) for k, m in MARGINAL_KG_MOL.items()},
        'stage_operation': (count * pyunits.m**3 / pyunits.hr, a.extraction_stage_operation_auxiliary * usd_per(pyunits.m**3)),
        'electricity': (kw * pyunits.kW, a.electricity_price * usd_per(pyunits.kWh)),
        'saponification_NaOH': (naoh * NAOH_KG_MOL * RAW_L_H * pyunits.kg, a.naoh_price * usd_per(pyunits.kg)),
    }
    li = model._feeds[0]['lithium-cation'] * fs.design.li_extraction * LI_MOLAR_MASS_G_MOL / 1000
    return attach(fs, flows, li * RAW_L_H * pyunits.kg)


def hydraulics(aqueous_m3_h, organic_m3_h):
    """Mixer-settler screening sizing of legacy `contactor_design.add_hydraulic_design`, from the existing registers.

    Accepts numbers or Pyomo expressions; results are m3, m2 and kW for one contact.
    """
    a = load_screening_cost_assumptions()
    row = mixer_settler()
    total = aqueous_m3_h + organic_m3_h
    holdup = total * mixing_time_h()
    area = total / float(row['settler_capacity_m3_h_m2'])
    mass_kg_h = 1000 * aqueous_m3_h + 1000 * a.formulated_organic_density * organic_m3_h
    return {'mixing_holdup_m3': holdup, 'settler_area_m2': area, 'active_volume_m3': holdup + area * a.settler_depth,
            'mixer_power_kw': holdup * a.mixer_power_density,
            'pump_power_kw': mass_kg_h / 3600 * GRAVITY_M_S2 * a.pump_head / a.pump_efficiency / 1000,
            'throughput_m3_h': total}


def contacts(washed_l, ratio, stages, scope):
    """Aqueous and organic m3/h of each modeled contact at 1 m3/h raw brine; #176 scrub and strip run at O/A 10.

    `washed_l` is the extraction aqueous litres per raw litre (1 for the extraction-only nominal feed).
    """
    organic = ratio * washed_l
    flows = {f'extraction_stage_{k}': (washed_l, organic) for k in range(1, stages + 1)}
    if scope == 'full_process':
        flows.update(scrub=(organic / 10, organic), strip=(organic / 10, organic))
    return flows


def hydraulic_report(washed_l, ratio, stages, scope):
    return {'basis': 'one mixer-settler contact each, normalized to 1 m3/h raw brine (sizes scale linearly with '
                     'raw-brine flow); register screening values and legacy equations. The A1 operating accounts do '
                     'not use them; A3-A4 scale them to the plant capacity for equipment capital and electricity',
            'contacts': {name: {k: value(n) for k, n in hydraulics(*flows).items()}
                         for name, flows in contacts(washed_l, ratio, stages, scope).items()},
            'unavailable': {} if scope != 'full_process' else {
                'saponification': 'NaOH solution volume is not defined by the #176 accounting'}}


# Lithium #211 A3-A4: PrOMMiS REECosting (UKy Fire Clay accounts, IDAES QGESS) as a screening transfer from
# rare-earth correlations. CEPCI 2023 keeps the face-value source-period prices unescalated; equipment
# reference costs are CEPCI-converted from $2016. Each A1 account enters QGESS once as a priced resource.
EQUIPMENT = (  # name, UKy account, contact sized by `hydraulics`, size key, units
    ('extraction_mixer_settler', '5.4', 'extraction_stage_1', 'active_volume_m3', pyunits.m**3),
    ('extraction_pump', '5.3', 'extraction_stage_1', 'throughput_m3_h', pyunits.m**3 / pyunits.hr),
    ('scrub_strip_mixer_settler', '6.4', 'scrub', 'active_volume_m3', pyunits.m**3),
    ('scrub_strip_pump', '6.3', 'scrub', 'throughput_m3_h', pyunits.m**3 / pyunits.hr))
OMITTED = {
    'UKy 5.1/5.2 and 6.1/6.2 tanks and tank mixers': 'no tank duty is defined',
    'saponification (UKy 7.x)': 'NaOH solution volume is not defined by the #176 accounting',
    'softening precipitation, sludge filtration and washing': 'precipitation residence time and solids-handling duties are not defined',
    'strip-water concentration and unused-acid recovery': 'declared separations without equipment or energy duties',
    'Li2CO3 carbonation, filtration, drying and mother-liquor recycle': 'reactor residence time, heating and solids-handling duties are not defined',
    'organic and reagent first fills, land': 'not quantified for this flowsheet'}
EXCLUDED = {  # agreed no-tax finance scenario
    'UKy property taxes and insurance': '1% of total plant cost per year, removed from fixed operating cost',
    'income tax, royalties and production credits': 'QGESS taxes and credits not enabled'}
LI2CO3_PER_LI = .07389 / (2 * .00694)  # kg Li2CO3 per kg Li: flowsheet.costs_and_balances lce_factor masses
LI2CO3_SALE_USD_KG = 20.9  # upper retained base revenue (flowsheet.costs_and_balances); enters only the 0.5% sales cost


def add_plant_costing(model, capacity_m3_h):
    """Full-process REECosting at `capacity_m3_h` raw brine; requires the A1 block from `full_process.connect_units`."""
    from idaes.core import UnitModelBlock
    from prommis.costing.ree_costing import REECosting, REECostingData, REEUnitModelCostingBlock

    a, fs, q = load_screening_cost_assumptions(), model.fs, capacity_m3_h
    fs.plant_costing = REECosting(CEPCI_year='2023', has_fixed_OM=True, has_variable_OM=True)
    plant = fs.plant_costing
    model._plant_capacity_m3_h = q
    plant.capacity_factor.set_value(a.annual_operating_hours / value(pyunits.convert(1 * pyunits.year, pyunits.hr)))
    flows = contacts(fs.washed_feed_l_per_raw_l, model._decisions['organic_to_aqueous_ratio'], len(fs.stages), 'full_process')
    sizes = {name: hydraulics(*f) for name, f in flows.items()}
    counts = {'extraction_stage_1': len(fs.stages), 'scrub': 2}  # serial stages; scrub and strip are equal contacts
    for name, account, contact, key, units in EQUIPMENT:
        unit = UnitModelBlock()
        fs.add_component('plant_' + name, unit)
        unit.size = Expression(expr=sizes[contact][key] * q * units)
        unit.costing = REEUnitModelCostingBlock(flowsheet_costing_block=plant,
            costing_method=REECostingData.get_equipment_costing,
            costing_method_arguments={'cost_accounts': [account], 'scaled_param': unit.size,
                                      'n_equip': counts[contact], 'CEPCI_year': '2023'})
    a1 = fs.costing
    fs.plant_rates = Block()
    for f in a1.used_flows:  # each A1 aggregate flow, kg/h at 1 m3/h raw brine, enters once
        flow = getattr(a1, 'aggregate_flow_' + f)
        fs.plant_rates.add_component(f, Expression(fs.time, rule=lambda b, t, flow=flow: flow * q))
    kw = sum(s['mixer_power_kw'] + s['pump_power_kw'] for s in sizes.values())
    # Register price as its own resource: PrOMMiS replaces user prices for its 'power' key.
    fs.plant_rates.electricity = Expression(fs.time, rule=lambda b, t: kw * q * pyunits.kW)
    # Li2CO3 after purge at full capacity, kg/y; QGESS applies the capacity factor in cost_of_production.
    fs.plant_production = Expression(expr=pyunits.convert(a1.li_kg_h * q * LI2CO3_PER_LI, pyunits.kg / pyunits.year))
    plant.build_REE_process_costs(
        production_rate=fs.plant_production,
        pure_product_output_rates={'Li2CO3': fs.plant_production},
        sale_prices={'Li2CO3': LI2CO3_SALE_USD_KG * usd_per(pyunits.kg)},
        resources={f: getattr(fs.plant_rates, f) for f in (*a1.used_flows, 'electricity')},
        resource_prices={f: getattr(a1, f + '_cost') for f in a1.used_flows} | {
            'electricity': a.electricity_price * usd_per(pyunits.kWh)},
        chemicals=[f for f in a1.used_flows if f != 'sludge'], waste=['sludge'])
    REECostingData.initialize(plant)
    plant.property_taxes_and_insurance_cost_eq.deactivate()  # EXCLUDED under the no-tax scenario
    plant.property_taxes_and_insurance_cost.fix(0)  # after initialize, which resets it
    return plant


def plant_report(model):
    """Capital, fixed and variable operating costs and cost of production, with an independent numerator check."""
    fs = model.fs
    plant, a1 = fs.plant_costing, fs.costing
    year_h = value(pyunits.convert(1 * pyunits.year, pyunits.hr))
    cf = value(plant.capacity_factor)
    equipment = {}
    for name, account, *_ in EQUIPMENT:
        unit = getattr(fs, 'plant_' + name)
        args = unit.costing.costing_method_arguments
        reference = args['additional_costing_params'][0]['10']['A'][account]
        size = value(pyunits.convert(unit.size, getattr(pyunits, reference['Units'].split('/')[0]) /
                                     (getattr(pyunits, reference['Units'].split('/')[1]) if '/' in reference['Units'] else 1)))
        equipment[name] = {'account': account, 'account_name': reference['Account Name'],
                           'scaled_param_per_unit': size, 'units': reference['Units'], 'n_equip': args['n_equip'],
                           'reference_param': reference['RP Value'], 'scale_ratio': size / reference['RP Value'],
                           'exponent': reference['Exponent'], 'bare_erected_cost_musd': value(unit.costing.bare_erected_cost[account])}
    resources = {r: value(plant.variable_operating_costs[0, r]) for r in (*a1.used_flows, 'electricity')}
    production_kg_y = value(fs.plant_production) * cf
    numerator = (value(plant.annualized_cost) + value(plant.total_fixed_OM_cost) +
                 value(plant.total_variable_OM_cost[0]) * cf)
    q = model._plant_capacity_m3_h
    a1_usd_y = sum(a1.aggregate_flow_costs[f].value for f in a1.used_flows) * q * year_h
    checks = {'resources_minus_a1_lines_musd_y': sum(v for r, v in resources.items() if r != 'electricity') - a1_usd_y / 1e6,
              'cost_of_production_relative': value(plant.cost_of_production) / (numerator * 1e6 / production_kg_y) - 1}
    if any(not math.isfinite(n) or abs(n) > 1e-9 for n in checks.values()):
        raise RuntimeError(f'QGESS numerator does not reconcile with the A1 accounts: {checks}')
    fixed = {k: value(getattr(plant, k)) for k in (
        'annual_operating_labor_cost', 'annual_technical_labor_cost', 'maintenance_and_material_cost',
        'quality_assurance_and_control_cost', 'sales_patenting_and_research_cost', 'admin_and_support_labor_cost',
        'property_taxes_and_insurance_cost')}
    # Labour and its proportional lines: QA 10% and admin 20% of operating labour, then 20% overhead at the capacity factor.
    labour = fixed['annual_operating_labor_cost'] + fixed['annual_technical_labor_cost'] + fixed[
        'quality_assurance_and_control_cost'] + fixed['admin_and_support_labor_cost']
    labour_driven = labour * (1 + .2 * cf)
    li2co3 = 1e6 / production_kg_y
    subtotals = {
        'cost_of_recovery': value(plant.cost_of_production),
        'cost_of_recovery_without_labour': (numerator - labour_driven) * li2co3,
        'labour_and_proportional_lines': labour_driven * li2co3,
        'annualized_capital': value(plant.annualized_cost) * li2co3,
        'operating_reagent_sludge_solvent_a1': value(a1.usd_kg_li) / LI2CO3_PER_LI}
    return {'claim': 'priced screening subtotal; screening transfer from UKy rare-earth correlations; capital, labour and energy are screening '
                     'values; priced screening subtotal excluding the omitted operations; not a commercial estimate',
            'capacity_m3_h_raw_brine': q, 'omitted_operations': OMITTED, 'equipment': equipment,
            'finance_scenario': {
                'costing': 'prommis.costing.ree_costing.REECosting on idaes.models.costing.QGESS, tech 10 (UKy), no CCS',
                'currency': 'capital: UKy $2016 references CEPCI-converted to 2023 USD; operating prices: source-period '
                            'nominal USD at face value, carried as the 2023 unit and not escalated; labour rates: '
                            'QGESS defaults read as 2023 USD/h. The total is not a common-year cost.',
                'annual_operating_hours': cf * year_h, 'capacity_factor': cf, 'pyomo_year_h': year_h,
                'staffing_per_shift': {k: value(v) for k, v in plant.operators_per_shift.items()},
                'labor_rates_usd_h': {k: value(v) for k, v in plant.labor_rates.items()},
                'labor_burden_percent': value(plant.labor_burden),
                'labor_basis': 'transferred rare-earth (UKy) staffing scenario: QGESS defaults, no project source',
                'lang_factor': value(plant.Lang_factor), 'location_factor': value(plant.location_factor),
                'tasc_toc_factor': value(plant.tasc_toc_factor), 'fixed_charge_factor': value(plant.fixed_charge_factor),
                'electricity': 'register electricity_price per kWh of A2 mixer and pump power only; a separate '
                               'resource that receives no PrOMMiS plant overhead',
                'product_sale_price_usd_kg_li2co3': LI2CO3_SALE_USD_KG, 'excluded_charges': EXCLUDED,
                'excluded_property_taxes_and_insurance_musd_y': .01 * value(plant.total_TPC)},
            'total_bare_erected_cost_musd': value(plant.total_BEC), 'total_plant_cost_musd': value(plant.total_TPC),
            'total_overnight_capital_musd': value(plant.total_overnight_capital),
            'annualized_capital_musd_y': value(plant.annualized_cost),
            'fixed_operating_musd_y': fixed | {'total': value(plant.total_fixed_OM_cost)},
            'variable_operating_full_capacity_musd_y': resources | {
                'plant_overhead': value(plant.plant_overhead_cost[0]), 'total': value(plant.total_variable_OM_cost[0])},
            'li2co3_production_kg_y': production_kg_y,
            'labour_driven_musd_y': labour_driven,
            'usd_kg_li2co3': subtotals, 'usd_kg_li': {k: n * LI2CO3_PER_LI for k, n in subtotals.items()},
            'reconciliation': checks | {'limit': 1e-9}}
