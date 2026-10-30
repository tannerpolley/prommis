"""Source-located raw brines, mol per raw litre; no softening or extraction solve."""

import csv
import math

from lithium_extraction.common.support.paths import REPO_ROOT

# Reaktoro 2.13 pitzer.dat values used by #176/#177, including its species masses.
MOLAR_MASS = {'H': 1.008, 'O': 16.0, 'C': 12.0111, 'Na': 22.9898, 'K': 39.0983,
              'Mg': 24.305, 'Ca': 40.08, 'Sr': 87.62, 'Ba': 137.33, 'Li': 6.941,
              'Cl': 35.453, 'Br': 79.904, 'S': 32.064, 'B': 10.81,
              'SO4': 96.06509715981777, 'HCO3': 61.01964857990889}
WATER_MOLAR_MASS_KG = .018016  # Reaktoro 2.13 pitzer.dat H2O, kg/mol; not the Engine's water approximation.
CHARGE = {'Li': 1, 'Na': 1, 'K': 1, 'Ca': 2, 'Mg': 2, 'Sr': 2, 'Ba': 2,
          'Cl': -1, 'Br': -1, 'SO4': -2, 'B': 0, 'HCO3': -1, 'CO2': 0,
          'H': 1, 'OH': -1, 'BTA': -1, 'water': 0}
PROCESS_PH = 5.77
CURATED = {'MS2': 'MS-2', 'BR2': 'BR-2', 'MarsHill': '17258', 'Westmoreland': '164'}
SOUTH_AR = 'data/reference/smackover_sar_li/input/usgs_southAR/southAR_brines_2022.txt'
PWGD = 'data/reference/produced_water/usgs_pwgd_v3_site_extract.csv'
ANALYTES = ('Li', 'Na', 'K', 'Ca', 'Mg', 'Sr', 'Ba', 'Cl', 'Br', 'SO4', 'B', 'HCO3', 'DIC')
REQUIRED = ('Li', 'Na', 'Ca', 'Mg', 'Cl')


def feed(mg_l, close_with='Cl', co2_mol=0.0):
    """The #176 reported-ion conversion and charge closure, unchanged."""
    mol = {k: v / 1000.0 / MOLAR_MASS[k] for k, v in mg_l.items()}
    charge = sum(CHARGE[k] * n for k, n in mol.items())
    mol[close_with] -= charge / CHARGE[close_with]
    if co2_mol:
        mol['CO2'] = co2_mol
    return mol


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def raw_feed(name, scale=1.0, *, reference_cations=None):
    """Curated feed or an ID in the retained USGS tables; missing optional analytes are labelled."""
    if not math.isfinite(scale) or scale < 0:
        raise ValueError('Ca/Mg scale must be finite and nonnegative')
    identity = CURATED.get(name, str(name))
    matches = []
    for path, key in ((SOUTH_AR, 'SITEID'), (PWGD, 'IDUSGS')):
        with (REPO_ROOT / path).open() as stream:
            matches.extend((path, row) for row in csv.DictReader(stream) if row[key] == identity)
    if len(matches) != 1:
        raise ValueError(f'USGS ID {identity}: expected one retained source row, found {len(matches)}')
    path, row = matches[0]
    missing = [e for e in REQUIRED if _number(row.get(e)) is None]
    if missing:
        raise ValueError(f'USGS ID {identity}: missing required analytes: {", ".join(missing)}')
    curated = name in CURATED
    elements = ANALYTES[:11] if path == SOUTH_AR else ANALYTES[:10]
    if not curated:
        elements = tuple(e for e in ANALYTES if e != 'DIC')
    ions = {e: _number(row.get(e)) for e in elements}
    omitted = [e for e in elements if ions[e] is None]
    ions = {e: v for e, v in ions.items() if v is not None or not curated}
    ions = {e: v if v is not None else 0.0 for e, v in ions.items()}
    dic = _number(row.get('DIC'))
    ph = _number(row.get('PH'))
    if path != SOUTH_AR and curated:
        omitted += ['B', 'DIC', 'pH']
    elif not curated:
        omitted += [e for e, value in (('DIC', dic), ('pH', ph)) if value is None]
    sg, tds = _number(row.get('SG')), _number(row.get('TDS'))
    water = None if sg is None or tds is None else sg - tds / 1e6
    if water is not None and water <= 0:
        raise ValueError(f'USGS ID {identity}: SG minus TDS gives nonpositive water mass')
    mol = feed(ions, co2_mol=(dic or 0.0) / 1000 / MOLAR_MASS['C'])
    if reference_cations is not None:
        for e, n in reference_cations.items():
            if e not in mol or CHARGE[e] <= 0 or not math.isfinite(n) or n < 0:
                raise ValueError('reference conversions require reported finite nonnegative cations')
            mol['Cl'] += CHARGE[e] * (n - mol[e])
            mol[e] = n
    delta_cl = mol['Cl'] - ions['Cl'] / 1000 / MOLAR_MASS['Cl']
    anions = sum(-CHARGE[e] * n / 1000 / MOLAR_MASS[e] for e, n in ions.items() if CHARGE[e] < 0)
    percent = 100 * delta_cl / anions if anions else None
    if mol['Cl'] < 0:
        raise ValueError(f'USGS ID {identity}: charge closure requires negative chloride')
    labels = ['pH: not reported' if e == 'pH' else
              f'{e}: curated omission (#{177})' if curated else
              f'{e}: not reported; assumed zero' for e in omitted]
    if percent is None or abs(percent) > 5:
        labels.append('large charge-closure adjustment')
    if water is None:
        labels.append('water mass unavailable: SG or TDS not reported')
    for e in ('Ca', 'Mg'):
        original = mol[e]
        mol[e] *= scale
        mol['Cl'] += 2 * (mol[e] - original)
    if mol['Cl'] < 0:
        raise ValueError(f'USGS ID {identity}: Ca/Mg scaling requires negative chloride')
    locator = f'{path}: ' + (f'SITEID={identity}; IDDB={row["IDDB"]}' if path == SOUTH_AR
                               else f'IDUSGS={identity}; {row["REFERENCE"]}')
    result = {'name': name, 'source_locator': locator, 'source_row': row,
              'printed_mg_l': ions, 'dic_mg_c_l': dic, 'measured_pH': ph, 'water_kg': water,
              'mol': mol, 'omitted': omitted, 'labels': labels, 'pH': PROCESS_PH,
              'pH_basis': 'process assumption, not measured feed pH', 'divalent_scale': scale,
              'charge_closure_mol_l': delta_cl, 'charge_closure_percent_anion_eq': percent,
              'ca_mg_li_mass_ratio': scale * (ions['Ca'] + ions['Mg']) / ions['Li'] if ions['Li'] else None}
    if name == 'MS2':
        result['T_C'] = float(row['TEMP'])
    return result
