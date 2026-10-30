"""Cheng 2013 Pitzer replay and Ge 2025 empirical mother-liquor solubility (#176)."""

import csv
import json
import math

import numpy as np

from lithium_extraction.common.support.paths import REPO_ROOT

GE_TABLE = REPO_ROOT / 'data/reference/thermodynamics/ge2025_table4.csv'
EXTENSION = REPO_ROOT / 'data/reference/thermodynamics/reaktoro_li2co3_pitzer_extension.dat'


def pitzer_system(mode='corrected'):
    """Explicitly attach extension parameters; Cheng Table 2 supplies phase Ksp.

    Diagnostic modes preserve the original phase and/or no-argument Pitzer setup.
    The shared extension and the #166 softening implementation remain unchanged.
    """
    import reaktoro as rkt
    from lithium_extraction.common.precipitation_payload import custom_pitzer_contents

    contents = custom_pitzer_contents(REPO_ROOT)
    if mode == 'corrected':
        # Cheng Eq. 3 / Table 2: ln Ksp = A + B/T + C ln T + D T.
        expression = [-32.672 / math.log(10), -0.11259 / math.log(10), -3653.6 / math.log(10), 12.591]
        contents = contents.replace('log_k -1.680', '-analytical_expression ' + ' '.join(map(str, expression)))
        contents = contents.replace('delta_h -4.3 kJ', '')
    db = rkt.PhreeqcDatabase.fromContents(contents)
    aq = rkt.AqueousPhase(rkt.speciate('H O Li Na Cl C'))
    if mode == 'default':
        aq.set(rkt.ActivityModelPitzer())
    else:
        data = json.loads(rkt.Params.embedded('Pitzer.json').data().dumpJson())
        params = data['ActivityModelParams']['Pitzer']
        sections = {'-B0': ('Beta0', 2), '-B1': ('Beta1', 2), '-C0': ('Cphi', 2),
                    '-THETA': ('Theta', 2), '-PSI': ('Psi', 3)}
        for line in EXTENSION.read_text().splitlines():
            fields = line.split()
            if not fields or fields[0].startswith('#'):
                continue
            if fields[0] in sections:
                key, count = sections[fields[0]]
            elif fields[0] in ('Li+', 'Na+', 'Cl-'):
                formulas = fields[:count]
                params[key] = [p for p in params[key] if set(p['Formulas']) != set(formulas)]
                params[key].append({'Formulas': formulas, 'CorrectionModel': 'Phreeqc',
                                    'Parameters': list(map(float, fields[count:]))})
        aq.set(rkt.ActivityModelPitzer(rkt.Params.fromStringJSON(json.dumps(data))))
    return rkt.ChemicalSystem(db, aq, rkt.MineralPhase('Li2CO3'))


def solubility(system, temperature_k, carbonate, salt, solid_mol=10.0):
    """Mol Li2CO3/kg equilibrium water with a remaining saturated solid; charge/element checks."""
    import reaktoro as rkt

    state = rkt.ChemicalState(system)
    state.temperature(temperature_k, 'kelvin')
    state.pressure(77.3, 'kPa')  # Ge Table 4
    for species, amount, unit in [('H2O', 1.0, 'kg'), ('Na+', 2 * carbonate + salt, 'mol'),
                                 ('CO3-2', carbonate, 'mol'), ('Cl-', salt, 'mol'), ('Li2CO3', solid_mol, 'mol')]:
        state.set(species, amount, unit)
    initial = np.array(state.elementAmounts(), dtype=float)
    result = rkt.EquilibriumSolver(system).solve(state)
    if not result.succeeded():
        raise RuntimeError('Li2CO3 saturation equilibrium did not converge')
    props, aq = rkt.ChemicalProps(state), rkt.AqueousProps(state)
    water = float(props.speciesAmount('H2O')) * system.species().get('H2O').molarMass()
    remaining = float(props.speciesAmount('Li2CO3'))
    si = float(aq.saturationIndex('Li2CO3'))
    closure = float(np.max(np.abs(np.array(props.elementAmounts(), dtype=float) - initial)))
    charge = float(props.chargeInPhase('AqueousPhase'))
    if remaining <= 0 or abs(si) > 1e-7 or max(closure, abs(charge)) > 1e-9:
        raise RuntimeError(f'Unsaturated or unclosed solubility state: solid={remaining}, SI={si}, closure={closure}')
    return {'temperature_k': temperature_k, 'carbonate_mol_kg': carbonate, 'nacl_mol_kg': salt,
            'li2co3_mol_kg': float(props.elementAmountInPhase('Li', 'AqueousPhase')) / (2 * water),
            'water_kg': water, 'solid_mol': remaining, 'saturation_index': si, 'closure_mol': closure,
            'charge_mol': charge, 'aqueous_na_mol': float(props.elementAmountInPhase('Na', 'AqueousPhase')),
            'aqueous_cl_mol': float(props.elementAmountInPhase('Cl', 'AqueousPhase')),
            'co3_mol': float(props.speciesAmount('CO3-2')), 'hco3_mol': float(props.speciesAmount('HCO3-'))}


def empirical_solubility(temperature_k, salt):
    """Linear interpolation in Ge's NaCl series at 0.5 mol/kg added Na2CO3; no extrapolation."""
    with GE_TABLE.open() as stream:
        rows = [r for r in csv.DictReader(stream) if r['series'] == 'NaCl']
    temperatures = [303.15, 313.15, 323.15, 333.15, 343.15, 353.15]
    if not temperatures[0] <= temperature_k <= temperatures[-1] or not 0 <= salt <= 3:
        raise ValueError('Empirical carbonation conditions outside Ge Table 4 interpolation domain')
    values = [np.interp(temperature_k, temperatures, [float(r[f'T{t:.2f}'.replace('.', '_')]) for t in temperatures])
              for r in rows]
    return float(np.interp(salt, [float(r['nacl_mol_kg']) for r in rows], values))
