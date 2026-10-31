"""Retain the complete washed state for each curated brine without changing the #176 equilibrium calculation."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--brine', choices=('MS2', 'BR2', 'MarsHill', 'Westmoreland'), default='MS2')
    parser.add_argument('--case', choices=('base', 'optimistic'))
    parser.add_argument('--output-file', type=Path, default=Path(__file__).resolve().parents[1] / 'results/washed_ms2.json')
    args = parser.parse_args()
    from analyses.prototypes.hbta_topo_ms2_prototype.scripts import precipitative_softening as ps
    from analyses.prototypes.hbta_topo_ms2_prototype.scripts.full_chain import PARAMETERS
    from lithium_extraction.prototypes.produced_water_sx_prototype import flowsheet
    from lithium_extraction.common.support.paths import REPO_ROOT

    from lithium_extraction.prototypes.produced_water_sx_prototype.brines import raw_feed
    raw = raw_feed(args.brine)
    source = REPO_ROOT / ('analyses/prototypes/hbta_topo_ms2_prototype/results/full_chain_softening.json'
        if args.brine == 'MS2' else f'analyses/prototypes/produced_water_sx_brines/results/{args.brine}_1/softening.json')
    previous = json.loads(source.read_text())
    output = {'type': 'prototype', 'raw_feed': raw, 'reference': str(source.relative_to(REPO_ROOT)),
              'reference_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'basis': 'mol per raw litre; aqueous components reconstruct every Reaktoro element',
              'component_basis': 'H is total acid protons on CO3-2/B(OH)4-/SO4-2 bases, not free H+; water is the '
                                 'corresponding H2O component; original aqueous species are retained separately'}
    # The retained values come from Linux; another platform's Reaktoro build agrees to its convergence tolerance, not
    # to the last bits (on Windows, MS-2 sludge differs by 3e-12 and residual Ca + Mg by 1e-9, relative).
    relative = ps.rkt.EquilibriumOptions().optima.convergence.tolerance
    for case in (args.case,) if args.case else ('base', 'optimistic'):
        t = time.monotonic()
        result = flowsheet.washed_feed(raw,
                                     PARAMETERS[case]['moisture'], PARAMETERS[case]['wash_volumes'])
        for name, expected in previous[case].items():
            for key, target in (expected.items() if isinstance(expected, dict) else [(name, expected)]):
                actual = result[name][key] if isinstance(expected, dict) else result[name]
                absolute = 1e-9 if key == 'HCl' or name == 'pH' else 1e-12
                assert math.isclose(actual, target, rel_tol=relative, abs_tol=absolute), (case, name, key)
        assert abs(result['pH'] - raw['pH']) <= 1e-9
        result['reagent_elements_mol'] = {e: sum(n * ps.REAGENT_ELEMENTS[k].get(e, 0)
            for k, n in result['reagent_mol'].items()) for e in result['raw_mol']}
        result['elapsed_wall_s'] = time.monotonic() - t
        output[case] = result
        print(case, 'P1 comparison; max element residual', max(map(abs, result['closure_mol'].values())),
              's', result['elapsed_wall_s'], flush=True)
    output['elapsed_wall_s'] = time.monotonic() - started
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
