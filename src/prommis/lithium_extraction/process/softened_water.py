"""Retain the complete washed state of a water without changing the #176 equilibrium calculation.

A curated water is compared with its retained #176/#177 softening result. Any other USGS sample ID has no reference:
its acceptance is the owner's element and charge closure (`flowsheet.washed_feed`, 1e-9 mol or eq per raw litre) and
the process pH within the limit derived from the solvers' precision (`flowsheet.ph_limit`), and its result goes to the site x solvent matrix (#341), never into a retained file.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from prommis.lithium_extraction.paths import REPO_ROOT


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--brine", default="MS2", help="curated water name or USGS sample ID"
    )
    parser.add_argument("--case", choices=("base", "optimistic"))
    parser.add_argument("--output-file", type=Path)
    args = parser.parse_args()
    from prommis.lithium_extraction.process.waters import CURATED, raw_feed

    curated = args.brine in CURATED
    if args.output_file is None:
        args.output_file = (
            REPO_ROOT
            / "analyses/prototypes/modular_process/results"
            / (
                "washed_ms2.json"
                if curated
                else f"site_solvent_matrix/softening/{args.brine}{'_' + args.case if args.case else ''}.json"
            )
        )
    from prommis.lithium_extraction.process import precipitative_softening as ps
    from prommis.lithium_extraction.process.softening_units import PARAMETERS
    from prommis.lithium_extraction.process import flowsheet

    raw = raw_feed(args.brine)
    output = {
        "type": "prototype",
        "raw_feed": raw,
        "basis": "mol per raw litre; aqueous components reconstruct every Reaktoro element",
        "component_basis": "H is total acid protons on CO3-2/B(OH)4-/SO4-2 bases, not free H+; water is the "
        "corresponding H2O component; original aqueous species are retained separately",
    }
    previous = None
    if curated:
        source = REPO_ROOT / (
            "analyses/prototypes/hbta_topo_ms2_prototype/results/full_chain_softening.json"
            if args.brine == "MS2"
            else f"analyses/prototypes/produced_water_sx_brines/results/{args.brine}_1/softening.json"
        )
        previous = json.loads(source.read_text())
        output.update(
            reference=str(source.relative_to(REPO_ROOT)),
            reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        )
    else:
        output["acceptance"] = (
            "no retained reference: owner element and charge closure within 1e-9 mol or eq per raw "
            "litre (flowsheet.washed_feed) and washed pH within the limit derived from the solvers' "
            "precision (flowsheet.ph_limit, #342)"
        )
    # Relative limit from the solver's precision (#219): Reaktoro's convergence tolerance. On Windows, MS-2 sludge
    # differs from the retained Linux values by 3e-12 and residual Ca + Mg by 1e-9.
    relative = ps.rkt.EquilibriumOptions().optima.convergence.tolerance
    for case in (args.case,) if args.case else ("base", "optimistic"):
        t = time.monotonic()
        result = flowsheet.washed_feed(
            raw, PARAMETERS[case]["moisture"], PARAMETERS[case]["wash_volumes"]
        )
        for name, expected in (previous[case] if previous else {}).items():
            for key, target in (
                expected.items() if isinstance(expected, dict) else [(name, expected)]
            ):
                actual = (
                    result[name][key] if isinstance(expected, dict) else result[name]
                )
                strict = name == "reagent_mol" or "sludge" in name
                absolute = (
                    1e-9 if key == "HCl" or name == "pH" else 0.0 if strict else 1e-12
                )
                assert math.isclose(
                    actual, target, rel_tol=relative, abs_tol=absolute
                ), (case, name, key)
        limit = result["pH_acceptance"]["limit_ph"]
        assert (
            abs(result["pH"] - raw["pH"]) <= limit
        ), f"{case}: washed pH {result['pH']!r} is not within {limit:.3g} of {raw['pH']}"
        result["reagent_elements_mol"] = {
            e: sum(
                n * ps.REAGENT_ELEMENTS[k].get(e, 0)
                for k, n in result["reagent_mol"].items()
            )
            for e in result["raw_mol"]
        }
        result["elapsed_wall_s"] = time.monotonic() - t
        output[case] = result
        print(
            case,
            "P1 comparison" if previous else "closure only",
            "; max element residual",
            max(map(abs, result["closure_mol"].values())),
            "s",
            result["elapsed_wall_s"],
            flush=True,
        )
    output["elapsed_wall_s"] = time.monotonic() - started
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
