"""Read the retained whole-process selections on the paper's levelized cost basis."""

import csv

from prommis.lithium_extraction.paths import REPO_ROOT

RESULTS = REPO_ROOT / "analyses/prototypes/modular_process/results/levelized_cost"
SELECTED = RESULTS / "priority_7/selected_designs.csv"
FITS = {
    "all_data": "D2EHDTPA/BuPhen (all-data fit)",
    "fig7_exchange": "D2EHDTPA/BuPhen (three-response fit)",
    "low": "HBTA/TOPO (single-stage fit)",
    "high": "HBTA/TOPO (cascade fit)",
}


def selected_designs():
    with SELECTED.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))
