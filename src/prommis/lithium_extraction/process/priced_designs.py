"""Read the retained heat-priced nominal process designs by design ID and price."""

import csv
import json

from prommis.lithium_extraction.paths import REPO_ROOT

RESULTS = REPO_ROOT / "analyses/prototypes/modular_process/results/heat_priced_designs"
PRIMARY, SECONDARY, REPRICE = 9000, 17500, 20900
FITS = {
    "fitted": "D2EHDTPA/BuPhen (Figiel)",
    "low": "HBTA/TOPO (single-stage fit)",
    "high": "HBTA/TOPO (cascade fit)",
}
WATERS = {"BR2": "BR-2", "MS2": "MS-2", "MarsHill": "Mars Hill"}


def load_matrix(results=RESULTS):
    with (results / "designs.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    designs = {(r["design_id"], float(r["price_usd_t_lce"])): r for r in rows}
    if len(designs) != len(rows):
        raise ValueError("Duplicate design-and-price rows in designs.csv")

    cells = json.loads((results / "selections.json").read_text())
    return designs, cells


def selected_design(designs, cell, price):
    if cell["winner"] is None:
        return None
    row = designs[cell["winner"], price]
    if (row["water"], row["route"], row["calibration"]) != (
        cell["water"],
        cell["route"],
        cell["fit"],
    ):
        raise ValueError(f"Selection does not match design {cell['winner']}")
    return row


def missing_reason(cell):
    if exclusions := cell["analytical_exclusions"]:
        exclusion = next(iter(exclusions.values()))
        return f"{exclusion['reason']} ({exclusion['inequality']} mol/L organic)"
    if cell["target_released_fallback"]:
        return "Closed N=5 fallback below 95% extraction with the target released"
    return "Initial scrub/Ge starts unavailable; N=5 fallback did not launch"


def selected_designs(designs, cells):
    for cell in cells:
        status = cell["status"].replace("_", " ")
        row = selected_design(designs, cell, cell["price"])
        yield {
            **{k: cell[k] for k in ("water", "route", "fit")},
            "price": int(cell["price"]),
            "status": status,
            "reason": "" if row else missing_reason(cell),
            "design": row,
        }
        if cell["price"] == PRIMARY and row:
            yield {
                "water": cell["water"],
                "route": cell["route"],
                "fit": cell["fit"],
                "price": REPRICE,
                "status": status,
                "reason": "",
                "design": selected_design(designs, cell, REPRICE),
            }
