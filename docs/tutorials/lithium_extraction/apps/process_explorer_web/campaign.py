"""Transport campaign records to the app; the campaign owns every numerical solve."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "analyses/prototypes/modular_process"
CAMPAIGN = STUDY / "results/campaign"
SOLVENT_KEYS = {"set_a": "A", "set_c": "C",
                "d2_held": "D2EHDTPA-Held", "d2_figiel": "D2EHDTPA-Figiel"}
SCOPE_KEYS = {"extraction": "extraction_only", "full": "full_process"}


def reason(causes):
    """Plain descriptions of the campaign diagnosis classes, without changing dispositions."""
    text = " ".join(causes).lower()
    if "capacity" in text:
        return ("The organic phase cannot hold 95 % of this water's lithium within the allowed "
                "extractant concentrations and organic-to-water ratios.")
    if "nominal-acidity" in text or "hydrogen" in text:
        return ("Infeasible under the model's assumed acidity of raw brine: the calculated hydrogen "
                "amount is negative. Buffering and additional acidity conditioning were not evaluated.")
    if "carbonation" in text or "acid inventory" in text or "acid bounds" in text:
        return ("No accepted design was found within the measured carbonation solubility range "
                "and the allowed acid amounts. Unresolved alternatives remain within the solve limits.")
    return "No accepted design was found within the solve limits."


def app_record(record, source):
    """Copy only app quantities; retain the candidate's original field names and units."""
    fields = ("candidate", "status", "fixed_design", "li_extraction_fraction", "product_recovery_fraction",
              "strip_li_recovery_fraction", "cost_usd_kg_li", "cost_usd_m3_raw_brine", "cost_basis",
              "full_process", "costs", "accounts", "sites_mol_l_organic")
    return {**{k: record[k] for k in fields if k in record}, "source": source}


if __name__ == "__main__":
    # A fresh process keeps each live candidate's Engine state separate, as in the campaign.
    sys.path.insert(0, str(ROOT))
    from analyses.prototypes.modular_process.scripts.run_campaign import worker

    job = json.loads(sys.argv[1])
    result = worker(job, persist=False)
    Path(sys.argv[2]).write_text(json.dumps(result, allow_nan=False))
