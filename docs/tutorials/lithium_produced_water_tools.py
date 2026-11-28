"""Display the retained whole-process comparison in the tutorial."""

import pandas as pd

from prommis.lithium_extraction.process.levelized_designs import FITS, selected_designs

WATERS = {"MS2": "MS-2", "MarsHill": "Mars Hill", "BR2": "BR-2"}


def paper_headline():
    """Display the retained fourteen-water comparison at the paper's precision."""
    return pd.DataFrame(
        [
            {
                "Water": (
                    f"USGS {row['water']}"
                    if row["water"].isdecimal()
                    else WATERS.get(row["water"], row["water"])
                ),
                "Calibration": FITS[row["constants"]],
                "Stages": int(row["n"]) if row["n"] else pd.NA,
                "Li extraction (%)": (
                    f"{100 * float(row['extraction']):.1f}" if row["n"] else ""
                ),
                "Raw Li recovery (%)": (
                    f"{100 * float(row['recovery']):.1f}" if row["n"] else ""
                ),
                "Levelized cost (USD/t LCE)": (
                    f"{int(row['usd_t']):,}" if row["n"] else ""
                ),
                "Disposition": row["reason"] or row["status"],
            }
            for row in selected_designs()
        ]
    )
