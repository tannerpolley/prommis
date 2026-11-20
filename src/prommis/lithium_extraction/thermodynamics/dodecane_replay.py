import csv
from dataclasses import replace
import math

from prommis.lithium_extraction.paths import REPO_ROOT

from prommis.lithium_extraction.thermodynamics.apparent_reactive_lle import (
    SMACKOVER,
    ZHANG_2017,
)
from prommis.lithium_extraction.thermodynamics.reaction_networks import load_constants

ROOT = REPO_ROOT
TRAINING = (
    ROOT
    / "analyses/v3/hbta_topo_ms2_idaes_optimization/results/lhs_training_matrix.csv"
)
FIT = (
    ROOT
    / "analyses/shared/hbta_topo_calibrated_surrogate/results/application_parameter_fits.json"
)


def _condition(row):
    carrier = float(row["carrier_mol_l_organic"])
    return replace(
        SMACKOVER,
        identity=row["sample_id"],
        role=row.get("role", "retained_blind_surrogate_comparison"),
        aqueous_volume_l=1.0,
        organic_volume_l=float(row["organic_to_aqueous_ratio"]),
        li_mol_l_aqueous=float(row["li_mol_l_aqueous"]),
        na_mol_l_aqueous=float(row["na_mol_l_aqueous"]),
        hydrogen_mol_l_aqueous=float(row["hydrogen_mol_l_aqueous"]),
        hbta_mol_l_organic=carrier,
        topo_mol_l_organic=carrier,
        temperature_k=float(row["temperature_k"]),
        initial_ph=-math.log10(float(row["hydrogen_mol_l_aqueous"])),
        li_mol_l_organic=carrier * float(row["incoming_li_loading_fraction"]),
        na_mol_l_organic=carrier * float(row["incoming_na_loading_fraction"]),
        saponification_fraction=0.0,
    )


def conditions():
    with TRAINING.open(newline="") as stream:
        row = next(
            r
            for r in csv.DictReader(stream)
            if r["sample_id"] == "training_fresh_oa_1_carrier_0.015_remaining_1"
        )
    return (
        ZHANG_2017,
        replace(
            SMACKOVER, identity="smackover_fresh_one_stage", saponification_fraction=0.0
        ),
        _condition(row),
    )


def fit_coordinates():
    fit = next(
        row for row in load_constants(FIT)["fits"] if row["diluent_id"] == "dodecane"
    )
    return float(fit["ln_k_exchange"]), float(fit["ln_k_na_uptake"])
