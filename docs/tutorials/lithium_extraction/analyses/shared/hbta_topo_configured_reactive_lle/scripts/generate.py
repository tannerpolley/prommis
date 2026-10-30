"""Engine-wheel identity checks and the configured HBTA/TOPO reaction topology.

The readiness-receipt generator that ran the manufactured oracle was retired with the
superseded Engine interface; `results/topology_receipt.json` and its report are historical.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import epcsaft
import numpy as np
from epcsaft import equilibrium



ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
SOURCE = REPO / "data/reference/epcsaft_regression/hbta_topo_extraction"
SPECIES_IDS = (
    "water",
    "li_plus",
    "na_plus",
    "chloride",
    "hbta",
    "topo",
    "na_bta",
    "na_bta_topo",
    "li_bta_topo",
)
AQUEOUS_SUPPORT = ("water", "li_plus", "na_plus", "chloride")
ORGANIC_SUPPORT = (
    "water",
    "hbta",
    "topo",
    "na_bta",
    "na_bta_topo",
    "li_bta_topo",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _verify_wheel(wheel: Path, expected_sha256: str, commit: str) -> dict[str, str]:
    wheel = wheel.resolve()
    if not wheel.is_file():
        raise ValueError(f"Engine wheel does not exist: {wheel}")
    expected = expected_sha256.removeprefix("sha256:")
    actual = _sha256(wheel)
    if actual != expected:
        raise ValueError(f"Engine wheel hash differs: expected {expected}, received {actual}")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("Engine commit must be a full lowercase Git object identity")
    distribution = importlib.metadata.distribution("epcsaft")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    installed = Path(unquote(urlparse(str(direct.get("url", ""))).path)).resolve()
    if direct.get("dir_info", {}).get("editable", False) or installed != wheel:
        raise ValueError("installed Engine is not the declared immutable wheel")
    installed_root = Path(distribution.locate_file("")).resolve()
    for module in (epcsaft, equilibrium):
        if installed_root not in Path(module.__file__).resolve().parents:
            raise ValueError(f"{module.__name__} does not originate from the Engine wheel")
    extras = []
    for candidate in importlib.metadata.distributions():
        name = candidate.metadata.get("Name", "")
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        if normalized.startswith("epcsaft") and normalized != "epcsaft":
            extras.append(name)
    if extras:
        raise ValueError(f"retired ePC-SAFT distributions are installed: {sorted(extras)}")
    return {
        "filename": wheel.name,
        "sha256": actual,
        "source_commit": commit,
        "version": distribution.version,
    }


def _source_contract() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    species_rows = {row["species_id"]: row for row in _rows(SOURCE / "species_registry.csv")}
    reaction_rows = {
        row["reaction_id"]: row for row in _rows(SOURCE / "reaction_network.csv")
    }
    missing_species = tuple(identity for identity in SPECIES_IDS if identity not in species_rows)
    missing_reactions = tuple(
        identity
        for identity in ("na_bta_topo_association", "li_na_exchange")
        if identity not in reaction_rows
    )
    if missing_species or missing_reactions:
        raise ValueError(
            f"source contract changed: missing species={missing_species}, reactions={missing_reactions}"
        )
    if any(
        reaction_rows[identity]["evidence_label"] != "source_backed"
        for identity in ("na_bta_topo_association", "li_na_exchange")
    ):
        raise ValueError("configured reactions must remain source-backed")
    return species_rows, reaction_rows


def configured_topology_receipt() -> dict[str, Any]:
    species_rows, reaction_rows = _source_contract()
    charges = tuple(float(species_rows[identity]["charge"]) for identity in SPECIES_IDS)
    balances = (
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # water identity
        (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),  # lithium
        (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0),  # sodium
        (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # chloride
        (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0),  # BTA moiety
        (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0),  # TOPO moiety
    )
    phases = ("aqueous", "organic")
    reactions = {
        "na_bta_topo_association": (
            ("organic", "na_bta", -1.0),
            ("organic", "topo", -1.0),
            ("organic", "na_bta_topo", 1.0),
        ),
        "li_na_exchange": (
            ("aqueous", "li_plus", -1.0),
            ("aqueous", "na_plus", 1.0),
            ("organic", "na_bta_topo", -1.0),
            ("organic", "li_bta_topo", 1.0),
        ),
    }
    supports = {"aqueous": AQUEOUS_SUPPORT, "organic": ORGANIC_SUPPORT}
    # Phase-major, species-minor stacked reaction directions.
    directions = np.zeros((len(reactions), len(phases) * len(SPECIES_IDS)))
    for row, (identity, terms) in enumerate(reactions.items()):
        for phase, species, coefficient in terms:
            if species not in supports[phase]:
                raise ValueError(f"{identity} uses {species} outside support")
            directions[row, phases.index(phase) * len(SPECIES_IDS) + SPECIES_IDS.index(species)] = coefficient
    per_phase = directions.reshape(len(reactions), len(phases), len(SPECIES_IDS))
    # Every source-backed direction must conserve each element/moiety balance across phases.
    if np.any(per_phase.sum(axis=1) @ np.array(balances).T != 0.0):
        raise ValueError("configured reactions violate a conserved balance")
    phase_charges = per_phase @ np.array(charges)
    return {
        "phase_ids": phases,
        "species_ids": SPECIES_IDS,
        "supports": supports,
        "declared_exact_absences": {
            phase: tuple(identity for identity in SPECIES_IDS if identity not in supports[phase]) for phase in phases
        },
        "reaction_ids": tuple(reactions),
        "reaction_source_locators": {identity: reaction_rows[identity]["source_locator"] for identity in reactions},
        "direction_rank": int(np.linalg.matrix_rank(directions)),
        "direction_count": len(reactions),
        "stacked_dimension": directions.shape[1],
        "phase_charge_changes": {
            f"{identity}/{phase}": float(phase_charges[row, column])
            for row, identity in enumerate(reactions)
            for column, phase in enumerate(phases)
        },
        "conserved_quantity_count": len(balances),
    }
