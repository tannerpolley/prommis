"""Canonical repository paths shared by the named flowsheet owners."""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath


def project_root(start: Path | None = None) -> Path:
    current = Path(__file__).resolve() if start is None else Path(start).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "lithium_extraction").is_dir() and (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not locate Lithium_Extraction repository root from {current}")


REPO_ROOT = project_root()
# The Engine's AMPL external-function library, next to its Python package.
ENGINE_ASL_LIBRARY = "epcsaft_asl.dll" if os.name == "nt" else "libepcsaft_asl.so"
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"
FLOWSHEETS_CATALOG = REPO_ROOT / "docs" / "flowsheets.md"
TBAC_DOCS_DIR = REPO_ROOT / "lithium_extraction" / "legacy" / "tbac_da_topo_produced_water" / "docs"
TBAC_FIGURES_DIR = TBAC_DOCS_DIR / "figures"
SLIDE_FIGURES_DIR = REPO_ROOT / "docs" / "slides" / "case_study_tbac_da_topo_produced_water" / "figures"
SOLVENT_SCREENING_SOURCE = REFERENCE_DIR / "produced_water" / "canonical_flowsheet_solvent_screening_source.csv"
ALAMO_SURROGATE_JSON = PROCESSED_DIR / "canonical_flowsheet_alamo_surrogate.json"


def repo_relative(path: Path) -> str:
    return Path(path).resolve().relative_to(REPO_ROOT).as_posix()


# On 2026-10-08 analyses moved into analyses/<bucket>/ and flowsheet packages into
# lithium_extraction/{common,v3,prototypes,legacy}/. Retained results keep the paths they were written with.
ANALYSIS_BUCKETS = ("shared", "v3", "prototypes", "screening", "engine_benchmarks", "legacy")
_COMMON_PACKAGES = ("support", "economics", "evidence", "thermodynamics")
_MOVED_FILES = {
    "lithium_extraction/flowsheets/hbta_topo_ms2_saponified/pretreatment.py": "lithium_extraction/common/pretreatment.py",
    "lithium_extraction/flowsheets/tbac_da_topo_produced_water/precipitation_payload.py":
        "lithium_extraction/common/precipitation_payload.py",
}


def current_path(recorded: str) -> str:
    """The current location of a repository-relative path recorded before the bucket move."""
    if recorded in _MOVED_FILES:
        return _MOVED_FILES[recorded]
    parts = PurePosixPath(recorded).parts
    if len(parts) > 1 and parts[0] == "analyses" and not (REPO_ROOT / "analyses" / parts[1]).exists():
        found = [bucket for bucket in ANALYSIS_BUCKETS if (REPO_ROOT / "analyses" / bucket / parts[1]).is_dir()]
        if len(found) == 1:
            return PurePosixPath("analyses", found[0], *parts[1:]).as_posix()
    if len(parts) > 2 and parts[:2] == ("lithium_extraction", "flowsheets"):
        found = [bucket for bucket in ("v3", "prototypes", "legacy") if (REPO_ROOT / "lithium_extraction" / bucket / parts[2]).is_dir()]
        if len(found) == 1:
            return PurePosixPath("lithium_extraction", found[0], *parts[2:]).as_posix()
    if len(parts) > 1 and parts[0] == "lithium_extraction" and parts[1] in _COMMON_PACKAGES:
        return PurePosixPath("lithium_extraction", "common", *parts[1:]).as_posix()
    return recorded


def pre_bucket_path(current: str) -> str:
    """The path a current repository file had before the bucket move, for reading historical revisions."""
    for old, new in _MOVED_FILES.items():
        if current == new:
            return old
    current = re.sub(r"^analyses/(" + "|".join(ANALYSIS_BUCKETS) + ")/", "analyses/", current)
    current = re.sub(r"^lithium_extraction/common/(" + "|".join(_COMMON_PACKAGES) + ")/", r"lithium_extraction/\1/", current)
    return re.sub(r"^lithium_extraction/(v3|prototypes|legacy)/", "lithium_extraction/flowsheets/", current)
