"""Supported Li2CO3 database-extension loader for sourced Reaktoro-PSE runs."""

from __future__ import annotations

from pathlib import Path

from lithium_extraction.common.support.paths import REPO_ROOT


SELECTED_DATABASE = "pitzer.dat"
ACTIVITY_MODEL = "ActivityModelPitzer"
CUSTOM_EXTENSION_RELATIVE = Path("data/reference/thermodynamics/reaktoro_li2co3_pitzer_extension.dat")


def project_root() -> Path:
    return REPO_ROOT


def custom_pitzer_contents(root: Path) -> str:
    import reaktoro

    base_contents = reaktoro.PhreeqcDatabase.contents(SELECTED_DATABASE)
    extension = (root / CUSTOM_EXTENSION_RELATIVE).read_text(encoding="utf-8").strip()
    lines = base_contents.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "END":
            return "\n".join(lines[:index] + ["", extension, ""] + lines[index:]) + "\n"
    raise ValueError(f"{SELECTED_DATABASE} embedded contents do not contain an END marker")
