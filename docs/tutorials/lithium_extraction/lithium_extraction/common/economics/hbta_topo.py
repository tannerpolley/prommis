"""Shared HBTA/TOPO screening-cost assumptions."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from lithium_extraction.common.support.paths import REPO_ROOT


ASSUMPTIONS_RELATIVE_PATH = Path("data/reference/tea/hbta_topo_screening_cost_assumptions.csv")


@dataclass(frozen=True)
class ScreeningCostAssumptions:
    annual_operating_hours: float
    hcl_31pct_solution_price: float
    naoh_price: float
    formulated_organic_price: float
    formulated_organic_density: float
    extraction_stage_operation_auxiliary: float
    electricity_price: float
    scrub_strip_regeneration_auxiliary: float
    waste_handling_allowance: float
    settler_depth: float
    mixer_power_density: float
    pump_head: float
    pump_efficiency: float
    acid_recovery_fraction: float
    organic_entrainment_loss_fraction: float
    organic_degradation_loss_fraction: float
    organic_purge_fraction: float
    fixed_opex_fraction: float
    capital_contingency_fraction: float
    capital_recovery_factor: float

    def __post_init__(self) -> None:
        if any(
            not isfinite(getattr(self, name))
            for name in self.__dataclass_fields__
        ):
            raise ValueError("screening-cost assumptions must be finite")
        positive = (
            "annual_operating_hours",
            "formulated_organic_density",
            "settler_depth",
            "mixer_power_density",
            "pump_head",
            "capital_recovery_factor",
        )
        nonnegative = (
            "hcl_31pct_solution_price",
            "naoh_price",
            "formulated_organic_price",
            "extraction_stage_operation_auxiliary",
            "electricity_price",
            "scrub_strip_regeneration_auxiliary",
            "waste_handling_allowance",
        )
        fractions = (
            "acid_recovery_fraction",
            "organic_entrainment_loss_fraction",
            "organic_degradation_loss_fraction",
            "organic_purge_fraction",
            "fixed_opex_fraction",
            "capital_contingency_fraction",
        )
        if any(getattr(self, name) <= 0.0 for name in positive):
            raise ValueError("positive screening-cost assumptions must exceed zero")
        if any(getattr(self, name) < 0.0 for name in nonnegative):
            raise ValueError("screening-cost prices and allowances must be nonnegative")
        if any(not 0.0 <= getattr(self, name) <= 1.0 for name in fractions):
            raise ValueError("screening-cost fractions must be between zero and one")
        if self.annual_operating_hours > 8784.0:
            raise ValueError("annual operating hours cannot exceed a leap year")
        if not 0.0 < self.pump_efficiency <= 1.0:
            raise ValueError("pump efficiency must be between zero and one")
        if self.capital_recovery_factor > 1.0:
            raise ValueError("capital recovery factor must not exceed one")


def load_screening_cost_assumptions(
    root: Path | None = None,
    *,
    basis: str = "base",
) -> ScreeningCostAssumptions:
    """Load a named low/base/high HBTA/TOPO assumption row."""

    if basis not in {"base", "low", "high"}:
        raise ValueError("cost assumption basis must be base, low, or high")
    root = REPO_ROOT if root is None else Path(root)
    with (root / ASSUMPTIONS_RELATIVE_PATH).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        raw_bounds = tuple(row[name] for name in ("low", "base", "high"))
        if not any(raw_bounds):
            continue
        if not all(raw_bounds):
            raise ValueError("screening-cost bounds must provide low, base, and high together")
        bounds = tuple(float(number) for number in raw_bounds)
        if not all(isfinite(number) for number in bounds) or not bounds[0] <= bounds[1] <= bounds[2]:
            raise ValueError("screening-cost bounds must be finite and ordered low <= base <= high")
    values = {row["parameter"]: float(row[basis]) for row in rows if row[basis]}
    return ScreeningCostAssumptions(**values)
