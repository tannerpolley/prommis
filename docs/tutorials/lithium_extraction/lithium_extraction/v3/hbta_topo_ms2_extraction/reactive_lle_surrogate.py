"""IDAES interface for the calibrated HBTA/TOPO reactive-LLE surrogate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from idaes.core.surrogate.alamopy import AlamoSurrogate
from idaes.core.surrogate.surrogate_block import SurrogateBlock
from pyomo.util.calc_var_value import calculate_variable_from_constraint

from lithium_extraction.common.support.paths import REPO_ROOT

ARTIFACT = REPO_ROOT / "data/processed/hbta_topo_reactive_lle_surrogate/surrogate.json"
MODEL_ID = "zhang2017_calibrated_h1_epcsaft_reactive_lle_v1"
INPUT_LABEL = "organic_to_aqueous_ratio"


def load_reactive_lle_contract(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the immutable application-surrogate contract."""

    artifact = ARTIFACT if path is None else Path(path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("schema_id") != "hbta-topo-reactive-lle-surrogate":
        raise ValueError("unsupported reactive-LLE surrogate schema")
    if payload.get("schema_version") != 1 or payload.get("model_id") != MODEL_ID:
        raise ValueError("unsupported reactive-LLE surrogate model")
    if payload.get("input_labels") != [INPUT_LABEL]:
        raise ValueError("reactive-LLE surrogate input identity changed")
    domain = payload.get("domain")
    if not isinstance(domain, dict):
        raise ValueError("reactive-LLE surrogate domain is missing")
    bounds = domain.get(INPUT_LABEL)
    if not isinstance(bounds, list) or len(bounds) != 2 or not 0.0 < bounds[0] < bounds[1]:
        raise ValueError("reactive-LLE surrogate O/A bounds are invalid")
    if not payload.get("accepted"):
        raise ValueError("reactive-LLE surrogate did not pass its retained verification")
    if payload["verification_max_abs_fraction_error"] > payload["acceptance_tolerance_abs_fraction"]:
        raise ValueError("reactive-LLE surrogate verification exceeds its retained tolerance")
    return payload


def _labels(stage_count: int) -> list[str]:
    if stage_count != 1:
        raise ValueError("the accepted reactive-LLE surrogate supports one extraction stage")
    return [
        f"stage_{stage}_{component}_fraction"
        for stage in range(1, stage_count + 1)
        for component in ("li", "na")
    ]


def _polynomial(coefficients: list[float]) -> str:
    return " + ".join(
        f"({coefficient:.17g})*{INPUT_LABEL}**{power}"
        for power, coefficient in enumerate(coefficients)
    )


def load_reactive_lle_surrogate(
    stage_count: int,
    path: Path | None = None,
) -> tuple[AlamoSurrogate, float]:
    """Build the IDAES surrogate for a fixed number of fresh-organic stages."""

    payload = load_reactive_lle_contract(path)
    labels = _labels(stage_count)
    expressions = {}
    for label in labels:
        record = payload["expressions"].get(label)
        if not isinstance(record, dict):
            raise ValueError(f"reactive-LLE surrogate output is missing: {label}")
        coefficients = record.get("coefficients_in_ascending_power")
        if not isinstance(coefficients, list) or not coefficients:
            raise ValueError(f"reactive-LLE coefficients are missing: {label}")
        expressions[label] = f"{label} == {_polynomial(coefficients)}"
    bounds = tuple(payload["domain"][INPUT_LABEL])
    surrogate = AlamoSurrogate(
        surrogate_expressions=expressions,
        input_labels=[INPUT_LABEL],
        output_labels=labels,
        input_bounds={INPUT_LABEL: bounds},
    )
    return surrogate, float(payload["verification_max_abs_fraction_error"])


def evaluate_reactive_lle_surrogate(
    organic_to_aqueous_ratio: float,
    stage_count: int,
    path: Path | None = None,
) -> dict[str, float]:
    """Evaluate the retained polynomial independently of a Pyomo model."""

    payload = load_reactive_lle_contract(path)
    lower, upper = payload["domain"][INPUT_LABEL]
    if not lower <= organic_to_aqueous_ratio <= upper:
        raise ValueError(f"O/A lies outside the reactive-LLE surrogate domain [{lower}, {upper}]")
    values = {}
    for label in _labels(stage_count):
        coefficients = payload["expressions"][label]["coefficients_in_ascending_power"]
        values[label] = sum(
            coefficient * organic_to_aqueous_ratio**power
            for power, coefficient in enumerate(coefficients)
        )
    return values


def attach_reactive_lle_surrogate(
    parent: Any,
    *,
    organic_to_aqueous_ratio: Any,
    output_variables: Mapping[str, Any],
    stage_count: int,
    path: Path | None = None,
) -> SurrogateBlock:
    """Attach the surrogate to caller-owned flowsheet variables."""

    surrogate, _ = load_reactive_lle_surrogate(stage_count, path)
    labels = _labels(stage_count)
    if set(output_variables) != set(labels):
        missing = sorted(set(labels) - set(output_variables))
        extra = sorted(set(output_variables) - set(labels))
        raise ValueError(f"reactive-LLE output mapping mismatch: missing={missing}, extra={extra}")
    if hasattr(parent, "reactive_lle_surrogate"):
        raise ValueError("reactive-LLE surrogate is already attached")
    lower, upper = surrogate.input_bounds()[INPUT_LABEL]
    organic_to_aqueous_ratio.setlb(lower)
    organic_to_aqueous_ratio.setub(upper)
    block = parent.reactive_lle_surrogate = SurrogateBlock()
    outputs = [output_variables[label] for label in labels]
    block.build_model(
        surrogate,
        input_vars=[organic_to_aqueous_ratio],
        output_vars=outputs,
    )
    for label, variable in zip(labels, outputs, strict=True):
        calculate_variable_from_constraint(variable, block.alamo_constraint[label])
    return block
