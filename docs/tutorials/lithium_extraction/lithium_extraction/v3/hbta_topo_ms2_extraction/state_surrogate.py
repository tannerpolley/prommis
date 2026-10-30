"""Reusable single-stage HBTA/TOPO reactive-LLE surrogate.

The surrogate maps physical inlet state variables to equilibrium metal-transfer
fractions.  Stage count, process connectivity, and absolute throughput remain
caller-owned.  Distribution and loading statistics are exact transformations
of the two predicted transfer fractions and the declared inlet basis.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from idaes.core.surrogate.base.surrogate_base import SurrogateBase
from idaes.core.surrogate.surrogate_block import SurrogateBlock
from pyomo.environ import Constraint, exp, log

from lithium_extraction.common.support.paths import REPO_ROOT

ARTIFACT = REPO_ROOT / "data/processed/hbta_topo_reactive_lle_surrogate/state_surrogate.json"
SCHEMA_ID = "hbta-topo-reactive-lle-state-surrogate"


def load_state_surrogate_contract(path: Path | None = None) -> dict[str, Any]:
    artifact = ARTIFACT if path is None else Path(path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("schema_id") != SCHEMA_ID or payload.get("schema_version") != 1:
        raise ValueError("unsupported reactive-LLE state-surrogate schema")
    if not payload.get("accepted"):
        raise ValueError("reactive-LLE state surrogate did not pass retained verification")
    labels = payload.get("input_labels")
    bounds = payload.get("input_bounds")
    transforms = payload.get("input_transforms")
    if not isinstance(labels, list) or not labels:
        raise ValueError("reactive-LLE state-surrogate inputs are missing")
    if set(labels) != set(bounds or {}) or set(labels) != set(transforms or {}):
        raise ValueError("reactive-LLE state-surrogate input metadata changed")
    if payload.get("output_labels") != ["li_extraction_fraction", "na_extraction_fraction"]:
        raise ValueError("reactive-LLE state-surrogate outputs changed")
    if payload.get("model_form") != "one_hidden_layer_tanh_logit":
        raise ValueError("reactive-LLE state-surrogate model form changed")
    feature_labels = payload.get("feature_labels")
    feature_bounds = payload.get("feature_bounds")
    hidden_count = payload.get("hidden_neurons")
    if not isinstance(feature_labels, list) or set(feature_labels) != set(
        feature_bounds or {}
    ):
        raise ValueError("reactive-LLE state-surrogate feature metadata changed")
    if not isinstance(hidden_count, int) or hidden_count <= 0:
        raise ValueError("reactive-LLE state-surrogate hidden-layer size changed")
    arrays = (
        np.asarray(payload.get("hidden_weights")),
        np.asarray(payload.get("hidden_bias")),
        np.asarray(payload.get("output_weights")),
        np.asarray(payload.get("output_bias")),
    )
    expected_shapes = (
        (len(feature_labels), hidden_count),
        (hidden_count,),
        (hidden_count, 2),
        (2,),
    )
    if any(array.shape != shape for array, shape in zip(arrays, expected_shapes, strict=True)):
        raise ValueError("reactive-LLE state-surrogate coefficient shape changed")
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("reactive-LLE state-surrogate coefficients are non-finite")
    return payload


def _feature_numpy(dataframe: pd.DataFrame, payload: Mapping[str, Any]) -> np.ndarray:
    ratio = dataframe["organic_to_aqueous_ratio"].to_numpy(dtype=float)
    li = dataframe["li_mol_l_aqueous"].to_numpy(dtype=float)
    na = dataframe["na_mol_l_aqueous"].to_numpy(dtype=float)
    hydrogen = dataframe["hydrogen_mol_l_aqueous"].to_numpy(dtype=float)
    carrier = dataframe["available_carrier_mol_l_organic"].to_numpy(dtype=float)
    capacity = ratio * carrier
    raw = {
        "organic_to_aqueous_ratio": ratio,
        "log10_li": np.log10(li),
        "log10_na": np.log10(na),
        "log10_hydrogen": np.log10(hydrogen),
        "log10_carrier": np.log10(carrier),
        "temperature_k": dataframe["temperature_k"].to_numpy(dtype=float),
        "log10_li_over_capacity": np.log10(li / capacity),
        "log10_na_over_capacity": np.log10(na / capacity),
        "log10_hydrogen_over_capacity": np.log10(hydrogen / capacity),
        "log10_total_cations_over_capacity": np.log10((li + na + hydrogen) / capacity),
    }
    columns = []
    for label in payload["feature_labels"]:
        lower, upper = payload["feature_bounds"][label]
        columns.append(2.0 * (raw[label] - lower) / (upper - lower) - 1.0)
    return np.column_stack(columns)


class ReactiveLleStateSurrogate(SurrogateBase):
    """IDAES-compatible bounded neural-logit equilibrium surrogate."""

    def __init__(self, payload: Mapping[str, Any]):
        self.payload = dict(payload)
        super().__init__(
            input_labels=list(payload["input_labels"]),
            output_labels=list(payload["output_labels"]),
            input_bounds={key: tuple(value) for key, value in payload["input_bounds"].items()},
        )

    @classmethod
    def from_file(cls, path: Path | None = None) -> "ReactiveLleStateSurrogate":
        return cls(load_state_surrogate_contract(path))

    def evaluate_surrogate(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        features = _feature_numpy(dataframe, self.payload)
        hidden = np.tanh(
            features @ np.asarray(self.payload["hidden_weights"])
            + np.asarray(self.payload["hidden_bias"])
        )
        predictors = hidden @ np.asarray(self.payload["output_weights"]) + np.asarray(
            self.payload["output_bias"]
        )
        fractions = 1.0 / (1.0 + np.exp(-predictors))
        result = {
            label: fractions[:, index] for index, label in enumerate(self.output_labels())
        }
        return pd.DataFrame(result, index=dataframe.index)

    def populate_block(self, block: Any, additional_options: dict[str, Any] | None = None) -> None:
        if additional_options:
            raise ValueError(f"unused state-surrogate options: {sorted(additional_options)}")
        inputs = block.input_vars_as_dict()
        outputs = block.output_vars_as_dict()
        ratio = inputs["organic_to_aqueous_ratio"]
        li = inputs["li_mol_l_aqueous"]
        na = inputs["na_mol_l_aqueous"]
        hydrogen = inputs["hydrogen_mol_l_aqueous"]
        carrier = inputs["available_carrier_mol_l_organic"]
        capacity = ratio * carrier
        raw = {
            "organic_to_aqueous_ratio": ratio,
            "log10_li": log(li) / math.log(10.0),
            "log10_na": log(na) / math.log(10.0),
            "log10_hydrogen": log(hydrogen) / math.log(10.0),
            "log10_carrier": log(carrier) / math.log(10.0),
            "temperature_k": inputs["temperature_k"],
            "log10_li_over_capacity": log(li / capacity) / math.log(10.0),
            "log10_na_over_capacity": log(na / capacity) / math.log(10.0),
            "log10_hydrogen_over_capacity": log(hydrogen / capacity) / math.log(10.0),
            "log10_total_cations_over_capacity": (
                log((li + na + hydrogen) / capacity) / math.log(10.0)
            ),
        }
        features = []
        for label in self.payload["feature_labels"]:
            lower, upper = self.payload["feature_bounds"][label]
            features.append(2.0 * (raw[label] - lower) / (upper - lower) - 1.0)

        hidden = []
        for hidden_index, bias in enumerate(self.payload["hidden_bias"]):
            predictor = bias + sum(
                features[index] * self.payload["hidden_weights"][index][hidden_index]
                for index in range(len(features))
            )
            exponential = exp(2.0 * predictor)
            hidden.append((exponential - 1.0) / (exponential + 1.0))

        def rule(_block: Any, label: str):
            output_index = self.output_labels().index(label)
            predictor = self.payload["output_bias"][output_index] + sum(
                hidden[index] * self.payload["output_weights"][index][output_index]
                for index in range(len(hidden))
            )
            return outputs[label] == 1.0 / (1.0 + exp(-predictor))

        block.reactive_lle_state_constraints = Constraint(self.output_labels(), rule=rule)

    def save(self, stream: Any) -> None:
        json.dump(self.payload, stream, indent=2, sort_keys=True)
        stream.write("\n")

    @classmethod
    def load(cls, stream: Any) -> "ReactiveLleStateSurrogate":
        return cls(json.load(stream))


def equilibrium_statistics(
    inputs: Mapping[str, float],
    outputs: Mapping[str, float],
) -> dict[str, float]:
    ratio = float(inputs["organic_to_aqueous_ratio"])
    li_fraction = float(outputs["li_extraction_fraction"])
    na_fraction = float(outputs["na_extraction_fraction"])
    li_distribution = li_fraction / ((1.0 - li_fraction) * ratio)
    na_distribution = na_fraction / ((1.0 - na_fraction) * ratio)
    li_loaded = float(inputs["li_mol_l_aqueous"]) * li_fraction
    na_loaded = float(inputs["na_mol_l_aqueous"]) * na_fraction
    return {
        "li_extraction_fraction": li_fraction,
        "na_extraction_fraction": na_fraction,
        "li_distribution_coefficient": li_distribution,
        "na_distribution_coefficient": na_distribution,
        "li_na_selectivity": li_distribution / na_distribution,
        "loaded_organic_li_fraction": li_loaded / (li_loaded + na_loaded),
        "hydrogen_out_mol_l_aqueous": (
            float(inputs["hydrogen_mol_l_aqueous"]) + li_loaded + na_loaded
        ),
    }


def attach_state_surrogate(
    parent: Any,
    *,
    input_variables: Mapping[str, Any],
    output_variables: Mapping[str, Any],
    path: Path | None = None,
) -> SurrogateBlock:
    surrogate = ReactiveLleStateSurrogate.from_file(path)
    if set(input_variables) != set(surrogate.input_labels()):
        raise ValueError("reactive-LLE state-surrogate input mapping mismatch")
    if set(output_variables) != set(surrogate.output_labels()):
        raise ValueError("reactive-LLE state-surrogate output mapping mismatch")
    if hasattr(parent, "reactive_lle_state_surrogate"):
        raise ValueError("reactive-LLE state surrogate is already attached")
    block = parent.reactive_lle_state_surrogate = SurrogateBlock()
    block.build_model(
        surrogate,
        input_vars=[input_variables[label] for label in surrogate.input_labels()],
        output_vars=[output_variables[label] for label in surrogate.output_labels()],
    )
    return block
