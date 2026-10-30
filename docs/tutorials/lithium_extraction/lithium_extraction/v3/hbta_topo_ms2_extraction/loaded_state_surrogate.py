"""Loaded-organic HBTA/TOPO equilibrium surrogate for IDAES stages."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from idaes.core.surrogate.base.surrogate_base import SurrogateBase
from idaes.core.surrogate.surrogate_block import SurrogateBlock
from pyomo.environ import Constraint, Expression, exp, log

from lithium_extraction.common.support.paths import REPO_ROOT

ARTIFACT = REPO_ROOT / "data/processed/hbta_topo_ms2_engine_comparison/greenfield_current_v3/loaded_state_surrogate.json"
# Later Engine cores verified against the generating states; kept beside the surrogate so its bytes stay unchanged.
VERIFIED_CORES = ARTIFACT.with_name("verified_engine_cores.json")
MODEL_ID = "hbta_topo_ms2_engine_comparison_greenfield_current_v3_loaded_state_v1"
SCHEMA_ID = "hbta-topo-loaded-organic-state-surrogate"
LOADING_EPSILON = 1.0e-8
COEFFICIENTS = ("hidden_weights", "hidden_bias", "output_weights", "output_bias")
# Output maps. The legacy softmax over (Li, Na, vacant) keeps Li + Na loading <= 1 but lets a stage take up more Li
# than enters it (Lithium #159). The bounded form predicts the fraction s_Li of the stage's available Li held in
# the organic and the fraction s_Na of the capacity Li leaves:
#     Li loading = s_Li * softmin(a_Li, 1),  Na loading = s_Na * (1 - Li loading),
# where a_Li = incoming Li loading + aqueous Li / (O/A * carrier) is the stage's Li per unit capacity and s = sigmoid.
# softmin(a, 1) = -ln(exp(-k a) + exp(-k)) / k never exceeds min(a, 1), so Li uptake stays below the aqueous Li
# entering the stage and Li + Na loading below 1 for every input. Na uptake is below the entering Na because
# a_Na >= 46 over the declared input bounds. The mean of the eight networks keeps both bounds.
LEGACY_FORM = "one_hidden_layer_tanh_carrier_softmax"
BOUNDED_FORM = "one_hidden_layer_tanh_li_bounded"
SOFTMIN_SHARPNESS = 200.0  # softmin(a, 1) lies within ln(2)/200 = 0.0035 of min(a, 1)


def load_loaded_state_contract(path: Path | None = None) -> dict[str, Any]:
    artifact = ARTIFACT if path is None else Path(path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("schema_id") != SCHEMA_ID or payload.get("schema_version") != 1:
        raise ValueError("unsupported loaded-organic state-surrogate schema")
    if not payload.get("accepted"):
        raise ValueError("loaded-organic state surrogate did not pass independent verification")
    if payload.get("model_form") not in (LEGACY_FORM, BOUNDED_FORM):
        raise ValueError("loaded-organic state-surrogate model form changed")
    labels = payload.get("input_labels")
    if set(labels or ()) != set(payload.get("input_bounds") or {}):
        raise ValueError("loaded-organic state-surrogate input metadata changed")
    if payload.get("output_labels") != [
        "equilibrium_li_loading_fraction",
        "equilibrium_na_loading_fraction",
    ]:
        raise ValueError("loaded-organic state-surrogate outputs changed")
    feature_labels = payload.get("feature_labels")
    feature_bounds = payload.get("feature_bounds")
    hidden_count = payload.get("hidden_neurons")
    if not isinstance(feature_labels, list) or set(feature_labels) != set(feature_bounds or {}):
        raise ValueError("loaded-organic state-surrogate feature metadata changed")
    if not isinstance(hidden_count, int) or hidden_count <= 0:
        raise ValueError("loaded-organic state-surrogate hidden-layer size changed")
    expected = (
        (len(feature_labels), hidden_count),
        (hidden_count,),
        (hidden_count, 2),
        (2,),
    )
    for member in _members(payload):
        arrays = tuple(np.asarray(member.get(key)) for key in COEFFICIENTS)
        if any(array.shape != shape for array, shape in zip(arrays, expected, strict=True)):
            raise ValueError("loaded-organic state-surrogate coefficient shape changed")
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("loaded-organic state-surrogate coefficients are non-finite")
    return payload


def require_default_surrogate(payload: Mapping[str, Any]) -> None:
    """Refuse any surrogate other than accepted v3 evaluated with the Engine native core that generated it, or with
    a later core whose replay of every generating state reproduced it (VERIFIED_CORES, each with its receipt)."""
    import epcsaft

    if payload.get("model_id") != MODEL_ID or payload.get("accepted") is not True:
        raise RuntimeError(f"expected the accepted {MODEL_ID} surrogate")
    package = Path(epcsaft.__file__).parent
    cores = [payload.get("native_core_sha256")]
    verified = json.loads(VERIFIED_CORES.read_text(encoding="utf-8"))
    if verified["generating_native_core_sha256"] == cores[0]:
        cores += [entry["native_core_sha256"] for entry in verified["verified_engine_cores"]]
    if not any(core and all(hashlib.sha256((package / name).read_bytes()).hexdigest() == digest
                            for name, digest in core.items()) for core in cores):
        raise RuntimeError("installed Engine native core is neither the surrogate's generating nor a verified core")


def _members(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """An ensemble payload lists its networks; the prediction is their mean loading."""
    return payload.get("members") or [payload]


def li_available_numpy(frame: pd.DataFrame) -> np.ndarray:
    """a_Li: the stage's incoming organic plus aqueous Li per unit of organic carrier capacity."""
    capacity = frame["organic_to_aqueous_ratio"].to_numpy(float) * frame["carrier_mol_l_organic"].to_numpy(float)
    return frame["incoming_li_loading_fraction"].to_numpy(float) + frame["li_mol_l_aqueous"].to_numpy(float) / capacity


def softmin_one(available: np.ndarray) -> np.ndarray:
    return -np.logaddexp(-SOFTMIN_SHARPNESS * available, -SOFTMIN_SHARPNESS) / SOFTMIN_SHARPNESS


def member_loadings(arrays, features: np.ndarray, li_available: np.ndarray | None, form: str) -> np.ndarray:
    """One network's equilibrium (Li, Na) loading fractions; li_available is required by the bounded form."""
    hidden_weights, hidden_bias, output_weights, output_bias = (np.asarray(array) for array in arrays)
    logits = np.tanh(features @ hidden_weights + hidden_bias) @ output_weights + output_bias
    if form == LEGACY_FORM:
        exponentials = np.exp(np.clip(logits, -60.0, 60.0))
        return exponentials / (1.0 + np.sum(exponentials, axis=1, keepdims=True))
    fractions = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    li = fractions[:, 0] * softmin_one(li_available)
    return np.column_stack((li, fractions[:, 1] * (1.0 - li)))


def _raw_numpy(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    ratio = frame["organic_to_aqueous_ratio"].to_numpy(float)
    li = frame["li_mol_l_aqueous"].to_numpy(float)
    na = frame["na_mol_l_aqueous"].to_numpy(float)
    hydrogen = frame["hydrogen_mol_l_aqueous"].to_numpy(float)
    carrier = frame["carrier_mol_l_organic"].to_numpy(float)
    li_loading = frame["incoming_li_loading_fraction"].to_numpy(float)
    na_loading = frame["incoming_na_loading_fraction"].to_numpy(float)
    loading = li_loading + na_loading
    capacity = ratio * carrier
    return {
        "organic_to_aqueous_ratio": ratio,
        "log10_li": np.log10(li),
        "log10_na": np.log10(na),
        "log10_hydrogen": np.log10(hydrogen),
        "log10_carrier": np.log10(carrier),
        "incoming_li_loading_fraction": li_loading,
        "incoming_na_loading_fraction": na_loading,
        "incoming_total_loading_fraction": loading,
        "incoming_loaded_li_share": (li_loading + LOADING_EPSILON)
        / (loading + 2.0 * LOADING_EPSILON),
        "temperature_k": frame["temperature_k"].to_numpy(float),
        "log10_li_over_capacity": np.log10(li / capacity),
        "log10_na_over_capacity": np.log10(na / capacity),
        "log10_total_cations_over_capacity": np.log10((li + na + hydrogen) / capacity),
    }


def _features_numpy(frame: pd.DataFrame, payload: Mapping[str, Any]) -> np.ndarray:
    raw = _raw_numpy(frame)
    return np.column_stack(
        [
            2.0 * (raw[label] - payload["feature_bounds"][label][0])
            / (payload["feature_bounds"][label][1] - payload["feature_bounds"][label][0])
            - 1.0
            for label in payload["feature_labels"]
        ]
    )


class LoadedOrganicStateSurrogate(SurrogateBase):
    """Differentiable carrier-occupancy closure for one equilibrium stage."""

    def __init__(self, payload: Mapping[str, Any]):
        self.payload = dict(payload)
        super().__init__(
            input_labels=list(payload["input_labels"]),
            output_labels=list(payload["output_labels"]),
            input_bounds={key: tuple(value) for key, value in payload["input_bounds"].items()},
        )
        self._arrays = [
            tuple(np.asarray(member[key]) for key in ("hidden_weights", "hidden_bias", "output_weights", "output_bias"))
            for member in _members(self.payload)
        ]

    @classmethod
    def from_file(cls, path: Path | None = None) -> "LoadedOrganicStateSurrogate":
        return cls(load_loaded_state_contract(path))

    def evaluate_surrogate(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        features = _features_numpy(dataframe, self.payload)
        available = li_available_numpy(dataframe)
        form = self.payload["model_form"]
        loading = np.mean([member_loadings(arrays, features, available, form) for arrays in self._arrays], axis=0)
        return pd.DataFrame(
            {
                label: loading[:, index]
                for index, label in enumerate(self.output_labels())
            },
            index=dataframe.index,
        )

    def populate_block(self, block: Any, additional_options: dict[str, Any] | None = None) -> None:
        if additional_options:
            raise ValueError(f"unused loaded-state options: {sorted(additional_options)}")
        inputs = block.input_vars_as_dict()
        outputs = block.output_vars_as_dict()
        ratio = inputs["organic_to_aqueous_ratio"]
        li = inputs["li_mol_l_aqueous"]
        na = inputs["na_mol_l_aqueous"]
        hydrogen = inputs["hydrogen_mol_l_aqueous"]
        carrier = inputs["carrier_mol_l_organic"]
        li_loading = inputs["incoming_li_loading_fraction"]
        na_loading = inputs["incoming_na_loading_fraction"]
        loading = li_loading + na_loading
        capacity = ratio * carrier
        raw = {
            "organic_to_aqueous_ratio": ratio,
            "log10_li": log(li) / math.log(10.0),
            "log10_na": log(na) / math.log(10.0),
            "log10_hydrogen": log(hydrogen) / math.log(10.0),
            "log10_carrier": log(carrier) / math.log(10.0),
            "incoming_li_loading_fraction": li_loading,
            "incoming_na_loading_fraction": na_loading,
            "incoming_total_loading_fraction": loading,
            "incoming_loaded_li_share": (li_loading + LOADING_EPSILON)
            / (loading + 2.0 * LOADING_EPSILON),
            "temperature_k": inputs["temperature_k"],
            "log10_li_over_capacity": log(li / capacity) / math.log(10.0),
            "log10_na_over_capacity": log(na / capacity) / math.log(10.0),
            "log10_total_cations_over_capacity": (
                log((li + na + hydrogen) / capacity) / math.log(10.0)
            ),
        }
        scaled = []
        for label in self.payload["feature_labels"]:
            lower, upper = self.payload["feature_bounds"][label]
            scaled.append(2.0 * (raw[label] - lower) / (upper - lower) - 1.0)
        # Named expressions are written once to the solver file instead of inside every neuron.
        block.scaled_features = Expression(range(len(scaled)), rule=lambda _b, i: scaled[i])
        features = list(block.scaled_features.values())
        members = _members(self.payload)

        def hidden_rule(_block: Any, member: int, neuron: int):
            weights = members[member]["hidden_weights"]
            predictor = members[member]["hidden_bias"][neuron] + sum(
                feature * weights[index][neuron] for index, feature in enumerate(features)
            )
            exponential = exp(2.0 * predictor)
            return (exponential - 1.0) / (exponential + 1.0)

        block.hidden_units = Expression(
            [(m, j) for m, member in enumerate(members) for j in range(len(member["hidden_bias"]))],
            rule=hidden_rule,
        )

        def exponential_rule(_block: Any, member: int, output: int):
            weights = members[member]["output_weights"]
            return exp(members[member]["output_bias"][output] + sum(
                block.hidden_units[member, j] * weights[j][output] for j in range(len(weights))
            ))

        block.output_exponentials = Expression(
            [(m, k) for m in range(len(members)) for k in range(2)], rule=exponential_rule
        )
        if self.payload["model_form"] == BOUNDED_FORM:
            available = li_loading + li / capacity
            block.li_capacity_bound = Expression(expr=-log(
                exp(-SOFTMIN_SHARPNESS * available) + exp(-SOFTMIN_SHARPNESS)) / SOFTMIN_SHARPNESS)
            # exp(z) / (1 + exp(z)) is the sigmoid of the output logit z.
            block.member_li = Expression(range(len(members)), rule=lambda b, m: b.output_exponentials[m, 0] / (
                1.0 + b.output_exponentials[m, 0]) * b.li_capacity_bound)

            def rule(_block: Any, label: str):
                if label == "equilibrium_li_loading_fraction":
                    return outputs[label] == sum(block.member_li[m] for m in range(len(members))) / len(members)
                return outputs[label] == sum(
                    block.output_exponentials[m, 1] / (1.0 + block.output_exponentials[m, 1]) * (1.0 - block.member_li[m])
                    for m in range(len(members))
                ) / len(members)
        else:
            def rule(_block: Any, label: str):
                index = self.output_labels().index(label)
                return outputs[label] == sum(
                    block.output_exponentials[m, index]
                    / (1.0 + block.output_exponentials[m, 0] + block.output_exponentials[m, 1])
                    for m in range(len(members))
                ) / len(members)

        block.loaded_state_constraints = Constraint(self.output_labels(), rule=rule)

    def save(self, stream: Any) -> None:
        json.dump(self.payload, stream, indent=2, sort_keys=True)
        stream.write("\n")

    @classmethod
    def load(cls, stream: Any) -> "LoadedOrganicStateSurrogate":
        return cls(json.load(stream))


def input_domain_violations(
    inputs: Mapping[str, float], payload: Mapping[str, Any]
) -> tuple[str, ...]:
    violations = [
        label
        for label, value in inputs.items()
        if label in payload["input_bounds"]
        and not payload["input_bounds"][label][0] <= float(value) <= payload["input_bounds"][label][1]
    ]
    if (
        float(inputs["incoming_li_loading_fraction"])
        + float(inputs["incoming_na_loading_fraction"])
        > float(payload["maximum_incoming_total_loading_fraction"])
    ):
        violations.append("incoming_total_loading_fraction")
    return tuple(violations)


def attach_loaded_state_surrogate(
    parent: Any,
    *,
    input_variables: Mapping[str, Any],
    output_variables: Mapping[str, Any],
    path: Path | None = None,
) -> SurrogateBlock:
    surrogate = LoadedOrganicStateSurrogate.from_file(path)
    if set(input_variables) != set(surrogate.input_labels()):
        raise ValueError("loaded-organic state-surrogate input mapping mismatch")
    if set(output_variables) != set(surrogate.output_labels()):
        raise ValueError("loaded-organic state-surrogate output mapping mismatch")
    block = SurrogateBlock()
    parent.add_component("loaded_state_surrogate", block)
    block.build_model(
        surrogate,
        input_vars=[input_variables[label] for label in surrogate.input_labels()],
        output_vars=[output_variables[label] for label in surrogate.output_labels()],
    )
    return block
