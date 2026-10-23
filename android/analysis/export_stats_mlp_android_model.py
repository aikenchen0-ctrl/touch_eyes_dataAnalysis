#!/usr/bin/env python3
"""Validate and export the statistics MLP as Android JSON assets."""

import argparse
import json
import math
from pathlib import Path

try:
    from . import stats_mlp_contract as contract
except ImportError:
    import stats_mlp_contract as contract


WEIGHT_SHAPES = {
    "shared.0.weight": [24, 60],
    "shared.0.bias": [24],
    "shared.3.weight": [24, 24],
    "shared.3.bias": [24],
    "finger.weight": [2, 24],
    "finger.bias": [2],
    "strength.weight": [2, 24],
    "strength.bias": [2],
}


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def validate_tensor(name, record):
    shape = record.get("shape")
    values = record.get("values")
    if shape != WEIGHT_SHAPES[name] or not isinstance(values, list):
        raise RuntimeError("invalid_weight_shape_" + name.replace(".", "_"))
    expected_count = 1
    for dimension in shape:
        expected_count *= dimension
    if len(values) != expected_count:
        raise RuntimeError("invalid_weight_value_count_" + name.replace(".", "_"))
    if any(
        type(value) not in (int, float) or not math.isfinite(value)
        for value in values
    ):
        raise RuntimeError("invalid_weight_value_" + name.replace(".", "_"))


def validate_contact_size_contract(model_contract):
    """Require the causal contact-area proxy metadata used by the runtime."""
    if model_contract.get("sizeSemantics") != contract.CONTACT_SIZE_SEMANTICS:
        raise RuntimeError("invalid_size_semantics")
    if model_contract.get("forceMeasurement") is not False:
        raise RuntimeError("size_must_not_be_force_measurement")
    if model_contract.get("missingValuePolicy") != (
        contract.CONTACT_SIZE_PROXY_POLICY["missingValuePolicy"]
    ):
        raise RuntimeError("invalid_missing_value_policy")
    if model_contract.get("sizeBaseline") != (
        contract.CONTACT_SIZE_BASELINE_POLICY
    ):
        raise RuntimeError("invalid_size_baseline_policy")


def export_model(source_dir, asset_dir, model_id):
    source_dir = Path(source_dir)
    asset_dir = Path(asset_dir)
    weights = read_json(source_dir / "weights.json")
    model_contract = read_json(source_dir / "model_contract.json")
    if set(weights) != set(WEIGHT_SHAPES):
        raise RuntimeError("invalid_weight_keys")
    for name in WEIGHT_SHAPES:
        validate_tensor(name, weights[name])
    runtime_architecture = model_contract.get("runtimeArchitecture")
    if runtime_architecture not in (
        contract.RUNTIME_ARCHITECTURE,
        contract.LEGACY_RUNTIME_ARCHITECTURE,
    ):
        raise RuntimeError("invalid_runtime_architecture")
    if model_contract.get("featureContract") != contract.FEATURE_CONTRACT:
        raise RuntimeError("invalid_feature_contract")
    if runtime_architecture != contract.RUNTIME_ARCHITECTURE:
        raise RuntimeError("invalid_causal_size_architecture")
    if model_contract.get("sourceFeatureContract") != contract.SOURCE_FEATURE_CONTRACT:
        raise RuntimeError("invalid_source_feature_contract")
    if model_contract.get("sourceFeatureNames") != list(
        contract.RUNTIME_SOURCE_FEATURE_NAMES
    ):
        raise RuntimeError("invalid_source_feature_names")
    validate_contact_size_contract(model_contract)
    if model_contract.get("featureNames") != list(contract.PHYSICAL_FEATURE_NAMES):
        raise RuntimeError("invalid_sequence_feature_names")
    if model_contract.get("sourceFeatureIndices") != list(contract.SOURCE_FEATURE_INDICES):
        raise RuntimeError("invalid_source_feature_indices")
    if model_contract.get("statsFeatureNames") != list(contract.STATS_FEATURE_NAMES):
        raise RuntimeError("invalid_stats_feature_names")
    if model_contract.get("matrixShape") != [
        contract.FRAME_COUNT,
        len(contract.PHYSICAL_FEATURE_NAMES),
    ]:
        raise RuntimeError("invalid_matrix_shape")
    if model_contract.get("frameCount") != contract.FRAME_COUNT:
        raise RuntimeError("invalid_frame_count")
    if model_contract.get("sequenceChannelCount") != len(
        contract.PHYSICAL_FEATURE_NAMES
    ):
        raise RuntimeError("invalid_sequence_channel_count")
    if model_contract.get("statsFeatureCount") != contract.STATS_FEATURE_COUNT:
        raise RuntimeError("invalid_stats_feature_count")
    if model_contract.get("inputCount") != contract.STATS_FEATURE_COUNT:
        raise RuntimeError("invalid_input_count")
    runtime_architecture = model_contract.get("runtimeArchitecture")
    expected_label_order = (
        contract.GATE_LABELS
        if runtime_architecture == contract.RUNTIME_ARCHITECTURE
        else contract.CLASS_LABELS
    )
    if model_contract.get("labelOrder") != list(expected_label_order):
        raise RuntimeError("invalid_label_order")
    if model_contract.get("classCount") != len(expected_label_order):
        raise RuntimeError("invalid_class_count")
    if model_contract.get("fingerLabelOrder") != list(contract.FINGER_LABELS):
        raise RuntimeError("invalid_finger_label_order")
    if model_contract.get("strengthLabelOrder") != list(contract.STRENGTH_LABELS):
        raise RuntimeError("invalid_strength_label_order")
    if runtime_architecture == contract.RUNTIME_ARCHITECTURE:
        if model_contract.get("primaryTask") != contract.PRIMARY_TASK:
            raise RuntimeError("invalid_primary_task")
        if model_contract.get("gateCount") != len(contract.GATE_LABELS):
            raise RuntimeError("invalid_gate_count")
        if model_contract.get("gateLabelOrder") != list(contract.GATE_LABELS):
            raise RuntimeError("invalid_gate_label_order")
        if model_contract.get("negativeClassLabels") != list(
            contract.NEGATIVE_CLASS_LABELS
        ):
            raise RuntimeError("invalid_negative_class_labels")
        if model_contract.get("positiveClassLabels") != list(
            contract.POSITIVE_CLASS_LABELS
        ):
            raise RuntimeError("invalid_positive_class_labels")
        if model_contract.get("diagnosticClassCount") != len(
            contract.CLASS_LABELS
        ):
            raise RuntimeError("invalid_diagnostic_class_count")
        if model_contract.get("diagnosticClassLabelOrder") != list(
            contract.CLASS_LABELS
        ):
            raise RuntimeError("invalid_diagnostic_class_label_order")
    if model_contract.get("hiddenCount") != 24:
        raise RuntimeError("invalid_hidden_count")
    if model_contract.get("dropoutAtInference") is not False:
        raise RuntimeError("dropout_must_be_disabled_at_inference")
    if model_contract.get("deploymentAllowed") is not False:
        raise RuntimeError("experimental_model_cannot_be_deployment_allowed")
    if model_contract.get("evaluationScope") != "all_data_fit_only":
        raise RuntimeError("invalid_evaluation_scope")
    sequence_channels = model_contract.get("sequenceChannels")
    if not isinstance(sequence_channels, list) or len(sequence_channels) != len(
        contract.PHYSICAL_FEATURE_NAMES
    ):
        raise RuntimeError("invalid_sequence_channels")
    for index, name in enumerate(contract.PHYSICAL_FEATURE_NAMES):
        channel = sequence_channels[index]
        if channel.get("name") != name:
            raise RuntimeError("invalid_sequence_channel_order")
        center = channel.get("center")
        scale = channel.get("scale")
        clip_min = channel.get("clipMin")
        clip_max = channel.get("clipMax")
        if type(center) not in (int, float) or not math.isfinite(center):
            raise RuntimeError("invalid_sequence_channel_center")
        if (
            type(scale) not in (int, float)
            or not math.isfinite(scale)
            or scale <= 0
        ):
            raise RuntimeError("invalid_sequence_channel_scale")
        if (
            type(clip_min) not in (int, float)
            or type(clip_max) not in (int, float)
            or not math.isfinite(clip_min)
            or not math.isfinite(clip_max)
            or clip_min >= clip_max
        ):
            raise RuntimeError("invalid_sequence_channel_clip")
    stats_scaler = model_contract.get("statsScaler")
    if not isinstance(stats_scaler, dict):
        raise RuntimeError("missing_stats_scaler")
    stats_center = stats_scaler.get("center", [])
    stats_scale = stats_scaler.get("scale", [])
    if (
        not isinstance(stats_center, list)
        or len(stats_center) != contract.STATS_FEATURE_COUNT
        or any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in stats_center
        )
    ):
        raise RuntimeError("invalid_stats_scaler_center")
    if (
        not isinstance(stats_scale, list)
        or len(stats_scale) != contract.STATS_FEATURE_COUNT
        or any(
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value <= 0
            for value in stats_scale
        )
    ):
        raise RuntimeError("invalid_stats_scaler_scale")
    if stats_scaler.get("clipMin") != contract.STATS_CLIP_MIN:
        raise RuntimeError("invalid_stats_clip_min")
    if stats_scaler.get("clipMax") != contract.STATS_CLIP_MAX:
        raise RuntimeError("invalid_stats_clip_max")

    weights_name = "AispectStatsMlpWeights_" + model_id + ".json"
    scaler_name = "AispectStatsMlpScaler_" + model_id + ".json"
    write_json(asset_dir / weights_name, weights)
    write_json(asset_dir / scaler_name, model_contract)
    return weights_name, scaler_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--asset-dir", required=True)
    parser.add_argument("--model-id", required=True)
    arguments = parser.parse_args()
    export_model(arguments.source_dir, arguments.asset_dir, arguments.model_id)


if __name__ == "__main__":
    main()
