#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


LABEL_ORDER = ["thumb_light", "thumb_heavy", "index_light", "index_heavy"]
RUNTIME_ARCHITECTURE = "time_grid_groupnorm_cnn_v1"
FEATURE_CONTRACT = "causal_time_grid_imu11_touch7_mask5_v3"
SUPPORTED_FEATURE_CONTRACTS = {
    FEATURE_CONTRACT,
    "causal_time_grid_touch7_mask5_no_gravity_v1",
    "causal_time_grid_imu11_touch7_mask5_delta_gravity_v1",
}


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def export_model(source_dir, asset_dir, model_id):
    source_dir = Path(source_dir)
    asset_dir = Path(asset_dir)
    weights = read_json(source_dir / "weights.json")
    scaler = read_json(source_dir / "scaler.json")
    contract = read_json(source_dir / "model_contract.json")
    feature_names = contract["featureNames"]
    matrix_shape = contract.get("matrixShape")
    if not isinstance(matrix_shape, list) or len(matrix_shape) != 2 or matrix_shape[0] != 9:
        raise RuntimeError("invalid_time_grid_source_contract")
    feature_count = int(matrix_shape[1])
    if feature_count not in (20, 23) or len(feature_names) != feature_count:
        raise RuntimeError("invalid_time_grid_source_contract")
    if scaler.get("featureNames") != feature_names or len(scaler.get("channels", [])) != feature_count:
        raise RuntimeError("invalid_time_grid_source_scaler")
    runtime_contract = str(contract.get("featureContract") or "")
    if runtime_contract == "all_data_channel_ablation_touch23_v1":
        runtime_contract = FEATURE_CONTRACT
    if runtime_contract not in SUPPORTED_FEATURE_CONTRACTS:
        raise RuntimeError("unsupported_time_grid_feature_contract")
    required_weights = {
        "conv1.weight": [32, feature_count, 3],
        "conv1.bias": [32],
        "norm1.weight": [32],
        "norm1.bias": [32],
        "conv2.weight": [32, 32, 3],
        "conv2.bias": [32],
        "norm2.weight": [32],
        "norm2.bias": [32],
        "output.weight": [4, 32],
        "output.bias": [4],
    }
    for name, shape in required_weights.items():
        if weights.get(name, {}).get("shape") != shape:
            raise RuntimeError("invalid_weight_shape_" + name.replace(".", "_"))
    output_scaler = {
        "runtimeArchitecture": RUNTIME_ARCHITECTURE,
        "featureContract": runtime_contract,
        "featureNames": feature_names,
        "matrixShape": [9, feature_count],
        "frameCount": 9,
        "classCount": 4,
        "labelOrder": LABEL_ORDER,
        "channels": scaler["channels"],
        "evaluationScope": "all_data_fit_only",
        "deploymentAllowed": False,
    }
    asset_dir.mkdir(parents=True, exist_ok=True)
    weights_name = "AispectTimeGridGroupNormWeights_" + model_id + ".json"
    scaler_name = "AispectTimeGridGroupNormScaler_" + model_id + ".json"
    write_json(asset_dir / weights_name, weights)
    write_json(asset_dir / scaler_name, output_scaler)
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
