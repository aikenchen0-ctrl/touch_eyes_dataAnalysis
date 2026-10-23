#!/usr/bin/env python3
"""Fit the fixed-budget statistics MLP candidate and write export inputs."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

try:
    from . import contact_size_proxy
    from . import gate_threshold_metrics
    from . import run_lightweight_time_grid_models as lightweight
    from . import stats_mlp_contract as contract
except ImportError:
    import contact_size_proxy
    import gate_threshold_metrics
    import run_lightweight_time_grid_models as lightweight
    import stats_mlp_contract as contract


RUNTIME_STATE_NAMES = frozenset((
    "shared.0.weight",
    "shared.0.bias",
    "shared.3.weight",
    "shared.3.bias",
    "finger.weight",
    "finger.bias",
    "strength.weight",
    "strength.bias",
))
FIXED_FPR_TARGETS = (0.01, 0.05)


def configure_determinism(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def source_matrix(row):
    """Validate the persisted v5 matrix before deriving the v6 runtime view."""
    if row.get("featureContract") != contract.LEGACY_V3_SOURCE_FEATURE_CONTRACT:
        raise RuntimeError("invalid_source_feature_contract")
    if row.get("featureNames") != list(contract.SOURCE_FEATURE_NAMES):
        raise RuntimeError("invalid_source_feature_names")
    matrix = np.asarray(row.get("matrix"), dtype=np.float64)
    if matrix.shape != (
        contract.FRAME_COUNT,
        len(contract.SOURCE_FEATURE_NAMES),
    ):
        raise RuntimeError("invalid_source_matrix")
    return matrix


def prepare_causal_source_matrices(rows):
    """Build v6 physical matrices through the shared causal transform."""
    for row in rows:
        source_matrix(row)
    return contact_size_proxy.transform_rows(rows)


def finite_fit_values(values):
    """Select finite samples for training-only scalers."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return flat[np.isfinite(flat)]


def prepare_all_data(rows):
    """Fit scalers on the complete candidate set and preserve their values."""
    raw = prepare_causal_source_matrices(rows)
    center, scale = contact_size_proxy.fit_robust_scaler(raw)
    normalized = contact_size_proxy.transform(raw, center, scale)
    stats = contact_size_proxy.stats_features(normalized)
    stats_center = []
    stats_scale = []
    for channel in range(stats.shape[1]):
        selected = finite_fit_values(stats[:, channel])
        if selected.size == 0:
            raise RuntimeError("empty_stats_scaler_channel")
        stats_center.append(float(selected.mean()))
        value = float(selected.std())
        stats_scale.append(value if value >= 1e-6 else 1.0)
    stats_center = np.asarray(stats_center, dtype=np.float32)
    stats_scale = np.asarray(stats_scale, dtype=np.float32)
    normalized_stats = np.clip(
        (stats - stats_center) / stats_scale,
        contract.STATS_CLIP_MIN,
        contract.STATS_CLIP_MAX,
    ).astype(np.float32)
    return {
        "sequenceCenter": center,
        "sequenceScale": scale,
        "statsCenter": stats_center.astype(np.float32),
        "statsScale": stats_scale.astype(np.float32),
        "stats": normalized_stats,
    }


def strength_probabilities(model, values, batch_size=1024):
    """Return strength probabilities in the fixed light, heavy order."""
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            model_output = model(values[start:start + batch_size])
            _, strength_logits, _ = lightweight.unpack_multitask_output(
                model_output
            )
            output.extend(
                torch.softmax(strength_logits, dim=1).cpu().numpy().tolist()
            )
    return np.asarray(output, dtype=np.float64)


def masked_context_loss(logits, targets):
    """Skip unknown metadata instead of teaching an invented unknown class."""
    valid = targets != contract.CONTEXT_IGNORE_INDEX
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return torch.nn.functional.cross_entropy(
        logits[valid],
        targets[valid],
    )


def gate_metrics(truths, probabilities):
    """Evaluate the product gate without depending on finger classification."""
    truths = np.asarray(truths, dtype=np.int64)
    predictions = np.asarray(probabilities, dtype=np.float64).argmax(axis=1)
    confusion = [
        [0 for _ in contract.GATE_LABELS]
        for _ in contract.GATE_LABELS
    ]
    for truth, prediction in zip(truths, predictions):
        confusion[int(truth)][int(prediction)] += 1
    recalls = []
    for index in range(len(contract.GATE_LABELS)):
        total = sum(confusion[index])
        recalls.append(confusion[index][index] / total if total else 0.0)
    correct = sum(
        confusion[index][index]
        for index in range(len(contract.GATE_LABELS))
    )
    return {
        "accuracy": correct / len(truths),
        "balancedAccuracy": sum(recalls) / len(recalls),
        "recallByClass": dict(zip(contract.GATE_LABELS, recalls)),
        "confusionMatrix": confusion,
    }


def gate_sample_weights(samples):
    """Balance sessions and the two gate classes without four-class quotas."""
    group_counts = {}
    for sample in samples:
        group_id = sample["groupId"]
        group_counts[group_id] = group_counts.get(group_id, 0) + 1
    raw = np.asarray(
        [
            1.0 / np.sqrt(group_counts[sample["groupId"]])
            for sample in samples
        ],
        dtype=np.float32,
    )
    labels = np.asarray([sample["label"] for sample in samples])
    target = len(samples) / len(contract.GATE_LABELS)
    for label in range(len(contract.GATE_LABELS)):
        selected = labels == label
        if not np.any(selected):
            raise RuntimeError("missing_gate_class")
        raw[selected] *= target / raw[selected].sum()
    return np.clip(raw, 0.25, 4.0) / raw.mean()


def train_candidate(
        values,
        labels,
        gate_labels,
        context_targets,
        sample_weights,
        hidden_count,
        seed,
        epochs,
        context_heads,
        context_loss_weight,
):
    configure_determinism(seed)
    model = lightweight.StatsMlp(
        contract.STATS_FEATURE_COUNT,
        hidden_count,
        True,
        contract.CONTEXT_CLASS_COUNTS if context_heads else None,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.002,
        weight_decay=0.003,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=0.00005,
    )
    x = torch.from_numpy(values)
    class_targets = torch.tensor(labels, dtype=torch.long)
    gate_targets = torch.tensor(gate_labels, dtype=torch.long)
    context_targets_tensor = torch.tensor(
        context_targets,
        dtype=torch.long,
    )
    weights = torch.from_numpy(sample_weights)
    for epoch in range(1, epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(order), 128):
            batch = order[start:start + 128]
            # 中文注释：力度门控是唯一主任务，轻触为负类、重触为正类。
            # 手指头只提供低权重辅助监督，不参与产品触发决策。
            model_output = model(x[batch])
            finger_logits, strength_logits, context_logits = (
                lightweight.unpack_multitask_output(model_output)
            )
            gate_losses = torch.nn.functional.cross_entropy(
                strength_logits,
                gate_targets[batch],
                reduction="none",
            )
            loss = (
                gate_losses * weights[batch]
            ).sum() / weights[batch].sum()
            loss += 0.20 * torch.nn.functional.cross_entropy(
                finger_logits,
                class_targets[batch] // 2,
            )
            if context_heads:
                if len(context_logits) != len(contract.CONTEXT_CLASS_COUNTS):
                    raise RuntimeError("invalid_context_head_count")
                # 中文注释：场景头只塑造共享编码器，权重低于手指辅助任务；
                # 导出时全部剔除，端侧不增加推理成本。
                context_loss = sum(
                    masked_context_loss(
                        logits,
                        context_targets_tensor[batch, index],
                    )
                    for index, logits in enumerate(context_logits)
                )
                loss += context_loss_weight * context_loss / len(context_logits)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        if epoch == epochs:
            print(json.dumps({"epoch": epoch, "totalEpochs": epochs}), flush=True)
    model.eval()
    gate_probabilities = strength_probabilities(model, x)
    diagnostics = lightweight.metrics(
        labels,
        lightweight.predict(model, x),
    )
    metrics = {
        "primaryTask": contract.PRIMARY_TASK,
        "gate": gate_metrics(gate_labels, gate_probabilities),
        "diagnosticFourClass": diagnostics,
        "contextAuxiliary": context_metrics(
            model,
            x,
            context_targets_tensor,
        ) if context_heads else None,
    }
    return model, metrics


def context_metrics(model, values, targets, batch_size=1024):
    """Report auxiliary accuracy only where a real metadata label exists."""
    model.eval()
    correct = [0] * len(contract.CONTEXT_TASK_NAMES)
    total = [0] * len(contract.CONTEXT_TASK_NAMES)
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            output = model(values[start:start + batch_size])
            _, _, context_logits = lightweight.unpack_multitask_output(output)
            for index, logits in enumerate(context_logits):
                selected = targets[start:start + batch_size, index]
                valid = selected != contract.CONTEXT_IGNORE_INDEX
                if not bool(valid.any()):
                    continue
                predictions = logits[valid].argmax(dim=1)
                truth = selected[valid]
                correct[index] += int((predictions == truth).sum().item())
                total[index] += int(truth.numel())
    return {
        name: {
            "validSampleCount": total[index],
            "accuracy": (
                correct[index] / total[index]
                if total[index]
                else None
            ),
        }
        for index, name in enumerate(contract.CONTEXT_TASK_NAMES)
    }


def tensor_record(value):
    tensor = value.detach().cpu().to(torch.float32)
    return {
        "shape": list(tensor.shape),
        "values": tensor.reshape(-1).tolist(),
    }


def runtime_state_dict(state):
    """Strip training-only heads and reject any runtime contract drift."""
    runtime_state = {
        name: value
        for name, value in state.items()
        if not name.startswith("context.")
    }
    if set(runtime_state) != RUNTIME_STATE_NAMES:
        raise RuntimeError("unexpected_stats_mlp_state_dict")
    return runtime_state


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=55)
    parser.add_argument("--hidden-count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--context-heads", action="store_true")
    parser.add_argument("--context-loss-weight", type=float, default=0.08)
    args = parser.parse_args()
    if args.epochs <= 0 or args.hidden_count <= 0:
        raise RuntimeError("invalid_training_budget")
    if not 0.0 < args.context_loss_weight <= 0.20:
        raise RuntimeError("invalid_context_loss_weight")

    rows = lightweight.read_rows(args.dataset_dir)
    if not rows:
        raise RuntimeError("empty_training_dataset")
    devices = sorted({str(row.get("deviceName") or "") for row in rows})
    invalid_labels = sorted({
        str(row.get("classLabel") or "")
        for row in rows
        if row.get("classLabel") not in contract.CLASS_LABELS
    })
    if invalid_labels:
        raise RuntimeError("invalid_training_class_label")
    labels = [
        lightweight.CLASS_LABELS.index(row["classLabel"])
        for row in rows
    ]
    gate_labels = [
        contract.class_strength_target(row["classLabel"])
        for row in rows
    ]
    context_labels = [
        contract.context_targets_for_row(row)
        for row in rows
    ]
    weighted_samples = [
        {"groupId": row["groupId"], "label": label}
        for row, label in zip(rows, gate_labels)
    ]
    prepared = prepare_all_data(rows)
    weights = gate_sample_weights(weighted_samples)
    model, train_metrics = train_candidate(
        prepared["stats"],
        labels,
        gate_labels,
        context_labels,
        weights,
        args.hidden_count,
        args.seed,
        args.epochs,
        args.context_heads,
        args.context_loss_weight,
    )
    runtime_state = runtime_state_dict(model.state_dict())
    gate_probabilities = strength_probabilities(
        model,
        torch.from_numpy(prepared["stats"]),
    )
    fixed_fpr = {
        str(target): gate_threshold_metrics.fixed_fpr_report(
            rows,
            gate_labels,
            gate_probabilities[:, contract.GATE_LABELS.index("heavy")],
            target_fpr=target,
        )
        for target in FIXED_FPR_TARGETS
    }

    output = Path(args.output_dir)
    weights_payload = {
        name: tensor_record(value)
        for name, value in sorted(runtime_state.items())
    }
    channels = []
    for name, center, scale in zip(
        contract.PHYSICAL_FEATURE_NAMES,
        prepared["sequenceCenter"],
        prepared["sequenceScale"],
    ):
        channels.append({
            "name": name,
            "center": float(center),
            "scale": float(scale),
            "clipMin": contract.STATS_CLIP_MIN,
            "clipMax": contract.STATS_CLIP_MAX,
        })
    contract_payload = {
        "runtimeArchitecture": contract.RUNTIME_ARCHITECTURE,
        "featureContract": contract.FEATURE_CONTRACT,
        "sourceFeatureContract": contract.SOURCE_FEATURE_CONTRACT,
        "sizeSemantics": contract.CONTACT_SIZE_SEMANTICS,
        "forceMeasurement": False,
        "missingValuePolicy": contract.CONTACT_SIZE_PROXY_POLICY[
            "missingValuePolicy"
        ],
        "sizeBaseline": dict(contract.CONTACT_SIZE_BASELINE_POLICY),
        "sourceFeatureNames": list(
            contract.RUNTIME_SOURCE_FEATURE_NAMES
        ),
        "featureNames": list(contract.PHYSICAL_FEATURE_NAMES),
        "sourceFeatureIndices": list(contract.SOURCE_FEATURE_INDICES),
        "statsFeatureNames": list(contract.STATS_FEATURE_NAMES),
        "matrixShape": [contract.FRAME_COUNT, len(contract.PHYSICAL_FEATURE_NAMES)],
        "frameCount": contract.FRAME_COUNT,
        "primaryTask": contract.PRIMARY_TASK,
        "gateCount": len(contract.GATE_LABELS),
        "gateLabelOrder": list(contract.GATE_LABELS),
        "negativeClassLabels": list(contract.NEGATIVE_CLASS_LABELS),
        "positiveClassLabels": list(contract.POSITIVE_CLASS_LABELS),
        "classCount": len(contract.GATE_LABELS),
        "labelOrder": list(contract.GATE_LABELS),
        "diagnosticClassCount": len(contract.CLASS_LABELS),
        "diagnosticClassLabelOrder": list(contract.CLASS_LABELS),
        "fingerLabelOrder": list(contract.FINGER_LABELS),
        "strengthLabelOrder": list(contract.STRENGTH_LABELS),
        "sequenceChannelCount": len(contract.PHYSICAL_FEATURE_NAMES),
        "statsFeatureCount": contract.STATS_FEATURE_COUNT,
        "inputCount": contract.STATS_FEATURE_COUNT,
        "hiddenCount": args.hidden_count,
        "dropoutAtInference": False,
        "sequenceChannels": channels,
        "statsScaler": {
            "center": prepared["statsCenter"].tolist(),
            "scale": prepared["statsScale"].tolist(),
            "clipMin": contract.STATS_CLIP_MIN,
            "clipMax": contract.STATS_CLIP_MAX,
        },
        "evaluationScope": "all_data_fit_only",
        "trainingAuxiliaryHeads": {
            "enabled": args.context_heads,
            "names": list(contract.CONTEXT_TASK_NAMES),
            "classLabelOrders": {
                name: list(labels)
                for name, labels in zip(
                    contract.CONTEXT_TASK_NAMES,
                    (
                        contract.MOTION_LABELS,
                        contract.SUPPORT_LABELS,
                        contract.POSTURE_LABELS,
                        contract.DEVICE_POSE_LABELS,
                    ),
                )
            },
            "ignoreIndex": contract.CONTEXT_IGNORE_INDEX,
            "lossWeight": args.context_loss_weight,
            "exportedToRuntime": False,
        },
        "singleDeviceCrossSessionEvidence": {
            "sampleCount": len(rows),
            "sessionCount": len({row["groupId"] for row in rows}),
            "deviceNames": devices,
        },
        "singleDeviceCrossSessionEvidenceOnly": len(devices) == 1,
        "deploymentAllowed": False,
    }
    report = {
        "candidate": "touch26_stats_mlp24_strength_gate_experimental",
        "scope": "all_data_fit_only",
        "singleDeviceCrossSessionEvidenceOnly": len(devices) == 1,
        "sampleCount": len(rows),
        "sessionCount": len({row["groupId"] for row in rows}),
        "deviceNames": devices,
        "epochs": args.epochs,
        "hiddenCount": args.hidden_count,
        "seed": args.seed,
        "trainingParameterCount": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "runtimeParameterCount": sum(
            parameter.numel() for parameter in runtime_state.values()
        ),
        "contextHeadsEnabled": args.context_heads,
        "contextLossWeight": args.context_loss_weight,
        "trainMetrics": train_metrics,
        "fixedFprMetrics": {
            "evaluationScope": "all_data_fit_only_optimistic",
            "reports": fixed_fpr,
        },
        "deploymentAllowed": False,
    }
    write_json(output / "weights.json", weights_payload)
    write_json(output / "model_contract.json", contract_payload)
    write_json(output / "training_report.json", report)
    print(
        json.dumps(
            {
                "output": str(output),
                "sampleCount": len(rows),
                "balancedAccuracy": train_metrics["gate"]["balancedAccuracy"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
