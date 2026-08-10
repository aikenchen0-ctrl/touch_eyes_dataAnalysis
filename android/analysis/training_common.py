import json
import math
import random


def import_torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("pytorch_not_installed") from error
    return torch


def select_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed, torch):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize(x, center, scale):
    return (x - center.view(1, -1, 1)) / scale.view(1, -1, 1)


def clone_state(model, torch):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def tensor_to_list(tensor):
    return [float(value) for value in tensor.detach().cpu().reshape(-1).tolist()]


def tensor_to_json(tensor):
    return {
        "shape": list(tensor.shape),
        "values": tensor_to_list(tensor),
    }


def write_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def parse_csv_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def finite_float(value, fallback=0.0):
    try:
        output = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(output):
        return fallback
    return output
