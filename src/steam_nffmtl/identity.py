"""Normalised training identities.

Two runs count as *seed repeats of the same model* only if their training
signatures are equal. The signature covers everything that changes the trained
model (task, architecture variant, hyperparameters, selection protocol, data
settings) and excludes the seed and settings that only affect speed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from .config import ExperimentConfig
from .data import data_config_identity

# Training fields that do not change the trained model: the repeat variable and a scoring chunk size.
EXCLUDED_TRAIN_KEYS = ("seed", "item_chunk")


def nffmtl_training_settings(cfg: ExperimentConfig, variant: str) -> dict[str, Any]:
    """Everything that determines an NFF-MTL model except its seed."""
    train = {k: v for k, v in asdict(cfg.train).items() if k not in EXCLUDED_TRAIN_KEYS}
    return {
        "model": "NFF-MTL", "variant": variant, "train_task": cfg.task.primary, "train": train,
        "selection": {"metric": cfg.train.select_metric, "val_user_sample": cfg.eval.val_user_sample,
                      "sample_seed": cfg.eval.sample_seed},
        "data": data_config_identity(cfg.data),
    }


def baseline_training_settings(name: str, params: dict[str, Any], cfg: ExperimentConfig) -> dict[str, Any]:
    """Everything that determines a fitted baseline except its seed."""
    return {"model": name, "train_task": cfg.task.primary, "params": dict(sorted(params.items())),
            "data": data_config_identity(cfg.data)}


def signature(settings: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON form of ``settings``."""
    return hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()


def config_id(model: str, sig: str) -> str:
    """Short, human-readable experiment identity, e.g. ``NFF-MTL@3fa2c91b0d``."""
    return f"{model}@{sig[:10]}"


def settings_diff(stored: dict[str, Any], requested: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flattened differences between two nested settings dictionaries."""
    out: dict[str, Any] = {}
    for key in sorted(set(stored) | set(requested)):
        a, b = stored.get(key), requested.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            out.update(settings_diff(a, b, f"{prefix}{key}."))
        elif a != b:
            out[f"{prefix}{key}"] = {"checkpoint": a, "requested": b}
    return out
