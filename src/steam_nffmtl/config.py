"""Typed experiment configuration loaded from YAML.

Every experiment is fully described by one YAML file. Unknown keys raise an
error so that a typo cannot silently fall back to a default.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Raw data location, temporal split and train-only preprocessing."""

    recommendations_csv: str = "data/recommendations.csv"
    metadata_json: str = "data/games_metadata.json"
    cache_dir: str = "cache"
    train_years: list[int] = field(default_factory=lambda: [2010, 2020])
    val_year: int = 2021
    test_year: int = 2022
    kcore: int = 5
    n_tags: int = 50
    dedup: str = "keep_last"  # keep the latest review per (user, game) inside a partition
    verify_source_hash: bool = False  # True: re-hash raw files on every cache load (slow, strict)


@dataclass
class TaskConfig:
    """Primary task definition.

    ``positive``: label 1 = positive review (is_recommended); relevance in the
    target year = positively reviewed games.  ``interaction``: label 1 = any
    review; relevance = any reviewed game.
    """

    primary: str = "positive"
    sensitivity_relevance: list[str] = field(default_factory=lambda: ["interaction"])


@dataclass
class EvalConfig:
    """Common evaluation protocol shared by every model."""

    ks: list[int] = field(default_factory=lambda: [10, 20])
    test_user_sample: int | None = None  # None = all eligible training-known users
    val_user_sample: int | None = 20000
    user_batch_size: int = 256
    auc_rows: int = 500000
    sample_seed: int = 20260101
    fail_on_error: bool = True
    head_share: float = 0.2  # most popular 20% of catalog items form the head
    short_lists: str = "report"  # users with < K finite candidates: 'report' (shorter list) or 'error'


@dataclass
class TrainConfig:
    """NFF-MTL architecture and optimisation."""

    seed: int = 0
    embedding_dim: int = 64
    hidden: list[int] = field(default_factory=lambda: [128, 64, 32])
    dropout: list[float] = field(default_factory=lambda: [0.3, 0.2])
    # Architecture variant (full / no_content / single_task) is chosen only by `train --variant`.
    alpha: float = 0.5
    neg_ratio: int = 4
    batch_size_total: int = 4096
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-5
    grad_clip: float = 1.0
    scheduler_t0: int = 2
    scheduler_tmult: int = 2
    max_epochs: int = 15
    patience: int = 3
    select_metric: str = "ndcg@20"
    pos_weight: str | float = "auto"
    reg_min_hours: float = 0.0  # regression loss on observed rows with hours > this value
    item_chunk: int = 4096
    deterministic: bool = True
    max_batches_per_epoch: int | None = None  # probe runs only: stop each epoch after this many batches


@dataclass
class TuningConfig:
    """Random search on the validation year with an equal number of trials per model.

    ``budget_policy='equal'``: every tuned model gets exactly ``n_trials`` distinct candidates;
    a grid with fewer points is an error. ``'cap'``: ``n_trials`` is an upper bound and smaller
    grids are used completely (the used budget is reported). Equal trial counts are not equal
    compute time.
    """

    n_trials: int = 10
    seed: int = 0
    metric: str = "ndcg@20"
    budget_policy: str = "equal"
    spaces: dict[str, dict[str, list[Any]]] = field(default_factory=dict)


@dataclass
class ExperimentMeta:
    """Experiment stage. Test-year evaluation is allowed only for 'final' (and synthetic 'smoke')."""

    stage: str = "pilot"  # 'pilot' | 'tuning' | 'final' | 'smoke'
    name: str = "steam-nffmtl"


@dataclass
class ExperimentConfig:
    """Top-level configuration."""

    experiment: ExperimentMeta = field(default_factory=ExperimentMeta)
    data: DataConfig = field(default_factory=DataConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    baselines: dict[str, dict[str, Any]] = field(default_factory=dict)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    runs_dir: str = "runs"

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, YAML-serialisable dictionary."""
        return asdict(self)


def _merge(obj: Any, updates: dict[str, Any], path: str = "") -> Any:
    """Recursively merge ``updates`` into a dataclass instance, rejecting unknown keys."""
    if not is_dataclass(obj):
        return updates
    known = {f.name: f for f in fields(obj)}
    for key, value in updates.items():
        if key not in known:
            raise KeyError(f"Unknown config key '{path}{key}'")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value, f"{path}{key}.")
        elif isinstance(current, dict) and isinstance(value, dict):
            setattr(obj, key, _merge_dicts(current, value))
        else:
            setattr(obj, key, copy.deepcopy(value))
    return obj


def _merge_dicts(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge plain dictionaries (used for the free-form ``baselines`` section)."""
    out = copy.deepcopy(base)
    for k, v in updates.items():
        out[k] = _merge_dicts(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else copy.deepcopy(v)
    return out


def validate(cfg: ExperimentConfig) -> None:
    """Raise ``ValueError`` for inconsistent settings."""
    if cfg.task.primary not in {"positive", "interaction"}:
        raise ValueError("task.primary must be 'positive' or 'interaction'")
    y0, y1 = cfg.data.train_years
    if not (y0 <= y1 < cfg.data.val_year < cfg.data.test_year):
        raise ValueError("Years must satisfy train <= val < test")
    if any(k <= 0 for k in cfg.eval.ks):
        raise ValueError("eval.ks must be positive")
    metric, _, k = cfg.train.select_metric.partition("@")
    if metric not in {"ndcg", "recall", "precision", "hr"} or not k.isdigit():
        raise ValueError("train.select_metric must look like 'ndcg@20'")
    if cfg.train.alpha <= 0:
        raise ValueError("train.alpha must be > 0 (use --variant single_task to drop the playtime head)")
    if cfg.experiment.stage not in {"pilot", "tuning", "final", "smoke"}:
        raise ValueError("experiment.stage must be pilot, tuning, final or smoke")
    if cfg.train.max_batches_per_epoch is not None and cfg.experiment.stage not in {"pilot", "smoke"}:
        raise ValueError("train.max_batches_per_epoch is for pilot/probe runs only")
    if cfg.tuning.budget_policy not in {"equal", "cap"}:
        raise ValueError("tuning.budget_policy must be 'equal' or 'cap'")
    if cfg.eval.short_lists not in {"report", "error"}:
        raise ValueError("eval.short_lists must be 'report' or 'error'")


def config_sha256(cfg: ExperimentConfig) -> str:
    """Stable hash of the resolved configuration."""
    import hashlib
    import json

    return hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True, default=str).encode()).hexdigest()


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """Load a YAML configuration, apply optional overrides and validate it."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = _merge(ExperimentConfig(), raw)
    if overrides:
        cfg = _merge(cfg, overrides)
    validate(cfg)
    return cfg


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    """Write the resolved configuration next to the run outputs."""
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False, allow_unicode=True)
