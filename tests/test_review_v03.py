"""Regression tests for the independent review of v0.2 (training identity and seed-repeat integrity)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from steam_nffmtl import cli
from steam_nffmtl.config import ExperimentConfig, load_config
from steam_nffmtl.synthetic import make_synthetic
from steam_nffmtl.train import build_model, load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"


def _cfg(tmp: Path, stage: str, **train) -> ExperimentConfig:
    d = tmp / "d"
    if not (d / "recommendations.csv").exists():
        make_synthetic(d, n_users=400, n_items=80)
    over = {"experiment": {"stage": stage}, "runs_dir": str(tmp / f"r_{stage}"),
            "data": {"recommendations_csv": str(d / "recommendations.csv"),
                     "metadata_json": str(d / "games_metadata.json"), "cache_dir": str(tmp / "c")},
            "train": {"max_epochs": 1, **train}}
    return load_config(SMOKE, over)


# ---------------------------------------------------------------- finding 1: training task kept with the checkpoint
def _tiny_ckpt(tiny_data, tmp_path: Path) -> tuple[Path, ExperimentConfig]:
    cfg = ExperimentConfig()
    cfg.train.embedding_dim, cfg.train.hidden = 8, [16, 8, 4]
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, build_model(tiny_data, cfg, "full"), tiny_data, cfg, "full", 0, 1, 0.1, [])
    return path, cfg


def test_checkpoint_refuses_other_task(tiny_data, tmp_path: Path) -> None:
    path, cfg = _tiny_ckpt(tiny_data, tmp_path)
    other = copy.deepcopy(cfg)
    other.task.primary = "interaction"
    with pytest.raises(ValueError, match="trained for task 'positive'"):
        load_checkpoint(path, tiny_data, other, torch.device("cpu"))


def test_checkpoint_refuses_other_training_settings(tiny_data, tmp_path: Path) -> None:
    path, cfg = _tiny_ckpt(tiny_data, tmp_path)
    changed = copy.deepcopy(cfg)
    changed.train.alpha = 1.0
    with pytest.raises(ValueError, match="train.alpha"):
        load_checkpoint(path, tiny_data, changed, torch.device("cpu"))
    harmless = copy.deepcopy(cfg)
    harmless.train.item_chunk, harmless.train.seed = 17, 99  # speed setting and repeat variable only
    model, ckpt = load_checkpoint(path, tiny_data, harmless, torch.device("cpu"))
    assert ckpt["train_task"] == "positive" and ckpt["config_id"].startswith("NFF-MTL@")


def test_checkpoint_without_identity_is_refused(tiny_data, tmp_path: Path) -> None:
    path, cfg = _tiny_ckpt(tiny_data, tmp_path)
    ckpt = torch.load(path, weights_only=False)
    for key in ("training_signature", "train_task"):
        ckpt.pop(key)
    torch.save(ckpt, path)
    with pytest.raises(ValueError, match="training identity"):
        load_checkpoint(path, tiny_data, cfg, torch.device("cpu"))


def test_stage_promotion_is_refused(tmp_path: Path) -> None:
    tuning = _cfg(tmp_path, "tuning")
    run = cli.cmd_train(tuning, "full", [0])[0]
    meta = json.loads(next((run / "results").glob("NFF-MTL__val-positive.json")).read_text())
    assert meta["train_task"] == "positive" and meta["train_stage"] == "tuning"
    final = copy.deepcopy(tuning)
    final.experiment.stage = "final"
    with pytest.raises(PermissionError, match="trained in stage 'tuning'"):
        cli.cmd_evaluate(final, str(run), "test")


# ---------------------------------------------------------------- finding 2: seed repeats need equal training settings
def test_compare_refuses_different_hyperparameters_as_seed_repeats(tmp_path: Path) -> None:
    smoke = _cfg(tmp_path, "smoke")
    base = cli.cmd_baselines(smoke, "test", 0, ["Popularity", "Random"])
    ev0 = cli.cmd_evaluate(smoke, str(cli.cmd_train(smoke, "full", [0])[0]), "test")
    alpha1 = copy.deepcopy(smoke)
    alpha1.train.alpha = 1.0
    ev1 = cli.cmd_evaluate(alpha1, str(cli.cmd_train(alpha1, "full", [1])[0]), "test")
    with pytest.raises(ValueError, match="different training settings"):
        cli.cmd_compare(smoke, [str(base), str(ev0), str(ev1)], "NFF-MTL", None, ["ndcg@20"])
    out = cli.cmd_compare(smoke, [str(base), str(ev0), str(ev1)],
                          json.loads(next((ev0 / "results").glob("*__test-positive.json")).read_text())["config_id"],
                          None, ["ndcg@20"], allow_multiple_configs=True)
    import pandas as pd

    table = pd.read_csv(out / "comparison_table.csv", index_col=0)
    assert sum(i.startswith("NFF-MTL@") for i in table.index) == 2 and (table["runs"] == 1).all()
    same = cli.cmd_evaluate(smoke, str(cli.cmd_train(smoke, "full", [1])[0]), "test")  # genuine seed repeat
    ok = cli.cmd_compare(smoke, [str(base), str(ev0), str(same)], "NFF-MTL", None, ["ndcg@20"])
    stats = json.loads((ok / "comparison_stats.json").read_text())
    assert "NFF-MTL vs Popularity | ndcg@20" in stats["seed_level"]  # two seeds -> seed-level test


# ---------------------------------------------------------------- equal-budget tuning, probe runs, registry scope
def test_tune_equal_budget_and_frozen_selection(tmp_path: Path) -> None:
    pytest.importorskip("implicit", reason="BPR-MF/ALS need implicit (preflight fails without it)")
    cfg = _cfg(tmp_path, "tuning")
    cfg.tuning.n_trials = 2
    cfg.tuning.spaces = {"itemknn": {"k": [5, 10, 20], "shrink": [10.0]}, "bpr": {"iterations": [5, 10]},
                         "nffmtl": {"alpha": [0.5, 1.0]}}
    run = cli.cmd_tune(cfg, ["itemknn", "bpr", "nffmtl"])
    sel = json.loads((run / "selection.json").read_text())
    assert {k: v["trials_used"] for k, v in sel.items()} == {"itemknn": 2, "bpr": 2, "nffmtl": 2}
    assert {v["budget_policy"] for v in sel.values()} == {"equal"}
    assert sel["itemknn"]["grid_size"] == 3
    trials = [json.loads(p.read_text()) for p in run.glob("trials/*/results/*.json")]
    for key in ("ItemKNN-CF", "BPR-MF", "NFF-MTL"):
        ids = {t["config_id"] for t in trials if t["model"] == key}
        assert len(ids) == 2, f"{key}: candidates must carry distinct experiment identities"
    final = load_config(run / "final_selected.yaml")
    assert final.experiment.stage == "final" and final.data.verify_source_hash
    assert final.baselines["itemknn"]["k"] == sel["itemknn"]["best_candidate"]["k"]
    assert final.train.alpha == sel["nffmtl"]["best_candidate"]["alpha"]
    with pytest.raises(PermissionError):
        cli.cmd_tune(final, ["itemknn"])


def test_probe_limits_batches_and_logs_resources(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "pilot", max_batches_per_epoch=2)
    run = cli.cmd_train(cfg, "full", [0])[0]
    h = json.loads((run / "training.json").read_text())["history"][0]
    assert h["batches"] == 2 and h["full_epoch_batches"] > 2
    assert {"seconds_per_batch", "peak_gpu_memory_gb", "host_max_rss_gb"} <= set(h)
    with pytest.raises(ValueError):
        load_config(SMOKE, {"experiment": {"stage": "final"}, "train": {"max_batches_per_epoch": 5}})


def test_prepare_writes_test_protocols_only_in_final(tmp_path: Path) -> None:
    run = cli.cmd_prepare(_cfg(tmp_path, "tuning"), force=False)
    assert {p.name for p in run.glob("protocol_*.json")} == {"protocol_val-positive.json",
                                                             "protocol_val-interaction.json"}


def test_equal_budget_refuses_small_grid_before_any_trial(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "tuning")
    cfg.tuning.n_trials = 3
    cfg.tuning.spaces = {"itemknn": {"k": [5, 10, 20]}, "tagknn": {"k": [5, 10]}}
    with pytest.raises(ValueError, match="smaller grids"):
        cli.cmd_tune(cfg, ["itemknn", "tagknn"])
    assert not Path(cfg.runs_dir).exists()  # refused before creating a run folder or fitting anything
    cfg.tuning.budget_policy = "cap"
    sel = json.loads((cli.cmd_tune(cfg, ["itemknn", "tagknn"]) / "selection.json").read_text())
    assert sel["tagknn"]["trials_used"] == 2 and sel["itemknn"]["trials_used"] == 3
    assert sel["tagknn"]["budget_policy"] == "cap"
