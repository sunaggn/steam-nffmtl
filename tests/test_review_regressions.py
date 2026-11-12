"""Regression tests for the independent code review of v0.1 (one section per finding)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from steam_nffmtl.baselines import EASEScorer, PopularityScorer, RandomScorer
from steam_nffmtl.config import DataConfig, ExperimentConfig, load_config
from steam_nffmtl.data import CacheMismatchError, cache_path, prepare_and_cache
from steam_nffmtl.evaluate import evaluate_ranking
from steam_nffmtl.metrics import beyond_accuracy, ranking_metrics, topk_indices
from steam_nffmtl.protocol import build_split, profile_matrix
from steam_nffmtl.repro import make_run_dir
from steam_nffmtl.stats import paired_user_comparison
from steam_nffmtl.synthetic import make_synthetic

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- 1. cache must match config and sources
def _dcfg(tmp: Path, **kw) -> DataConfig:
    rec, meta = tmp / "recommendations.csv", tmp / "games_metadata.json"
    if not rec.exists():
        make_synthetic(tmp, n_users=300, n_items=60)
    return DataConfig(recommendations_csv=str(rec), metadata_json=str(meta), cache_dir=str(tmp / "cache"), **kw)


def test_cache_is_keyed_by_preprocessing_settings(tmp_path: Path) -> None:
    a = prepare_and_cache(_dcfg(tmp_path, kcore=1, n_tags=3))
    b = prepare_and_cache(_dcfg(tmp_path, kcore=3, n_tags=1))
    assert cache_path(_dcfg(tmp_path, kcore=1, n_tags=3)) != cache_path(_dcfg(tmp_path, kcore=3, n_tags=1))
    assert len(a.tag_names) == 3 and len(b.tag_names) == 1
    assert a.card["kcore"]["k"] == 1 and b.card["kcore"]["k"] == 3
    assert a.fingerprint() != b.fingerprint()
    with pytest.raises(ValueError, match="removed every training row"):
        prepare_and_cache(_dcfg(tmp_path, kcore=99))


def test_cache_rejects_changed_source_file(tmp_path: Path) -> None:
    cfg = _dcfg(tmp_path, kcore=1)
    prepare_and_cache(cfg)
    rec = Path(cfg.recommendations_csv)
    rec.write_text(rec.read_text() + rec.read_text().splitlines()[1] + "\n")  # one extra row
    with pytest.raises(CacheMismatchError):
        prepare_and_cache(cfg)
    assert prepare_and_cache(cfg, force=True).card["raw"]["rows"] > 0  # explicit rebuild works


def test_cache_rejects_edited_manifest_and_missing_manifest(tmp_path: Path) -> None:
    cfg = _dcfg(tmp_path, kcore=1)
    prepare_and_cache(cfg)
    mf = cache_path(cfg) / "cache_manifest.json"
    manifest = json.loads(mf.read_text())
    manifest["data_config"]["kcore"] = 7
    mf.write_text(json.dumps(manifest))
    with pytest.raises(CacheMismatchError):
        prepare_and_cache(cfg)
    mf.unlink()
    with pytest.raises(CacheMismatchError):
        prepare_and_cache(cfg)


def test_same_mtime_change_detected_with_verify_hash(tmp_path: Path) -> None:
    cfg = _dcfg(tmp_path, kcore=1, verify_source_hash=True)
    prepare_and_cache(cfg)
    meta = Path(cfg.metadata_json)
    st = meta.stat()
    text = meta.read_text()
    meta.write_text(text.replace("tag_", "tAg_", 1))  # same size, different content
    os.utime(meta, ns=(st.st_atime_ns, st.st_mtime_ns))
    with pytest.raises(CacheMismatchError):
        prepare_and_cache(cfg)


# ---------------------------------------------------------------- 2. fewer finite candidates than K
def test_topk_never_returns_masked_items() -> None:
    s = np.array([[0.5, -np.inf, -np.inf, -np.inf], [0.1, 0.2, 0.3, 0.4]])
    assert topk_indices(s, 2).tolist() == [[0, -1], [3, 2]]


def test_short_lists_in_metrics_and_beyond_accuracy() -> None:
    import scipy.sparse as sp

    top = np.array([[0, -1], [3, 2]])
    rel = sp.csr_matrix(np.array([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32))
    m = ranking_metrics(top, rel, [2])
    assert m["hits@2"].tolist() == [1, 1]
    assert m["precision@2"].tolist() == [0.5, 0.5] and m["recall@2"].tolist() == [1.0, 1.0]
    # the -1 slot must not be read as the last item (index -1 = item 3)
    rel2 = sp.csr_matrix(np.array([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=np.float32))
    assert ranking_metrics(top, rel2, [2])["hits@2"].tolist() == [0, 1]
    b = beyond_accuracy(top, [2], np.array([10.0, 5.0, 2.0, 1.0]), None)
    assert b["coverage@2"] == pytest.approx(3 / 4) and b["mean_list_length@2"] == pytest.approx(1.5)
    assert b["arp@2"] == pytest.approx((10 + 1 + 2) / 3)


def test_evaluator_reports_or_rejects_short_lists(tiny_data) -> None:
    s = build_split(tiny_data, "test", "positive")  # user 2: catalog of 4 games, 3 in history
    assert s.candidate_counts().tolist() == [1] and s.report["min_candidates_per_user"] == 1
    pop = np.ones(tiny_data.n_items)
    res = evaluate_ranking(PopularityScorer(profile_matrix(tiny_data, "positive")), s, [2], 8, pop, None,
                           keep_topk=True)
    i30 = int(np.flatnonzero(tiny_data.item_ids == 30)[0])
    assert res.topk.tolist() == [[i30, -1]]
    assert res.coverage["short_list_users@2"] == 1
    assert res.per_user["precision@2"].tolist() == [0.5] and res.per_user["recall@2"].tolist() == [1.0]
    with pytest.raises(ValueError):
        evaluate_ranking(RandomScorer(tiny_data.n_items, 0), s, [2], 8, pop, None, short_lists="error")


# ---------------------------------------------------------------- 3. paired statistics need identical users
def test_paired_comparison_refuses_different_user_sets() -> None:
    a = pd.Series([0.1, 0.2, 0.3], index=[1, 2, 3])
    with pytest.raises(ValueError):
        paired_user_comparison(a, pd.Series([0.2, 0.3, 0.4], index=[2, 3, 4]), n_boot=10)
    with pytest.raises(ValueError):
        paired_user_comparison(a, pd.Series([0.2, 0.3, 0.4], index=[1, 1, 3]), n_boot=10)
    shuffled = pd.Series([0.3, 0.1, 0.2], index=[3, 1, 2])
    r = paired_user_comparison(a, shuffled, n_boot=10)  # same users, other order: re-aligned
    assert r["n_users"] == 3 and r["mean_diff"] == pytest.approx(0.0)


def test_compare_refuses_duplicates_and_mismatched_rows(tmp_path: Path) -> None:
    pytest.importorskip("implicit", reason="BPR-MF/ALS need implicit (preflight fails without it)")
    from steam_nffmtl import cli

    work = cli.cmd_smoke(str(tmp_path), str(ROOT / "configs" / "smoke.yaml"))
    runs = sorted((work / "r").iterdir())
    base = next(r for r in runs if "baselines" in r.name)
    ev = next(r for r in runs if "evaluate" in r.name)
    cfg = load_config(base / "config.yaml")
    with pytest.raises(ValueError, match="Duplicate"):
        cli.cmd_compare(cfg, [str(base), str(ev), str(ev)], "NFF-MTL", None, ["ndcg@20"])
    csv = next((ev / "results").glob("NFF-MTL__test-positive.csv.gz"))
    frame = pd.read_csv(csv)
    frame.iloc[:-1].to_csv(csv, index=False)  # drop one user
    with pytest.raises(ValueError):
        cli.cmd_compare(cfg, [str(base), str(ev)], "NFF-MTL", None, ["ndcg@20"])


# ---------------------------------------------------------------- 4. one authority for the architecture
def test_removed_architecture_flags_are_rejected() -> None:
    with pytest.raises(KeyError):
        load_config(ROOT / "configs" / "default.yaml", {"train": {"use_content": False}})


def test_checkpoint_architecture_is_verified(tiny_data, tmp_path: Path) -> None:
    from steam_nffmtl.train import build_model, load_checkpoint, model_spec, save_checkpoint

    cfg = ExperimentConfig()
    cfg.train.embedding_dim, cfg.train.hidden = 8, [16, 8, 4]
    model = build_model(tiny_data, cfg, "no_content")
    assert model.content is None and model_spec(tiny_data, cfg, "no_content")["use_content"] is False
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, tiny_data, cfg, "no_content", 0, 1, 0.1, [])
    loaded, _ = load_checkpoint(path, tiny_data, cfg, torch.device("cpu"))
    assert loaded.content is None
    with pytest.raises(ValueError):
        load_checkpoint(path, tiny_data, cfg, torch.device("cpu"), variant="full")
    cfg.train.embedding_dim = 16
    with pytest.raises(ValueError):
        load_checkpoint(path, tiny_data, cfg, torch.device("cpu"))


# ---------------------------------------------------------------- additional protections
def test_test_year_is_locked_outside_final_stage(tmp_path: Path) -> None:
    from steam_nffmtl import cli

    cfg = ExperimentConfig()
    cfg.experiment.stage, cfg.runs_dir = "tuning", str(tmp_path)
    with pytest.raises(PermissionError):
        cli.cmd_baselines(cfg, "test", 0, None)
    c1 = ExperimentConfig(runs_dir=str(tmp_path / "runs_a"))
    c1.data.cache_dir = str(tmp_path / "cache")
    reg = cli.TestRegistry(c1)
    reg.claim("k", {"model": "X"}, False)
    reg.finish("k")
    c2 = ExperimentConfig(runs_dir=str(tmp_path / "runs_b"))  # another runs_dir, same prepared data
    c2.data.cache_dir = str(tmp_path / "cache")
    with pytest.raises(PermissionError):
        cli.TestRegistry(c2).claim("k", {"model": "X"}, False)


def test_ease_memory_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    import scipy.sparse as sp

    from steam_nffmtl import baselines

    monkeypatch.setattr(baselines, "available_memory_gb", lambda device: 1e-6)
    with pytest.raises(MemoryError):
        EASEScorer(sp.csr_matrix(np.eye(50, dtype=np.float32)), l2=1.0)


def test_fingerprints_cover_tag_order_and_real_ids(tiny_data) -> None:
    fp = tiny_data.fingerprint()
    s = build_split(tiny_data, "test", "interaction")
    tiny_data.__dict__.pop("_fingerprint", None)
    tiny_data.tag_names = list(reversed(tiny_data.tag_names))
    assert tiny_data.fingerprint() != fp
    s.user_ids = s.user_ids + 1
    assert s.fingerprint() != build_split(tiny_data, "test", "interaction").fingerprint()


def test_run_dirs_sort_chronologically(tmp_path: Path) -> None:
    a = make_run_dir(tmp_path, "x")
    b = make_run_dir(tmp_path, "x")
    time.sleep(1.05)
    c = make_run_dir(tmp_path, "x")
    names = [p.name for p in (a, b, c)]
    assert sorted(names) == names and b.name.endswith("-02")


@pytest.mark.parametrize("variant", ["full", "no_content", "single_task"])
def test_estimate_reflects_the_built_model(tiny_data, variant: str) -> None:
    from steam_nffmtl.cli import estimate_parameters
    from steam_nffmtl.train import build_model, count_parameters, model_spec

    cfg = ExperimentConfig()
    spec = model_spec(tiny_data, cfg, variant)
    assert estimate_parameters(spec) == count_parameters(build_model(tiny_data, cfg, variant))
