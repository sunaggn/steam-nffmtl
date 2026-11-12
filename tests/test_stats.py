"""Paired statistics and Holm adjustment."""
import numpy as np
import pandas as pd
import pytest

from steam_nffmtl.stats import holm, paired_user_comparison, seed_level_comparison


def test_holm_by_hand() -> None:
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_paired_user_comparison_detects_shift() -> None:
    rng = np.random.default_rng(0)
    b = pd.Series(rng.random(500))
    a = b + 0.05 + rng.normal(0, 0.01, 500)
    r = paired_user_comparison(a, b, n_boot=200)
    assert r["mean_diff"] == pytest.approx(0.05, abs=0.005)
    assert r["p_wilcoxon"] < 1e-6 and r["rank_biserial"] > 0.9
    lo, hi = r["ci95_user_bootstrap"]
    assert lo < 0.05 < hi


def test_seed_level_comparison() -> None:
    r = seed_level_comparison([0.30, 0.31, 0.29, 0.32, 0.30], [0.28, 0.29, 0.28, 0.30, 0.29])
    assert r["n_seeds"] == 5 and r["mean_diff"] == pytest.approx(0.016)
    assert r["p_paired_t"] < 0.01


def test_config_overrides_deep_merge() -> None:
    from pathlib import Path

    from steam_nffmtl.cli import _parse_overrides
    from steam_nffmtl.config import load_config

    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    cfg = load_config(cfg_path, _parse_overrides(["baselines.ease.enabled=true", "train.max_epochs=2"]))
    assert cfg.baselines["ease"]["enabled"] is True and cfg.baselines["ease"]["l2"] == 500.0
    assert cfg.baselines["bpr"]["factors"] == 64 and cfg.train.max_epochs == 2
    import pytest
    with pytest.raises(KeyError):
        load_config(cfg_path, {"train": {"max_epoch": 3}})
