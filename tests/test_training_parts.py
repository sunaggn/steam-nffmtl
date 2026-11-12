"""Negative sampler, full-catalog scorer equivalence, checkpoint guards and k-core."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from steam_nffmtl.data import _kcore
from steam_nffmtl.model import NFFMTL, NFFMTLScorer
from steam_nffmtl.train import ExactNegativeSampler, pos_weight_value


def test_negatives_never_hit_training_interactions() -> None:
    rng = np.random.default_rng(0)
    dense = rng.random((50, 30)) < 0.6
    m = sp.csr_matrix(dense.astype(np.float32))
    m.sort_indices()
    g = torch.Generator().manual_seed(1)
    s = ExactNegativeSampler(m, torch.device("cpu"), g)
    users = torch.arange(50).repeat_interleave(40)
    items = s.sample(users)
    assert not dense[users.numpy(), items.numpy()].any()
    g2 = torch.Generator().manual_seed(1)
    again = ExactNegativeSampler(m, torch.device("cpu"), g2).sample(users)
    assert torch.equal(items, again)  # seeded and reproducible


def test_sampler_rejects_user_with_full_catalog() -> None:
    m = sp.csr_matrix(np.ones((2, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        ExactNegativeSampler(m, torch.device("cpu"), torch.Generator().manual_seed(0))


@pytest.mark.parametrize("use_content,use_mtl", [(True, True), (False, True), (True, False)])
def test_full_catalog_scores_equal_pairwise_forward(use_content: bool, use_mtl: bool) -> None:
    torch.manual_seed(0)
    tags = (np.random.default_rng(0).random((23, 5)) < 0.4).astype(np.float32)
    model = NFFMTL(11, 23, tags, embedding_dim=8, hidden=(16, 8, 4), use_content=use_content, use_mtl=use_mtl)
    model.train()
    for _ in range(3):  # populate BatchNorm running statistics
        model(torch.randint(0, 11, (64,)), torch.randint(0, 23, (64,)))
    model.eval()
    users = np.array([0, 3, 10])
    full = NFFMTLScorer(model, "cpu", item_chunk=7).score(users)
    uu = np.repeat(users, 23)
    ii = np.tile(np.arange(23), len(users))
    pairs = NFFMTLScorer(model, "cpu").score_pairs(uu, ii).reshape(len(users), 23)
    np.testing.assert_allclose(full, pairs, rtol=1e-5, atol=1e-5)


def test_pos_weight_definition() -> None:
    labels = np.array([1] * 845 + [0] * 155)
    # (observed zeros + 4 sampled negatives per observed row) / positives
    assert pos_weight_value(labels, 4, "auto") == pytest.approx((155 + 4000) / 845)
    assert pos_weight_value(labels, 4, "none") is None


def test_kcore_on_training_rows() -> None:
    df = pd.DataFrame({"user_id": [1, 1, 1, 2, 2, 3], "app_id": [10, 20, 30, 10, 20, 10]})
    out, iters = _kcore(df, 2)
    assert sorted(out["user_id"].unique()) == [1, 2] and sorted(out["app_id"].unique()) == [10, 20]
    assert iters >= 2


def test_legacy_checkpoint_is_refused(tiny_data, tmp_path) -> None:
    from steam_nffmtl.train import inspect_legacy_checkpoint

    assert inspect_legacy_checkpoint(tmp_path / "missing.pth", tiny_data)["found"] is False
    # legacy layout: global mapping with padding index 0 (ids start at 1)
    legacy = {"user2idx": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}, "item2idx": {10: 1, 20: 2, 30: 3, 40: 4, 99: 5},
              "model_state_dict": {}}
    torch.save(legacy, tmp_path / "nff_mtl_best.pth")
    rep = inspect_legacy_checkpoint(tmp_path / "nff_mtl_best.pth", tiny_data)
    assert rep["found"] and not rep["compatible"]
