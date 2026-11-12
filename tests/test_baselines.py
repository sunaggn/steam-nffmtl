"""Baselines on toy data with known answers, including implicit's matrix orientation."""
import numpy as np
import pytest
import scipy.sparse as sp

from steam_nffmtl.baselines import (EASEScorer, ImplicitMFScorer, ItemKNNCFScorer, PopularityScorer,
                                    RandomScorer, TagKNNScorer, ease_cost_estimate)


def _toy(n_users: int = 40, n_items: int = 12, seed: int = 0) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    m = sp.csr_matrix((rng.random((n_users, n_items)) < 0.3).astype(np.float32))
    m.sort_indices()
    return m


@pytest.mark.parametrize("kind", ["bpr", "als"])
def test_implicit_orientation_and_scores(kind: str) -> None:
    pytest.importorskip("implicit")
    x = _toy()  # 40 users x 12 games: a transposed fit would give 12 user rows
    s = ImplicitMFScorer(kind, x, {"factors": 4, "iterations": 5}, seed=0)
    assert s.user_factors.shape[0] == 40 and s.item_factors.shape[0] == 12
    sc = s.score(np.array([0, 39]))
    assert sc.shape == (2, 12) and np.isfinite(sc).all()
    np.testing.assert_allclose(s.score_pairs(np.array([0, 39]), np.array([3, 5])), [sc[0, 3], sc[1, 5]], rtol=1e-5)


def test_als_learns_block_structure() -> None:
    pytest.importorskip("implicit")
    x = np.zeros((40, 10), dtype=np.float32)
    x[:20, :5] = 1
    x[20:, 5:] = 1
    x[0, 4] = 0  # user 0 has not seen game 4 of its own block
    s = ImplicitMFScorer("als", sp.csr_matrix(x), {"factors": 2, "iterations": 15, "regularization": 0.01}, seed=0)
    row = s.score(np.array([0]))[0]
    assert row[4] > row[5:].max()


def test_popularity_and_random() -> None:
    x = _toy()
    pop = PopularityScorer(x)
    np.testing.assert_allclose(pop.score(np.array([0]))[0], np.asarray(x.sum(0)).ravel())
    r = RandomScorer(12, 7)
    a, b = r.score(np.array([0, 1])), r.score(np.array([1]))
    np.testing.assert_allclose(a[1], b[0])  # independent of batching
    assert (a >= 0).all() and (a < 1).all()


def test_itemknn_cf_neighbours() -> None:
    x = np.zeros((6, 4), dtype=np.float32)
    x[:4, 0] = x[:4, 1] = 1  # games 0 and 1 always co-occur
    x[4:, 2] = 1
    x[5, 3] = 1
    s = ItemKNNCFScorer(sp.csr_matrix(x), k=2, shrink=0.0)
    profile_user = sp.csr_matrix(np.array([[1, 0, 0, 0]], dtype=np.float32))
    s.profile = profile_user
    row = s.score(np.array([0]))[0]
    assert row.argmax() == 1 and row[1] == pytest.approx(1.0)


def test_tagknn_prefers_shared_tags() -> None:
    tags = np.array([[1, 0], [1, 0], [0, 1], [0, 0]], dtype=np.float32)
    s = TagKNNScorer(tags, sp.csr_matrix(np.array([[1, 0, 0, 0]], dtype=np.float32)), k=2)
    row = s.score(np.array([0]))[0]
    assert row[1] == pytest.approx(1.0) and row[2] == 0 and row[3] == 0


def test_ease_matches_closed_form() -> None:
    x = _toy(30, 8, 1)
    lam = 5.0
    s = EASEScorer(x, l2=lam)
    g = (x.T @ x).toarray().astype(np.float64) + lam * np.eye(8)
    p = np.linalg.inv(g)
    b = -p / np.diag(p)[None, :]
    np.fill_diagonal(b, 0.0)
    np.testing.assert_allclose(s.B, b, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(s.score_pairs(np.array([2, 5]), np.array([1, 7])),
                               [s.score(np.array([2]))[0, 1], s.score(np.array([5]))[0, 7]], rtol=1e-5)
    est = ease_cost_estimate(25000, "cpu")
    assert est["gram_float64_gb"] == pytest.approx(5.0)       # float64, as computed by EASEScorer
    assert est["peak_host_gb"] == pytest.approx(17.5)          # 28 n^2 bytes on the CPU path
    gpu = ease_cost_estimate(25000, "cuda")
    assert gpu["peak_gpu_gb"] == pytest.approx(15.0) and gpu["peak_host_gb"] == pytest.approx(7.5)
    with pytest.raises(MemoryError):
        EASEScorer(x, max_items=4)


def test_transposed_fit_is_rejected() -> None:
    """The original notebook fitted on user_item.T; the corrected wrapper refuses that matrix."""
    pytest.importorskip("implicit")
    x = _toy()
    with pytest.raises(AssertionError):
        ImplicitMFScorer("bpr", x.T.tocsr(), {"factors": 4, "iterations": 2}, seed=0,
                         expected_users=40, expected_items=12)
