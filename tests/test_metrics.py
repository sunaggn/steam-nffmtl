"""Hand-computed checks of the metric definitions."""
import math

import numpy as np
import pytest
import scipy.sparse as sp

from steam_nffmtl.metrics import beyond_accuracy, binary_auc, gini, ranking_metrics, topk_indices


def _rel(rows: list[list[int]], n: int) -> sp.csr_matrix:
    r = [i for i, row in enumerate(rows) for _ in row]
    c = [j for row in rows for j in row]
    return sp.csr_matrix((np.ones(len(c)), (r, c)), shape=(len(rows), n))


def test_ranking_metrics_by_hand() -> None:
    # user 0: item 0 is history (masked); relevant {1, 4}; user 1: relevant {0}, ranked last
    scores = np.array([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    scores[0, 0] = -np.inf
    top = topk_indices(scores, 4)
    assert top.tolist() == [[1, 2, 3, 4], [5, 4, 3, 2]]
    m = ranking_metrics(top, _rel([[1, 4], [0]], 6), [2, 4])
    idcg2 = 1 + 1 / math.log2(3)
    assert m["hits@2"].tolist() == [1, 0]
    assert m["precision@2"] == pytest.approx([0.5, 0.0])
    assert m["recall@2"] == pytest.approx([0.5, 0.0])          # standard: hits / |R_u|
    assert m["hr@2"] == pytest.approx([1.0, 0.0])
    assert m["ndcg@2"] == pytest.approx([1 / idcg2, 0.0])
    assert m["precision@4"] == pytest.approx([0.5, 0.0])
    assert m["recall@4"] == pytest.approx([1.0, 0.0])
    assert m["ndcg@4"] == pytest.approx([(1 + 1 / math.log2(5)) / idcg2, 0.0])


def test_recall_is_not_capped() -> None:
    # 12 relevant items, all 10 recommendations hit: standard Recall@10 = 10/12, not 10/min(12, 10) = 1
    scores = np.tile(np.arange(20, 0, -1, dtype=float), (1, 1))
    m = ranking_metrics(topk_indices(scores, 10), _rel([list(range(12))], 20), [10])
    assert m["recall@10"][0] == pytest.approx(10 / 12)
    assert m["precision@10"][0] == pytest.approx(1.0)


def test_ties_break_by_lower_index() -> None:
    assert topk_indices(np.array([[1.0, 1.0, 1.0, 0.0]]), 2).tolist() == [[0, 1]]
    assert topk_indices(np.array([[0.0, 2.0, 2.0, 2.0, 1.0]]), 2).tolist() == [[1, 2]]


def test_empty_relevant_set_is_rejected() -> None:
    with pytest.raises(ValueError):
        ranking_metrics(np.array([[0, 1]]), _rel([[]], 3), [2])


def test_auc_by_hand() -> None:
    assert binary_auc(np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.4, 0.4])) == pytest.approx(0.875)
    with pytest.raises(ValueError):
        binary_auc(np.array([1, 1]), np.array([0.1, 0.2]))


def test_gini_and_beyond_accuracy() -> None:
    assert gini(np.array([1, 1, 1, 1])) == pytest.approx(0.0)
    assert gini(np.array([0, 0, 0, 4])) == pytest.approx(0.75)
    pop = np.array([100.0, 50.0, 10.0, 5.0, 1.0])
    tags = np.array([[1, 0], [1, 0], [0, 1], [0, 0], [0, 1]], dtype=np.float32)
    b = beyond_accuracy(np.array([[0, 1], [2, 3]]), [2], pop, tags, head_share=0.2)
    assert b["coverage@2"] == pytest.approx(4 / 5)
    assert b["long_tail_share@2"] == pytest.approx(3 / 4)       # only item 0 is head
    assert b["arp@2"] == pytest.approx((100 + 50 + 10 + 5) / 4)
    assert b["tag_ild@2"] == pytest.approx(0.0)                 # list 0: same tags; list 1 has 1 tagged item
    assert b["tag_ild_valid_share@2"] == pytest.approx(0.5)
    assert b["untagged_share@2"] == pytest.approx(1 / 4)
