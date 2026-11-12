"""Ranking, classification and beyond-accuracy metrics.

Definitions (binary relevance, per user u with eligible relevant set R_u, top-K list L_K):

* Precision@K = |L_K ∩ R_u| / K
* Recall@K    = |L_K ∩ R_u| / |R_u|            (standard; not min(|R_u|, K))
* HR@K        = 1[|L_K ∩ R_u| > 0]
* NDCG@K      = DCG@K / IDCG@K, DCG = sum_r rel_r / log2(r + 1) for ranks r = 1..K,
                IDCG over min(|R_u|, K) relevant items at the top.

Ties in scores are broken by ascending item index so that results are deterministic.
Only finite scores are candidates: when a user has fewer than K finite candidates
(history masked with -inf), the list is shorter and the empty slots are -1. Empty
slots never count as hits or as recommended items; Precision@K still divides by K.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` highest finite scores per row; ties go to the lower item index.

    Slots without a finite candidate are ``-1``. Rows without a tie at the k-th score
    use a vectorised path; rows with a boundary tie are resolved exactly.
    """
    if scores.ndim != 2:
        raise ValueError("scores must be 2-D (users x items)")
    n_rows, n = scores.shape
    k = min(k, n)
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    part.sort(axis=1)  # index order, so the stable sort below breaks ties by index
    vals = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-vals, axis=1, kind="stable")
    out = np.take_along_axis(part, order, axis=1)
    kth = vals.min(axis=1)
    n_ge = (scores >= kth[:, None]).sum(axis=1)
    for r in np.flatnonzero((n_ge > k) & np.isfinite(kth)):  # finite boundary ties
        cand = np.flatnonzero(scores[r] >= kth[r])
        o = np.lexsort((cand, -scores[r, cand]))
        out[r] = cand[o[:k]]
    finite = np.isfinite(np.take_along_axis(scores, out, axis=1))
    out[~finite] = -1
    return out


def ranking_metrics(topk: np.ndarray, relevant: sp.csr_matrix, ks: list[int]) -> dict[str, np.ndarray]:
    """Per-user Precision, Recall, NDCG and HR at each K.

    ``topk`` is (n_users, K_max) item indices; ``relevant`` is the matching CSR rows.
    Every user must have at least one relevant item (guaranteed by the protocol).
    """
    n_rel = np.diff(relevant.indptr)
    if (n_rel == 0).any():
        raise ValueError("Every evaluated user needs at least one eligible relevant item")
    kmax = topk.shape[1]
    if max(ks) > kmax:
        raise ValueError("topk has fewer columns than max(ks)")
    rel_dense = relevant.toarray().astype(bool)
    hit = np.take_along_axis(rel_dense, np.maximum(topk, 0), axis=1) & (topk >= 0)
    discounts = 1.0 / np.log2(np.arange(2, kmax + 2))
    cum_disc = np.cumsum(discounts)
    out: dict[str, np.ndarray] = {"n_relevant": n_rel.astype(np.int64)}
    for k in ks:
        h = hit[:, :k]
        hits = h.sum(1)
        dcg = (h * discounts[:k]).sum(1)
        idcg = cum_disc[np.minimum(n_rel, k) - 1]
        out[f"hits@{k}"] = hits.astype(np.int64)
        out[f"precision@{k}"] = hits / k
        out[f"recall@{k}"] = hits / n_rel
        out[f"ndcg@{k}"] = dcg / idcg
        out[f"hr@{k}"] = (hits > 0).astype(np.float64)
    return out


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney statistic with average ranks for ties."""
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs both classes")
    ranks = rankdata(scores, method="average")
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def gini(counts: np.ndarray) -> float:
    """Gini coefficient of recommendation counts over the full catalog (0 = uniform)."""
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def beyond_accuracy(topk: np.ndarray, ks: list[int], item_popularity: np.ndarray,
                    tag_matrix: np.ndarray | None, head_share: float = 0.2) -> dict[str, float]:
    """Catalog coverage, Gini, average recommendation popularity, long-tail share,
    novelty and tag intra-list diversity at each K.

    ``item_popularity`` = number of training interactions per catalog game (> 0 after k-core).
    Long tail = games outside the ``head_share`` most popular games. Novelty =
    mean -log2(pop_i / sum pop). Tag ILD = mean pairwise cosine distance between
    recommended games that have at least one selected tag; lists with fewer than
    two tagged games are excluded and their share is reported.
    """
    n_items = len(item_popularity)
    pop = item_popularity.astype(np.float64)
    order = np.argsort(-pop, kind="stable")
    head = np.zeros(n_items, dtype=bool)
    head[order[: int(np.ceil(head_share * n_items))]] = True
    selfinfo = -np.log2(pop / pop.sum())
    out: dict[str, float] = {}
    tag_norm = None
    if tag_matrix is not None and tag_matrix.shape[1] > 0:
        norms = np.linalg.norm(tag_matrix, axis=1, keepdims=True)
        tag_norm = np.divide(tag_matrix, norms, out=np.zeros_like(tag_matrix), where=norms > 0)
        tagged = norms[:, 0] > 0
    for k in ks:
        lists = topk[:, :k]
        valid = lists >= 0
        rec = lists[valid]
        counts = np.bincount(rec, minlength=n_items)
        out[f"coverage@{k}"] = float((counts > 0).mean())
        out[f"gini@{k}"] = gini(counts)
        out[f"arp@{k}"] = float(pop[rec].mean()) if len(rec) else float("nan")
        out[f"long_tail_share@{k}"] = float((~head[rec]).mean()) if len(rec) else float("nan")
        out[f"novelty@{k}"] = float(selfinfo[rec].mean()) if len(rec) else float("nan")
        out[f"mean_list_length@{k}"] = float(valid.sum(1).mean()) if len(lists) else 0.0
        if tag_norm is not None:
            ild, n_valid = [], 0
            for row in lists:
                row = row[row >= 0]
                rows = row[tagged[row]]
                if len(rows) < 2:
                    continue
                v = tag_norm[rows]
                sim = v @ v.T
                iu = np.triu_indices(len(rows), 1)
                ild.append(float((1.0 - sim[iu]).mean()))
                n_valid += 1
            out[f"tag_ild@{k}"] = float(np.mean(ild)) if ild else float("nan")
            out[f"tag_ild_valid_share@{k}"] = n_valid / len(lists) if len(lists) else 0.0
            out[f"untagged_share@{k}"] = float((~tagged[rec]).mean()) if len(rec) else float("nan")
    return out
