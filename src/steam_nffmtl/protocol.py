"""Common evaluation protocol shared by every model.

For a target year the protocol fixes, once and for all models:

* the evaluated users: training-known users with at least one eligible relevant game;
* the candidate catalog: every game in the training catalog (full-catalog ranking);
* the history filter: every game the user reviewed before the target year
  (training years for validation; training years and 2021 for test);
* the eligible relevant set: games reviewed in the target year that are in the
  training catalog, satisfy the relevance definition and are not in the
  history filter. Recall uses this set as its denominator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from .data import PreparedData
from .repro import sha256_arrays


def _csr(rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int]) -> sp.csr_matrix:
    """Binary CSR matrix with duplicate entries collapsed and sorted indices."""
    m = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=shape)
    m.sum_duplicates()
    m.data[:] = 1.0
    m.sort_indices()
    return m


def profile_matrix(data: PreparedData, kind: str) -> sp.csr_matrix:
    """User x game training matrix used as model input.

    ``kind='positive'`` keeps positively reviewed games, ``'interaction'`` keeps all
    training reviews. Used by Popularity, kNN, EASE, BPR and ALS.
    """
    tr = data.train
    mask = tr["label"] == 1 if kind == "positive" else np.ones(len(tr["u"]), dtype=bool)
    return _csr(tr["u"][mask], tr["i"][mask], (data.n_users, data.n_items))


def history_matrix(data: PreparedData, target: str) -> sp.csr_matrix:
    """All known (user, game) interactions before the target year."""
    parts = ["train"] if target == "val" else ["train", "val"]
    rows, cols = [], []
    for p in parts:
        d = data.partition(p)
        ok = (d["u"] >= 0) & (d["i"] >= 0)
        rows.append(d["u"][ok])
        cols.append(d["i"][ok])
    return _csr(np.concatenate(rows), np.concatenate(cols), (data.n_users, data.n_items))


@dataclass
class EvalSplit:
    """Frozen evaluation protocol for one target year and one relevance definition."""

    name: str
    target: str
    relevance: str
    users: np.ndarray
    relevant: sp.csr_matrix
    exclude: sp.csr_matrix
    n_items: int
    data_fingerprint: str
    user_ids: np.ndarray  # original Steam user IDs of ``users``
    item_ids: np.ndarray  # original app IDs of the catalog (index order)
    report: dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Hash binding the prepared data, real IDs, users, relevant sets and history masks."""
        meta = np.frombuffer(f"{self.data_fingerprint}|{self.target}|{self.relevance}".encode(), dtype=np.uint8)
        return sha256_arrays(meta, self.user_ids, self.item_ids, self.users, self.relevant.indptr,
                             self.relevant.indices, self.exclude.indptr, self.exclude.indices)

    def candidate_counts(self) -> np.ndarray:
        """Number of finite candidates (catalog minus history) per evaluated user."""
        return self.n_items - np.diff(self.exclude.indptr)


def build_split(data: PreparedData, target: str, relevance: str,
                user_sample: int | None = None, seed: int = 0) -> EvalSplit:
    """Construct the evaluation protocol for ``target`` in {'val', 'test'}.

    Rows of the target year are filtered step by step and every exclusion is
    counted in ``report`` so that the paper can state exactly what was evaluated.
    """
    if target not in {"val", "test"}:
        raise ValueError("target must be 'val' or 'test'")
    if relevance not in {"positive", "interaction"}:
        raise ValueError("relevance must be 'positive' or 'interaction'")
    d = data.partition(target)
    hist = history_matrix(data, target)
    n_rows = len(d["u"])
    known_u = d["u"] >= 0
    warm = known_u & (d["i"] >= 0)
    rel_mask = warm & (d["label"] == 1) if relevance == "positive" else warm.copy()
    u, i = d["u"][rel_mask], d["i"][rel_mask]
    in_hist = np.zeros(len(u), dtype=bool)
    if len(u):
        # membership test of (u, i) in the history CSR via sorted global keys
        hk = np.repeat(np.arange(hist.shape[0], dtype=np.int64), np.diff(hist.indptr)) * data.n_items + hist.indices
        q = u.astype(np.int64) * data.n_items + i
        pos = np.searchsorted(hk, q)
        pos = np.minimum(pos, max(len(hk) - 1, 0))
        in_hist = (hk[pos] == q) if len(hk) else in_hist
    u_ok, i_ok = u[~in_hist], i[~in_hist]
    eligible_users = np.unique(u_ok)
    report: dict[str, Any] = {
        "target": target,
        "relevance": relevance,
        "target_rows": int(n_rows),
        "target_users": int(len(np.unique(d["user_id"]))) if n_rows else 0,
        "rows_cold_user_excluded": int((~known_u).sum()),
        "rows_cold_game_excluded": int((known_u & ~warm).sum()),
        "rows_not_relevant_label": int((warm & ~rel_mask).sum()),
        "rows_relevant_but_in_history_excluded": int(in_hist.sum()),
        "eligible_relevant_pairs": int(len(np.unique(u_ok * data.n_items + i_ok))) if len(u_ok) else 0,
        "known_users_in_target": int(len(np.unique(d["u"][known_u]))),
        "users_with_eligible_relevant": int(len(eligible_users)),
        "history_cutoff": "end of training years" if target == "val" else "end of validation year",
        "candidates": "full training catalog minus history",
        "n_candidate_items": int(data.n_items),
    }
    users = eligible_users
    if user_sample is not None and user_sample < len(users):
        rng = np.random.default_rng(seed)
        users = np.sort(rng.choice(users, size=user_sample, replace=False))
        report["user_sample"] = {"size": int(user_sample), "seed": int(seed)}
    report["evaluated_users"] = int(len(users))
    rel_full = _csr(u_ok, i_ok, (data.n_users, data.n_items))
    relevant = rel_full[users]
    exclude = hist[users]
    relevant.sort_indices()
    exclude.sort_indices()
    # Sanity: no relevant game may be masked by the history filter.
    if relevant.multiply(exclude).nnz:
        raise AssertionError("Relevant item found inside the history filter")
    report["mean_eligible_relevant_per_user"] = float(np.diff(relevant.indptr).mean()) if len(users) else 0.0
    name = f"{target}-{relevance}"
    split = EvalSplit(name, target, relevance, users.astype(np.int64), relevant, exclude, data.n_items,
                      data.fingerprint(), data.user_ids[users], data.item_ids, report)
    cand = split.candidate_counts()
    report["min_candidates_per_user"] = int(cand.min()) if len(cand) else 0
    report["users_with_fewer_than_20_candidates"] = int((cand < 20).sum())
    return split


def classification_rows(data: PreparedData, target: str, n: int | None, seed: int) -> dict[str, np.ndarray]:
    """Observed warm reviews of the target year for review-label AUC (same rows for all models)."""
    d = data.partition(target)
    warm = np.flatnonzero((d["u"] >= 0) & (d["i"] >= 0))
    if n is not None and n < len(warm):
        rng = np.random.default_rng(seed)
        warm = np.sort(rng.choice(warm, size=n, replace=False))
    return {"u": d["u"][warm], "i": d["i"][warm], "label": d["label"][warm].astype(np.int8),
            "hours_log": d["hours_log"][warm]}
