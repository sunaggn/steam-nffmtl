"""Model-agnostic full-catalog evaluation.

A model is anything that implements :class:`Scorer`: ``score(users)`` returns a
dense (len(users) x n_items) array over the *whole* training catalog. The
evaluator, not the model, applies the history mask, so every model sees the
same candidates. Scoring errors are recorded; with ``fail_on_error`` they abort
the run, otherwise the result is marked invalid and cannot enter a comparison.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .metrics import beyond_accuracy, binary_auc, ranking_metrics, topk_indices
from .protocol import EvalSplit


class Scorer(Protocol):
    """Interface every recommender must implement."""

    name: str

    def score(self, users: np.ndarray) -> np.ndarray:
        """Scores for every catalog game, shape (len(users), n_items)."""
        ...

    def score_pairs(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Scores for aligned (user, game) pairs."""
        ...


def score_pairs_via_rows(scorer: Scorer, users: np.ndarray, items: np.ndarray,
                         batch: int = 256) -> np.ndarray:
    """Generic pair scorer: score each distinct user's full row once and gather."""
    out = np.empty(len(users), dtype=np.float64)
    order = np.argsort(users, kind="stable")
    uniq, start = np.unique(users[order], return_index=True)
    bounds = list(start) + [len(order)]
    for b in range(0, len(uniq), batch):
        ub = uniq[b:b + batch]
        rows = scorer.score(ub)
        for j in range(len(ub)):
            sel = order[bounds[b + j]:bounds[b + j + 1]]
            out[sel] = rows[j, items[sel]]
    return out


@dataclass
class RankingResult:
    """Outcome of one model on one frozen protocol."""

    model: str
    split_name: str
    split_fingerprint: str
    data_fingerprint: str
    per_user: pd.DataFrame
    summary: dict[str, float]
    beyond: dict[str, float]
    coverage: dict[str, Any]
    errors: list[dict[str, Any]] = field(default_factory=list)
    topk: np.ndarray | None = None

    @property
    def valid(self) -> bool:
        """True when every protocol user was evaluated without error."""
        return not self.errors and self.coverage["evaluated_users"] == self.coverage["protocol_users"]


def evaluate_ranking(scorer: Scorer, split: EvalSplit, ks: list[int], batch_size: int,
                     item_popularity: np.ndarray, tag_matrix: np.ndarray | None,
                     user_ids: np.ndarray | None = None, fail_on_error: bool = True,
                     head_share: float = 0.2, short_lists: str = "report",
                     keep_topk: bool = False) -> RankingResult:
    """Rank the full training catalog for every protocol user and compute metrics.

    Invariants checked on every batch: score shape and finiteness, and that no
    history game appears in a recommendation list. Users with fewer finite
    candidates than ``max(ks)`` get shorter lists (``short_lists='report'``, counted
    in ``coverage``) or abort the run (``'error'``).
    """
    kmax = max(ks)
    n_items = split.n_items
    n_cand = split.candidate_counts()
    if short_lists == "error" and (n_cand < kmax).any():
        raise ValueError(f"{int((n_cand < kmax).sum())} users have fewer than {kmax} candidates")
    per_user: list[dict[str, np.ndarray]] = []
    tops: list[np.ndarray] = []
    errors: list[dict[str, Any]] = []
    t0 = time.time()
    for start in range(0, len(split.users), batch_size):
        users = split.users[start:start + batch_size]
        try:
            s = np.asarray(scorer.score(users), dtype=np.float64)
            if s.shape != (len(users), n_items):
                raise ValueError(f"score() returned shape {s.shape}, expected {(len(users), n_items)}")
            if not np.isfinite(s).all():
                raise ValueError("score() returned non-finite values")
        except Exception as exc:  # recorded, never silently skipped
            errors.append({"batch_start": int(start), "n_users": int(len(users)),
                           "error": repr(exc), "traceback": traceback.format_exc(limit=3)})
            if fail_on_error:
                raise RuntimeError(f"[{scorer.name}] scoring failed at batch {start}: {exc!r}") from exc
            continue
        ex = split.exclude[start:start + batch_size]
        rows = np.repeat(np.arange(len(users)), np.diff(ex.indptr))
        s[rows, ex.indices] = -np.inf
        top = topk_indices(s, kmax)
        in_hist = np.take_along_axis(ex.toarray() > 0, np.maximum(top, 0), axis=1) & (top >= 0)
        if in_hist.any():
            raise AssertionError(f"[{scorer.name}] a history game entered a recommendation list")
        m = ranking_metrics(top, split.relevant[start:start + batch_size], ks)
        m["u"] = users
        m["n_candidates"] = n_cand[start:start + batch_size]
        per_user.append(m)
        tops.append(top)
    elapsed = time.time() - t0
    if per_user:
        frame = pd.DataFrame({k: np.concatenate([m[k] for m in per_user]) for k in per_user[0]})
        top_all = np.concatenate(tops)
    else:
        frame = pd.DataFrame()
        top_all = np.zeros((0, kmax), dtype=np.int64)
    if user_ids is not None and len(frame):
        frame.insert(1, "user_id", user_ids[frame["u"].to_numpy()])
    metric_cols = [c for c in frame.columns if "@" in c and not c.startswith("hits")]
    summary = {c: float(frame[c].mean()) for c in metric_cols} if len(frame) else {}
    beyond = beyond_accuracy(top_all, ks, item_popularity, tag_matrix, head_share) if len(top_all) else {}
    coverage = {"protocol_users": int(len(split.users)), "evaluated_users": int(len(frame)),
                "failed_users": int(sum(e["n_users"] for e in errors)), "seconds": round(elapsed, 1)}
    for k in ks:
        coverage[f"short_list_users@{k}"] = int((n_cand < k).sum())
    return RankingResult(scorer.name, split.name, split.fingerprint(), split.data_fingerprint, frame, summary,
                         beyond, coverage, errors, top_all if keep_topk else None)


def evaluate_classification_auc(scorer: Scorer, rows: dict[str, np.ndarray]) -> dict[str, Any]:
    """Review-label (is_recommended) AUC on a fixed set of observed warm rows."""
    pair_fn = getattr(scorer, "score_pairs", None)
    s = pair_fn(rows["u"], rows["i"]) if pair_fn is not None else score_pairs_via_rows(scorer, rows["u"], rows["i"])
    s = np.asarray(s, dtype=np.float64)
    if not np.isfinite(s).all():
        raise ValueError(f"[{scorer.name}] non-finite pair scores")
    return {"auc_review_label": binary_auc(rows["label"], s), "n_rows": int(len(s)),
            "positive_rate": float(rows["label"].mean())}


def comparison_table(results: list[RankingResult]) -> pd.DataFrame:
    """Side-by-side summary; refuses results from different protocols or with errors."""
    if not results:
        raise ValueError("No results")
    fps = {r.split_fingerprint for r in results}
    if len(fps) != 1:
        raise ValueError(f"Results come from different protocols: {fps}")
    if len({r.data_fingerprint for r in results}) != 1:
        raise ValueError("Results come from different prepared data")
    invalid = [r.model for r in results if not r.valid]
    if invalid:
        raise ValueError(f"Invalid (incomplete) results cannot be compared: {invalid}")
    users0 = results[0].per_user["u"].to_numpy()
    for r in results[1:]:
        if not np.array_equal(users0, r.per_user["u"].to_numpy()):
            raise ValueError(f"User sets differ between {results[0].model} and {r.model}")
    rows = []
    for r in results:
        row = {"model": r.model, "users": r.coverage["evaluated_users"]}
        row.update(r.summary)
        row.update(r.beyond)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
