"""Paired uncertainty analysis.

Two sources of uncertainty are kept apart:

* **user sampling** (one trained model, many users): paired per-user differences,
  user bootstrap CI, Wilcoxon signed-rank test, rank-biserial effect size;
* **training randomness** (several seeds, one protocol): seed-level means, paired
  t-test over seeds and Cohen's d_z. With five seeds the smallest attainable
  two-sided Wilcoxon p is 0.0625, so the seed-level test is a paired t-test.

Holm's step-down procedure adjusts a family of p-values.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def paired_user_comparison(a: pd.Series, b: pd.Series, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Compare per-user metric values of model A and model B on the same users (A - B).

    Both series must be indexed by unique user indices with identical membership;
    a different order is re-aligned, different membership raises ``ValueError``.
    """
    for name, x in (("a", a), ("b", b)):
        if not x.index.is_unique:
            raise ValueError(f"Series {name} has duplicate user indices")
    if not a.index.equals(b.index):
        if len(a) != len(b) or not a.index.sort_values().equals(b.index.sort_values()):
            raise ValueError(f"User sets differ (|a|={len(a)}, |b|={len(b)}, "
                             f"common={len(a.index.intersection(b.index))}); refusing to intersect silently")
        b = b.reindex(a.index)
    d = (a - b).to_numpy(dtype=np.float64)
    n = len(d)
    if n < 2:
        raise ValueError("Need at least two paired users")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for j in range(n_boot):
        boots[j] = d[rng.integers(0, n, n)].mean()
    nonzero = d[d != 0]
    if len(nonzero):
        w = stats.wilcoxon(d, zero_method="zsplit", alternative="two-sided")
        p_w = float(w.pvalue)
        ranks = stats.rankdata(np.abs(nonzero))
        r_plus, r_minus = ranks[nonzero > 0].sum(), ranks[nonzero < 0].sum()
        rank_biserial = float((r_plus - r_minus) / (r_plus + r_minus))
    else:
        p_w, rank_biserial = 1.0, 0.0
    sd = d.std(ddof=1)
    return {
        "n_users": n, "mean_a": float(a.mean()), "mean_b": float(b.mean()), "mean_diff": float(d.mean()),
        "ci95_user_bootstrap": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "p_wilcoxon": p_w, "rank_biserial": rank_biserial,
        "d_z": float(d.mean() / sd) if sd > 0 else 0.0,
        "share_users_a_better": float((d > 0).mean()), "share_users_tied": float((d == 0).mean()),
    }


def seed_level_comparison(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> dict[str, Any]:
    """Paired comparison of seed-level aggregate metrics (seed j of A paired with seed j of B)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or len(a) < 2:
        raise ValueError("Need >= 2 paired seeds of equal length")
    d = a - b
    sd = d.std(ddof=1)
    if sd > 0:
        t = stats.ttest_rel(a, b)
        p_t, t_stat = float(t.pvalue), float(t.statistic)
    else:
        p_t, t_stat = (1.0 if d.mean() == 0 else 0.0), float("inf") if d.mean() else 0.0
    half = stats.t.ppf(0.975, len(d) - 1) * sd / np.sqrt(len(d)) if sd > 0 else 0.0
    return {"n_seeds": int(len(d)), "mean_a": float(a.mean()), "sd_a": float(a.std(ddof=1)),
            "mean_b": float(b.mean()), "sd_b": float(b.std(ddof=1)), "mean_diff": float(d.mean()),
            "ci95_seed_t": [float(d.mean() - half), float(d.mean() + half)],
            "t": t_stat, "p_paired_t": p_t, "d_z": float(d.mean() / sd) if sd > 0 else 0.0}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[key] = running
    return adjusted
