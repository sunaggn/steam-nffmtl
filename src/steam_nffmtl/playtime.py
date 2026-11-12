"""Evaluation of the auxiliary playtime head against simple predictors on identical rows.

Target: y = log1p(hours). The head is trained on observed training reviews with
hours > ``reg_min_hours``; evaluation uses the same rule on warm test rows.
Hours-space errors use expm1(max(prediction, 0)): a log-space prediction below
zero would imply negative playtime, so it is clipped and the clipped share is
reported. expm1 of a conditional mean in log space is not a conditional mean
in hours (retransformation bias), so hours-space MAE is descriptive only.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .data import PreparedData


def playtime_rows(data: PreparedData, target: str, reg_min_hours: float) -> dict[str, np.ndarray]:
    """Warm observed rows of ``target`` with hours above the regression threshold."""
    d = data.partition(target)
    ok = (d["u"] >= 0) & (d["i"] >= 0) & (d["hours_log"] > np.log1p(reg_min_hours))
    return {"u": d["u"][ok], "i": d["i"][ok], "y": d["hours_log"][ok].astype(np.float64)}


def fit_simple_predictors(data: PreparedData, reg_min_hours: float) -> dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    """Global mean, per-game median and additive user + game mean, fitted on training rows."""
    tr = playtime_rows(data, "train", reg_min_hours)
    mu = float(tr["y"].mean())
    frame = pd.DataFrame(tr)
    game_median = frame.groupby("i")["y"].median()
    gm = np.full(data.n_items, mu)
    gm[game_median.index.to_numpy()] = game_median.to_numpy()
    game_mean = frame.groupby("i")["y"].mean()
    bi = np.zeros(data.n_items)
    bi[game_mean.index.to_numpy()] = game_mean.to_numpy() - mu
    resid = frame["y"].to_numpy() - mu - bi[frame["i"].to_numpy()]
    user_res = pd.Series(resid).groupby(frame["u"].to_numpy()).mean()
    bu = np.zeros(data.n_users)
    bu[user_res.index.to_numpy()] = user_res.to_numpy()
    return {
        "GlobalMean": lambda u, i: np.full(len(u), mu),
        "GameMedian": lambda u, i: gm[i],
        "UserGameMean": lambda u, i: mu + bu[u] + bi[i],
    }


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Log-space MAE/RMSE, correlations and clipped hours-space errors."""
    err = y_pred - y_true
    hours_true = np.expm1(y_true)
    hours_pred = np.expm1(np.clip(y_pred, 0.0, 50.0))
    out = {
        "n": int(len(y_true)),
        "mae_log": float(np.abs(err).mean()),
        "rmse_log": float(np.sqrt((err ** 2).mean())),
        "median_ae_hours": float(np.median(np.abs(hours_pred - hours_true))),
        "mae_hours_descriptive": float(np.abs(hours_pred - hours_true).mean()),
        "share_negative_log_predictions": float((y_pred < 0).mean()),
    }
    if np.ptp(y_pred) > 1e-9:  # correlations are undefined for constant predictors
        out["pearson_r_log"] = float(pearsonr(y_pred, y_true)[0])
        out["spearman_rho_log"] = float(spearmanr(y_pred, y_true)[0])
    return out


def per_user_abs_error(rows: dict[str, np.ndarray], y_pred: np.ndarray) -> pd.Series:
    """Mean absolute log error per user (used for paired, user-clustered comparisons)."""
    return pd.Series(np.abs(y_pred - rows["y"])).groupby(rows["u"]).mean()
