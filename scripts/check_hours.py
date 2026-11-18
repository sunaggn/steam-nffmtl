"""Empirical checks of the `hours` column of the Kaggle *Game Recommendations on Steam* data.

Background (dataset author's code, github.com/kaydotdev/steam-game-recommendations, 2023 versions):
hours were scraped from review cards with the regex ``([\\d.]+) hrs on record``. Two consequences
are tested here on the real file:

1. "hrs on record" may be the reviewer's total playtime when the page was crawled, not at
   review time. The audit flags reviews written shortly after a game's recorded release whose
   hours exceed the time since release; this pattern is consistent with later accumulation,
   but release-date errors and pre-release access can also contribute.
2. The regex has no thousands separator, so "1,234.5 hrs on record" could be parsed as 234.5.
   The audit checks whether any value exceeds 1000 and reports heavy-game distributions.
   These checks do not identify the parsing mechanism by themselves.

Usage:
    python check_hours.py --recs recommendations.csv [--games games.csv] [--meta games_metadata.json] --out report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HEAVY_GAMES = {570: "Dota 2", 730: "Counter-Strike: GO", 440: "Team Fortress 2", 578080: "PUBG"}


def hours_profile(recs: pd.DataFrame) -> dict[str, Any]:
    """Distribution of `hours`, including an explicit check for values above 1000 h."""
    h = recs["hours"].to_numpy()
    q = np.quantile(h, [0.5, 0.9, 0.99, 0.999])
    return {"rows": int(len(h)), "min": float(h.min()), "max": float(h.max()),
            "q50_q90_q99_q999": [round(float(x), 1) for x in q],
            "rows_ge_1000": int((h >= 1000).sum()), "rows_eq_0": int((h == 0).sum()),
            "share_lt_1h": float((h < 1).mean()),
            "no_values_above_1000h": bool(not np.any(h > 1000))}


def heavy_game_check(recs: pd.DataFrame) -> dict[str, Any]:
    """Share of reviews under 1 h for games with many heavy players versus all games."""
    overall = float((recs["hours"] < 1).mean())
    out: dict[str, Any] = {"share_lt_1h_all_games": overall, "games": {}}
    for app, name in HEAVY_GAMES.items():
        g = recs.loc[recs["app_id"] == app, "hours"]
        if len(g):
            out["games"][name] = {"reviews": int(len(g)), "share_lt_1h": float((g < 1).mean()),
                                  "share_ge_500h": float((g >= 500).mean()), "max": float(g.max())}
    return out


def release_check(recs: pd.DataFrame, games: pd.DataFrame, early_access: set[int], max_days: int = 14) -> dict[str, Any]:
    """Reviews within ``max_days`` of release whose hours exceed the time elapsed since release."""
    g = games[["app_id", "date_release"]].copy()
    g["date_release"] = pd.to_datetime(g["date_release"], errors="coerce")
    df = recs.merge(g.dropna(), on="app_id", how="inner")
    df = df[~df["app_id"].isin(early_access)]
    days = (df["date"] - df["date_release"]).dt.days
    near = df[(days >= 0) & (days <= max_days)].copy()
    near_days = days[(days >= 0) & (days <= max_days)]
    possible = 24.0 * (near_days + 1)           # hours that could have been played by the end of the review day
    impossible = near["hours"] > possible
    by_day = (pd.DataFrame({"d": near_days, "bad": impossible}).groupby("d")["bad"].mean().round(4).to_dict())
    return {"window_days": max_days, "reviews_in_window": int(len(near)),
            "reviews_with_more_hours_than_possible": int(impossible.sum()),
            "share_impossible": float(impossible.mean()) if len(near) else None,
            "share_impossible_by_days_since_release": {int(k): v for k, v in by_day.items()},
            "early_access_games_excluded": len(early_access)}


def year_trend(recs: pd.DataFrame) -> dict[str, Any]:
    """Median hours and share >= 100 h by review year (crawl-time totals accumulate for older reviews)."""
    y = recs["date"].dt.year
    grp = recs.groupby(y)["hours"]
    return {int(k): {"rows": int(n), "median": float(m), "share_ge_100h": float(s)}
            for (k, n), m, s in zip(grp.size().items(), grp.median(), grp.apply(lambda x: (x >= 100).mean()))}


def early_access_ids(meta_path: Path | None) -> set[int]:
    """App IDs tagged 'Early Access' in the metadata file (empty set if not available)."""
    if meta_path is None or not meta_path.exists():
        return set()
    ids = set()
    with open(meta_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                obj = json.loads(line)
                if "Early Access" in (obj.get("tags") or []):
                    ids.add(int(obj["app_id"]))
    return ids


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Run all checks and write a JSON report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--recs", type=Path, required=True)
    ap.add_argument("--games", type=Path)
    ap.add_argument("--meta", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    recs = pd.read_csv(a.recs, usecols=["app_id", "date", "hours"], dtype={"app_id": "int32", "hours": "float32"})
    recs["date"] = pd.to_datetime(recs["date"])
    report: dict[str, Any] = {"hours_profile": hours_profile(recs), "heavy_games": heavy_game_check(recs),
                              "by_review_year": year_trend(recs)}
    if a.games is not None and a.games.exists():
        report["release_check"] = release_check(recs, pd.read_csv(a.games), early_access_ids(a.meta))
    else:
        report["release_check"] = "games.csv not given: release-date test skipped"
    a.out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("hours_profile", "heavy_games")}, indent=1))
    if isinstance(report["release_check"], dict):
        print(json.dumps({k: v for k, v in report["release_check"].items()
                          if k != "share_impossible_by_days_since_release"}, indent=1))
    return report


if __name__ == "__main__":
    main()
