"""Synthetic data with the schema of the Kaggle Steam files, for tests and smoke runs only.

No real user data is used. Users have latent genre preferences; games have a
genre, a skewed popularity and genre-related tags (10% have no tags, mimicking
untagged popular titles). Review probability, sentiment and playtime depend on
the user-genre match, so models with a real signal beat Random.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def make_synthetic(out_dir: str | Path, n_users: int = 1200, n_items: int = 200, n_genres: int = 5,
                   tags_per_genre: int = 5, years: tuple[int, int] = (2014, 2022), seed: int = 0,
                   duplicate_share: float = 0.01) -> tuple[Path, Path]:
    """Write ``recommendations.csv`` and ``games_metadata.json`` and return their paths."""
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    genre = rng.integers(n_genres, size=n_items)
    pop = np.exp(rng.normal(0.0, 1.0, n_items))
    pref = rng.dirichlet(np.full(n_genres, 0.5), size=n_users)
    meta_lines = []
    for i in range(n_items):
        if rng.random() < 0.10:
            tags: list[str] = []
        else:
            own = rng.choice(tags_per_genre, size=rng.integers(2, 5), replace=False)
            tags = [f"tag_{genre[i]}_{t}" for t in own]
            if rng.random() < 0.5:
                tags.append(str(rng.choice(["Indie", "Casual"])))
        meta_lines.append({"app_id": int(1000 + 10 * i), "description": "", "tags": tags})
    rows = []
    year_list = np.arange(years[0], years[1] + 1)
    for u in range(n_users):
        n_int = int(min(80, 3 + rng.geometric(1 / 12)))
        w = pop * (0.15 + pref[u, genre])
        items = rng.choice(n_items, size=min(n_int, n_items), replace=False, p=w / w.sum())
        for i in items:
            match = pref[u, genre[i]] * n_genres
            p_pos = 1.0 / (1.0 + np.exp(-(1.5 * (match - 1.0) + 1.2)))
            y = int(rng.choice(year_list))
            day = pd.Timestamp(year=y, month=1, day=1) + pd.Timedelta(days=int(rng.integers(0, 365)))
            hours = 0.0 if rng.random() < 0.02 else float(np.round(np.exp(rng.normal(1.0 + 0.8 * match, 1.0)), 1))
            rows.append((1000 + 10 * int(i), str(day.date()), "true" if rng.random() < p_pos else "false",
                         hours, 10_000_000 + u))
    df = pd.DataFrame(rows, columns=["app_id", "date", "is_recommended", "hours", "user_id"])
    dup = df.sample(frac=duplicate_share, random_state=seed)
    df = pd.concat([df, dup], ignore_index=True)
    df.insert(1, "helpful", 0)
    df.insert(2, "funny", 0)
    df["review_id"] = np.arange(len(df))
    rec_path, meta_path = out / "recommendations.csv", out / "games_metadata.json"
    df.to_csv(rec_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as fh:
        for m in meta_lines:
            fh.write(json.dumps(m) + "\n")
    return rec_path, meta_path
