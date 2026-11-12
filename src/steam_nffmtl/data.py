"""Data loading and leakage-controlled preprocessing.

All learned preprocessing decisions (k-core filtering, identifier mappings,
tag vocabulary) are derived from the training years only. Validation and test
rows are mapped onto the training vocabulary; rows whose user or game is not in
that vocabulary are kept, marked with index ``-1`` and counted in the data card.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DataConfig
from .repro import save_json, sha256_arrays, sha256_file

PREPROCESS_VERSION = "pp2"  # bump whenever build_prepared changes its output
REQUIRED_COLUMNS = ["user_id", "app_id", "date", "is_recommended", "hours"]
PARTITIONS = ("train", "val", "test")


def _to_int8_bool(series: pd.Series) -> np.ndarray:
    """Convert a boolean-like column (bool, 0/1 or 'true'/'false') to int8."""
    if series.dtype == bool:
        return series.to_numpy().astype(np.int8)
    if pd.api.types.is_numeric_dtype(series):
        return (series.to_numpy() > 0).astype(np.int8)
    mapped = series.astype(str).str.strip().str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0})
    if mapped.isna().any():
        bad = series[mapped.isna()].unique()[:5]
        raise ValueError(f"Unrecognised is_recommended values: {bad}")
    return mapped.to_numpy().astype(np.int8)


def load_metadata(path: str | Path) -> pd.DataFrame:
    """Read the JSON-lines game metadata file and keep ``app_id`` and ``tags``."""
    rows: list[dict[str, Any]] = []
    n_bad = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            tags = obj.get("tags")
            rows.append({"app_id": int(obj["app_id"]), "tags": tags if isinstance(tags, list) else []})
    meta = pd.DataFrame(rows, columns=["app_id", "tags"])
    meta.attrs["n_bad_lines"] = n_bad
    return meta


def load_raw(dcfg: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load review rows and game metadata from the paths in the configuration."""
    recs = pd.read_csv(dcfg.recommendations_csv, usecols=REQUIRED_COLUMNS)
    recs["date"] = pd.to_datetime(recs["date"])
    recs["is_recommended"] = _to_int8_bool(recs["is_recommended"])
    recs["hours"] = recs["hours"].astype("float32")
    recs["user_id"] = recs["user_id"].astype("int64")
    recs["app_id"] = recs["app_id"].astype("int64")
    meta = load_metadata(dcfg.metadata_json)
    return recs, meta


@dataclass
class PreparedData:
    """Preprocessed, index-mapped interactions plus the data card.

    ``train`` holds index arrays ``u``, ``i`` (all valid), ``label`` (int8
    is_recommended) and ``hours_log``. ``val`` and ``test`` hold the same keys,
    where ``u``/``i`` are ``-1`` for identifiers unknown to the training
    vocabulary, plus original ``user_id``/``item_id``.
    """

    user_ids: np.ndarray
    item_ids: np.ndarray
    tag_names: list[str]
    tag_matrix: np.ndarray
    train: dict[str, np.ndarray]
    val: dict[str, np.ndarray]
    test: dict[str, np.ndarray]
    card: dict[str, Any] = field(default_factory=dict)

    @property
    def n_users(self) -> int:
        """Number of training-known users."""
        return int(len(self.user_ids))

    @property
    def n_items(self) -> int:
        """Number of games in the training catalog."""
        return int(len(self.item_ids))

    def partition(self, name: str) -> dict[str, np.ndarray]:
        """Return the arrays of ``train``, ``val`` or ``test``."""
        if name not in PARTITIONS:
            raise KeyError(name)
        return getattr(self, name)

    def fingerprint(self) -> str:
        """SHA-256 over all arrays (and the tag order) that define the experiment data; cached."""
        cached = self.__dict__.get("_fingerprint")
        if cached is not None:
            return cached
        tags = np.frombuffer("\x1f".join(self.tag_names).encode("utf-8"), dtype=np.uint8)
        arrays = [self.user_ids, self.item_ids, self.tag_matrix, tags]
        for part in PARTITIONS:
            d = self.partition(part)
            arrays += [d[k] for k in sorted(d)]
        self.__dict__["_fingerprint"] = sha256_arrays(*arrays)
        return self.__dict__["_fingerprint"]

    def save(self, directory: str | Path) -> None:
        """Persist arrays (``data.npz``), tag order and data card."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "user_ids": self.user_ids,
            "item_ids": self.item_ids,
            "tag_matrix": self.tag_matrix,
        }
        for part in PARTITIONS:
            for key, arr in self.partition(part).items():
                payload[f"{part}__{key}"] = arr
        np.savez_compressed(directory / "data.npz", **payload)
        save_json({"tag_names": self.tag_names}, directory / "tag_order.json")
        card = dict(self.card)
        card["fingerprint"] = self.fingerprint()
        save_json(card, directory / "data_card.json")

    @classmethod
    def load(cls, directory: str | Path) -> "PreparedData":
        """Load a dataset written by :meth:`save` and verify its fingerprint."""
        directory = Path(directory)
        with np.load(directory / "data.npz") as z:
            arrays = {k: z[k] for k in z.files}
        parts: dict[str, dict[str, np.ndarray]] = {p: {} for p in PARTITIONS}
        for key, arr in arrays.items():
            if "__" in key:
                part, name = key.split("__", 1)
                parts[part][name] = arr
        with open(directory / "tag_order.json", encoding="utf-8") as fh:
            tag_names = json.load(fh)["tag_names"]
        with open(directory / "data_card.json", encoding="utf-8") as fh:
            card = json.load(fh)
        data = cls(arrays["user_ids"], arrays["item_ids"], tag_names, arrays["tag_matrix"],
                   parts["train"], parts["val"], parts["test"], card)
        expected = card.get("fingerprint")
        if expected and expected != data.fingerprint():
            raise ValueError("Cached data fingerprint mismatch; rebuild the cache")
        return data


def _kcore(train: pd.DataFrame, k: int) -> tuple[pd.DataFrame, int]:
    """Iteratively keep users and games with at least ``k`` training interactions."""
    iterations = 0
    while True:
        iterations += 1
        n_before = len(train)
        uc = train["user_id"].value_counts()
        train = train[train["user_id"].isin(uc.index[uc >= k])]
        ic = train["app_id"].value_counts()
        train = train[train["app_id"].isin(ic.index[ic >= k])]
        if len(train) == n_before:
            return train, iterations


def _pair_key(user: np.ndarray, item: np.ndarray, base: int) -> np.ndarray:
    """Encode (user_id, app_id) pairs as unique int64 keys."""
    return user.astype(np.int64) * np.int64(base) + item.astype(np.int64)


def build_prepared(recs: pd.DataFrame, meta: pd.DataFrame, dcfg: DataConfig) -> PreparedData:
    """Split by calendar year, de-duplicate, apply train-only k-core and map identifiers."""
    y0, y1 = dcfg.train_years
    card: dict[str, Any] = {
        "dataset": "Game Recommendations on Steam (A. Kozyriev, Kaggle)",
        "raw": {
            "rows": int(len(recs)),
            "users": int(recs["user_id"].nunique()),
            "games_in_reviews": int(recs["app_id"].nunique()),
            "metadata_games": int(len(meta)),
            "metadata_bad_lines": int(meta.attrs.get("n_bad_lines", 0)),
            "date_min": str(recs["date"].min().date()),
            "date_max": str(recs["date"].max().date()),
        },
        "split": {"unit": "global calendar-year split", "train_years": [y0, y1],
                  "val_year": dcfg.val_year, "test_year": dcfg.test_year},
    }
    year = recs["date"].dt.year.to_numpy()
    part = np.full(len(recs), -1, dtype=np.int8)
    part[(year >= y0) & (year <= y1)] = 0
    part[year == dcfg.val_year] = 1
    part[year == dcfg.test_year] = 2
    card["raw"]["rows_outside_split_years"] = int((part < 0).sum())
    df = recs.loc[part >= 0, REQUIRED_COLUMNS].copy()
    df["part"] = part[part >= 0]
    df = df.sort_values(["date"], kind="stable")

    # --- de-duplicate repeated (user, game) pairs inside each partition ---
    if dcfg.dedup != "keep_last":
        raise ValueError("Only dedup='keep_last' is implemented")
    dup = df.duplicated(subset=["part", "user_id", "app_id"], keep="last")
    card["duplicates_removed"] = {p: int(dup[df["part"] == j].sum()) for j, p in enumerate(PARTITIONS)}
    df = df[~dup]
    base = int(df["app_id"].max()) + 1
    keys = {j: _pair_key(df.loc[df["part"] == j, "user_id"].to_numpy(),
                         df.loc[df["part"] == j, "app_id"].to_numpy(), base) for j in range(3)}
    card["repeated_pairs_across_partitions"] = {
        "val_pairs_also_in_train": int(np.isin(keys[1], keys[0]).sum()),
        "test_pairs_also_in_train_or_val": int(np.isin(keys[2], np.concatenate([keys[0], keys[1]])).sum()),
    }
    card["rows_after_dedup"] = {p: int((df["part"] == j).sum()) for j, p in enumerate(PARTITIONS)}
    total = sum(card["rows_after_dedup"].values())
    card["split"]["row_share_after_dedup"] = {p: round(v / total, 4) for p, v in card["rows_after_dedup"].items()}

    # --- train-only k-core ---
    train_raw = df[df["part"] == 0]
    train, iters = _kcore(train_raw, dcfg.kcore)
    if len(train) == 0:
        raise ValueError(f"k-core with k={dcfg.kcore} removed every training row")
    user_ids = np.sort(train["user_id"].unique()).astype(np.int64)
    item_ids = np.sort(train["app_id"].unique()).astype(np.int64)
    n_u, n_i = len(user_ids), len(item_ids)
    card["kcore"] = {
        "k": dcfg.kcore, "applied_to": "training years only", "iterations": iters,
        "rows_before": int(len(train_raw)), "rows_after": int(len(train)),
        "users_before": int(train_raw["user_id"].nunique()), "users_after": int(n_u),
        "games_before": int(train_raw["app_id"].nunique()), "games_after": int(n_i),
        "sparsity_after": float(1.0 - len(train) / (n_u * n_i)) if n_u and n_i else None,
    }
    uidx = pd.Index(user_ids)
    iidx = pd.Index(item_ids)

    def _arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
        return {
            "u": uidx.get_indexer(frame["user_id"]).astype(np.int64),
            "i": iidx.get_indexer(frame["app_id"]).astype(np.int64),
            "label": frame["is_recommended"].to_numpy().astype(np.int8),
            "hours_log": np.log1p(frame["hours"].to_numpy().astype(np.float32)).astype(np.float32),
            "user_id": frame["user_id"].to_numpy().astype(np.int64),
            "item_id": frame["app_id"].to_numpy().astype(np.int64),
        }

    tr = _arrays(train)
    assert (tr["u"] >= 0).all() and (tr["i"] >= 0).all()
    card["train"] = {"rows": int(len(train)), "users": n_u, "games": n_i,
                     "positive_rate": float(tr["label"].mean()),
                     "rows_hours_gt_0": int((tr["hours_log"] > 0).sum())}
    parts: dict[str, dict[str, np.ndarray]] = {"train": tr}
    for j, name in ((1, "val"), (2, "test")):
        frame = df[df["part"] == j]
        arr = _arrays(frame)
        known_u, known_i = arr["u"] >= 0, arr["i"] >= 0
        card[name] = {
            "rows": int(len(frame)),
            "users": int(frame["user_id"].nunique()),
            "games": int(frame["app_id"].nunique()),
            "positive_rate": float(arr["label"].mean()) if len(frame) else None,
            "rows_cold_user": int((~known_u).sum()),
            "rows_known_user_cold_game": int((known_u & ~known_i).sum()),
            "rows_warm": int((known_u & known_i).sum()),
            "users_known": int(len(np.unique(arr["u"][known_u]))),
            "share_users_known": float(len(np.unique(arr["u"][known_u])) / max(frame["user_id"].nunique(), 1)),
        }
        parts[name] = arr

    # --- tag vocabulary from training-catalog games only ---
    meta_cat = meta[meta["app_id"].isin(item_ids)].drop_duplicates("app_id")
    counts: dict[str, int] = {}
    for tags in meta_cat["tags"]:
        for t in set(tags):
            counts[t] = counts.get(t, 0) + 1
    tag_names = [t for t, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[: dcfg.n_tags]]
    tpos = {t: j for j, t in enumerate(tag_names)}
    tag_matrix = np.zeros((n_i, len(tag_names)), dtype=np.float32)
    rows = iidx.get_indexer(meta_cat["app_id"])
    for r, tags in zip(rows, meta_cat["tags"]):
        for t in tags:
            j = tpos.get(t)
            if j is not None:
                tag_matrix[r, j] = 1.0
    card["tags"] = {
        "vocabulary": "top-n tags by document frequency over training-catalog games",
        "n_tags": len(tag_names),
        "catalog_games_with_metadata": int(len(meta_cat)),
        "catalog_games_without_any_selected_tag": int((tag_matrix.sum(1) == 0).sum()),
        "note": "Tags come from a metadata snapshot collected after the test year (static-feature assumption).",
    }
    card["unverified_assumptions"] = [
        "Game metadata (tags) is a single snapshot taken after 2022, not a historical record; tag features may "
        "contain information from after the training years.",
        "It is not verified whether `hours` is playtime at review time or cumulative playtime at data collection; "
        "in the latter case the regression target of training reviews contains post-review activity.",
    ]
    return PreparedData(user_ids, item_ids, tag_names, tag_matrix, parts["train"], parts["val"], parts["test"], card)


class CacheMismatchError(RuntimeError):
    """Raised when a cache does not match the requested configuration or source files."""


def data_config_identity(dcfg: DataConfig) -> dict[str, Any]:
    """Settings that determine the prepared data (paths and verification switches excluded)."""
    return {"preprocess_version": PREPROCESS_VERSION, "train_years": list(dcfg.train_years),
            "val_year": dcfg.val_year, "test_year": dcfg.test_year, "kcore": dcfg.kcore,
            "n_tags": dcfg.n_tags, "dedup": dcfg.dedup}


def cache_path(dcfg: DataConfig) -> Path:
    """``cache_dir/<version>-<hash>``: different preprocessing settings never share a cache."""
    ident = json.dumps(data_config_identity(dcfg), sort_keys=True).encode()
    import hashlib

    return Path(dcfg.cache_dir) / f"{PREPROCESS_VERSION}-{hashlib.sha256(ident).hexdigest()[:12]}"


def _source_identity(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    st = p.stat()
    return {"path": str(p.resolve()), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns),
            "sha256": sha256_file(p)}


def _check_sources(manifest: dict[str, Any], dcfg: DataConfig) -> None:
    """Compare current raw files with the identities stored at build time."""
    for key, path in (("recommendations_csv", dcfg.recommendations_csv), ("metadata_json", dcfg.metadata_json)):
        stored = manifest["sources"][key]
        p = Path(path)
        if not p.exists():
            raise CacheMismatchError(f"Source file missing: {p}")
        st = p.stat()
        quick_same = st.st_size == stored["size"] and st.st_mtime_ns == stored["mtime_ns"]
        if dcfg.verify_source_hash or not quick_same:
            if st.st_size != stored["size"] or sha256_file(p) != stored["sha256"]:
                raise CacheMismatchError(f"{key} differs from the file used to build the cache ({stored['path']})")


def prepare_and_cache(dcfg: DataConfig, force: bool = False) -> PreparedData:
    """Build the prepared dataset, or load a cache whose manifest matches config and source files."""
    cache = cache_path(dcfg)
    manifest_file = cache / "cache_manifest.json"
    if manifest_file.exists() and not force:
        with open(manifest_file, encoding="utf-8") as fh:
            manifest = json.load(fh)
        if manifest.get("data_config") != data_config_identity(dcfg):
            raise CacheMismatchError(f"Cache {cache} was built with {manifest.get('data_config')}")
        _check_sources(manifest, dcfg)
        data = PreparedData.load(cache)
        if data.fingerprint() != manifest.get("fingerprint"):
            raise CacheMismatchError("Cache arrays do not match the manifest fingerprint")
        return data
    if (cache / "data.npz").exists() and not manifest_file.exists() and not force:
        raise CacheMismatchError(f"{cache} has data but no manifest; rebuild with --force")
    sources = {"recommendations_csv": _source_identity(dcfg.recommendations_csv),
               "metadata_json": _source_identity(dcfg.metadata_json)}
    recs, meta = load_raw(dcfg)
    data = build_prepared(recs, meta, dcfg)
    data.card["raw_sha256"] = {k: v["sha256"] for k, v in sources.items()}
    data.card["data_config"] = data_config_identity(dcfg)
    data.save(cache)
    save_json({"data_config": data_config_identity(dcfg), "sources": sources,
               "fingerprint": data.fingerprint()}, manifest_file)
    return data


def write_card(data: PreparedData, path: str | Path) -> None:
    """Write the data card with the dataset fingerprint."""
    card = dict(data.card)
    card["fingerprint"] = data.fingerprint()
    save_json(card, path)
