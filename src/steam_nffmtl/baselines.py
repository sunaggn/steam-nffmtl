"""Baseline recommenders that implement the common :class:`~steam_nffmtl.evaluate.Scorer` interface.

All baselines return dense scores over the full training catalog; the evaluator
masks history. None of them falls back to another model: a user the model
cannot score raises an error instead of silently receiving Popularity.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import scipy.sparse as sp


class RandomScorer:
    """Uniform random ranking; scores are a deterministic hash of (seed, user, item)."""

    def __init__(self, n_items: int, seed: int, name: str = "Random") -> None:
        self.name, self.n_items, self.seed = name, n_items, np.uint64(seed)

    def _hash(self, u: np.ndarray, i: np.ndarray) -> np.ndarray:
        x = (u.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) ^ (i.astype(np.uint64) * np.uint64(0xBF58476D1CE4E5B9))
        x ^= self.seed * np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
        x *= np.uint64(0xD6E8FEB86659FD93)
        x ^= x >> np.uint64(32)
        return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)

    def score(self, users: np.ndarray) -> np.ndarray:
        """Random scores, identical for a (seed, user, item) triple regardless of batching."""
        with np.errstate(over="ignore"):
            return self._hash(users[:, None], np.arange(self.n_items)[None, :])

    def score_pairs(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Random scores for aligned pairs."""
        with np.errstate(over="ignore"):
            return self._hash(users, items)


class PopularityScorer:
    """Non-personalised ranking by training popularity (count in the profile matrix)."""

    def __init__(self, profile: sp.csr_matrix, name: str = "Popularity") -> None:
        self.name = name
        self.pop = np.asarray(profile.sum(axis=0)).ravel().astype(np.float64)

    def score(self, users: np.ndarray) -> np.ndarray:
        """Same popularity vector for every user."""
        return np.broadcast_to(self.pop, (len(users), len(self.pop))).copy()

    def score_pairs(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Popularity of each paired game."""
        return self.pop[items]


def _topk_sparse(sim_block: np.ndarray, row_offset: int, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the ``k`` largest positive similarities per row, excluding the diagonal."""
    rows_idx = np.arange(sim_block.shape[0])
    sim_block[rows_idx, rows_idx + row_offset] = 0.0
    k = min(k, sim_block.shape[1] - 1)
    part = np.argpartition(-sim_block, kth=k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(sim_block, part, axis=1)
    keep = vals > 0
    r = np.repeat(rows_idx + row_offset, k)[keep.ravel()]
    return r, part.ravel()[keep.ravel()], vals.ravel()[keep.ravel()]


class _NeighbourScorer:
    """Shared scoring for item-neighbourhood models: score = profile_row @ S."""

    name: str
    profile: sp.csr_matrix
    sim: sp.csr_matrix

    def score(self, users: np.ndarray) -> np.ndarray:
        """Sum of similarities between candidate games and the user's profile games."""
        return np.asarray((self.profile[users] @ self.sim).todense(), dtype=np.float64)

    def score_pairs(self, users: np.ndarray, items: np.ndarray, chunk: int = 20000) -> np.ndarray:
        """Pair scores without materialising full rows (processed in chunks)."""
        out = np.empty(len(users), dtype=np.float64)
        for s in range(0, len(users), chunk):
            p = self.profile[users[s:s + chunk]]
            sim = self.sim[:, items[s:s + chunk]].T.tocsr()
            out[s:s + chunk] = np.asarray(p.multiply(sim).sum(axis=1)).ravel()
        return out


class TagKNNScorer(_NeighbourScorer):
    """Content-based kNN: cosine similarity between binary tag vectors (top-k neighbours)."""

    def __init__(self, tag_matrix: np.ndarray, profile: sp.csr_matrix, k: int = 100,
                 block: int = 2048, name: str = "TagKNN") -> None:
        self.name, self.profile = name, profile
        norms = np.linalg.norm(tag_matrix, axis=1, keepdims=True)
        x = np.divide(tag_matrix, norms, out=np.zeros_like(tag_matrix, dtype=np.float32), where=norms > 0)
        n = x.shape[0]
        rs, cs, vs = [], [], []
        for b in range(0, n, block):
            sim = x[b:b + block] @ x.T
            r, c, v = _topk_sparse(sim, b, k)
            rs.append(r); cs.append(c); vs.append(v)
        self.sim = sp.csr_matrix((np.concatenate(vs), (np.concatenate(rs), np.concatenate(cs))), shape=(n, n))


class ItemKNNCFScorer(_NeighbourScorer):
    """Collaborative item kNN: shrunk cosine on co-occurrence counts (top-k neighbours)."""

    def __init__(self, profile: sp.csr_matrix, k: int = 100, shrink: float = 10.0,
                 block: int = 1024, name: str = "ItemKNN-CF") -> None:
        self.name, self.profile = name, profile
        xt = profile.T.tocsr().astype(np.float32)
        x = profile.tocsr().astype(np.float32)
        deg = np.asarray(profile.sum(axis=0)).ravel().astype(np.float32)
        n = profile.shape[1]
        rs, cs, vs = [], [], []
        for b in range(0, n, block):
            co = np.asarray((xt[b:b + block] @ x).todense(), dtype=np.float32)
            denom = np.sqrt(deg[b:b + block, None] * deg[None, :]) + shrink
            sim = np.divide(co, denom, out=np.zeros_like(co), where=denom > 0)
            r, c, v = _topk_sparse(sim, b, k)
            rs.append(r); cs.append(c); vs.append(v)
        self.sim = sp.csr_matrix((np.concatenate(vs), (np.concatenate(rs), np.concatenate(cs))), shape=(n, n))


class ImplicitMFScorer:
    """BPR-MF or ALS from the ``implicit`` library with explicit orientation checks.

    ``implicit`` >= 0.5 expects a *user x item* matrix in ``fit``. After fitting we
    assert that ``user_factors`` has one row per user and ``item_factors`` one row
    per game; scores are computed as dot products, never via ``recommend``.
    """

    def __init__(self, kind: str, profile: sp.csr_matrix, params: dict[str, Any], seed: int,
                 use_gpu: bool = False, name: str | None = None,
                 expected_users: int | None = None, expected_items: int | None = None) -> None:
        import implicit

        major, minor = (int(x) for x in implicit.__version__.split(".")[:2])
        if (major, minor) < (0, 5):
            raise RuntimeError("implicit >= 0.5 is required (user x item fit convention)")
        self.kind = kind
        self.name = name or {"bpr": "BPR-MF", "als": "ALS"}[kind]
        n_users = expected_users if expected_users is not None else profile.shape[0]
        n_items = expected_items if expected_items is not None else profile.shape[1]
        if profile.shape != (n_users, n_items):
            raise AssertionError(f"{kind}: fit matrix has shape {profile.shape}, expected users x items "
                                 f"({n_users}, {n_items}); a transposed matrix is not allowed")
        p = dict(params)
        if kind == "bpr":
            model = implicit.bpr.BayesianPersonalizedRanking(
                factors=p.get("factors", 64), learning_rate=p.get("learning_rate", 0.01),
                regularization=p.get("regularization", 0.01), iterations=p.get("iterations", 100),
                random_state=seed, use_gpu=use_gpu)
            mat = profile
        elif kind == "als":
            model = implicit.als.AlternatingLeastSquares(
                factors=p.get("factors", 64), regularization=p.get("regularization", 0.1),
                alpha=p.get("alpha", 1.0), iterations=p.get("iterations", 15),
                random_state=seed, use_gpu=use_gpu)
            mat = profile
        else:
            raise ValueError(kind)
        model.fit(mat.astype(np.float32).tocsr(), show_progress=False)
        if hasattr(model, "to_cpu"):
            model = model.to_cpu()
        uf, itf = np.asarray(model.user_factors), np.asarray(model.item_factors)
        if uf.shape[0] != n_users or itf.shape[0] != n_items:
            raise AssertionError(
                f"{self.name}: factor shapes {uf.shape}/{itf.shape} do not match users={n_users}, items={n_items}")
        self.user_factors, self.item_factors = uf.astype(np.float32), itf.astype(np.float32)

    def score(self, users: np.ndarray) -> np.ndarray:
        """Dot products with every game."""
        return (self.user_factors[users] @ self.item_factors.T).astype(np.float64)

    def score_pairs(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Row-wise dot products for aligned pairs."""
        return np.einsum("ij,ij->i", self.user_factors[users], self.item_factors[items]).astype(np.float64)


def ease_cost_estimate(n_items: int, device: str = "cpu") -> dict[str, float]:
    """Peak memory (GB) of :class:`EASEScorer` as implemented (float64 Gram and inverse).

    CPU path: Gram (8n²) + inverse (8n², NumPy/LAPACK also copies its input: +8n²) +
    float32 result (4n²) ≈ 28n² bytes. GPU path: host Gram (8n²) + float32 result (4n²)
    on the CPU, and Gram copy + inverse + solver workspace (≈ 3 × 8n²) on the GPU.
    """
    n2 = float(n_items) ** 2
    gb = 1e9
    est = {"n_items": n_items, "device": device, "gram_float64_gb": 8 * n2 / gb,
           "result_float32_gb": 4 * n2 / gb, "inversion_tflop_approx": 2.0 * n_items ** 3 / 1e12}
    if device == "cpu":
        est["peak_host_gb"] = 28 * n2 / gb
        est["peak_gpu_gb"] = 0.0
    else:
        est["peak_host_gb"] = 12 * n2 / gb
        est["peak_gpu_gb"] = 24 * n2 / gb
    return est


def available_memory_gb(device: str) -> float:
    """Free host RAM (Linux sysconf) or free GPU memory, in GB; ``inf`` if unknown."""
    if device == "cpu":
        try:
            import os

            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
        except (ValueError, OSError, AttributeError):
            return float("inf")
    import torch

    free, _ = torch.cuda.mem_get_info(torch.device(device))
    return free / 1e9


class EASEScorer:
    """EASE (Steck, 2019): B = I - P / diag(P), P = (X^T X + λI)^-1, diag(B) = 0.

    Refuses to start when the estimated peak memory exceeds ``memory_fraction`` of the
    currently free memory on the host or GPU.
    """

    def __init__(self, profile: sp.csr_matrix, l2: float = 500.0, max_items: int = 40000,
                 device: str = "cpu", memory_fraction: float = 0.8, name: str = "EASE") -> None:
        n = profile.shape[1]
        est = ease_cost_estimate(n, device)
        if n > max_items:
            raise MemoryError(f"EASE: n_items={n} exceeds max_items={max_items} ({est})")
        host_free = available_memory_gb("cpu")
        if est["peak_host_gb"] > memory_fraction * host_free:
            raise MemoryError(f"EASE needs ~{est['peak_host_gb']:.2f} GB host RAM, {host_free:.2f} GB free")
        if device != "cpu":
            gpu_free = available_memory_gb(device)
            if est["peak_gpu_gb"] > memory_fraction * gpu_free:
                raise MemoryError(f"EASE needs ~{est['peak_gpu_gb']:.2f} GB on {device}, {gpu_free:.2f} GB free")
        self.name, self.profile, self.estimate = name, profile.astype(np.float32).tocsr(), est
        gram = (self.profile.T @ self.profile).toarray().astype(np.float64)
        gram[np.diag_indices(n)] += l2
        if device == "cpu":
            p = np.linalg.inv(gram)
            del gram
            d = np.diag(p).copy()
            p /= -d[None, :]
            np.fill_diagonal(p, 0.0)
            self.B = p.astype(np.float32)
        else:
            import torch

            g = torch.from_numpy(gram).to(device)
            del gram
            p = torch.linalg.inv(g)
            del g
            d = torch.diagonal(p).clone()
            p.div_(-d[None, :])
            p.fill_diagonal_(0.0)
            self.B = p.float().cpu().numpy()
            del p
            torch.cuda.empty_cache()

    def score(self, users: np.ndarray) -> np.ndarray:
        """Profile rows times the item-item weight matrix."""
        return np.asarray(self.profile[users] @ self.B, dtype=np.float64)

    def score_pairs(self, users: np.ndarray, items: np.ndarray, chunk: int = 1024) -> np.ndarray:
        """Pair scores without full rows (processed in chunks)."""
        out = np.empty(len(users), dtype=np.float64)
        for s in range(0, len(users), chunk):
            p = self.profile[users[s:s + chunk]]
            cols = np.ascontiguousarray(self.B[:, items[s:s + chunk]].T)
            out[s:s + chunk] = np.asarray(p.multiply(cols).sum(axis=1)).ravel()
        return out


BASELINE_KEYS = {"Random": "random", "Popularity": "popularity", "TagKNN": "tagknn", "ItemKNN-CF": "itemknn",
                 "BPR-MF": "bpr", "ALS": "als", "EASE": "ease"}
DEFAULT_ENABLED = {"random": True, "popularity": True, "tagknn": True, "itemknn": True, "bpr": True,
                   "als": True, "ease": False}


def make_baseline(key: str, profile: sp.csr_matrix, tag_matrix: np.ndarray, params: dict[str, Any],
                  seed: int, use_gpu: bool = False) -> Any:
    """Fit one baseline identified by its config key with explicit hyperparameters."""
    n_users, n_items = profile.shape
    p = dict(params)
    if key == "random":
        return RandomScorer(n_items, seed)
    if key == "popularity":
        return PopularityScorer(profile)
    if key == "tagknn":
        return TagKNNScorer(tag_matrix, profile, **p)
    if key == "itemknn":
        return ItemKNNCFScorer(profile, **p)
    if key in ("bpr", "als"):
        return ImplicitMFScorer(key, profile, p, seed, use_gpu, expected_users=n_users, expected_items=n_items)
    if key == "ease":
        return EASEScorer(profile, **p)
    raise ValueError(f"Unknown baseline key {key!r}")


def baseline_factories(profile: sp.csr_matrix, tag_matrix: np.ndarray, cfg: dict[str, dict[str, Any]],
                       seed: int, use_gpu: bool = False) -> list[tuple[str, dict[str, Any], Callable[[], Any]]]:
    """(name, hyperparameters, factory) for every enabled baseline; fitting happens when called."""
    out: list[tuple[str, dict[str, Any], Callable[[], Any]]] = []
    for name, key in BASELINE_KEYS.items():
        section = dict(cfg.get(key, {}))
        if not section.pop("enabled", DEFAULT_ENABLED[key]):
            continue
        out.append((name, section, (lambda k=key, prm=section: make_baseline(k, profile, tag_matrix, prm, seed,
                                                                           use_gpu))))
    return out


def build_baselines(profile: sp.csr_matrix, tag_matrix: np.ndarray, cfg: dict[str, dict[str, Any]],
                    seed: int, use_gpu: bool = False) -> dict[str, Any]:
    """Fit every enabled baseline at once (convenient for tests and small data)."""
    return {name: make() for name, _, make in baseline_factories(profile, tag_matrix, cfg, seed, use_gpu)}


def seed_dependent(name: str) -> bool:
    """Whether repeated seeds change this baseline's output."""
    return name in {"Random", "BPR-MF", "ALS"}


__all__ = ["RandomScorer", "PopularityScorer", "TagKNNScorer", "ItemKNNCFScorer", "ImplicitMFScorer",
           "EASEScorer", "ease_cost_estimate", "available_memory_gb", "BASELINE_KEYS", "make_baseline", "baseline_factories", "build_baselines", "seed_dependent"]
