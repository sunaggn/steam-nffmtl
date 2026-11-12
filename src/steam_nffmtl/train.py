"""Training of NFF-MTL with exact negative sampling and validation-metric model selection."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from .config import ExperimentConfig
from .data import PreparedData
from .evaluate import evaluate_ranking
from .model import NFFMTL, NFFMTLScorer
from .identity import config_id, nffmtl_training_settings, settings_diff, signature
from .protocol import EvalSplit, _csr
from .repro import code_sha256, environment_info, set_seed

VARIANTS = {
    "full": {"use_content": True, "use_mtl": True},
    "no_content": {"use_content": False, "use_mtl": True},
    "single_task": {"use_content": True, "use_mtl": False},
}


class ExactNegativeSampler:
    """Uniform negatives from the training catalog that are never a training interaction of the user.

    Membership is tested exactly against sorted ``user * n_items + item`` keys;
    rejected draws are resampled until none remain.
    """

    def __init__(self, interactions: sp.csr_matrix, device: torch.device, generator: torch.Generator) -> None:
        self.n_items = interactions.shape[1]
        deg = np.diff(interactions.indptr)
        if (deg >= self.n_items).any():
            raise ValueError("A user has interacted with every catalog game; no negative exists")
        keys = np.repeat(np.arange(interactions.shape[0], dtype=np.int64), deg) * self.n_items + interactions.indices
        self.keys = torch.as_tensor(np.sort(keys), device=device)
        self.device, self.g = device, generator

    def contains(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Boolean mask: True where (user, item) is a training interaction."""
        q = users * self.n_items + items
        pos = torch.searchsorted(self.keys, q).clamp_(max=len(self.keys) - 1)
        return self.keys[pos] == q

    def sample(self, users: torch.Tensor, max_rounds: int = 1000) -> torch.Tensor:
        """One negative game per entry of ``users``."""
        items = torch.randint(self.n_items, users.shape, generator=self.g, device=self.device)
        for _ in range(max_rounds):
            bad = self.contains(users, items)
            n_bad = int(bad.sum())
            if n_bad == 0:
                return items
            items[bad] = torch.randint(self.n_items, (n_bad,), generator=self.g, device=self.device)
        raise RuntimeError("Negative sampling did not converge")


def training_tensors(data: PreparedData, task: str, reg_min_hours: float,
                     device: torch.device) -> dict[str, torch.Tensor]:
    """Observed training rows as device tensors with task-specific labels.

    ``positive`` task: label = is_recommended (negative reviews are observed zeros).
    ``interaction`` task: every observed review has label 1.
    ``has_hours`` marks observed rows that enter the regression loss (hours > reg_min_hours).
    """
    tr = data.train
    label = tr["label"].astype(np.float32) if task == "positive" else np.ones(len(tr["u"]), dtype=np.float32)
    has_hours = tr["hours_log"] > np.log1p(reg_min_hours)
    return {
        "u": torch.as_tensor(tr["u"], device=device),
        "i": torch.as_tensor(tr["i"], device=device),
        "y": torch.as_tensor(label, device=device),
        "h": torch.as_tensor(tr["hours_log"], device=device),
        "hm": torch.as_tensor(has_hours, device=device),
    }


def pos_weight_value(labels: np.ndarray, neg_ratio: int, setting: str | float) -> float | None:
    """BCE positive-class weight = (#label-0 rows incl. sampled negatives) / (#label-1 rows)."""
    if setting in (None, "none"):
        return None
    if setting != "auto":
        return float(setting)
    n_pos = float((labels == 1).sum())
    n_zero = float((labels == 0).sum()) + neg_ratio * len(labels)
    return n_zero / n_pos


def model_spec(data: PreparedData, cfg: ExperimentConfig, variant: str) -> dict[str, Any]:
    """Complete description of the network that ``build_model`` creates (the single authority)."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    t = cfg.train
    return {"variant": variant, **VARIANTS[variant], "n_users": data.n_users, "n_items": data.n_items,
            "n_tags": int(data.tag_matrix.shape[1]), "embedding_dim": t.embedding_dim,
            "hidden": list(t.hidden), "dropout": list(t.dropout)}


def build_model(data: PreparedData, cfg: ExperimentConfig, variant: str) -> NFFMTL:
    """Instantiate NFF-MTL; architecture flags come only from ``variant``."""
    spec = model_spec(data, cfg, variant)
    return NFFMTL(spec["n_users"], spec["n_items"], data.tag_matrix, spec["embedding_dim"], spec["hidden"],
                  spec["dropout"], use_content=spec["use_content"], use_mtl=spec["use_mtl"])


def count_parameters(model: NFFMTL) -> int:
    """Number of trainable parameters."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def save_checkpoint(path: Path, model: NFFMTL, data: PreparedData, cfg: ExperimentConfig,
                    variant: str, seed: int, epoch: int, val_metric: float,
                    history: list[dict[str, Any]]) -> None:
    """Store weights together with everything needed to reuse them safely."""
    settings = nffmtl_training_settings(cfg, variant)
    sig = signature(settings)
    torch.save({
        "state_dict": model.state_dict(),
        "model_spec": model_spec(data, cfg, variant), "stage": cfg.experiment.stage,
        "train_task": cfg.task.primary, "training_settings": settings, "training_signature": sig,
        "config_id": config_id("NFF-MTL" if variant == "full" else f"NFF-MTL[{variant}]", sig),
        "code_sha256": code_sha256(),
        "variant": variant, "seed": seed, "epoch": epoch, "val_metric": val_metric,
        "select_metric": cfg.train.select_metric, "history": history,
        "config": cfg.to_dict(), "data_fingerprint": data.fingerprint(),
        "user_ids": data.user_ids, "item_ids": data.item_ids, "tag_names": data.tag_names,
        "env": environment_info(),
    }, path)


def load_checkpoint(path: Path, data: PreparedData, cfg: ExperimentConfig, device: torch.device,
                    variant: str | None = None) -> tuple[NFFMTL, dict[str, Any]]:
    """Load a checkpoint, refusing it if data, mappings, tag order or architecture differ.

    The stored ``model_spec`` must equal the spec implied by ``cfg`` (and ``variant``
    when given); weights are never loaded into a differently configured network.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    stored = ckpt.get("model_spec")
    if stored is None or "training_signature" not in ckpt or "train_task" not in ckpt:
        raise ValueError("Checkpoint lacks model_spec/training identity (not produced by steam_nffmtl >= 0.3)")
    if ckpt["train_task"] != cfg.task.primary:
        raise ValueError(f"Checkpoint was trained for task '{ckpt['train_task']}' but the config says "
                         f"task.primary='{cfg.task.primary}'. Keep task.primary as trained; evaluate other "
                         "relevance definitions through task.sensitivity_relevance.")
    requested_settings = nffmtl_training_settings(cfg, variant or stored["variant"])
    if signature(requested_settings) != ckpt["training_signature"]:
        diff = settings_diff(ckpt["training_settings"], requested_settings)
        raise ValueError(f"Evaluation config differs from the checkpoint's training config: {diff}")
    requested = model_spec(data, cfg, variant or stored["variant"])
    if stored != requested:
        diff = {k: (stored.get(k), requested.get(k)) for k in requested if stored.get(k) != requested.get(k)}
        raise ValueError(f"Checkpoint architecture differs from the requested configuration: {diff}")
    if ckpt.get("data_fingerprint") != data.fingerprint():
        raise ValueError("Checkpoint was trained on different prepared data")
    if not (np.array_equal(ckpt["user_ids"], data.user_ids) and np.array_equal(ckpt["item_ids"], data.item_ids)
            and list(ckpt["tag_names"]) == list(data.tag_names)):
        raise ValueError("Checkpoint mappings or tag order differ from the current data")
    model = build_model(data, cfg, stored["variant"]).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, ckpt


def inspect_legacy_checkpoint(path: str | Path, data: PreparedData) -> dict[str, Any]:
    """Report whether an old ``nff_mtl_best.pth`` can be reused with the corrected pipeline.

    The legacy checkpoint used a global (all-years) mapping with a padding index 0;
    the corrected pipeline uses a training-only mapping, so embedding rows refer to
    different users and games and the weights must not be combined.
    """
    path = Path(path)
    if not path.exists():
        return {"found": False, "path": str(path), "decision": "not found: controlled retraining required"}
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    users = ckpt.get("user2idx", {})
    items = ckpt.get("item2idx", {})
    same_users = len(users) == data.n_users and all(users.get(int(u)) == j for j, u in enumerate(data.user_ids))
    same_items = len(items) == data.n_items and all(items.get(int(i)) == j for j, i in enumerate(data.item_ids))
    compatible = bool(same_users and same_items and "tag_names" in ckpt)
    return {"found": True, "path": str(path), "legacy_users": len(users), "legacy_items": len(items),
            "current_users": data.n_users, "current_items": data.n_items, "compatible": compatible,
            "decision": "reusable" if compatible else "incompatible mappings/tag order: retrain"}


def _max_rss_gb() -> float | None:
    """Peak resident memory of this process in GB (Linux/macOS), or None."""
    try:
        import resource
        import sys

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(rss / (1e9 if sys.platform == "darwin" else 1e6), 3)
    except (ImportError, AttributeError):
        return None


def train_nffmtl(data: PreparedData, cfg: ExperimentConfig, variant: str, seed: int,
                 val_split: EvalSplit, run_dir: Path, logger: logging.Logger,
                 device: torch.device) -> tuple[NFFMTL, dict[str, Any]]:
    """Train one variant with one seed; select the epoch with the best validation metric.

    The best checkpoint is written to ``run_dir/checkpoint_best.pt`` and reloaded
    from disk before returning, so the returned model is exactly the saved one.
    """
    t = cfg.train
    set_seed(seed, t.deterministic)
    model = build_model(data, cfg, variant).to(device)
    flags = VARIANTS[variant]
    tensors = training_tensors(data, cfg.task.primary, t.reg_min_hours, device)
    interactions = _csr(data.train["u"], data.train["i"], (data.n_users, data.n_items))
    gen = torch.Generator(device=device).manual_seed(seed)
    sampler = ExactNegativeSampler(interactions, device, gen)
    labels_np = tensors["y"].cpu().numpy()
    pw = pos_weight_value(labels_np, t.neg_ratio, t.pos_weight)
    bce = nn.BCEWithLogitsLoss(pos_weight=None if pw is None else torch.tensor(pw, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=t.scheduler_t0, T_mult=t.scheduler_tmult)
    n_real = len(tensors["u"])
    real_per_batch = max(1, t.batch_size_total // (1 + t.neg_ratio))
    metric_name = t.select_metric
    select_k = int(metric_name.split("@")[1])
    ckpt_path = run_dir / "checkpoint_best.pt"
    history: list[dict[str, Any]] = []
    best, best_epoch, bad = -np.inf, 0, 0
    item_pop = np.asarray(interactions.sum(axis=0)).ravel()
    logger.info("variant=%s seed=%d pos_weight=%s rows=%d real/batch=%d", variant, seed, pw, n_real, real_per_batch)
    cuda = device.type == "cuda"
    for epoch in range(1, t.max_epochs + 1):
        t0 = time.time()
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        perm = torch.randperm(n_real, generator=gen, device=device)
        tot, tot_rec, tot_reg, nb = 0.0, 0.0, 0.0, 0
        for s in range(0, n_real, real_per_batch):
            idx = perm[s:s + real_per_batch]
            u_r, i_r = tensors["u"][idx], tensors["i"][idx]
            u_n = u_r.repeat_interleave(t.neg_ratio)
            i_n = sampler.sample(u_n)
            users, items = torch.cat([u_r, u_n]), torch.cat([i_r, i_n])
            y = torch.cat([tensors["y"][idx], torch.zeros(len(u_n), device=device)])
            logits, hours = model(users, items)
            loss_rec = bce(logits, y)
            loss = loss_rec
            loss_reg = torch.zeros((), device=device)
            if flags["use_mtl"]:
                # mean absolute error over observed rows with hours > reg_min_hours (no host sync)
                m = tensors["hm"][idx].float()
                err = (hours[: len(idx)] - tensors["h"][idx]).abs()
                loss_reg = (err * m).sum() / m.sum().clamp(min=1.0)
                loss = loss + t.alpha * loss_reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
            opt.step()
            tot = tot + loss.detach(); tot_rec = tot_rec + loss_rec.detach(); tot_reg = tot_reg + loss_reg.detach()
            nb += 1
            if t.max_batches_per_epoch is not None and nb >= t.max_batches_per_epoch:
                break
        if cuda:
            torch.cuda.synchronize(device)
        sched.step()
        tot, tot_rec, tot_reg = float(tot), float(tot_rec), float(tot_reg)
        train_s = time.time() - t0
        model.eval()
        scorer = NFFMTLScorer(model, device, t.item_chunk)
        res = evaluate_ranking(scorer, val_split, [select_k], cfg.eval.user_batch_size, item_pop, None,
                               fail_on_error=True)
        val = res.summary[metric_name]
        rec = {"epoch": epoch, "train_loss": tot / nb, "train_bce": tot_rec / nb, "train_l1": tot_reg / nb,
               f"val_{metric_name}": val, "lr": opt.param_groups[0]["lr"],
               "train_seconds": round(train_s, 1), "val_seconds": res.coverage["seconds"],
               "batches": nb, "full_epoch_batches": int(np.ceil(n_real / real_per_batch)),
               "seconds_per_batch": round(train_s / max(nb, 1), 5),
               "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 3) if cuda else None,
               "host_max_rss_gb": _max_rss_gb(), "val_users": res.coverage["evaluated_users"]}
        history.append(rec)
        logger.info("epoch %d | loss %.4f (bce %.4f, l1 %.4f) | val %s %.5f | %.0fs + %.0fs", epoch, rec["train_loss"],
                    rec["train_bce"], rec["train_l1"], metric_name, val, train_s, res.coverage["seconds"])
        if val > best + 1e-7:
            best, best_epoch, bad = val, epoch, 0
            save_checkpoint(ckpt_path, model, data, cfg, variant, seed, epoch, val, history)
        else:
            bad += 1
            if bad >= t.patience:
                logger.info("early stopping after epoch %d (best epoch %d)", epoch, best_epoch)
                break
    model, ckpt = load_checkpoint(ckpt_path, data, cfg, device, variant)
    if ckpt["epoch"] != best_epoch:
        raise AssertionError("Reloaded checkpoint is not the selected epoch")
    info = {"variant": variant, "seed": seed, "best_epoch": best_epoch, f"best_val_{metric_name}": best,
            "pos_weight": pw, "history": history, "checkpoint": str(ckpt_path),
            "model_spec": ckpt["model_spec"], "parameters": count_parameters(model),
            "train_task": ckpt["train_task"], "training_signature": ckpt["training_signature"],
            "training_settings": ckpt["training_settings"], "config_id": ckpt["config_id"],
            "stage": ckpt["stage"]}
    return model, info
