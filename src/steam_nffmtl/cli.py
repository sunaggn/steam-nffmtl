"""Command-line entry point (v0.2). Every command writes ``runs/<date>-<time>-<slug>/`` with
``manifest.json``, ``config.yaml``, ``env.json``, ``run.log`` and its outputs, and prints
``RUN_DIR=<path>`` as its last line.

Stages: ``pilot`` and ``tuning`` may only touch the validation year; ``final`` (and the
synthetic ``smoke`` stage) may evaluate the test year, and each model/checkpoint is
evaluated on a test split at most once (``runs_dir/test_registry.json``).
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baselines import baseline_factories, ease_cost_estimate, make_baseline, seed_dependent
from .config import ExperimentConfig, _merge, config_sha256, load_config, save_config, validate
from .data import PreparedData, cache_path, prepare_and_cache, write_card
from .identity import baseline_training_settings, config_id, signature
from .evaluate import RankingResult, evaluate_classification_auc, evaluate_ranking
from .protocol import EvalSplit, build_split, classification_rows, profile_matrix
from .repro import (code_sha256, environment_info, get_logger, make_run_dir, save_json, set_seed,
                    sha256_file, write_manifest)
from .stats import holm, paired_user_comparison, seed_level_comparison

TEST_STAGES = {"final", "smoke"}


# ----------------------------------------------------------------- helpers
def _start(cfg: ExperimentConfig, slug: str, command: str, extra: dict[str, Any] | None = None) -> tuple[Path, Any]:
    run_dir = make_run_dir(cfg.runs_dir, f"{cfg.experiment.stage}-{slug}")
    logger = get_logger(run_dir)
    save_config(cfg, run_dir / "config.yaml")
    save_json(environment_info(), run_dir / "env.json")
    write_manifest(run_dir, command, {"stage": cfg.experiment.stage, "config_sha256": config_sha256(cfg),
                                      **(extra or {})})
    logger.info("run directory: %s", run_dir)
    return run_dir, logger


def _done(run_dir: Path) -> Path:
    print(f"RUN_DIR={run_dir}", flush=True)
    return run_dir


def _guard_target(cfg: ExperimentConfig, target: str) -> None:
    """Only the final (or synthetic smoke) stage may look at the test year."""
    if target == "test" and cfg.experiment.stage not in TEST_STAGES:
        raise PermissionError(f"Test-year evaluation is locked for stage '{cfg.experiment.stage}'. "
                              "Select models on validation, then set experiment.stage=final.")


class TestRegistry:
    """Record of test evaluations kept next to the prepared-data cache (shared by every runs_dir).

    A model identity (checkpoint hash, or baseline signature + seed) may be evaluated on the
    test year once, whatever the user sample or relevance settings. This is a guard against
    accidental repeated use of the test year, not a tamper-proof lock: deleting the file resets it.
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.path = cache_path(cfg.data) / "test_registry.json"
        self.entries: dict[str, Any] = json.loads(self.path.read_text()) if self.path.exists() else {}

    def claim(self, key: str, info: dict[str, Any], allow_reevaluate: bool) -> None:
        """Register ``key`` before evaluating; refuse a second evaluation unless explicitly allowed."""
        if key in self.entries and self.entries[key].get("status") == "done" and not allow_reevaluate:
            raise PermissionError(f"Already evaluated on test: {self.entries[key]}. "
                                  "Use --allow-reevaluate only to repeat an identical, interrupted run.")
        self.entries[key] = {**info, "status": "started", "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self._write()

    def finish(self, key: str) -> None:
        """Mark an evaluation as completed."""
        self.entries[key]["status"] = "done"
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        save_json(self.entries, self.path)


def _splits(data: PreparedData, cfg: ExperimentConfig, target: str) -> dict[str, EvalSplit]:
    """Primary relevance first, then sensitivity relevance definitions."""
    sample = cfg.eval.test_user_sample if target == "test" else cfg.eval.val_user_sample
    out: dict[str, EvalSplit] = {}
    for rel in [cfg.task.primary] + [r for r in cfg.task.sensitivity_relevance if r != cfg.task.primary]:
        s = build_split(data, target, rel, sample, cfg.eval.sample_seed)
        out[s.name] = s
    return out


def _item_popularity(data: PreparedData) -> np.ndarray:
    """Training interaction counts per game: reference popularity for beyond-accuracy metrics."""
    return np.asarray(profile_matrix(data, "interaction").sum(axis=0)).ravel()


def _identity(cfg: ExperimentConfig, data: PreparedData, settings: dict[str, Any], train_stage: str,
              train_code: str, name: str) -> dict[str, Any]:
    """Result identity. Training fields come from what was trained (checkpoint or baseline params)."""
    sig = signature(settings)
    return {"stage": cfg.experiment.stage, "train_stage": train_stage, "train_task": settings["train_task"],
            "data_fingerprint": data.fingerprint(), "config_sha256": config_sha256(cfg),
            "code_sha256": code_sha256(), "train_code_sha256": train_code,
            "training_signature": sig, "training_settings": settings, "config_id": config_id(name, sig)}


def _save_result(run_dir: Path, res: RankingResult, identity: dict[str, Any], extra: dict[str, Any]) -> None:
    out = run_dir / "results"
    out.mkdir(exist_ok=True)
    stem = f"{res.model}__{res.split_name}".replace("/", "_")
    res.per_user.to_csv(out / f"{stem}.csv.gz", index=False)
    payload = {"model": res.model, "split": res.split_name, "split_fingerprint": res.split_fingerprint,
               "summary": res.summary, "beyond": res.beyond, "coverage": res.coverage,
               "errors": res.errors, "valid": res.valid, **identity, **extra}
    if payload["data_fingerprint"] != res.data_fingerprint:
        raise AssertionError("Result data fingerprint mismatch")
    save_json(payload, out / f"{stem}.json")


def _write_protocol_reports(run_dir: Path, splits: dict[str, EvalSplit]) -> None:
    for name, s in splits.items():
        save_json({**s.report, "fingerprint": s.fingerprint(), "data_fingerprint": s.data_fingerprint},
                  run_dir / f"protocol_{name}.json")


def _device() -> Any:
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _eval_all(scorer: Any, splits: dict[str, EvalSplit], data: PreparedData, cfg: ExperimentConfig) -> list[RankingResult]:
    pop = _item_popularity(data)
    return [evaluate_ranking(scorer, s, cfg.eval.ks, cfg.eval.user_batch_size, pop, data.tag_matrix, data.user_ids,
                             cfg.eval.fail_on_error, cfg.eval.head_share, cfg.eval.short_lists)
            for s in splits.values()]


# ----------------------------------------------------------------- commands
def cmd_prepare(cfg: ExperimentConfig, force: bool) -> Path:
    """Build (or validate and load) the cached dataset; write data card and protocol reports."""
    run_dir, log = _start(cfg, "prepare", "prepare")
    t0 = time.time()
    data = prepare_and_cache(cfg.data, force=force)
    write_card(data, run_dir / "data_card.json")
    # Pilot/tuning stages write validation protocols only; test-year protocol reports (counts, no scores)
    # are produced in the final stage.
    for target in (("val", "test") if cfg.experiment.stage in TEST_STAGES else ("val",)):
        _write_protocol_reports(run_dir, _splits(data, cfg, target))
    log.info("prepared in %.0fs: %d users, %d games, %d training rows; fingerprint %s",
             time.time() - t0, data.n_users, data.n_items, len(data.train["u"]), data.fingerprint()[:12])
    return _done(run_dir)


def estimate_parameters(spec: dict[str, Any]) -> int:
    """Trainable parameters of the network described by ``model_spec``."""
    d, (h1, h2, h3) = spec["embedding_dim"], spec["hidden"]
    n = (spec["n_users"] + spec["n_items"]) * d
    if spec["use_content"]:
        n += spec["n_tags"] * d + d
    n += (d * (3 if spec["use_content"] else 2) + 1) * h1 + 2 * h1 + (h1 + 1) * h2 + 2 * h2 + (h2 + 1) * h3
    n += h3 + 1
    if spec["use_mtl"]:
        n += h3 + 1
    return int(n)


def cmd_estimate(cfg: ExperimentConfig, variant: str) -> dict[str, Any]:
    """Sizes and memory estimates for the model that `train --variant` would build (no training)."""
    from .train import model_spec

    data = prepare_and_cache(cfg.data)
    test = build_split(data, "test", cfg.task.primary, cfg.eval.test_user_sample, cfg.eval.sample_seed)
    val = build_split(data, "val", cfg.task.primary, cfg.eval.val_user_sample, cfg.eval.sample_seed)
    spec = model_spec(data, cfg, variant)
    ease_device = cfg.baselines.get("ease", {}).get("device", "cpu")
    est = {"model_spec": spec, "parameters": estimate_parameters(spec),
           "training_rows": int(len(data.train["u"])),
           "rows_per_epoch_incl_negatives": int(len(data.train["u"]) * (1 + cfg.train.neg_ratio)),
           "test_users": int(len(test.users)), "val_users": int(len(val.users)),
           "test_pairs_scored_per_split": int(len(test.users)) * data.n_items,
           "val_pairs_scored_per_epoch": int(len(val.users)) * data.n_items,
           "ease": ease_cost_estimate(data.n_items, ease_device)}
    print(json.dumps(est, indent=2))
    return est


def cmd_baselines(cfg: ExperimentConfig, target: str, seed: int, only: list[str] | None,
                  allow_reevaluate: bool = False) -> Path:
    """Fit each enabled baseline on training years and evaluate it on the common protocol."""
    _guard_target(cfg, target)
    run_dir, log = _start(cfg, f"baselines-{target}-seed{seed}", "baselines", {"target": target, "seed": seed})
    set_seed(seed)
    data = prepare_and_cache(cfg.data)
    splits = _splits(data, cfg, target)
    _write_protocol_reports(run_dir, splits)
    profile = profile_matrix(data, cfg.task.primary)
    auc_rows = classification_rows(data, target, cfg.eval.auc_rows, cfg.eval.sample_seed)
    use_gpu = bool(cfg.baselines.get("use_gpu", {}).get("enabled", False))
    bcfg = {k: v for k, v in cfg.baselines.items() if k != "use_gpu"}
    registry = TestRegistry(cfg) if target == "test" else None
    for name, params, make in baseline_factories(profile, data.tag_matrix, bcfg, seed, use_gpu):
        if only and name not in only:
            continue
        settings = baseline_training_settings(name, params, cfg)
        ident = _identity(cfg, data, settings, cfg.experiment.stage, code_sha256(), name)
        key = f"baseline:{ident['training_signature']}:seed{seed if seed_dependent(name) else '-'}"
        if registry:
            registry.claim(key, {"model": name, "seed": seed, "run_dir": str(run_dir)}, allow_reevaluate)
        t0 = time.time()
        scorer = make()
        fit_s = time.time() - t0
        auc = evaluate_classification_auc(scorer, auc_rows)
        for res in _eval_all(scorer, splits, data, cfg):
            _save_result(run_dir, res, ident, {"seed": seed, "seed_dependent": seed_dependent(name),
                                              "model_params": params, "fit_seconds": round(fit_s, 1), "auc": auc})
            log.info("%-11s %-17s ndcg@20=%.5f recall@20=%.5f users=%d (%.0fs)", name, res.split_name,
                     res.summary.get("ndcg@20", float("nan")), res.summary.get("recall@20", float("nan")),
                     res.coverage["evaluated_users"], res.coverage["seconds"])
        log.info("%-11s review-label AUC=%.4f on %d rows; fit %.0fs", name, auc["auc_review_label"], auc["n_rows"], fit_s)
        if registry:
            registry.finish(key)
        del scorer
    return _done(run_dir)


def cmd_train(cfg: ExperimentConfig, variant: str, seeds: list[int]) -> list[Path]:
    """Train per seed with validation-based epoch selection; evaluates the selected checkpoint on validation only."""
    from .model import NFFMTLScorer
    from .train import train_nffmtl

    data = prepare_and_cache(cfg.data)
    device = _device()
    dirs = []
    for seed in seeds:
        run_dir, log = _start(cfg, f"train-{variant}-seed{seed}", "train", {"variant": variant, "seed": seed})
        val_splits = _splits(data, cfg, "val")
        primary = val_splits[f"val-{cfg.task.primary}"]
        _write_protocol_reports(run_dir, val_splits)
        model, info = train_nffmtl(data, cfg, variant, seed, primary, run_dir, log, device)
        info["checkpoint_sha256"] = sha256_file(info["checkpoint"])
        save_json(info, run_dir / "training.json")
        ident = _identity(cfg, data, info["training_settings"], info["stage"], code_sha256(), _model_name(variant))
        scorer = NFFMTLScorer(model, device, cfg.train.item_chunk, name=_model_name(variant))
        for res in _eval_all(scorer, val_splits, data, cfg):
            _save_result(run_dir, res, ident, {"seed": seed, "seed_dependent": True, "variant": variant,
                                              "model_params": info["model_spec"], "best_epoch": info["best_epoch"],
                                              "checkpoint_sha256": info["checkpoint_sha256"]})
        log.info("selected epoch %d; checkpoint %s", info["best_epoch"], info["checkpoint"])
        dirs.append(_done(run_dir))
    return dirs


def _model_name(variant: str) -> str:
    return "NFF-MTL" if variant == "full" else f"NFF-MTL[{variant}]"


def cmd_evaluate(cfg: ExperimentConfig, train_run: str, target: str, allow_reevaluate: bool = False) -> Path:
    """Evaluate a saved checkpoint (reloaded and verified) on the given year, once per checkpoint for test."""
    from .model import NFFMTLScorer
    from .playtime import fit_simple_predictors, per_user_abs_error, playtime_rows, regression_report
    from .train import VARIANTS, load_checkpoint

    _guard_target(cfg, target)
    ckpt_path = Path(train_run) / "checkpoint_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt_sha = sha256_file(ckpt_path)
    data = prepare_and_cache(cfg.data)
    device = _device()
    model, ckpt = load_checkpoint(ckpt_path, data, cfg, device)  # verifies task, training settings, architecture
    variant, seed = ckpt["variant"], int(ckpt["seed"])
    if ckpt["stage"] != cfg.experiment.stage:
        raise PermissionError(f"Checkpoint was trained in stage '{ckpt['stage']}' but is being evaluated in stage "
                              f"'{cfg.experiment.stage}'. Retrain with the frozen final config instead of "
                              "promoting a pilot/tuning checkpoint.")
    run_dir, log = _start(cfg, f"evaluate-{target}-{variant}-seed{seed}", "evaluate",
                          {"target": target, "variant": variant, "seed": seed, "checkpoint": str(ckpt_path),
                           "checkpoint_sha256": ckpt_sha})
    registry = TestRegistry(cfg) if target == "test" else None
    key = f"checkpoint:{ckpt_sha}"
    if registry:
        registry.claim(key, {"model": _model_name(variant), "seed": seed, "run_dir": str(run_dir)}, allow_reevaluate)
    name = _model_name(variant)
    ident = _identity(cfg, data, ckpt["training_settings"], ckpt["stage"], ckpt["code_sha256"], name)
    scorer = NFFMTLScorer(model, device, cfg.train.item_chunk, name=name)
    splits = _splits(data, cfg, target)
    _write_protocol_reports(run_dir, splits)
    auc = evaluate_classification_auc(scorer, classification_rows(data, target, cfg.eval.auc_rows, cfg.eval.sample_seed))
    for res in _eval_all(scorer, splits, data, cfg):
        _save_result(run_dir, res, ident, {"seed": seed, "seed_dependent": True, "variant": variant, "auc": auc,
                                          "model_params": ckpt["model_spec"], "best_epoch": ckpt["epoch"],
                                          "checkpoint_sha256": ckpt_sha})
        log.info("%s %s ndcg@20=%.5f recall@20=%.5f", name, res.split_name,
                 res.summary.get("ndcg@20", float("nan")), res.summary.get("recall@20", float("nan")))
    if VARIANTS[variant]["use_mtl"]:
        rows = playtime_rows(data, target, cfg.train.reg_min_hours)
        pred = scorer.predict_hours(rows["u"], rows["i"])
        report = {name: regression_report(rows["y"], pred)}
        model_err = per_user_abs_error(rows, pred)
        comparisons = {}
        for bname, fn in fit_simple_predictors(data, cfg.train.reg_min_hours).items():
            bp = fn(rows["u"], rows["i"])
            report[bname] = regression_report(rows["y"], bp)
            comparisons[f"{name} - {bname} (per-user log-MAE)"] = paired_user_comparison(
                model_err, per_user_abs_error(rows, bp), seed=cfg.eval.sample_seed)
        save_json({"metrics": report, "paired_user_comparisons": comparisons,
                   "rows": f"warm {target} reviews with hours > reg_min_hours"}, run_dir / "playtime.json")
        log.info("playtime log-MAE: %s", {k: round(v["mae_log"], 4) for k, v in report.items()})
    if registry:
        registry.finish(key)
    return _done(run_dir)


def _collect(result_dirs: list[str]) -> pd.DataFrame:
    rows = []
    for d in result_dirs:
        for js in sorted(Path(d).glob("results/*.json")):
            meta = json.loads(js.read_text(encoding="utf-8"))
            rows.append({"model": meta["model"], "split": meta["split"], "seed": int(meta.get("seed", 0)),
                         "fingerprint": meta["split_fingerprint"], "data_fingerprint": meta["data_fingerprint"],
                         "stage": meta["stage"], "code": meta["code_sha256"], "task": meta["train_task"],
                         "signature": meta["training_signature"], "config_id": meta["config_id"],
                         "valid": meta["valid"], "seed_dependent": meta.get("seed_dependent", True),
                         "run_dir": str(d), "csv": Path(str(js)[:-5] + ".csv.gz"), "meta": meta})
    if not rows:
        raise FileNotFoundError("No results/*.json found in the given directories")
    return pd.DataFrame(rows)


def _check_comparable(res: pd.DataFrame, allow_mixed_code: bool, allow_multiple_configs: bool = False) -> None:
    """Refuse mixed data, protocols, stages, training tasks or code, invalid results, runs of one model
    with different training settings, and duplicate runs. Sets ``res['key']`` (model or config_id)."""
    for col, what in (("data_fingerprint", "prepared data"), ("fingerprint", "protocol"), ("stage", "stage"),
                      ("task", "training task")):
        if res[col].nunique() != 1:
            raise ValueError(f"Results mix different {what}: {sorted(res[col].unique())}")
    if not allow_mixed_code and res["code"].nunique() != 1:
        raise ValueError("Results were produced by different code versions (use --allow-mixed-code to override)")
    if not res["valid"].all():
        raise ValueError(f"Invalid results: {res.loc[~res['valid'], 'model'].tolist()}")
    n_sig = res.groupby("model")["signature"].nunique()
    if (n_sig > 1).any() and not allow_multiple_configs:
        raise ValueError(f"Runs of {n_sig[n_sig > 1].index.tolist()} have different training settings and are not "
                         "seed repeats; compare them as separate configurations with --allow-multiple-configs")
    res["key"] = res["config_id"] if (n_sig > 1).any() else res["model"]
    dup = res[res.duplicated(["key", "seed"], keep=False)]
    if len(dup):
        raise ValueError("Duplicate model/seed results:\n" + dup[["key", "seed", "run_dir"]].to_string())
    fixed = res[~res["seed_dependent"]].groupby("key").size()
    if (fixed > 1).any():
        raise ValueError(f"Seed-independent models must appear once: {fixed[fixed > 1].to_dict()}")


def _load_per_user(row: pd.Series, metrics: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(row["csv"])
    if not frame["u"].is_unique:
        raise ValueError(f"Duplicate users in {row['csv']}")
    if len(frame) != row["meta"]["coverage"]["evaluated_users"]:
        raise ValueError(f"{row['csv']} has {len(frame)} rows, manifest says {row['meta']['coverage']['evaluated_users']}")
    missing = [m for m in metrics if m not in frame.columns]
    if missing:
        raise KeyError(f"{row['csv']} lacks metrics {missing}")
    return frame.set_index("u")


def cmd_compare(cfg: ExperimentConfig, result_dirs: list[str], reference: str, split: str | None,
                metrics: list[str], allow_mixed_code: bool = False, allow_multiple_configs: bool = False) -> Path:
    """Seed-level table and paired tests of every model against the reference model."""
    run_dir, log = _start(cfg, "compare", "compare", {"inputs": result_dirs})
    res = _collect(result_dirs)
    split = split or f"test-{cfg.task.primary}"
    res = res[res["split"] == split].reset_index(drop=True)
    if res.empty:
        raise ValueError(f"No results for split {split}")
    if split.startswith("test") and res["stage"].iloc[0] not in TEST_STAGES:
        raise PermissionError("Test results must come from the final stage")
    _check_comparable(res, allow_mixed_code, allow_multiple_configs)
    if reference not in set(res["key"]):
        raise ValueError(f"Reference {reference} not found; available: {sorted(res['key'].unique())}")
    per_user = {i: _load_per_user(r, metrics) for i, r in res.iterrows()}
    users0 = per_user[0].index.sort_values()
    for i, frame in per_user.items():
        if not frame.index.sort_values().equals(users0):
            raise ValueError(f"User set of {res.loc[i, 'model']} (seed {res.loc[i, 'seed']}) differs")
    table = []
    for model, g in res.groupby("key"):
        row: dict[str, Any] = {"model": model, "config_id": g.iloc[0]["config_id"], "runs": len(g),
                               "seeds": sorted(g["seed"].tolist())}
        beyond_keys = [c for c in g.iloc[0]["meta"]["beyond"]
                       if c.split("@")[0] in {"coverage", "gini", "arp", "long_tail_share", "novelty", "tag_ild"}]
        for m in metrics + beyond_keys:
            vals = [r["meta"]["summary"].get(m, r["meta"]["beyond"].get(m)) for _, r in g.iterrows()]
            row[f"{m}_mean"] = float(np.mean(vals))
            row[f"{m}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        aucs = [r["meta"].get("auc", {}).get("auc_review_label") for _, r in g.iterrows()]
        row["auc_review_label_mean"] = float(np.mean(aucs)) if all(a is not None for a in aucs) else float("nan")
        table.append(row)
    table_df = pd.DataFrame(table).set_index("model")
    table_df.to_csv(run_dir / "comparison_table.csv")
    ref = res[res["key"] == reference].sort_values("seed")
    first = ref.index[0]
    out: dict[str, Any] = {"split": split, "reference": reference, "fingerprint": res["fingerprint"].iloc[0],
                           "data_fingerprint": res["data_fingerprint"].iloc[0], "user_level": {}, "seed_level": {}}
    p_user: dict[str, dict[str, float]] = {m: {} for m in metrics}
    p_seed: dict[str, dict[str, float]] = {m: {} for m in metrics}
    for model, g in res.groupby("key"):
        if model == reference:
            continue
        g = g.sort_values("seed")
        for m in metrics:
            key = f"{reference} vs {model} | {m}"
            cmp_ = paired_user_comparison(per_user[first][m], per_user[g.index[0]][m], seed=cfg.eval.sample_seed)
            cmp_["note"] = (f"reference seed {res.loc[first, 'seed']} vs {model} seed {int(g.iloc[0]['seed'])}; "
                            "user-sampling uncertainty only")
            out["user_level"][key] = cmp_
            p_user[m][model] = cmp_["p_wilcoxon"]
            if len(ref) < 2:
                continue
            if bool(g.iloc[0]["seed_dependent"]):
                common = sorted(set(ref["seed"]) & set(g["seed"]))
                if len(common) < 2:
                    continue
                a = [ref.set_index("seed").loc[s, "meta"]["summary"][m] for s in common]
                b = [g.set_index("seed").loc[s, "meta"]["summary"][m] for s in common]
            else:  # deterministic baseline: one value, identical for every seed
                a = [r["meta"]["summary"][m] for _, r in ref.iterrows()]
                b = [g.iloc[0]["meta"]["summary"][m]] * len(a)
            sl = seed_level_comparison(a, b)
            out["seed_level"][key] = sl
            p_seed[m][model] = sl["p_paired_t"]
    out["holm_user_level"] = {m: holm(p) for m, p in p_user.items() if p}
    out["holm_seed_level"] = {m: holm(p) for m, p in p_seed.items() if p}
    save_json(out, run_dir / "comparison_stats.json")
    log.info("\n%s", table_df[[c for c in table_df.columns if c.endswith("_mean") or c == "runs"]].round(5).to_string())
    return _done(run_dir)


def cmd_preflight(out: str | None, require_cuda: bool = False) -> dict[str, Any]:
    """Check interpreter, packages, implicit orientation and (if present) CUDA code paths. Exit 1 on failure."""
    import importlib
    from importlib import metadata

    rep: dict[str, Any] = {"env": environment_info(), "checks": {}, "warnings": []}
    ok = True
    if sys.version_info < (3, 11):
        rep["checks"]["python>=3.11"] = False
        ok = False
    pins = {}
    req = Path(__file__).resolve().parents[2] / "requirements.txt"
    if req.exists():
        for line in req.read_text().splitlines():
            if "==" in line and not line.startswith("#"):
                name, ver = line.split("==")
                pins[name.strip()] = ver.strip()
    for mod, dist in (("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"), ("torch", "torch"),
                      ("implicit", "implicit"), ("yaml", "PyYAML")):
        try:
            importlib.import_module(mod)
            ver = metadata.version(dist)
            rep["checks"][f"import {mod}"] = True
            if dist in pins and pins[dist] != ver.split("+")[0]:
                rep["warnings"].append(f"{dist} {ver} differs from tested {pins[dist]}")
        except Exception as exc:  # noqa: BLE001 - report every failure
            rep["checks"][f"import {mod}"] = f"FAILED: {exc!r}"
            ok = False
    try:
        import scipy.sparse as sp

        from .baselines import ImplicitMFScorer

        x = sp.csr_matrix((np.random.default_rng(0).random((30, 7)) < 0.4).astype(np.float32))
        s = ImplicitMFScorer("als", x, {"factors": 3, "iterations": 2}, 0, expected_users=30, expected_items=7)
        rep["checks"]["implicit user x item orientation"] = s.user_factors.shape[0] == 30
    except Exception as exc:  # noqa: BLE001
        rep["checks"]["implicit user x item orientation"] = f"FAILED: {exc!r}"
        ok = False
    try:
        import torch

        if torch.cuda.is_available():
            from .model import NFFMTL, NFFMTLScorer
            from .train import ExactNegativeSampler

            dev = torch.device("cuda")
            m = sp.csr_matrix((np.random.default_rng(1).random((40, 25)) < 0.5).astype(np.float32))
            m.sort_indices()
            neg = ExactNegativeSampler(m, dev, torch.Generator(device=dev).manual_seed(0)).sample(
                torch.arange(40, device=dev).repeat_interleave(20))
            dense = m.toarray() > 0
            rep["checks"]["cuda exact negative sampling"] = not dense[
                np.repeat(np.arange(40), 20), neg.cpu().numpy()].any()
            tags = (np.random.default_rng(2).random((25, 4)) < 0.5).astype(np.float32)
            torch.manual_seed(0)
            model = NFFMTL(40, 25, tags, 8, (16, 8, 4)).eval()
            cpu = NFFMTLScorer(model, "cpu", 7).score(np.arange(5))
            gpu = NFFMTLScorer(model.to(dev), dev, 7).score(np.arange(5))
            rep["checks"]["cuda scoring equals cpu"] = bool(np.allclose(cpu, gpu, atol=1e-4))
        elif require_cuda:
            rep["checks"]["cuda available"] = "FAILED: --require-cuda but no GPU is visible"
            ok = False
        else:
            rep["warnings"].append("CUDA not available: GPU code paths not checked")
    except Exception as exc:  # noqa: BLE001
        rep["checks"]["cuda checks"] = f"FAILED: {exc!r}"
        ok = False
    ok = ok and all(v is True for v in rep["checks"].values())
    rep["ok"] = ok
    print(json.dumps({"ok": ok, "checks": rep["checks"], "warnings": rep["warnings"]}, indent=2))
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        save_json(rep, Path(out) / "preflight.json")
    if not ok:
        raise SystemExit(1)
    return rep


TUNABLE = {"tagknn": "TagKNN", "itemknn": "ItemKNN-CF", "bpr": "BPR-MF", "als": "ALS", "ease": "EASE",
           "nffmtl": "NFF-MTL"}


def tuning_candidates(space: dict[str, list[Any]], n_trials: int, seed: int, key: str) -> tuple[list[dict[str, Any]], int]:
    """Up to ``n_trials`` distinct grid points drawn without replacement (seeded per model)."""
    if n_trials < 1:
        raise ValueError("tuning.n_trials must be >= 1")
    names = sorted(space)
    grid = list(itertools.product(*[space[n] for n in names]))
    rng = np.random.default_rng([seed, zlib.crc32(key.encode())])
    order = rng.permutation(len(grid))[: min(n_trials, len(grid))]
    return [dict(zip(names, grid[i])) for i in order], len(grid)


def cmd_tune(cfg: ExperimentConfig, models: list[str]) -> Path:
    """Equal-budget random search on the validation year; writes selection.json and final_selected.yaml."""
    from .model import NFFMTLScorer
    from .train import train_nffmtl

    if cfg.experiment.stage not in {"tuning", "smoke"}:
        raise PermissionError("tune runs only in stage 'tuning' (validation year)")
    grids: dict[str, int] = {}
    for key in models:  # validate every space before the first (possibly long) trial
        if key not in TUNABLE:
            raise ValueError(f"Not tunable: {key}; choose from {sorted(TUNABLE)}")
        space = cfg.tuning.spaces.get(key)
        if not space:
            raise ValueError(f"No search space for '{key}' in tuning.spaces")
        grids[key] = int(np.prod([len(v) for v in space.values()]))
    small = {k: g for k, g in grids.items() if g < cfg.tuning.n_trials}
    if small and cfg.tuning.budget_policy == "equal":
        raise ValueError(f"budget_policy 'equal' needs >= {cfg.tuning.n_trials} candidates per model; smaller grids: "
                         f"{small}. Enlarge these grids, lower tuning.n_trials, or set budget_policy: cap.")
    run_dir, log = _start(cfg, "tune", "tune", {"models": models, "n_trials": cfg.tuning.n_trials,
                                               "tuning_seed": cfg.tuning.seed, "metric": cfg.tuning.metric})
    data = prepare_and_cache(cfg.data)
    split = build_split(data, "val", cfg.task.primary, cfg.eval.val_user_sample, cfg.eval.sample_seed)
    _write_protocol_reports(run_dir, {split.name: split})
    metric = cfg.tuning.metric
    ks = sorted(set(cfg.eval.ks) | {int(metric.split("@")[1])})
    profile = profile_matrix(data, cfg.task.primary)
    pop = _item_popularity(data)
    use_gpu = bool(cfg.baselines.get("use_gpu", {}).get("enabled", False))
    device = _device()
    rows: list[dict[str, Any]] = []
    selection: dict[str, Any] = {}
    for key in models:
        space = cfg.tuning.spaces[key]
        cands, grid_size = tuning_candidates(space, cfg.tuning.n_trials, cfg.tuning.seed, key)
        name = TUNABLE[key]
        for j, cand in enumerate(cands):
            t0 = time.time()
            trial_dir = run_dir / "trials" / f"{key}-{j:03d}"
            trial_dir.mkdir(parents=True)
            if key == "nffmtl":
                tcfg = copy.deepcopy(cfg)
                _merge(tcfg.train, cand, "train.")
                validate(tcfg)
                model, info = train_nffmtl(data, tcfg, "full", tcfg.train.seed, split, trial_dir, log, device)
                scorer = NFFMTLScorer(model, device, tcfg.train.item_chunk, name=name)
                settings, params = info["training_settings"], cand
            else:
                tcfg = cfg
                params = {**{a: b for a, b in cfg.baselines.get(key, {}).items() if a != "enabled"}, **cand}
                scorer = make_baseline(key, profile, data.tag_matrix, params, cfg.train.seed, use_gpu)
                settings = baseline_training_settings(name, params, cfg)
            res = evaluate_ranking(scorer, split, ks, cfg.eval.user_batch_size, pop, data.tag_matrix, data.user_ids,
                                   cfg.eval.fail_on_error, cfg.eval.head_share, cfg.eval.short_lists)
            ident = _identity(tcfg, data, settings, cfg.experiment.stage, code_sha256(), name)
            _save_result(trial_dir, res, ident, {"seed": cfg.train.seed, "seed_dependent": True, "trial": j,
                                                "candidate": cand, "model_params": params})
            rows.append({"model": name, "key": key, "trial": j, "config_id": ident["config_id"],
                         metric: res.summary[metric], "seconds": round(time.time() - t0, 1),
                         **{f"param.{a}": json.dumps(b) for a, b in cand.items()}})
            log.info("tune %-10s trial %d/%d %s=%.5f %s", name, j + 1, len(cands), metric, res.summary[metric], cand)
            del scorer
        mine = [r for r in rows if r["key"] == key]
        best = max(mine, key=lambda r: (r[metric], -r["trial"]))  # ties: earliest trial
        selection[key] = {"model": name, "best_candidate": cands[best["trial"]], "metric": metric,
                          "value": best[metric], "config_id": best["config_id"], "trials_used": len(cands),
                          "grid_size": grid_size, "budget": cfg.tuning.n_trials,
                          "budget_policy": cfg.tuning.budget_policy, "seed": cfg.train.seed}
    pd.DataFrame(rows).to_csv(run_dir / "tuning_results.csv", index=False)
    save_json(selection, run_dir / "selection.json")
    final = copy.deepcopy(cfg)
    final.experiment.stage = "final"
    final.data.verify_source_hash = True
    for key, sel in selection.items():
        if key == "nffmtl":
            _merge(final.train, sel["best_candidate"], "train.")
        else:
            final.baselines.setdefault(key, {}).update(sel["best_candidate"])
            final.baselines[key]["enabled"] = True
    final.runs_dir = str(Path(cfg.runs_dir).parent / "runs_final")
    validate(final)
    save_config(final, run_dir / "final_selected.yaml")
    log.info("selection: %s", {k: (v["best_candidate"], round(v["value"], 5)) for k, v in selection.items()})
    return _done(run_dir)


def cmd_inspect_legacy(cfg: ExperimentConfig, checkpoint: str) -> dict[str, Any]:
    """Check whether an old checkpoint matches the corrected data pipeline."""
    from .train import inspect_legacy_checkpoint

    rep = inspect_legacy_checkpoint(checkpoint, prepare_and_cache(cfg.data))
    print(json.dumps(rep, indent=2))
    return rep


def cmd_smoke(out_root: str, config_path: str) -> Path:
    """Synthetic end-to-end run (stage 'smoke'): prepare, baselines, train, evaluate, compare."""
    from .synthetic import make_synthetic

    work = make_run_dir(out_root, "smoke")  # short paths: Windows MAX_PATH
    rec, meta = make_synthetic(work / "d")
    overrides = {"experiment": {"stage": "smoke"},
                 "data": {"recommendations_csv": str(rec), "metadata_json": str(meta), "cache_dir": str(work / "c")},
                 "runs_dir": str(work / "r")}
    cfg = load_config(config_path, overrides)
    cmd_prepare(cfg, force=True)
    b = cmd_baselines(cfg, "test", cfg.train.seed, None)
    t = cmd_train(cfg, "full", [cfg.train.seed])
    e = cmd_evaluate(cfg, str(t[0]), "test")
    cmd_compare(cfg, [str(b), str(e)], "NFF-MTL", None, ["ndcg@20", "recall@20"])
    (work / "README.txt").write_text("Synthetic smoke run (no real data). Not a research result.\n", encoding="utf-8")
    return work


def _parse_overrides(pairs: list[str] | None) -> dict[str, Any]:
    """Turn ``["eval.test_user_sample=200000", "train.max_epochs=5"]`` into a nested dict."""
    import yaml

    out: dict[str, Any] = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not key or not sep:
            raise ValueError(f"Override must look like section.key=value, got {pair!r}")
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return out


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch to a command."""
    ap = argparse.ArgumentParser(prog="steam_nffmtl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "estimate", "baselines", "train", "evaluate", "tune", "compare", "inspect-legacy"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--set", nargs="*", default=[], metavar="SECTION.KEY=VALUE",
                       help="override config values; the resolved config is saved in the run folder")
        if name == "prepare":
            p.add_argument("--force", action="store_true", help="rebuild even if a matching cache exists")
        if name == "estimate":
            p.add_argument("--variant", default="full", choices=["full", "no_content", "single_task"])
        if name in ("baselines", "evaluate"):
            p.add_argument("--split", default="val", choices=["val", "test"])
            p.add_argument("--allow-reevaluate", action="store_true")
        if name == "baselines":
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--only", nargs="*")
        if name == "train":
            p.add_argument("--variant", default="full", choices=["full", "no_content", "single_task"])
            p.add_argument("--seeds", type=int, nargs="+", default=[0])
        if name == "evaluate":
            p.add_argument("--run", required=True, help="training run folder containing checkpoint_best.pt")
        if name == "compare":
            p.add_argument("--results", nargs="+", required=True)
            p.add_argument("--reference", default="NFF-MTL")
            p.add_argument("--split")
            p.add_argument("--allow-mixed-code", action="store_true")
            p.add_argument("--allow-multiple-configs", action="store_true",
                           help="show differently configured runs of one model as separate rows (config_id)")
            p.add_argument("--metrics", nargs="+", default=["ndcg@10", "ndcg@20", "recall@10", "recall@20",
                                                             "precision@10", "precision@20", "hr@10", "hr@20"])
        if name == "inspect-legacy":
            p.add_argument("--checkpoint", default="nff_mtl_best.pth")
        if name == "tune":
            p.add_argument("--models", nargs="+", default=["tagknn", "itemknn", "bpr", "als", "ease", "nffmtl"])
    p = sub.add_parser("smoke")
    p.add_argument("--out", default="runs_smoke")
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "configs" / "smoke.yaml"))
    p = sub.add_parser("preflight")
    p.add_argument("--out", default=None)
    p.add_argument("--require-cuda", action="store_true", help="fail if no GPU is visible (use on Colab)")
    args = ap.parse_args(argv)
    if args.cmd == "smoke":
        cmd_smoke(args.out, args.config)
        return
    if args.cmd == "preflight":
        cmd_preflight(args.out, args.require_cuda)
        return
    cfg = load_config(args.config, _parse_overrides(args.set))
    if args.cmd == "prepare":
        cmd_prepare(cfg, args.force)
    elif args.cmd == "estimate":
        cmd_estimate(cfg, args.variant)
    elif args.cmd == "baselines":
        cmd_baselines(cfg, args.split, args.seed, args.only, args.allow_reevaluate)
    elif args.cmd == "train":
        cmd_train(cfg, args.variant, args.seeds)
    elif args.cmd == "evaluate":
        cmd_evaluate(cfg, args.run, args.split, args.allow_reevaluate)
    elif args.cmd == "compare":
        cmd_compare(cfg, args.results, args.reference, args.split, args.metrics, args.allow_mixed_code,
                    args.allow_multiple_configs)
    elif args.cmd == "tune":
        cmd_tune(cfg, args.models)
    elif args.cmd == "inspect-legacy":
        cmd_inspect_legacy(cfg, args.checkpoint)


if __name__ == "__main__":
    main(sys.argv[1:])
