# Experiment protocol

Status (v0.3.1): implemented in `src/steam_nffmtl/` and covered by `tests/`. Values in brackets are filled from run folders; none are reported yet. The protocol is leakage-controlled, not leakage-free (Section 9).

## 1. Task definition

The data contain three distinct questions, which the original notebook mixed:

| Task | Label of a training row | Relevant in the target year | Role here |
| --- | --- | --- | --- |
| Future positive-preference retrieval | 1 = positive review; 0 = negative review or sampled unreviewed game | positively reviewed games | **Primary** (`task.primary: positive`) |
| Future interaction retrieval | 1 = any review; 0 = sampled unreviewed game | any reviewed game | Sensitivity (`sensitivity_relevance: [interaction]`) |
| Review-label classification | `is_recommended` of observed reviews | — | Separate AUC on observed rows |

**What the original model learned.** `NFF_MTL_Complete_extended.ipynb` trained with label 1 = positive review and label 0 = negative review or sampled unobserved game, i.e. a positive-preference objective, but evaluated relevance as *any* 2022 review. A negative review was therefore a negative during training and a relevant item during evaluation.

**Why positive preference is primary.** It is the objective the model already optimises, it matches the purpose of a recommender (suggest games the user will endorse), and Steam's explicit thumbs-up/down makes it observable. The interaction definition is reported as a sensitivity analysis, evaluated with the same users-and-candidates logic. A model trained on the interaction label (`task.primary: interaction`) is a separate experiment, not a re-scoring of the same weights.

## 2. Data and preprocessing (training years only)

* Split: global calendar-year split. Train 2010–2020, validation 2021, test 2022; rows outside these years are counted and dropped.
* Duplicates: within each partition, repeated (user, game) pairs keep the latest review; counts per partition and pairs repeated across partitions are reported in `data_card.json`.
* k-core (k = 5) is applied iteratively to **training rows only**. User and game mappings are built from the filtered training rows, so validation/test users or games unseen in training are *cold* (index −1). Cold rows are counted, not silently mapped to an existing embedding.
* Tag vocabulary: top 50 tags by document frequency over **training-catalog games**; the order is stored with every checkpoint. Limitation: the metadata snapshot itself was collected after 2022 (static-feature assumption).
* Data card fields: raw rows, users, games in reviews, metadata games, rows outside years, duplicates, k-core before/after, sparsity, per-partition rows/users/positive rate, cold-user and cold-game rows, share of known users, untagged catalog games, SHA-256 of raw files, the preprocessing settings, the list of unverified assumptions and a fingerprint of the prepared arrays (including tag order).
* Cache: stored under `cache_dir/pp2-<hash of preprocessing settings>`; `cache_manifest.json` records settings and the raw files' size, mtime and SHA-256. Every load compares settings, size and mtime; SHA-256 is recomputed when size or mtime differ, and on every load when `verify_source_hash: true` (final configs). A mismatch raises `CacheMismatchError`; `prepare --force` rebuilds explicitly.

## 3. Model and training

* Architecture unchanged from the 64-d experiment: 64-d user and game embeddings, tags → Linear(50, 64) + ReLU, fusion MLP 192 → 128 → 64 → 32 with BatchNorm and dropout 0.3/0.2, logit head and linear playtime head.
* Negatives: four per observed row, drawn uniformly from the **training catalog** and resampled until they are not any training interaction of that user (exact rejection; test `test_negatives_never_hit_training_interactions`).
* Loss: `BCEWithLogits(pos_weight) + α · L1`. `pos_weight = (#label-0 rows incl. sampled negatives) / (#label-1 rows)`; L1 on log1p(hours) over observed rows with hours > `reg_min_hours` (default 0, i.e. zero-hour reviews and sampled negatives are excluded). α = 0.5 is not claimed to balance the losses.
* Architecture: chosen only by `train --variant` (`full`, `no_content`, `single_task`); the resulting `model_spec` is stored in the checkpoint and must match on reload.
* Selection: early stopping on **validation NDCG@20** (full-catalog ranking of a fixed, seeded sample of 20 000 validation users, history = training years). The best checkpoint is written to disk and **reloaded** before any test evaluation. The test set is evaluated once, after selection.
* Reproducibility: Python, NumPy and PyTorch (CPU/CUDA) seeds; deterministic cuDNN; `use_deterministic_algorithms(warn_only=True)` (embedding backward has no deterministic CUDA kernel, so bitwise identity on GPU is not guaranteed); seeded sampler generator; environment, package versions and GPU in `env.json`.

## 4. Common evaluation protocol

Fixed once per target year and relevance definition (`protocol.build_split`), identical for all models (verified by a fingerprint that `compare` checks):

* **Users:** training-known users with at least one eligible relevant game (all of them by default; a seeded sample only if `eval.test_user_sample` is set and reported).
* **Candidates:** the full training catalog. Test games are never injected.
* **History filter:** games the user reviewed before the target year are removed from candidates: training years for validation; training years and 2021 for test.
* **Eligible relevant set:** target-year games in the training catalog that satisfy the relevance definition and are not in the history filter. Relevant reviews of history games, cold games and cold users are excluded *and counted* in `protocol_<split>.json`; Recall uses the eligible set as denominator.
* **Profile input** for history-based baselines (Popularity, kNN, EASE, BPR, ALS): training-year interactions of the primary label, the same information NFF-MTL's embeddings see.
* **Errors:** a scoring error aborts the run (`fail_on_error: true`); results with failed users are marked invalid and `compare` refuses them.
* **Short lists:** a user with fewer finite candidates than K receives a shorter list; empty slots are never counted as recommendations or hits, Precision@K still divides by K, and the number of such users is reported (`short_list_users@K`, `min_candidates_per_user`). `eval.short_lists: error` aborts instead. On every batch the evaluator asserts that no history game is recommended.
* **Identity:** the protocol fingerprint binds the prepared-data fingerprint, target year, relevance definition, original user and app IDs, relevant sets and history masks; every result also stores config and code hashes.

## 4a. Stages, training identity and the test registry

* **Stages.** `pilot` (probe: 1 epoch × 300 training batches, 2 000 validation users — data preparation, GPU tensors and baseline fits still use the full data; pilot: full training data, 2 epochs) and `tuning` compute validation-year scores only. Test-year rows are still read for descriptive counts in the data card; test-year protocols and scores are produced only in `final`.
* **Training identity.** A normalised training signature covers the task, variant, every training setting except the seed and the scoring chunk size, the selection protocol (metric, validation sample, sampling seed) and the data settings. It is stored in each checkpoint with the training task, stage and code hash. Loading a checkpoint under a different task, different training settings or without an identity raises; `evaluate` refuses checkpoints trained in another stage. Results record `train_task`, `train_stage`, `training_signature`, `config_id` and the training settings taken from the checkpoint, separately from the evaluated relevance.
* **Seed repeats.** In `compare`, runs of one model are seed repeats only if their signatures are equal; different hyperparameters are refused or, with `--allow-multiple-configs`, reported as separate configurations.
* **Test registry.** `<cache>/test_registry.json` (shared by all runs directories on the same prepared data) admits one test evaluation per checkpoint hash, and per baseline signature and seed, independent of user sample and relevance settings. It prevents accidental repeated use of the test year; it can be deleted and is not a tamper-proof lock.

## 4b. Equal-budget tuning

`tune` (stage `tuning` only) draws exactly `n_trials` distinct candidates without replacement from each model's grid in `tuning.spaces` (seeded per model). With `budget_policy: equal` (default) a grid smaller than `n_trials` is an error raised before any trial; `budget_policy: cap` turns `n_trials` into an upper bound and reports the used budget. Equal trial counts do not mean equal compute time. Each candidate is trained or fitted on the training years and scored on the fixed validation sample with `tuning.metric` (NDCG@20). NFF-MTL trials select their epoch on the same validation sample, so their tuning score is the selected-epoch validation score. All trials use one seed; seed variability is assessed only in the final stage. The selection, the grids and the budget are written to `selection.json`; `final_selected.yaml` is the frozen final configuration (stage `final`, strict source hashing). The embedding size (64) and BPR/ALS factors (64) are fixed to the studied architecture rather than tuned.

## 5. Metrics

Per user, then averaged: Precision@K, Recall@K = hits / |eligible relevant|, NDCG@K (binary gains, log2 discount, ideal DCG over min(|R|, K)), HR@K, for K ∈ {10, 20}. Ties are broken by lower item index. Beyond accuracy at each K: catalog coverage, Gini of recommendation counts, average recommendation popularity, long-tail share (games outside the 20% most popular), novelty (−log2 popularity share) and tag intra-list diversity over tagged games (share of untagged recommendations reported). Review-label AUC uses one fixed seeded sample of 500 000 observed warm test reviews for every model. All metric functions are checked against hand-computed examples (`tests/test_metrics.py`).

## 6. Baselines

Random, Popularity, TagKNN (content, tag cosine), ItemKNN-CF (collaborative, shrunk cosine), BPR-MF and ALS (`implicit` 0.7.3, user × item fit verified), EASE (enabled after `estimate`; float64 Gram and inverse: for 25 000 games ≈ 17.5 GB host RAM on the CPU path, or ≈ 7.5 GB host + 15 GB GPU on the CUDA path; the scorer refuses to start if free memory is insufficient). Hyperparameters are selected with the equal-budget search in Section 4b; untuned defaults are used only in probe and pilot runs. NCF and LightGCN are excluded until they run under this protocol; the previous LightGCN was trained on a 67 652-edge subgraph of evaluation users and is not comparable.

## 7. Statistics

User-sampling and training-seed uncertainty are reported separately (`stats.py`): per-user paired differences with user bootstrap CI, Wilcoxon signed-rank (zero-split) and rank-biserial effect size; seed-level means ± SD with a paired t-test and Cohen's d_z (five seeds; Wilcoxon cannot reach p < 0.0625). Holm correction within each metric family. Deterministic baselines contribute one value for all seeds.

## 8. Phases and compute

1. `preflight --require-cuda` and tests (GPU tests on Colab). 2. Synthetic end-to-end run (CPU, seconds). 3. `prepare`, `estimate`, probe (limited batches), pilot (full data, 2 epochs). 4. `tune` on validation. 5. Final with `final_selected.yaml`: seeds 0–4 for NFF-MTL and its two ablations, `evaluate` once per checkpoint, baselines on test, `compare`.

Estimates, not measurements on the real data: top-k and metrics cost 0.07 s per 256 users on a single slow CPU core, i.e. a few minutes per model and split for ~10⁶ users; NFF-MTL training uses a vectorised GPU loader instead of the original per-sample `Dataset` (≈2 000 s/epoch), so an epoch should take minutes. Replace these with the real `train_seconds` and `val_seconds` from the pilot's `training.json` before planning the final stage (≈15 NFF-MTL runs).

## 9. Limitations to state in the paper

These are recorded as `unverified_assumptions` in every data card and must not be described as resolved.


Missing-not-at-random reviews and store exposure favour popular games; offline gains do not imply online gains; tags are a post-hoc snapshot; whether `hours` is measured at review time or at collection time is not documented in the files used here and must be checked against the dataset documentation; cold-start users and games are outside the main evaluation and need their own stratum if claims about them are made.
