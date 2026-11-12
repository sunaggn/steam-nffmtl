# Changes and verification report

Versions: v0.1 (sections 1–4), v0.2 (section 5, first independent review), v0.3 (section 6, second independent review) and v0.3.1 (section 7, third independent review). Earlier packages are kept unchanged; each version is a new package.

Scope: corrections to the 64-d NFF-MTL experiment (`NFF_MTL_Complete_extended.ipynb`, manuscript `GameRS_64_Makale_tamamlanmis_taslak.docx`). The 32-d archive experiment is not used. Original files are unchanged; all corrections live in this repository.

## 1. Defects of the original notebook and what replaced them

"Evidence" refers to cells of the original notebook, inspected directly. "Verified by" names the test in `tests/` that would fail if the defect returned.

| # | Original behaviour | Evidence | Correction | Verified by |
| --- | --- | --- | --- | --- |
| 1 | Training label = positive review, evaluation relevance = any 2022 review (negative reviews were negatives in training and relevant in evaluation) | CELL 3 dataset, CELL 5 `evaluate_topk` | Primary task = positive preference for both; interaction relevance only as sensitivity; review-label AUC kept separate | `test_test_positive_protocol`, `test_interaction_relevance_and_val_cutoff` |
| 2 | 5-core, user/game mapping (1 910 450 users, 34 369 games) built on all years | CELL 2 | k-core, mappings and tag vocabulary from training years only; cold IDs = −1 and counted | `test_dedup_and_unknown_ids`, `test_tags_from_training_catalog_only`, `test_kcore_on_training_rows` |
| 3 | Negatives drawn from the all-years catalog with at most 5 retries (not guaranteed negatives; future-only games used as negatives) | CELL 3 `SteamDatasetEfficient` | Training catalog only; exact rejection against all training interactions of the user | `test_negatives_never_hit_training_interactions`, `test_sampler_rejects_user_with_full_catalog` |
| 4 | Test games injected into 1 000 sampled candidates for NFF-MTL/NCF/LightGCN, full catalog for the others | CELL 5 `recommend_nff_mtl`, T2 | One protocol: full training catalog, history masked by the evaluator, never by the model | `test_same_protocol_for_all_models` |
| 5 | Validation-year history not removed for test | CELL 5 | History cutoff end of 2020 for validation, end of 2021 for test | `test_test_positive_protocol` |
| 6 | "Recall" = hits / min(\|G\|, K) | CELL 5 | Standard Recall = hits / \|eligible relevant\|; NDCG and HR added; K = 10 and 20 | `test_ranking_metrics_by_hand`, `test_recall_is_not_capped` |
| 7 | BPR/ALS fitted on the transposed matrix; `recommend_bpr/als` failed for most users, errors swallowed by `except: continue`; cold users received Popularity (≈1 830 of 10 000 users evaluated) | CELL 4, CELL 5 | User × item fit with shape assertions; scores by dot product; no fallback; errors abort | `test_implicit_orientation_and_scores`, `test_transposed_fit_is_rejected`, `test_als_learns_block_structure`, `test_errors_are_not_silent` |
| 8 | "ItemKNN" was tag-based content kNN; no collaborative kNN, no EASE | CELL 4 | `TagKNN` and `ItemKNN-CF` separately; EASE with cost guard | `test_tagknn_prefers_shared_tags`, `test_itemknn_cf_neighbours`, `test_ease_matches_closed_form` |
| 9 | AUC gave 0 to out-of-vocabulary pairs; same 500k rows but cold pairs included | CELL 5 `compute_auc` | Fixed seeded sample of warm observed rows for every model; non-finite scores rejected | `test_auc_by_hand` |
| 10 | Epoch selected by a validation loss that differed from the training loss (unweighted BCE, L1 including zero-hour rows); best val AUC at epoch 2, saved epoch 1 | CELL 3 output | Selection on validation NDCG@20 (full catalog); best checkpoint reloaded from disk and checked | `train_nffmtl` assertion; end-to-end test |
| 11 | No torch seed in main training; `pip install implicit` unpinned | CELL 1, CELL 3 | All seeds set and logged; pinned requirements; env.json per run | `test_negatives_never_hit_training_interactions` (seeded sampler) |
| 12 | Diversity = 0 for Popularity/BPR/ALS because popular games have no tags; diagnostic cells never ran | CELL 5, cells 11–13 | Tag ILD over tagged games only, with valid-list and untagged shares reported | `test_gini_and_beyond_accuracy` |
| 13 | LightGCN trained on a 67 652-edge subgraph of the evaluation users | T2 | Removed from the comparison until re-implemented under the protocol | — |
| 14 | Activation × loss ablation interrupted after 2 epochs of one configuration; Fig. 4 of the original manuscript has no recorded run | cell 9 | Not reused; any ablation must come from a run folder | — |
| 15 | Publication-chart cell failed (NameError); Figs. 5–8 values (0.0905, 0.0162, 0.5853) appear in no recorded output | cell 10 | Figures must be generated from `comparison_table.csv` of a run folder | — |
| 16 | Corrected AUC (T4) failed after a kernel restart; case study (T5), significance (T6) and summary (T7) never ran | T4–T7 | Statistics in `compare`; case studies only from seeded random users of a reported checkpoint | `test_paired_user_comparison_detects_shift`, `test_holm_by_hand` |
| 17 | Playtime head reported without a baseline; no lower clipping before expm1 | T3 | Global mean, game median, user + game mean on identical rows; negative log predictions clipped and counted; user-clustered paired comparison | end-to-end test (`playtime.json`) |
| 18 | `nff_mtl_best.pth` not available | — | `inspect-legacy` reports it as missing or incompatible (all-years mapping); controlled retraining path | `test_legacy_checkpoint_is_refused` |

## 2. What was executed

* v0.1: `python -m pytest -q`: **33 passed** (Python 3.12.3, CPU, packages in `requirements.txt`), including a synthetic end-to-end CLI run (prepare → baselines incl. EASE → NFF-MTL 3 epochs → compare).
* Synthetic smoke outputs are in `examples/synthetic-smoke/` (synthetic data generated by `synthetic.py`; no real user data). They show that every model is evaluated on one protocol fingerprint with zero failed users. Their numbers carry no scientific meaning.
* Micro-benchmark: top-k plus metrics take 0.07 s per 256 users × 25 000 games on one CPU core.

## 3. What was not executed

* Nothing was run on the real Steam data or on a GPU. CUDA-specific paths (CUDA generator for the sampler, `searchsorted` on GPU, GPU memory of full-catalog scoring) are untested here.
* No tuning: baselines and NFF-MTL use configured defaults. An equal-budget search is still to be implemented.
* NCF, LightGCN, cold-start strata and an exposure/IPS sensitivity analysis are not implemented.
* The target journal's (TJMCS) official pages were not reachable from this environment; no journal requirement has been verified.

## 4. Audit notes moved out of the manuscript

The revised manuscript contained provenance notes that belong here, not in the paper body: the discrepancy between the earlier Overleaf values (P@10 0.0905, capped retrieval 0.2724, novelty 9.86, AUC 0.5853) and the stored notebook values (0.0885, 0.2637, 9.91, 0.5839); the renaming of the capped measure to "capped retrieval@10"; statements that values were "transcribed from the supplied notebook outputs"; and requests for author confirmation. None of these archived values may be reported as results of the corrected pipeline.

## 5. v0.2 — response to the independent code review

### 5.1 Reproduced defects and fixes

All four reported behaviours were reproduced on the v0.1 code before fixing (cache: by code inspection; the other three by direct calls: `topk_indices` returned a masked item, `paired_user_comparison` silently used 2 of 3 users, `use_content=False` still built the content layer).

| Finding | Fix | Regression tests (`tests/test_review_regressions.py`) |
| --- | --- | --- |
| 1. Cache loaded regardless of preprocessing settings and source files | Cache folder named by a hash of the preprocessing settings and `PREPROCESS_VERSION`; `cache_manifest.json` stores settings and raw-file size, mtime and SHA-256; validated on every load (`verify_source_hash` re-hashes always); data without manifest, edited manifests and changed sources raise `CacheMismatchError`; k-core that removes all rows raises | `test_cache_is_keyed_by_preprocessing_settings`, `test_cache_rejects_changed_source_file`, `test_cache_rejects_edited_manifest_and_missing_manifest`, `test_same_mtime_change_detected_with_verify_hash` |
| 2. Masked (history) games returned when fewer than K finite candidates | Only finite scores are candidates; empty slots are −1 and never count as hits or recommendations; per-batch assertion that no history game is recommended; `short_list_users@K` and `min_candidates_per_user` reported; `eval.short_lists: error` aborts | `test_topk_never_returns_masked_items`, `test_short_lists_in_metrics_and_beyond_accuracy`, `test_evaluator_reports_or_rejects_short_lists`; the protocol test now asserts the lists |
| 3. Paired statistics silently intersected different user sets | Unique indices and identical membership required (re-ordering allowed, different membership raises); `compare` checks unique users, row count vs manifest and identical user sets across all runs | `test_paired_comparison_refuses_different_user_sets`, `test_compare_refuses_duplicates_and_mismatched_rows` |
| 4. `train.use_content/use_mtl` coexisted with `--variant` and were ignored | Flags removed (config loader rejects them); `model_spec` from `--variant` is the single authority, used by `estimate`, stored in the checkpoint and verified on reload | `test_removed_architecture_flags_are_rejected`, `test_checkpoint_architecture_is_verified`, `test_estimate_reflects_the_built_model` |

### 5.2 Additional gaps

| Gap | Change |
| --- | --- |
| Notebook did not stop on failure | No `!` cells; every step runs through `subprocess` with a non-zero exit raising `CalledProcessError`; verified locally (a locked test evaluation raised and stopped) |
| First real-data step was not a pilot | `configs/pilot.yaml` (stage `pilot`: 5 000 validation users, 2 epochs, validation only, separate `runs_pilot`) |
| Duplicate model/seed runs could enter `compare`; text-sorted folder names | `compare` refuses duplicate model/seed and repeated seed-independent baselines; folders are `<date>-<HHMMSS>-<stage>-<slug>` with zero-padded suffixes; every run writes `manifest.json`; each command prints `RUN_DIR=` and the notebook passes those exact paths |
| Incomplete result identity | Data fingerprint now includes the tag order; protocol fingerprint binds data fingerprint, target, relevance and original user/app IDs; results store data, protocol, config and code hashes, stage and task; `compare` refuses mixtures |
| EASE memory underestimated (4-byte assumption, float64 computation) | Estimate follows the implementation (float64; CPU ≈ 28n² bytes, CUDA ≈ 12n² host + 24n² GPU); in-place operations; refuses to start if free host/GPU memory is insufficient |
| No test lock; no separate evaluate command; tuning missing | Stages `pilot`/`tuning` cannot read the test year; `train` evaluates validation only; `evaluate` reloads a checkpoint for test; `runs_dir/test_registry.json` allows one test evaluation per model identity. **Equal-budget tuning is still not implemented.** |
| No installation manifest | `pyproject.toml` declares runtime dependencies and a `test` extra; `cli preflight [--require-cuda]` checks imports, versions against `requirements.txt`, implicit orientation and CUDA code paths |
| Future information not fully resolved | Not claimed as resolved: `unverified_assumptions` in every data card (tag snapshot, timing of `hours`); README and protocol say "leakage-controlled", not "leakage-free" |

### 5.3 Test results (this environment)

Python 3.12.3, CPU only, numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, torch 2.5.1, implicit 0.7.3 (installed, so implicit-dependent tests ran).

* `python -m pytest -q -rs`: **51 passed, 3 skipped**. The 3 skipped tests are `tests/test_gpu.py` (CUDA not available). No implicit-dependent test was skipped.
* Of the 33 v0.1 tests, one was changed because it asserted the old, wrong behaviour (`test_ease_matches_closed_form` checked the 4-byte EASE estimate); the protocol test gained the missing assertion that history games are never recommended.
* `cli preflight`: ok (warning: CUDA not available); `cli preflight --require-cuda`: exits 1 as intended.
* Synthetic end-to-end run through the notebook helper: prepare → baselines (8 models incl. EASE) → train → evaluate → compare completed; a test evaluation under stage `tuning` raised and stopped the helper.

### 5.4 Still not executed or not implemented

* GPU tests and CUDA paths (run `pytest -m gpu` and `preflight --require-cuda` on Colab), the reviewer's Windows/Torch 2.11 environment, and anything on real Steam data.
* Equal-budget hyperparameter search, NCF/LightGCN under the protocol, cold-start strata, exposure/IPS sensitivity.
* Wheel availability was checked only for `implicit` 0.7.3 (cp312 manylinux wheel exists); other packages were not re-checked on PyPI.

## 6. v0.3 — response to the second independent review

### 6.1 Reproduced gaps and fixes

Both gaps were reproduced on the v0.2 code before fixing: a checkpoint trained for `positive` loaded under `task.primary: interaction` and `alpha: 1.0` without error, and `_check_comparable` accepted seed 0 / α 0.5 and seed 1 / α 1.0 as seed repeats of one model.

| Gap | Fix | Regression tests (`tests/test_review_v03.py`) |
| --- | --- | --- |
| 1. Training task not kept with the checkpoint; evaluation identity built from the current config | `identity.py`: normalised training settings (task, variant, all training settings except seed and scoring chunk, selection protocol, data settings) and their signature. Checkpoints store `train_task`, settings, signature, `config_id`, stage and code hash. `load_checkpoint` refuses another task, other training settings (with the differing keys) or a missing identity; `evaluate` refuses a checkpoint from another stage. Results take training fields from the checkpoint and keep `train_task` separate from the evaluated relevance | `test_checkpoint_refuses_other_task`, `test_checkpoint_refuses_other_training_settings`, `test_checkpoint_without_identity_is_refused`, `test_stage_promotion_is_refused` |
| 2. Different hyperparameters merged as seed repeats | Every result carries `training_signature` and `config_id`; `compare` requires one signature per model (else refuses, or with `--allow-multiple-configs` reports configurations as separate rows); duplicates are checked on (configuration, seed) | `test_compare_refuses_different_hyperparameters_as_seed_repeats` (also checks that a genuine seed repeat is accepted and yields a seed-level test) |

### 6.2 Corrections of earlier statements

The v0.2 report and my accompanying message overstated four points; the documentation now says:

* **Source hashing:** SHA-256 is recomputed only when size or mtime differ, unless `verify_source_hash: true` (now set in the final configs). It is not recomputed on every load by default.
* **Test year in early stages:** pilot and tuning never compute test-year scores, but test-year rows are read for descriptive counts. `prepare` now writes test protocol reports only in the final stage.
* **Test registry:** moved next to the data cache so that it is shared by all runs directories on the same prepared data, and keyed by checkpoint hash or baseline signature + seed (no longer by user sample or relevance). It remains a guard against accidental reuse, not a tamper-proof lock.
* **Pilot size:** the pilot uses the full training data and default baseline iterations. A separate probe (`configs/probe.yaml`: 300 batches, 2 000 validation users, 5 BPR / 2 ALS iterations) now precedes it; training logs record batches, seconds per batch, peak GPU memory and host peak RSS.
* The conclusion that only GPU tests remained before the final runs was wrong: equal-budget tuning was missing and the two gaps above were open.

### 6.3 Equal-budget tuning (new)

`tune` (stage `tuning` only) samples `tuning.n_trials` distinct candidates per model without replacement from `tuning.spaces`, fits or trains each on the training years, scores it on the fixed validation sample, and writes `tuning_results.csv`, `selection.json` (selected candidate, used budget, grid size) and `final_selected.yaml` (stage `final`, strict hashing). Each candidate has its own `config_id`. Test: `test_tune_equal_budget_and_frozen_selection` (equal used budget, distinct identities, frozen config content, refusal in the final stage).

### 6.4 Test results (this environment)

Python 3.12.3, CPU only; numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, torch 2.5.1, implicit 0.7.3.

* `python -m pytest -q -rs`: **59 passed, 3 skipped** (the three CUDA tests in `tests/test_gpu.py`). This is not an independent replication of the reviewer's environment (Torch 2.11 + RTX 4060 Ti: 47 passed, 5 skipped for missing implicit, 2 failed on long Windows paths and then on missing implicit).
* Multi-process CLI flow on synthetic data: `tune` → `final_selected.yaml` → baselines and two NFF-MTL seeds trained and evaluated on test → `compare` succeeded; a second test evaluation of the same checkpoint and an evaluation with a changed task were refused.
* Smoke runs now use short folder names (`<date>-<time>-smoke/{d,c,r}`) to stay within Windows' default path limit; enabling long paths on Windows is still advisable. Whether `implicit` 0.7.3 provides a Windows wheel was not checked.

### 6.5 Still open

* GPU tests and full-data runs (probe, pilot, tuning, final) on Colab; real run times and memory.
* The two data assumptions (`unverified_assumptions` in the data card) remain limitations of the study.
* NCF/LightGCN under the protocol, cold-start strata and exposure/IPS sensitivity are not implemented.

## 7. v0.3.1 — response to the third independent review

The review confirmed the v0.3 identity and seed-repeat fixes and raised four points before running.

| Point | Change | Verified by |
| --- | --- | --- |
| 1. "Run all" did not stop after the probe; pilot and tune cells were active | Two execution gates, off by default: `PROCEED_TO_PILOT = False` after the probe and `PROCEED_TO_TUNE = False` before tuning; each raises and stops "Run all" until changed | Gate cells executed locally with default switches: both raised |
| 2. "Equal budget" was an upper bound (`min(n_trials, grid)`); TagKNN had 4 candidates against 8 | `tuning.budget_policy: equal` (default): exactly `n_trials` candidates per model, and any smaller grid stops `tune` before a run folder is created or a model is fitted; `cap` keeps the old behaviour and records it. Default grids enlarged to ≥ 8 (TagKNN 8, EASE 8). Documentation says equal trial counts are not equal compute time | `test_equal_budget_refuses_small_grid_before_any_trial`, `test_tune_equal_budget_and_frozen_selection` |
| 3. Probe limits training batches only | Documented in the configs, README, protocol and notebook: data preparation, GPU tensors, the validation pass and baseline fits use the full data; the baseline probe is a separate, optional cell | — |
| 4. EASE off by default and not tuned; NCF/LightGCN absent | Notebook switch `INCLUDE_EASE` (adds EASE to tuning and enables it); the notes state that only models actually run may be reported | — |

**Correction of my earlier message.** I wrote that `tune` gives every model the same number of candidates; in v0.3 it gave at most `n_trials`, fewer for small grids. v0.3.1 enforces equality by default.

**Missing `implicit`.** The three tests that failed in the reviewer's environment (`test_cuda_training_epoch`, `test_compare_refuses_duplicates_and_mismatched_rows`, `test_tune_equal_budget_and_frozen_selection`) needed BPR/ALS through the smoke configuration. They now skip with an explicit reason when `implicit` cannot be imported, like the other implicit-dependent tests; `preflight` still exits with code 1 in that case, so the Colab notebook stops before the tests.

### 7.1 Test results (this environment)

* With `implicit` 0.7.3: **60 passed, 3 skipped** (the three CUDA tests).
* With `implicit` made unimportable (`sys.modules["implicit"] = None`, imitating the reviewer's environment): **53 passed, 10 skipped** (7 for `implicit`, 3 for CUDA), **0 failed**; `preflight` exited with code 1.
* These runs are on CPU with Torch 2.5.1; they do not replicate the reviewer's Torch 2.11 / CUDA results.
