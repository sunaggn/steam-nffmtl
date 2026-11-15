# Preliminary real-data results

Status recorded on 15 November 2026. These values come from the completed Colab logs and the comparison table supplied by the author. The raw run artifacts have not yet been bundled in this repository. The full model now has five evaluated seeds, but repeated baselines and most ablation runs are still pending.

## Protocol

Training years: 2010–2020. Validation: 2021. Test: 2022. Training-only iterative 5-core and training catalog mappings. Prepared training data contains 11,478,966 interactions, 1,000,980 users and 23,336 games. Features use 50 tags and 64-dimensional embeddings. Primary relevance is a positive Steam review. Ranking uses the full training catalog after history filtering. The seed-0 positive test contains 304,199 eligible users.

Three validation candidates were tried per tuned model. NFF-MTL candidates were limited to five epochs during the initial search. The selected configuration was checked for up to ten epochs on validation, before final testing. Selected values are alpha=0.5, learning rate=0.001, weight decay=0.0. Final full-model runs use a maximum of ten epochs, validation-based selection and patience three.

## First final comparison, seed 0

| Model | NDCG@20 | Recall@20 | Observed-review label AUC |
| --- | ---: | ---: | ---: |
| ALS | 0.05268 | 0.10561 | 0.55628 |
| ItemKNN-CF | 0.04875 | 0.09769 | 0.57841 |
| NFF-MTL | 0.03261 | 0.07192 | 0.64584 |
| BPR-MF | 0.03016 | 0.06097 | 0.53041 |
| Popularity | 0.02797 | 0.06140 | 0.56025 |
| TagKNN | 0.00663 | 0.01232 | 0.50358 |
| Random | 0.00037 | 0.00084 | 0.49877 |

In this run ALS and ItemKNN-CF lead the ranking metrics. NFF-MTL leads observed-review label AUC, which measures a different task. The single-run results do not establish variability across seeds.

## Full model, five seeds

The second full-model run selected epoch ten on 2021 validation (NDCG@20=0.03634). Its separate 2022 evaluation reported positive-review NDCG@20=0.03284 and Recall@20=0.07287; the any-review sensitivity task gave NDCG@20=0.03042 and Recall@20=0.06615. Playtime log-MAE was 0.8935 (GlobalMean 1.2425, GameMedian 0.9101, UserGameMean 0.9116). The reported run directory is `runs_final/2026-11-12-110556-final-evaluate-test-full-seed1` in the author's Drive. These figures are transcribed from the author's Colab output; the run artifacts have not been imported here.

The seed-2 2022 evaluation reported positive-review NDCG@20=0.03513 and Recall@20=0.07629; the any-review sensitivity task gave NDCG@20=0.03221 and Recall@20=0.06855. Playtime log-MAE was 0.8913. The reported run directory is `runs_final/2026-11-12-144748-final-evaluate-test-full-seed2` in the author's Drive. The seed-2 training log and raw artifacts have not been imported here.

The supplied notebook also records seed 3, selected at epoch eight, with 2022 positive-review NDCG@20=0.03252, Recall@20=0.07300 and playtime log-MAE=0.8836. Seed 4 selected epoch ten, with NDCG@20=0.03283, Recall@20=0.07310 and log-MAE=0.8980. Their evaluation run directories are `runs_final/2026-11-13-121946-final-evaluate-test-full-seed3` and `runs_final/2026-11-13-135138-final-evaluate-test-full-seed4` in the author's Drive.

| Full-model seed | Positive-review NDCG@20 | Positive-review Recall@20 | Playtime log-MAE |
| ---: | ---: | ---: | ---: |
| 0 | 0.03261 | 0.07192 | 0.8907 |
| 1 | 0.03284 | 0.07287 | 0.8935 |
| 2 | 0.03513 | 0.07629 | 0.8913 |
| 3 | 0.03252 | 0.07300 | 0.8836 |
| 4 | 0.03283 | 0.07310 | 0.8980 |
| Mean | 0.03319 | 0.07344 | 0.89142 |
| Sample SD | 0.00110 | 0.00166 | 0.00523 |

These are descriptive summaries of author-supplied logs, not paired comparisons with repeated baselines. A later attempted seed-2 training in the notebook shows only a partial run; it is excluded from this table, which uses the completed 12 November seed-2 checkpoint.

## No-content ablation, seeds 0 and 1

The no-content variant removes the tag/content branch while retaining the same training task. Seeds 0 and 1 were evaluated once each on 2022. The matched results were:

| Seed | Variant | Positive-review NDCG@20 | Positive-review Recall@20 | Playtime log-MAE |
| ---: | --- | ---: | ---: | ---: |
| 0 | Full | 0.03261 | 0.07192 | 0.8907 |
| 0 | No content | 0.02918 | 0.06685 | 0.8833 |
| 1 | Full | 0.03284 | 0.07287 | 0.8935 |
| 1 | No content | 0.03140 | 0.07101 | 0.8902 |

In both paired seeds the full model ranks positive-review games better, while the no-content model has slightly lower playtime error. The ranking differences are 0.00343 and 0.00144 NDCG@20 for seeds 0 and 1, respectively. Two paired seeds are insufficient for a stable ablation estimate. The author reported seed-0 run directories `runs_final/2026-11-13-175046-final-train-no_content-seed0` and `runs_final/2026-11-13-190729-final-evaluate-test-no_content-seed0`, and seed-1 run directories `runs_final/2026-11-15-092049-final-train-no_content-seed1` and `runs_final/2026-11-15-103653-final-evaluate-test-no_content-seed1`; the raw artifacts have not been imported.

Playtime log-MAE on the common eligible test rows was 0.8907 for NFF-MTL, 1.2425 for GlobalMean, 0.9101 for GameMedian and 0.9116 for UserGameMean. Lower log-MAE is better.

## Completed and pending

- Colab preflight and all 63 package tests passed, with an OpenBLAS threading warning.
- Probe, full-data two-epoch pilot and validation search completed.
- Full model seeds 0–4 trained and evaluated on 2022. Baseline seed 0 comparison completed. Full-model seed 1–4 AUC and diversity results have not yet been imported.
- No-content seeds 0 and 1 trained and evaluated on 2022. No-content seeds 2–4, all single-task seeds, repeated stochastic baselines, and the completed matched multi-seed comparison remain pending.
- EASE is disabled in these experiments. NCF and LightGCN have not been implemented under this protocol.

## Reproduction cautions

`configs/default.yaml` is the package search template (eight trials and up to fifteen epochs), not the exact three-trial run above. The real run used `--set tuning.n_trials=3 train.max_epochs=5`; final full-model training and evaluation used `--set train.max_epochs=10` with the selected configuration. `final_selected.yaml` from that run and its raw manifests are still to be imported. Do not present the synthetic example outputs as Steam results.

The tag metadata is a snapshot after the test year. The timing of the playtime measurements is unverified. These limitations remain part of the study.
