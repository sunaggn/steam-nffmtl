# Preliminary real-data results

Status recorded on 18 November 2026. These values come from the completed Colab logs and the comparison table supplied by the author. The raw run artifacts have not yet been bundled in this repository. The full model, no-content variant and single-task variant now each have five evaluated seeds. Repeated stochastic baselines are complete; the consolidated comparison is pending.

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

## No-content ablation, seeds 0–4

The no-content variant removes the tag/content branch while retaining the same training task. Seeds 0–4 were evaluated once each on 2022. The matched results were:

| Seed | Variant | Positive-review NDCG@20 | Positive-review Recall@20 | Playtime log-MAE |
| ---: | --- | ---: | ---: | ---: |
| 0 | Full | 0.03261 | 0.07192 | 0.8907 |
| 0 | No content | 0.02918 | 0.06685 | 0.8833 |
| 1 | Full | 0.03284 | 0.07287 | 0.8935 |
| 1 | No content | 0.03140 | 0.07101 | 0.8902 |
| 2 | Full | 0.03513 | 0.07629 | 0.8913 |
| 2 | No content | 0.03036 | 0.06827 | 0.8901 |
| 3 | Full | 0.03252 | 0.07300 | 0.8836 |
| 3 | No content | 0.02887 | 0.06489 | 0.8679 |
| 4 | Full | 0.03283 | 0.07310 | 0.8980 |
| 4 | No content | 0.03280 | 0.07392 | 0.8911 |
| Mean | Full | 0.03319 | 0.07344 | 0.89142 |
| Mean | No content | 0.03052 | 0.06899 | 0.88452 |

The full model's positive-review NDCG@20 is higher in all five matched seeds, but seed 4 is nearly tied (0.03283 versus 0.03280). The paired mean NDCG@20 difference is 0.00266 in favor of the full model. Its Recall@20 is higher in four of five seeds; no-content is slightly higher in seed 4 (0.07392 versus 0.07310). No-content has lower playtime log-MAE in all five seeds, with a paired mean difference of 0.00690. These are descriptive results from five seeds, not a significance claim. Seed 3 stopped training after epoch seven and selected the best validation checkpoint from epoch four. The author reported seed-0 run directories `runs_final/2026-11-13-175046-final-train-no_content-seed0` and `runs_final/2026-11-13-190729-final-evaluate-test-no_content-seed0`, seed-1 run directories `runs_final/2026-11-15-092049-final-train-no_content-seed1` and `runs_final/2026-11-15-103653-final-evaluate-test-no_content-seed1`, seed-2 run directories `runs_final/2026-11-15-115939-final-train-no_content-seed2` and `runs_final/2026-11-15-131526-final-evaluate-test-no_content-seed2`, seed-3 run directories `runs_final/2026-11-16-084447-final-train-no_content-seed3` and `runs_final/2026-11-16-094010-final-evaluate-test-no_content-seed3`, and seed-4 run directories `runs_final/2026-11-16-112817-final-train-no_content-seed4` and `runs_final/2026-11-16-124644-final-evaluate-test-no_content-seed4`; the raw artifacts have not been imported.

Playtime log-MAE on the common eligible test rows was 0.8907 for NFF-MTL, 1.2425 for GlobalMean, 0.9101 for GameMedian and 0.9116 for UserGameMean. Lower log-MAE is better.

## Single-task ablation, seed 0

The live Colab notebook now records one completed single-task run. It selected epoch 10 on 2021 validation (positive-review NDCG@20=0.04288). Its 2022 positive-review NDCG@20 was 0.03548 and Recall@20 was 0.07411; the any-review sensitivity scores were NDCG@20=0.03285 and Recall@20=0.06759. The same-seed full model scored 0.03261 and 0.07192 on the primary positive-review ranking task. This one-seed observation is not a multi-seed ablation conclusion. The single-task run has no playtime prediction result because that auxiliary task is disabled. The author’s Drive records are `runs_final/2026-11-17-095111-final-train-single_task-seed0` and `runs_final/2026-11-17-110731-final-evaluate-test-single_task-seed0`; the raw artifacts have not been imported.

## Completed and pending

- Colab preflight and all 63 package tests passed, with an OpenBLAS threading warning.
- Probe, full-data two-epoch pilot and validation search completed.
- Full model seeds 0–4 trained and evaluated on 2022. Baseline seed 0 comparison completed. Full-model seed 1–4 AUC and diversity results have not yet been imported.
- No-content and single-task seeds 0–4 trained and evaluated on 2022. Single-task seed 1 reported positive-review NDCG@20=0.03688 and Recall@20=0.07767; seeds 3 and 4 reported 0.03746/0.07947 and 0.03666/0.07839. Seed 2 completion was supplied, but its metrics and raw artifacts have not been imported here. The five-seed single-task summary remains pending.
- Repeated Random, BPR-MF and ALS baseline runs for seeds 1–4 completed. Seed 4 reported positive-review NDCG@20 of 0.00038, 0.02994 and 0.05262, respectively. A consolidated comparison using the raw run artifacts remains pending.
- An exploratory EASE search on 2021 validation completed three candidates and selected l2=100.0. One subsequent 2022 EASE attempt was interrupted without result files; a second attempt has no confirmed result in the supplied notebook. EASE is excluded from reported test comparisons. NCF and LightGCN have not been implemented under this protocol.

## Reproduction cautions

`configs/default.yaml` is the package search template (eight trials and up to fifteen epochs), not the exact three-trial run above. The real run used `--set tuning.n_trials=3 train.max_epochs=5`; final full-model training and evaluation used `--set train.max_epochs=10` with the selected configuration. `final_selected.yaml` from that run and its raw manifests are still to be imported. Do not present the synthetic example outputs as Steam results.

The tag metadata is a snapshot after the test year. The timing of the playtime measurements is unverified. These limitations remain part of the study.
