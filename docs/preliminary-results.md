# Preliminary real-data results

Status recorded on 12 November 2026. These values come from the completed Colab logs and the comparison table supplied by the author. The raw run artifacts have not yet been bundled in this repository. This summary is a partial experiment report, not a completed multi-seed benchmark.

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

## Full model, seeds 1 and 2

The second full-model run selected epoch ten on 2021 validation (NDCG@20=0.03634). Its separate 2022 evaluation reported positive-review NDCG@20=0.03284 and Recall@20=0.07287; the any-review sensitivity task gave NDCG@20=0.03042 and Recall@20=0.06615. Playtime log-MAE was 0.8935 (GlobalMean 1.2425, GameMedian 0.9101, UserGameMean 0.9116). The reported run directory is `runs_final/2026-11-12-110556-final-evaluate-test-full-seed1` in the author's Drive. These figures are transcribed from the author's Colab output; the run artifacts have not been imported here.

The seed-2 2022 evaluation reported positive-review NDCG@20=0.03513 and Recall@20=0.07629; the any-review sensitivity task gave NDCG@20=0.03221 and Recall@20=0.06855. Playtime log-MAE was 0.8913. The reported run directory is `runs_final/2026-11-12-144748-final-evaluate-test-full-seed2` in the author's Drive. The seed-2 training log and raw artifacts have not been imported here.

Across these three full-model seeds, the positive-review NDCG@20 mean is 0.03353 (sample standard deviation 0.00139), Recall@20 mean is 0.07369 (sample standard deviation 0.00230), and playtime log-MAE mean is 0.89183 (sample standard deviation 0.00147). These are descriptive summaries of author-supplied logs, not paired comparisons with repeated baselines.

Playtime log-MAE on the common eligible test rows was 0.8907 for NFF-MTL, 1.2425 for GlobalMean, 0.9101 for GameMedian and 0.9116 for UserGameMean. Lower log-MAE is better.

## Completed and pending

- Colab preflight and all 63 package tests passed, with an OpenBLAS threading warning.
- Probe, full-data two-epoch pilot and validation search completed.
- Full model seed 0 trained and evaluated on 2022. Baseline seed 0 comparison completed.
- Full model seeds 1 and 2 trained and evaluated on 2022. Seed 1's best validation NDCG@20 was 0.03634 at epoch ten. Their AUC and diversity results have not yet been imported.
- Full model seeds 3–4, no-content and single-task ablations, and the completed matched multi-seed comparison remain pending.
- EASE is disabled in these experiments. NCF and LightGCN have not been implemented under this protocol.

## Reproduction cautions

`configs/default.yaml` is the package search template (eight trials and up to fifteen epochs), not the exact three-trial run above. The real run used `--set tuning.n_trials=3 train.max_epochs=5`; final full-model training and evaluation used `--set train.max_epochs=10` with the selected configuration. `final_selected.yaml` from that run and its raw manifests are still to be imported. Do not present the synthetic example outputs as Steam results.

The tag metadata is a snapshot after the test year. The timing of the playtime measurements is unverified. These limitations remain part of the study.
