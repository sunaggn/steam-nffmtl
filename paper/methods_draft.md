# Manuscript draft — sections that the corrected code now determines

Status: draft text for `GameRS_64_Makale_tamamlanmis_taslak.docx`. Bracketed fields come from run folders; no result is claimed. Sections 1–5 describe the implemented protocol and can replace Sections 3–5 of the current draft once the numbers are filled.

## 1. Research questions and contributions (to be finalised after Section 6)

**RQ1.** Under a leakage-controlled temporal protocol with full-catalog ranking, does a hybrid model that fuses identifier embeddings with tag features retrieve games that users will review positively better than tuned collaborative and content baselines?
**RQ2.** Do tag features and an auxiliary playtime-regression objective each contribute to ranking quality, as measured by controlled ablations over several training seeds?
**RQ3.** How do the conclusions change when relevance is defined as any future review instead of a future positive review?

The contributions are (i) a temporal evaluation protocol for the Steam review data in which every model shares users, candidates, history cutoffs and metric code; (ii) a comparison of the hybrid multi-task model with [baselines] across five seeds with paired tests; (iii) ablations of tag fusion and playtime supervision; and (iv) a sensitivity analysis of the relevance definition. [If the hybrid model does not outperform the strongest baseline, state this in the abstract and frame the study as an empirical analysis of these components.]

## 2. Data and task definition

We use the Game Recommendations on Steam dataset [8] ([snapshot date], SHA-256 [hash]). Reviews from 2010–2020 form the training partition, 2021 the validation partition and 2022 the test partition (global calendar-year split). Within each partition, repeated reviews of the same game by the same user are reduced to the latest review ([n] removed). An iterative 5-core filter is applied to the training partition only, leaving [U] users, [I] games and [N] reviews (sparsity [s]). Identifier mappings and the tag vocabulary (the 50 most frequent tags among training-catalog games) are derived from this filtered training set; validation and test reviews by unseen users or of unseen games are excluded from the main evaluation and counted (Table [x]).

The primary task is future positive-preference retrieval: for each user, rank the games of the training catalog so that games the user will review positively in the target year appear at the top. During training, positive reviews have label 1, while negative reviews and sampled unreviewed games have label 0. Review-label classification (separating positive from negative observed reviews) is reported separately, and retrieval of any future review is reported as a sensitivity analysis.

## 3. Model and training

Integer identifiers index 64-dimensional user and game embeddings; binary tag vectors pass through a linear layer with ReLU to a 64-dimensional content representation. The concatenated 192-dimensional vector feeds a shared network of layers with 128, 64 and 32 units (batch normalisation after the first two, dropout 0.3 and 0.2). A linear head produces the recommendation logit and a second linear head predicts log(1 + hours). For every observed training review we draw four games uniformly from the training catalog, rejecting any game the user reviewed in the training years. The loss is binary cross-entropy with logits, with positive-class weight [w] equal to the ratio of label-0 to label-1 rows, plus α = 0.5 times the mean absolute error of log playtime over observed reviews with positive playtime. We train with Adam (learning rate 10⁻³, weight decay 10⁻⁵), batches of 4,096 rows, gradient clipping at 1.0 and cosine annealing with warm restarts (T₀ = 2, T_mult = 2). The epoch is selected by NDCG@20 on a fixed random sample of 20,000 validation users ranked over the full catalog; the selected checkpoint is reloaded and evaluated once on the test year, after all settings were fixed on validation data.

## 4. Evaluation protocol

All models are evaluated on the same users, candidate games, history filter and relevance sets. Evaluated users are training-known users with at least one eligible relevant game in 2022 ([n] users). Each user's candidates are all [I] training-catalog games except those the user reviewed before 2022; no test game is added to the candidate set. A test review is eligible when its game is in the training catalog, it satisfies the relevance definition and the game is not in the user's history. We report Precision, Recall (hits divided by the number of eligible relevant games), NDCG and hit rate at K = 10 and 20, together with catalog coverage, Gini index of recommendation counts, average recommendation popularity, long-tail share, novelty and tag intra-list diversity. Review-label AUC is computed on the same [500,000] observed test reviews for every model.

## 5. Baselines and statistical analysis

Baselines are Random, Popularity, a content-based tag kNN, collaborative item kNN with shrunk cosine similarity, BPR-MF, ALS [and EASE]. [Hyperparameters were selected on the validation year with an equal budget of [n] trials per model; search spaces are listed in Appendix [x].] Training randomness is assessed over five seeds (mean ± SD and paired t-tests over seeds); user-level uncertainty for a given seed is assessed with paired per-user differences, bootstrap confidence intervals and Wilcoxon signed-rank tests. p-values within a metric family are Holm-adjusted, and we report Cohen's d_z and rank-biserial correlations as effect sizes.

## 6. Results — pending

Table A: main comparison (models × metrics × K; mean ± SD over seeds). Table B: ablation (full, without tags, without playtime head). Table C: relevance sensitivity. Table D: playtime head versus simple predictors on identical rows. Figures: architecture (192 → 128 → 64 → 32, two heads) and protocol diagram. All values must be generated from `comparison_table.csv` and `comparison_stats.json`.

## 7. Limitations (draft)

Reviews are missing not at random and store exposure favours popular games, so offline accuracy may overstate the value of popularity-aligned recommendations; offline gains do not guarantee online gains. Tags come from a metadata snapshot collected after the test year. [Whether playtime is measured at review time or at data collection must be verified; if it is cumulative at collection, the regression target contains post-review activity.] Cold-start users and games are excluded from the main evaluation.

## 8. Remaining work before the manuscript can be updated

1. Run `preflight --require-cuda`, the tests (including GPU tests) and the pilot on Colab; inspect the data card, protocol reports and run times.
2. Implement the equal-budget tuning for every model on the validation year.
3. Run five seeds for NFF-MTL, its two ablations and the seed-dependent baselines.
4. Decide the framing (RQ1 answered positively or not) from the results, then rewrite the abstract, introduction and conclusion.
5. Replace audit notes with the change report, regenerate all tables and figures from run folders, and verify the literature from primary sources.
6. Check the TJMCS author guidelines and template directly on the journal website.
