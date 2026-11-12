# Steam NFF-MTL

**Neural Feature Fusion with Multi-Task Learning for game recommendation and playtime prediction.**

Leakage-controlled re-implementation of the NFF-MTL experiments (user/game embeddings + tag fusion,
recommendation head + auxiliary playtime head) on the *Game Recommendations on Steam* dataset.
The code replaces the evaluation and baseline logic of `NFF_MTL_Complete_extended.ipynb`; see
`docs/CHANGES_AND_VERIFICATION.md` for what changed and why, and `docs/experiment-protocol.md`
for the protocol.

**Status (v0.3.1):** revised after three independent code reviews (see `docs/CHANGES_AND_VERIFICATION.md`, sections 5–7).
Real-data preparation, validation tuning, and a first final comparison have now completed on Colab L4.
The full model has completed training and 2022 evaluation for seeds 0, 1 and 2. The seed-0 baseline comparison and all three full-model results are available in
[the preliminary results summary](docs/preliminary-results.md). The remaining seeds
and ablations are pending. Multi-seed conclusions and the manuscript results are still in progress.
The bundled `examples/synthetic-smoke/` files are synthetic package examples and carry no research claims. The pipeline is *leakage-controlled*, not leakage-free: two assumptions about
the source data remain unverified (see `unverified_assumptions` in every data card).

## Install

```bash
python -m venv .venv && source .venv/bin/activate      # Python >= 3.11
pip install -r requirements.txt && pip install -e .
python -m steam_nffmtl.cli preflight                     # imports, implicit orientation, CUDA paths if present
python -m pytest -q -rs                                  # 60 tests + 3 GPU tests (skipped without CUDA)
```

On Windows, if NumPy/PyTorch abort natively inside MKL/OpenMP, run the tests with
`MKL_THREADING_LAYER=SEQUENTIAL OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` (observed by the reviewer; not seen on Linux).

On Google Colab use `notebooks/NFF_MTL_Corrected_Colab.ipynb` (it installs `requirements-colab.txt`).

## Data

Download `recommendations.csv` and `games_metadata.json` from the Kaggle dataset
*Game Recommendations on Steam* (A. Kozyriev) and set their paths in the configs. The data are not redistributed.
The cache folder is named by the preprocessing settings (`cache_dir/pp2-<hash>`); its manifest stores the raw files'
size, mtime and SHA-256. On every load size and mtime are compared and SHA-256 is recomputed only if they differ,
unless `data.verify_source_hash: true` (set in the final configs), which recomputes it on every load.

## Stages and commands

| Stage | Config | Test-year scores |
| --- | --- | --- |
| `pilot` (probe) | `configs/probe.yaml`: 1 epoch × 300 training batches, 2 000 validation users, few BPR/ALS iterations; data preparation, GPU tensors and baseline fits still use the full data | locked |
| `pilot` | `configs/pilot.yaml`: **full training data**, 2 epochs, 5 000 validation users, default baseline iterations | locked |
| `tuning` | `configs/default.yaml` with `tuning.spaces` | locked |
| `final` | `final_selected.yaml` written by `tune` (template: `configs/final.yaml`) | once per model identity |

Test-year rows are read in every stage for descriptive counts (data card); only `final` builds test protocols and
computes test-year scores. The test registry (`<cache>/test_registry.json`, shared by every runs directory using
the same prepared data) allows one test evaluation per checkpoint or per baseline signature and seed. It guards
against accidental reuse; it is not a tamper-proof lock.

```bash
python -m steam_nffmtl.cli prepare   --config configs/default.yaml
python -m steam_nffmtl.cli estimate  --config configs/default.yaml --variant full
python -m steam_nffmtl.cli train     --config configs/probe.yaml --variant full --seeds 0     # seconds/batch, GPU memory
python -m steam_nffmtl.cli train     --config configs/pilot.yaml --variant full --seeds 0     # full-data epoch cost
python -m steam_nffmtl.cli tune      --config configs/default.yaml --models tagknn itemknn bpr als nffmtl
F=runs_tuning/<tune-run>/final_selected.yaml
python -m steam_nffmtl.cli train     --config $F --variant full --seeds 0 1 2 3 4
python -m steam_nffmtl.cli evaluate  --config $F --split test --run runs_final/<train-run>
python -m steam_nffmtl.cli baselines --config $F --split test --seed 0
python -m steam_nffmtl.cli compare   --config $F --reference NFF-MTL --results <run> <run> ...
```

* **Training identity.** Each checkpoint stores its training task, a normalised training signature (all training,
  selection and data settings except the seed and the scoring chunk size), its stage and code hash. Loading refuses a
  different `task.primary`, different training settings or a missing identity; `evaluate` refuses a checkpoint trained
  in another stage (no promotion of pilot/tuning checkpoints). Other relevance definitions are evaluated through
  `task.sensitivity_relevance`; results keep `train_task` and the split's relevance as separate fields.
* **Seed repeats.** `compare` treats runs of one model as seed repeats only if their training signatures are equal;
  otherwise it refuses, or with `--allow-multiple-configs` shows each configuration as its own row (`config_id`).
* **Tuning.** `tune` draws exactly `tuning.n_trials` distinct candidates per model without replacement from its grid
  (`budget_policy: equal`; a smaller grid stops `tune` before any trial; `cap` would make `n_trials` an upper bound),
  selects by validation `tuning.metric`, and writes `tuning_results.csv`, `selection.json` (trials used, grid size,
  policy) and `final_selected.yaml`. Equal trial counts are not equal compute time: one NFF-MTL trial is a full training run.

The architecture is chosen only by `--variant` (`full`, `no_content`, `single_task`). Other values can be overridden
with `--set section.key=value`; the resolved configuration is saved in the run folder. Every command prints
`RUN_DIR=<path>`; pass those paths to `compare` instead of globbing. `compare` also refuses mixed data, protocols,
stages, training tasks or code versions, duplicate model/seed runs and per-user files whose users differ.

## Run folders

Every command writes `runs/<YYYY-MM-DD>-<HHMMSS>-<stage>-<slug>/` containing `manifest.json` (run ID, command,
stage, config and code hashes), `config.yaml`, `env.json` (Python, packages, GPU),
`run.log`, `protocol_<split>.json` (users, candidates, history cutoff and every excluded row), and
`results/<model>__<split>.{json,csv.gz}` (aggregate and **per-user** metrics). Training runs add
`checkpoint_best.pt` (weights + model spec + mappings + tag order + config + seed + data fingerprint) and
`training.json`; `evaluate` adds `playtime.json`. Each result JSON records data, protocol, config and code hashes.
`compare` writes `comparison_table.csv` and `comparison_stats.json`.

## Layout

```text
src/steam_nffmtl/   config, data, identity, protocol, metrics, evaluate, baselines, model, train, playtime, stats, synthetic, cli
configs/            probe.yaml, pilot.yaml, default.yaml (tuning), final.yaml (template), smoke.yaml (synthetic)
tests/              hand-computed metric tests, protocol, sampler, model, baselines, statistics, end-to-end
notebooks/          Colab driver
docs/               experiment protocol, change and verification report
paper/              manuscript draft of the methods (results pending)
```

## Project lineage

This repository continues the Steam recommender research associated with
[melisguclu/game-recommender-system](https://github.com/melisguclu/game-recommender-system).
The earlier 32-dimensional experiment and the extended 64-dimensional notebook are historical versions.
The current package revises preprocessing and evaluation as documented in `docs/CHANGES_AND_VERIFICATION.md`.
The old and current scores use different protocols and should not be compared as an isolated embedding-size ablation.
