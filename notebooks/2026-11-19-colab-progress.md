# Colab progress — 19 November 2026

Source cells present in the live Colab notebook but absent from the previous repository progress snapshot. This records code only: no execution outputs, credentials, raw reviews, or model weights. The blocks are historical and should not be run together. The explicit comparison below was made before the exploratory EASE result was added; the final five-seed figures are documented in `docs/preliminary-results.md`.

## Mark incomplete EASE attempts (cell 44)

This cell checked result files and renamed unfinished EASE attempt folders. The completed EASE run is recorded separately.

```python
RF = DATA_DIR / 'runs_final'
for p in sorted(RF.glob('2026-11-18-*-final-baselines-test-seed0')):
    res = sorted(x.name for x in (p / 'results').glob('*.json')) if (p / 'results').exists() else []
    print(p.name, res)
    if all(r.startswith('EASE') for r in res):
        p.rename(p.with_name(p.name + '-INTERRUPTED')); print('  -> INTERRUPTED olarak işaretlendi')
```

## Explicit 20-run comparison (cell 48)

This enumerated five baselines, five full-model runs, and five runs for each of two ablations. It excluded the later exploratory EASE run.

```python
from pathlib import Path
import pandas as pd

folders = {
    "baselines": [
        "2026-11-10-114002-final-baselines-test-seed0",
        "2026-11-18-085122-final-baselines-test-seed1",
        "2026-11-18-091445-final-baselines-test-seed2",
        "2026-11-18-093726-final-baselines-test-seed3",
        "2026-11-18-100018-final-baselines-test-seed4",
    ],
    "full": [
        "2026-11-10-112426-final-evaluate-test-full-seed0",
        "2026-11-12-110556-final-evaluate-test-full-seed1",
        "2026-11-12-144748-final-evaluate-test-full-seed2",
        "2026-11-13-121946-final-evaluate-test-full-seed3",
        "2026-11-13-135138-final-evaluate-test-full-seed4",
    ],
    "no_content": [
        "2026-11-13-190729-final-evaluate-test-no_content-seed0",
        "2026-11-15-103653-final-evaluate-test-no_content-seed1",
        "2026-11-15-131526-final-evaluate-test-no_content-seed2",
        "2026-11-16-094010-final-evaluate-test-no_content-seed3",
        "2026-11-16-124644-final-evaluate-test-no_content-seed4",
    ],
    "single_task": [
        "2026-11-17-110731-final-evaluate-test-single_task-seed0",
        "2026-11-17-191017-final-evaluate-test-single_task-seed1",
        "2026-11-17-202246-final-evaluate-test-single_task-seed2",
        "2026-11-17-215014-final-evaluate-test-single_task-seed3",
        "2026-11-17-231725-final-evaluate-test-single_task-seed4",
    ],
}

run_dirs = [
    DATA_DIR / "runs_final" / name
    for group in folders.values()
    for name in group
]

missing = [
    str(path)
    for path in run_dirs
    if not list((path / "results").glob("*__test-positive.json"))
]
if missing:
    raise FileNotFoundError("Eksik veya tamamlanmamış koşular:\n" + "\n".join(missing))

cmp = Path(cli(
    "compare", "--config", FINAL_CFG,
    "--split", "test-positive",
    "--reference", "NFF-MTL",
    "--results", *run_dirs,
))

table = pd.read_csv(cmp / "comparison_table.csv", index_col=0)
display(table.filter(regex=r"(^runs$|_mean$)").round(5))
print("KARŞILAŞTIRMA KLASÖRÜ:", cmp)
```

## Inspect seed-level comparison statistics (cell 49)

```python
RF = DATA_DIR / 'runs_final'; C = RF / '2026-11-18-140154-final-compare'
dirs = [p for p in sorted(RF.glob('*-final-evaluate-test-*')) + sorted(RF.glob('*-final-baselines-test-seed*')) if 'INTERRUPTED' not in p.name]
for p in dirs:
    for js in (p / 'results').glob('NFF-MTL*__test-positive.json'):
        m = json.loads(js.read_text()); print(m['model'], m['seed'], round(m['summary']['ndcg@20'], 5))
st = json.loads((C / 'comparison_stats.json').read_text())
for k, v in st['seed_level'].items():
    if 'ndcg@20' in k: print(k, '| fark', round(v['mean_diff'], 5), '| p', round(v['p_paired_t'], 4), '| d_z', round(v['d_z'], 2))
print('Holm (seed düzeyi, ndcg@20):', st['holm_seed_level'].get('ndcg@20'))
```
