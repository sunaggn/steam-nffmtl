# Colab progress — 18 November 2026

Source cells added or changed in the live notebook since the previous repository snapshot. Outputs, credentials, raw data, and model weights are excluded. This is a historical code record; do not run every block in sequence. The single-task and baseline runs recorded below have already completed.

## Single-task seed 1 cell

```python
s = 1   # sonra 1, 2, 3, 4
tr = Path(cli('train', '--config', FINAL_CFG, '--variant', 'single_task', '--seeds', s, '--set', 'train.max_epochs=10'))
ev = Path(cli('evaluate', '--config', FINAL_CFG, '--split', 'test', '--run', tr, '--set', 'train.max_epochs=10'))
print('BİTTİ', s, tr, ev)
```

## Single-task seed loop

```python
for s in [1, 2, 3, 4]:
    tr = Path(cli(
        'train', '--config', FINAL_CFG,
        '--variant', 'single_task', '--seeds', str(s),
        '--set', 'train.max_epochs=10',
    ))
    ev = Path(cli(
        'evaluate', '--config', FINAL_CFG,
        '--split', 'test', '--run', tr,
        '--set', 'train.max_epochs=10',
    ))
    print('BİTTİ', s, tr, ev)
```

## Repeated stochastic baselines

```python
for s in [1, 2, 3, 4]:
    print('BİTTİ', s, cli('baselines', '--config', FINAL_CFG, '--split', 'test', '--seed', s, '--only', 'Random', 'BPR-MF', 'ALS', '--set', 'train.max_epochs=10'))
```

## Exploratory EASE attempt

The validation search selected `l2=100.0`. One later test attempt was interrupted without result files; a second has no confirmed result in the supplied notebook. EASE test metrics are not reported.

```python
ease_tune = cli('tune', '--config', 'configs/default.yaml', '--models', 'ease', '--set', 'baselines.ease.enabled=true', 'tuning.n_trials=3')
l2 = json.loads((Path(ease_tune) / 'selection.json').read_text())['ease']['best_candidate']['l2']
print('seçilen l2:', l2)
ease_test = cli('baselines', '--config', FINAL_CFG, '--split', 'test', '--seed', 0, '--only', 'EASE', '--set', 'baselines.ease.enabled=true', f'baselines.ease.l2={l2}', 'train.max_epochs=10')
```

The notebook also contains a diagnostic cell that checks the EASE process and output directory:

```python
import subprocess
print(subprocess.run(['bash', '-c', 'ps -eo pid,etime,cmd | grep "steam_nffmtl.cli baselines" | grep -v grep'], capture_output=True, text=True).stdout or 'çalışan EASE süreci yok')
d = DATA_DIR / 'runs_final' / '2026-11-18-112503-final-baselines-test-seed0'
print((d / 'run.log').read_text()[-800:])
print('sonuç dosyaları:', sorted(p.name for p in (d / 'results').glob('*')) if (d / 'results').exists() else 'yok')
```

A second EASE test attempt used the selected `l2=100.0`:

```python
ease_test = cli('baselines', '--config', FINAL_CFG, '--split', 'test', '--seed', 0, '--only', 'EASE', '--set', 'baselines.ease.enabled=true', 'baselines.ease.l2=100.0', 'train.max_epochs=10')
```
