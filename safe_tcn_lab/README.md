# SAFE-TCN Lab

Fresh wind power forecasting transfer-learning pipeline.

## Main ideas

- single-model transfer forecaster
- strong target-only backbone
- train-only source selection
- bounded residual transfer over source-conditioned forecasts
- sample-aware and horizon-aware transfer gate
- explicit safety metrics against negative transfer

## Commands

```powershell
Get-ChildItem safe_tcn_lab\*.py | ForEach-Object { python -m py_compile $_.FullName }
python safe_tcn_lab\smoke_test.py
python safe_tcn_lab\run_experiment.py --dataset gefcom --methods core
python safe_tcn_lab\run_experiment.py --dataset gefcom --methods paper_all
python safe_tcn_lab\run_benchmark.py --dataset gefcom --methods paper_all --seeds 42 43 44
python safe_tcn_lab\run_transfer_sweep.py --dataset sdwpf --methods paper_all --target_train_days_list 30 60 --seeds 42 43 44
```

## Comparator Sets

- `core`: `persistence`, `ridge`, `lightgbm`, `tcn`, `tcn_fine_tune`, `safe_tcn`
- `paper_all`: `persistence`, `ridge`, `lightgbm`, `lstm`, `gru`, `tcn`, `dlinear`, `nbeats`, `informer`, `fedformer`, `patchtst`, `timesnet`, `itransformer`, `tcn_fine_tune`, `tcn_multi_task`, `safe_tcn`
- `extended_all`: `paper_all` plus `lgbm_transfer`

Legacy aliases remain accepted:

- `ridge_local -> ridge`
- `lgbm_local -> lightgbm`
- `tcn_local -> tcn`
- `fine_tune -> tcn_fine_tune`
- `multi_task -> tcn_multi_task`

## Artifacts

Every run writes a timestamped folder with:

- `config.json`
- `report.json`
- `artifacts/per_target_metrics.parquet`
- `artifacts/source_selection.parquet`
- `artifacts/method_runtime.parquet`
- `artifacts/predictions/<method>/target_<id>_test.parquet`
- `artifacts/window_metrics/<method>/target_<id>_test.parquet`
- `artifacts/training_history/<method>/target_<id>.parquet`
- `artifacts/safe_sources/safe_tcn/target_<id>_test.parquet`

## Notes

- SDWPF uses the official-style `144 -> 288` setup and `153/16/15` day split.
- GEFCom uses a chronological calendar split inside the available processed parquet.
- Safety metrics are computed against `tcn`.
- Defaults are conservative on source count: `GEFCom=2` sources, `SDWPF=1` source.
- `SAFE-TCN` uses a frozen local backbone plus bounded residual transfer from pretrained sources.
- Transfer is calibrated on the validation split before test evaluation so harmful horizons can fall back to the local model.
- Limited-target-data experiments reuse the matching normalization stats during evaluation.
