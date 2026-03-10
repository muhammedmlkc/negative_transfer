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
python safe_tcn_lab\run_experiment.py --dataset gefcom
python safe_tcn_lab\run_experiment.py --dataset sdwpf
python safe_tcn_lab\run_benchmark.py --dataset gefcom --seeds 42 43 44
python safe_tcn_lab\run_transfer_sweep.py --dataset gefcom --target_train_days_list 7 14 30 60 --seeds 42 43 44
```

## Current baselines

- `persistence`
- `ridge_local`
- `lgbm_local`
- `lgbm_transfer`
- `tcn_local`
- `fine_tune`
- `safe_tcn`

## Notes

- SDWPF uses the official-style `144 -> 288` setup and `153/16/15` day split.
- GEFCom uses a chronological calendar split inside the available processed parquet.
- Safety metrics are computed against `tcn_local`.
- Defaults are conservative on source count: `GEFCom=2` sources, `SDWPF=1` source.
- `SAFE-TCN` uses a frozen local backbone plus bounded residual transfer from pretrained sources.
- Transfer is calibrated on the validation split before test evaluation so harmful horizons can fall back to the local model.
- Limited-target-data experiments reuse the matching normalization stats during evaluation.
