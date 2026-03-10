# negative_transfer

Wind power forecasting transfer-learning workspace centered on the `safe_tcn_lab` package.

## Repo layout

- `safe_tcn_lab/`: forecasting code, experiments, training, metrics
- `data/processed/`: expected location of processed parquet datasets
- `docs/`: notes plus RunPod setup instructions

## Datasets

This repo expects the following files to be placed locally and does not commit them to Git:

- `data/processed/sdwpf_long.parquet`
- `data/processed/gefcom_wind_long.parquet`

## Quick start

Use a PyTorch-ready environment, then install the non-PyTorch dependencies:

```bash
bash scripts/runpod_setup.sh
```

Smoke test:

```bash
python safe_tcn_lab/smoke_test.py
```

RunPod instructions:

- `docs/RUNPOD.md`
