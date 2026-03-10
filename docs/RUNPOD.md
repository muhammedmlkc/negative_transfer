# RunPod Setup

Use a PyTorch-based RunPod image so `torch` is already available.

## 1. Clone the repo

```bash
git clone https://github.com/muhammedmlkc/negative_transfer.git
cd negative_transfer
```

## 2. Install Python dependencies

```bash
bash scripts/runpod_setup.sh
```

## 3. Add datasets

Place the processed parquet files here:

- `data/processed/sdwpf_long.parquet`
- `data/processed/gefcom_wind_long.parquet`

The code uses repo-relative paths by default, so no local Windows path changes are needed.

## 4. Sanity check

```bash
python safe_tcn_lab/smoke_test.py
```

## 5. Example runs

Quick SDWPF benchmark:

```bash
python safe_tcn_lab/run_benchmark.py --dataset sdwpf --seeds 42 43 --target_ids 1 25 75 --methods tcn_local fine_tune safe_tcn --pretrain_epochs 12 --finetune_epochs 10 --safe_epochs 10
```

Quick GEFCom transfer sweep:

```bash
python safe_tcn_lab/run_transfer_sweep.py --dataset gefcom --target_train_days_list 30 60 --seeds 42 43 --methods tcn_local fine_tune safe_tcn
```

## Notes

- `torch` is intentionally not pinned in `requirements.txt`; install this repo inside a PyTorch image/container.
- Large datasets and generated experiment outputs are excluded from Git.
