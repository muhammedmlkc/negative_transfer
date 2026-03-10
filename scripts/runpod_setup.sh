#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p data/processed
mkdir -p safe_tcn_lab/outputs
mkdir -p safe_tcn_lab/benchmark_outputs
mkdir -p safe_tcn_lab/transfer_sweeps

python - <<'PY'
import importlib
missing = []
for module in ("torch", "numpy", "pandas", "pyarrow", "sklearn", "lightgbm"):
    try:
        importlib.import_module(module)
    except Exception:
        missing.append(module)
if missing:
    raise SystemExit(f"Missing modules after setup: {missing}")
print("RunPod setup complete.")
PY
