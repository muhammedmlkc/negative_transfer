from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safe_tcn_lab.run_experiment import build_parser, run_experiment


def save_json(payload: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=float)


def exact_sign_test_pvalue(wins: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    tail = sum(math.comb(n, k) for k in range(0, min(wins, n - wins) + 1))
    return float(min(1.0, 2.0 * tail / (2**n)))


def aggregate_runs(runs: List[Dict], baseline_method: str, primary_method: str) -> Dict[str, object]:
    methods = sorted(
        {
            method
            for run in runs
            for per_target in run["per_target"].values()
            for method in per_target
            if not method.startswith("_")
        }
    )
    records = []
    for run in runs:
        seed = run["config"]["seed"]
        for target_id, per_target in run["per_target"].items():
            for method, metrics in per_target.items():
                if method.startswith("_"):
                    continue
                row = {"seed": seed, "target_id": int(target_id), "method": method}
                row.update(metrics)
                records.append(row)

    aggregate: Dict[str, Dict[str, float]] = {}
    baseline_rows = [row for row in records if row["method"] == baseline_method]
    baseline_map = {(row["seed"], row["target_id"]): row for row in baseline_rows}

    for method in methods:
        rows = [row for row in records if row["method"] == method]
        aggregate[method] = {}
        for metric in ("MAE", "RMSE", "nMAE", "nRMSE", "SCORE", "WINDOW_HARM_RATE"):
            vals = np.array([row.get(metric, np.nan) for row in rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            aggregate[method][metric] = float(np.mean(vals)) if vals.size else float("nan")
            aggregate[method][f"{metric}_STD"] = float(np.std(vals)) if vals.size else float("nan")
        if method != baseline_method:
            paired = [
                (baseline_map[(row["seed"], row["target_id"])], row)
                for row in rows
                if (row["seed"], row["target_id"]) in baseline_map
            ]
            if paired:
                rmse_deltas = np.array([row["RMSE"] - base["RMSE"] for base, row in paired], dtype=np.float64)
                mae_deltas = np.array([row["MAE"] - base["MAE"] for base, row in paired], dtype=np.float64)
                aggregate[method]["NEG_TRANSFER_RATE_RMSE_PCT"] = float(100.0 * np.mean(rmse_deltas > 0))
                aggregate[method]["NEG_TRANSFER_RATE_MAE_PCT"] = float(100.0 * np.mean(mae_deltas > 0))
                aggregate[method]["MEAN_RMSE_DELTA"] = float(np.mean(rmse_deltas))
                aggregate[method]["WORST_RMSE_REGRET"] = float(np.max(rmse_deltas))

    significance = {}
    primary_rows = [row for row in records if row["method"] == primary_method]
    primary_map = {(row["seed"], row["target_id"]): row for row in primary_rows}
    for method in methods:
        if method == primary_method:
            continue
        rows = [row for row in records if row["method"] == method]
        deltas = []
        wins = 0
        for row in rows:
            key = (row["seed"], row["target_id"])
            if key not in primary_map:
                continue
            delta = primary_map[key]["RMSE"] - row["RMSE"]
            deltas.append(delta)
            wins += int(delta < 0)
        if deltas:
            arr = np.array(deltas, dtype=np.float64)
            significance[method] = {
                "mean_rmse_delta": float(np.mean(arr)),
                "wins_primary": int(wins),
                "n_pairs": int(arr.size),
                "sign_test_pvalue": exact_sign_test_pvalue(wins, int(arr.size)),
            }
    return {"aggregate": aggregate, "significance_vs_primary": significance, "records": records}


def build_benchmark_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--baseline_method", default="tcn")
    parser.add_argument("--primary_method", default="safe_tcn")
    parser.add_argument("--benchmark_root", default="safe_tcn_lab/benchmark_outputs")
    return parser


def main() -> None:
    args = build_benchmark_parser().parse_args()
    run_root = os.path.join(args.benchmark_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_root, exist_ok=True)
    runs = []
    for seed in args.seeds:
        run_args = vars(args).copy()
        run_args["seed"] = seed
        run_args["output_root"] = os.path.join(run_root, f"seed_{seed}")
        runs.append(run_experiment(SimpleNamespace(**run_args)))
    summary = aggregate_runs(runs, baseline_method=args.baseline_method, primary_method=args.primary_method)
    save_json({"config": vars(args), "runs": runs, "summary": summary}, os.path.join(run_root, "benchmark_report.json"))
    pd.DataFrame(summary["records"]).to_parquet(os.path.join(run_root, "benchmark_records.parquet"), index=False)
    pd.DataFrame(
        [
            {"method": method, **metrics}
            for method, metrics in summary["aggregate"].items()
        ]
    ).to_parquet(os.path.join(run_root, "benchmark_aggregate.parquet"), index=False)
    pd.DataFrame(
        [
            {"comparator": method, **metrics}
            for method, metrics in summary["significance_vs_primary"].items()
        ]
    ).to_parquet(os.path.join(run_root, "benchmark_significance.parquet"), index=False)

    print("\nBenchmark Summary")
    for method, metrics in summary["aggregate"].items():
        print(
            f"{method:<14} RMSE={metrics.get('RMSE', float('nan')):.4f} "
            f"RMSE_STD={metrics.get('RMSE_STD', float('nan')):.4f} "
            f"Harm={metrics.get('WINDOW_HARM_RATE', float('nan')):.4f}"
        )


if __name__ == "__main__":
    main()
