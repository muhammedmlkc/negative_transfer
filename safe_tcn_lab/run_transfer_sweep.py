from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safe_tcn_lab.run_benchmark import aggregate_runs, build_benchmark_parser
from safe_tcn_lab.run_experiment import run_experiment


def save_json(payload: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=float)


def build_parser() -> argparse.ArgumentParser:
    parser = build_benchmark_parser()
    parser.add_argument("--target_train_days_list", nargs="+", type=int, default=[7, 14, 30, 60])
    parser.add_argument("--sweep_root", default="safe_tcn_lab/transfer_sweeps")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sweep_root = os.path.join(args.sweep_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(sweep_root, exist_ok=True)

    all_days = {}
    print("\nTransfer Sweep")
    for target_days in args.target_train_days_list:
        day_root = os.path.join(sweep_root, f"days_{target_days}")
        os.makedirs(day_root, exist_ok=True)
        runs: List[Dict] = []
        for seed in args.seeds:
            run_args = vars(args).copy()
            run_args["seed"] = seed
            run_args["target_train_days"] = target_days
            run_args["output_root"] = os.path.join(day_root, f"seed_{seed}")
            runs.append(run_experiment(SimpleNamespace(**run_args)))
        summary = aggregate_runs(runs, baseline_method=args.baseline_method, primary_method=args.primary_method)
        all_days[target_days] = {"runs": runs, "summary": summary}
        save_json(
            {"config": vars(args), "target_train_days": target_days, "runs": runs, "summary": summary},
            os.path.join(day_root, "benchmark_report.json"),
        )
        pd.DataFrame(summary["records"]).to_parquet(os.path.join(day_root, "benchmark_records.parquet"), index=False)
        pd.DataFrame(
            [
                {"method": method, **metrics}
                for method, metrics in summary["aggregate"].items()
            ]
        ).to_parquet(os.path.join(day_root, "benchmark_aggregate.parquet"), index=False)

        primary = summary["aggregate"].get(args.primary_method, {})
        baseline = summary["aggregate"].get(args.baseline_method, {})
        print(
            f"{target_days:>3}d  "
            f"{args.primary_method} RMSE={primary.get('RMSE', float('nan')):.4f}  "
            f"{args.baseline_method} RMSE={baseline.get('RMSE', float('nan')):.4f}  "
            f"Harm={primary.get('WINDOW_HARM_RATE', float('nan')):.4f}"
        )

    save_json(
        {"config": vars(args), "results_by_target_days": all_days},
        os.path.join(sweep_root, "transfer_sweep_report.json"),
    )
    sweep_rows = []
    for target_days, payload in all_days.items():
        for method, metrics in payload["summary"]["aggregate"].items():
            row = {"target_train_days": int(target_days), "method": method}
            row.update(metrics)
            sweep_rows.append(row)
    if sweep_rows:
        pd.DataFrame(sweep_rows).to_parquet(os.path.join(sweep_root, "transfer_sweep_aggregate.parquet"), index=False)


if __name__ == "__main__":
    main()
