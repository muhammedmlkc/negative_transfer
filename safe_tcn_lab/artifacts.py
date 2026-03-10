from __future__ import annotations

import os
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd

from safe_tcn_lab.data import DatasetSpec, WindowDataset


def save_parquet(frame: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.to_parquet(path, index=False)


def build_prediction_frame(
    spec: DatasetSpec,
    dataset: WindowDataset,
    dataset_name: str,
    seed: int,
    target_id: int,
    method: str,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    extras: Dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    extras = extras or {}
    rows = []
    for window_id in range(len(dataset)):
        future = dataset.get_future_frame(window_id)
        timestamps = pd.to_datetime(future[spec.time_col]).reset_index(drop=True)
        forecast_start = timestamps.iloc[0] if len(timestamps) else pd.NaT
        for horizon in range(len(timestamps)):
            row = {
                "dataset": dataset_name,
                "seed": seed,
                "target_id": target_id,
                "method": method,
                "split": split,
                "window_id": window_id,
                "forecast_start_time": forecast_start,
                "timestamp": timestamps.iloc[horizon],
                "horizon": horizon + 1,
                "y_true": float(y_true[window_id, horizon]),
                "y_pred": float(y_pred[window_id, horizon]),
            }
            for key, value in extras.items():
                arr = np.asarray(value)
                if arr.ndim == 1:
                    row[key] = float(arr[horizon])
                elif arr.ndim == 2:
                    row[key] = float(arr[window_id, horizon])
                else:
                    raise ValueError(f"Unsupported extras shape for '{key}': {arr.shape}")
            rows.append(row)
    return pd.DataFrame(rows)


def build_safe_source_frame(
    spec: DatasetSpec,
    dataset: WindowDataset,
    dataset_name: str,
    seed: int,
    target_id: int,
    split: str,
    source_ids: Sequence[int],
    source_preds: np.ndarray,
    source_weights: np.ndarray,
    source_gates: np.ndarray,
) -> pd.DataFrame:
    rows = []
    if source_preds.size == 0 or not source_ids:
        return pd.DataFrame(
            columns=[
                "dataset",
                "seed",
                "target_id",
                "split",
                "window_id",
                "forecast_start_time",
                "timestamp",
                "horizon",
                "source_rank",
                "source_id",
                "source_pred",
                "source_weight",
                "source_gate",
            ]
        )
    for window_id in range(len(dataset)):
        future = dataset.get_future_frame(window_id)
        timestamps = pd.to_datetime(future[spec.time_col]).reset_index(drop=True)
        forecast_start = timestamps.iloc[0] if len(timestamps) else pd.NaT
        for source_rank, source_id in enumerate(source_ids):
            for horizon in range(len(timestamps)):
                rows.append(
                    {
                        "dataset": dataset_name,
                        "seed": seed,
                        "target_id": target_id,
                        "split": split,
                        "window_id": window_id,
                        "forecast_start_time": forecast_start,
                        "timestamp": timestamps.iloc[horizon],
                        "horizon": horizon + 1,
                        "source_rank": source_rank,
                        "source_id": source_id,
                        "source_pred": float(source_preds[window_id, source_rank, horizon]),
                        "source_weight": float(source_weights[window_id, source_rank, horizon]),
                        "source_gate": float(source_gates[window_id, source_rank, horizon]),
                    }
                )
    return pd.DataFrame(rows)


def build_source_selection_frame(
    dataset_name: str,
    seed: int,
    target_id: int,
    target_train_days: int | None,
    source_pairs: Iterable[tuple[int, float]],
) -> pd.DataFrame:
    rows = []
    for rank, (source_id, similarity) in enumerate(source_pairs):
        rows.append(
            {
                "dataset": dataset_name,
                "seed": seed,
                "target_id": target_id,
                "target_train_days": target_train_days,
                "source_rank": rank,
                "source_id": source_id,
                "similarity": float(similarity),
            }
        )
    return pd.DataFrame(rows)


def build_per_target_metrics_frame(dataset_name: str, seed: int, per_target: Dict[int, Dict[str, Dict[str, float]]]) -> pd.DataFrame:
    rows = []
    for target_id, target_results in per_target.items():
        for method, metrics in target_results.items():
            if method.startswith("_") or not isinstance(metrics, dict):
                continue
            row = {
                "dataset": dataset_name,
                "seed": seed,
                "target_id": int(target_id),
                "method": method,
            }
            for key, value in metrics.items():
                if key.startswith("_"):
                    continue
                row[key] = value
            rows.append(row)
    return pd.DataFrame(rows)


def build_training_history_frame(dataset_name: str, seed: int, target_id: int, method: str, model) -> pd.DataFrame:
    history = getattr(model, "_training_history", None) or []
    summary = getattr(model, "_training_summary", None) or {}
    rows = []
    for item in history:
        row = {
            "dataset": dataset_name,
            "seed": seed,
            "target_id": target_id,
            "method": method,
        }
        row.update(item)
        row["best_epoch"] = summary.get("best_epoch")
        row["best_val_loss"] = summary.get("best_val_loss")
        row["duration_sec"] = summary.get("duration_sec")
        row["epochs_ran"] = summary.get("epochs_ran")
        rows.append(row)
    return pd.DataFrame(rows)
