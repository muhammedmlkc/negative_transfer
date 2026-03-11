from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def negative_transfer_relative_impact(baseline_error: float, contender_error: float) -> float:
    return float((contender_error - baseline_error) / (baseline_error + 1e-8) * 100.0)


def sdwpf_valid_mask(raw_future: pd.DataFrame) -> np.ndarray:
    cond = (
        ((raw_future["Patv"] <= 0) & (raw_future["Wspd"] > 2.5))
        | (raw_future["Pab1"] > 89)
        | (raw_future["Pab2"] > 89)
        | (raw_future["Pab3"] > 89)
        | (raw_future["Wdir"] < -180)
        | (raw_future["Wdir"] > 180)
        | (raw_future["Ndir"] < -720)
        | (raw_future["Ndir"] > 720)
    )
    return (~cond.fillna(False)).to_numpy(dtype=bool)


def evaluate_sdwpf(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    future_frames: Iterable[pd.DataFrame],
    rated_capacity_kw: float = 1500.0,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    maes = []
    rmses = []
    for idx, raw_future in enumerate(future_frames):
        valid = sdwpf_valid_mask(raw_future)
        if not np.any(valid):
            continue
        true_i = y_true[idx][valid] / 1000.0
        pred_i = y_pred[idx][valid] / 1000.0
        maes.append(mae(true_i, pred_i))
        rmses.append(rmse(true_i, pred_i))
    mae_arr = np.asarray(maes, dtype=np.float64)
    rmse_arr = np.asarray(rmses, dtype=np.float64)
    if mae_arr.size == 0:
        mae_arr = np.array([math.nan], dtype=np.float64)
        rmse_arr = np.array([math.nan], dtype=np.float64)
    cap_mw = rated_capacity_kw / 1000.0
    metrics = {
        "MAE": float(np.nanmean(mae_arr)),
        "RMSE": float(np.nanmean(rmse_arr)),
        "nMAE": float(np.nanmean(mae_arr) / cap_mw),
        "nRMSE": float(np.nanmean(rmse_arr) / cap_mw),
        "SCORE": float((np.nanmean(mae_arr) + np.nanmean(rmse_arr)) / (2.0 * cap_mw)),
    }
    return metrics, mae_arr, rmse_arr


def evaluate_generic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rated_capacity: float | None = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    per_window_mae = np.mean(np.abs(y_true - y_pred), axis=1)
    per_window_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))
    metrics = {
        "MAE": float(np.mean(per_window_mae)),
        "RMSE": float(np.mean(per_window_rmse)),
    }
    if rated_capacity:
        metrics["nMAE"] = float(metrics["MAE"] / rated_capacity)
        metrics["nRMSE"] = float(metrics["RMSE"] / rated_capacity)
    return metrics, per_window_mae, per_window_rmse


def evaluate_predictions(
    dataset_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    future_frames: Iterable[pd.DataFrame],
    rated_capacity: float | None = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    if dataset_name == "sdwpf":
        return evaluate_sdwpf(y_true, y_pred, future_frames, rated_capacity_kw=rated_capacity or 1500.0)
    return evaluate_generic(y_true, y_pred, rated_capacity=rated_capacity)


def add_relative_safety(metrics: Dict[str, float], baseline: Dict[str, float], prefix: str = "") -> None:
    base_mae = baseline["MAE"]
    base_rmse = baseline["RMSE"]
    metrics[f"{prefix}NTRI_MAE"] = negative_transfer_relative_impact(base_mae, metrics["MAE"])
    metrics[f"{prefix}NTRI_RMSE"] = negative_transfer_relative_impact(base_rmse, metrics["RMSE"])
    metrics[f"{prefix}NEG_TRANSFER_RMSE"] = float(metrics["RMSE"] > base_rmse)
    metrics[f"{prefix}NEG_TRANSFER_MAE"] = float(metrics["MAE"] > base_mae)
    base_window_rmse = baseline.get("_WINDOW_RMSE")
    contender_rmse = metrics.get("_WINDOW_RMSE")
    if base_window_rmse is not None and contender_rmse is not None:
        base_arr = np.asarray(base_window_rmse, dtype=np.float64)
        contender_arr = np.asarray(contender_rmse, dtype=np.float64)
        paired = min(len(base_arr), len(contender_arr))
        if paired > 0:
            harm = contender_arr[:paired] > base_arr[:paired]
            metrics[f"{prefix}WINDOW_HARM_RATE"] = float(np.mean(harm))
            metrics[f"{prefix}WORST_RMSE_REGRET"] = float(np.max(contender_arr[:paired] - base_arr[:paired]))


def add_transfer_safety(per_target: Dict[str, Dict[str, float]], baseline_method: str = "tcn") -> None:
    if baseline_method not in per_target:
        return
    baseline = per_target[baseline_method]
    for method, metrics in per_target.items():
        if method.startswith("_") or method == baseline_method:
            continue
        add_relative_safety(metrics, baseline, prefix="")


def add_method_relative_safety(
    per_target: Dict[str, Dict[str, float]],
    *,
    method: str,
    baseline_method: str,
    prefix: str = "LOCAL_",
) -> None:
    if method not in per_target or baseline_method not in per_target:
        return
    add_relative_safety(per_target[method], per_target[baseline_method], prefix=prefix)


def summarize_results(all_results: Dict[int, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    methods = sorted(
        {
            method
            for per_target in all_results.values()
            for method in per_target
            if not method.startswith("_")
        }
    )
    summary: Dict[str, Dict[str, float]] = {}
    for method in methods:
        rows = [per_target[method] for per_target in all_results.values() if method in per_target]
        if not rows:
            continue
        summary[method] = {}
        for metric in (
            "MAE",
            "RMSE",
            "nMAE",
            "nRMSE",
            "SCORE",
            "NTRI_MAE",
            "NTRI_RMSE",
            "WINDOW_HARM_RATE",
            "WORST_RMSE_REGRET",
            "LOCAL_NTRI_MAE",
            "LOCAL_NTRI_RMSE",
            "LOCAL_WINDOW_HARM_RATE",
            "LOCAL_WORST_RMSE_REGRET",
        ):
            values = [row.get(metric, math.nan) for row in rows]
            finite = [value for value in values if np.isfinite(value)]
            summary[method][metric] = float(np.mean(finite)) if finite else float("nan")
        if method != "tcn":
            neg_rmse = [row.get("NEG_TRANSFER_RMSE", math.nan) for row in rows]
            neg_mae = [row.get("NEG_TRANSFER_MAE", math.nan) for row in rows]
            neg_rmse = [value for value in neg_rmse if np.isfinite(value)]
            neg_mae = [value for value in neg_mae if np.isfinite(value)]
            if neg_rmse:
                summary[method]["NEG_TRANSFER_RATE_RMSE_PCT"] = float(100.0 * np.mean(neg_rmse))
            if neg_mae:
                summary[method]["NEG_TRANSFER_RATE_MAE_PCT"] = float(100.0 * np.mean(neg_mae))
            local_neg_rmse = [row.get("LOCAL_NEG_TRANSFER_RMSE", math.nan) for row in rows]
            local_neg_mae = [row.get("LOCAL_NEG_TRANSFER_MAE", math.nan) for row in rows]
            local_neg_rmse = [value for value in local_neg_rmse if np.isfinite(value)]
            local_neg_mae = [value for value in local_neg_mae if np.isfinite(value)]
            if local_neg_rmse:
                summary[method]["LOCAL_NEG_TRANSFER_RATE_RMSE_PCT"] = float(100.0 * np.mean(local_neg_rmse))
            if local_neg_mae:
                summary[method]["LOCAL_NEG_TRANSFER_RATE_MAE_PCT"] = float(100.0 * np.mean(local_neg_mae))
    return summary
