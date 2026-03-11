from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from safe_tcn_lab.data import DatasetSpec
from safe_tcn_lab.nf_baselines import NFModelBundle, fit_nf_model, predict_nf_windows


@dataclass
class SafePatchTSTBundle:
    local_bundle: NFModelBundle
    source_bundles: list[NFModelBundle]
    source_ids: list[int]
    source_weights: np.ndarray
    transfer_strength_base: np.ndarray
    calibration_alpha: np.ndarray
    residual_cap: np.ndarray
    dispersion_scale: np.ndarray
    fit_duration_sec: float
    calibration_duration_sec: float
    reused_local_bundle: bool = False
    _training_history: list[dict[str, float]] = field(default_factory=list)
    _training_summary: dict[str, float] = field(default_factory=dict)


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean = np.sum(values * weights[None, :, :], axis=1)
    centered = values - mean[:, None, :]
    var = np.sum(weights[None, :, :] * centered * centered, axis=1)
    return np.sqrt(np.maximum(var, 0.0))


def _bounded_transfer(raw_transfer: np.ndarray, residual_cap: np.ndarray) -> np.ndarray:
    cap = residual_cap[None, :]
    return cap * np.tanh(raw_transfer / np.maximum(cap, 1e-6))


def _compute_source_weights(
    local_pred: np.ndarray,
    source_preds: np.ndarray,
    y_true: np.ndarray,
    similarities: np.ndarray,
) -> np.ndarray:
    local_se = (local_pred - y_true) ** 2
    source_se = (source_preds - y_true[:, None, :]) ** 2
    gains = np.maximum(local_se[:, None, :] - source_se, 0.0)
    sim = np.clip(np.asarray(similarities, dtype=np.float64), 0.0, None)[:, None]
    score = gains.mean(axis=0) * (0.5 + 0.5 * sim)
    fallback = np.maximum(sim, 1e-3)
    zero_cols = np.all(score <= 1e-8, axis=0)
    if np.any(zero_cols):
        score[:, zero_cols] = fallback
    score = np.maximum(score, 1e-8)
    return score / score.sum(axis=0, keepdims=True)


def _calibrate_parameters(
    local_pred: np.ndarray,
    source_preds: np.ndarray,
    y_true: np.ndarray,
    source_weights: np.ndarray,
    grid_size: int,
    harm_limit: float,
    residual_cap_scale: float,
    residual_cap_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_se = (local_pred - y_true) ** 2
    source_residuals = source_preds - local_pred[:, None, :]
    weighted_source_se = np.sum(source_weights[None, :, :] * (source_preds - y_true[:, None, :]) ** 2, axis=1)
    transfer_strength_base = np.clip(
        np.mean(np.maximum(local_se - weighted_source_se, 0.0), axis=0) / np.maximum(local_se.mean(axis=0), 1e-6),
        0.0,
        1.0,
    )
    raw_transfer = np.sum(source_weights[None, :, :] * source_residuals, axis=1)
    dispersion = _weighted_std(source_residuals, source_weights)
    dispersion_scale = np.quantile(dispersion, 0.6, axis=0) + 1e-6
    gate = 1.0 / (1.0 + dispersion / dispersion_scale[None, :])
    target_scale = np.std(y_true, axis=0) + 1e-6
    residual_cap = np.maximum(
        residual_cap_floor * target_scale,
        residual_cap_scale * np.quantile(np.abs(raw_transfer), 0.75, axis=0),
    )
    bounded = _bounded_transfer(raw_transfer, residual_cap)

    grid = np.linspace(0.0, 1.0, max(int(grid_size), 2), dtype=np.float64)
    calibration_alpha = np.zeros(local_pred.shape[1], dtype=np.float64)
    for horizon in range(local_pred.shape[1]):
        local_h = local_pred[:, horizon]
        truth_h = y_true[:, horizon]
        local_se_h = local_se[:, horizon]
        transfer_h = transfer_strength_base[horizon] * gate[:, horizon] * bounded[:, horizon]
        best_alpha = 0.0
        best_objective = float(np.mean(local_se_h))
        for alpha in grid:
            pred_h = local_h + alpha * transfer_h
            se_h = (pred_h - truth_h) ** 2
            harm_h = float(np.mean(se_h > local_se_h))
            regret_h = float(np.max(se_h - local_se_h))
            objective = float(np.mean(se_h))
            if harm_h > harm_limit:
                objective += float(np.mean(local_se_h)) * (harm_h - harm_limit) * 2.0
            if regret_h > 0.0:
                objective += 0.05 * regret_h
            if objective < best_objective:
                best_objective = objective
                best_alpha = float(alpha)
        calibration_alpha[horizon] = best_alpha
    return transfer_strength_base.astype(np.float32), calibration_alpha.astype(np.float32), residual_cap.astype(np.float32), dispersion_scale.astype(np.float32)


def fit_safe_patchtst(
    *,
    spec: DatasetSpec,
    feature_cols: Sequence[str],
    input_size: int,
    h: int,
    target_train_frame: pd.DataFrame,
    target_val_frame: pd.DataFrame,
    target_val_indices: Sequence[int],
    source_frames: Sequence[tuple[int, float, pd.DataFrame, pd.DataFrame]],
    seed: int,
    device: str,
    args,
    local_bundle: NFModelBundle | None = None,
) -> SafePatchTSTBundle:
    fit_start = time.perf_counter()
    reused_local_bundle = local_bundle is not None
    if local_bundle is None:
        local_bundle = fit_nf_model(
            "patchtst",
            train_frame=target_train_frame,
            val_frame=target_val_frame,
            spec=spec,
            feature_cols=feature_cols,
            input_size=input_size,
            h=h,
            seed=seed,
            device=device,
            args=args,
        )

    source_bundles: list[NFModelBundle] = []
    source_ids: list[int] = []
    similarities: list[float] = []
    for rank, (source_id, similarity, source_train_frame, source_val_frame) in enumerate(source_frames):
        bundle = fit_nf_model(
            "patchtst",
            train_frame=source_train_frame,
            val_frame=source_val_frame,
            spec=spec,
            feature_cols=feature_cols,
            input_size=input_size,
            h=h,
            seed=seed + 101 + rank,
            device=device,
            args=args,
        )
        source_bundles.append(bundle)
        source_ids.append(source_id)
        similarities.append(similarity)

    calibration_start = time.perf_counter()
    y_val_true, local_val_pred = predict_nf_windows(
        local_bundle,
        test_frame=target_val_frame,
        spec=spec,
        feature_cols=feature_cols,
        window_indices=target_val_indices,
        seq_len=input_size,
        pred_len=h,
    )

    if source_bundles:
        source_val_preds = []
        for bundle in source_bundles:
            _, pred = predict_nf_windows(
                bundle,
                test_frame=target_val_frame,
                spec=spec,
                feature_cols=feature_cols,
                window_indices=target_val_indices,
                seq_len=input_size,
                pred_len=h,
            )
            source_val_preds.append(pred)
        source_val_preds_arr = np.stack(source_val_preds, axis=1)
        source_weights = _compute_source_weights(local_val_pred, source_val_preds_arr, y_val_true, np.asarray(similarities))
        transfer_strength_base, calibration_alpha, residual_cap, dispersion_scale = _calibrate_parameters(
            local_val_pred,
            source_val_preds_arr,
            y_val_true,
            source_weights,
            grid_size=args.calibration_grid_size,
            harm_limit=args.calibration_harm_limit,
            residual_cap_scale=args.residual_cap_scale,
            residual_cap_floor=args.residual_cap_floor,
        )
    else:
        source_weights = np.zeros((0, h), dtype=np.float32)
        transfer_strength_base = np.zeros(h, dtype=np.float32)
        calibration_alpha = np.zeros(h, dtype=np.float32)
        residual_cap = np.zeros(h, dtype=np.float32)
        dispersion_scale = np.ones(h, dtype=np.float32)

    calibration_duration = time.perf_counter() - calibration_start
    total_duration = time.perf_counter() - fit_start

    history = list(getattr(local_bundle.fitted_model, "_training_history", []) or [])
    if history:
        history.append(
            {
                "epoch": float(len(history) + 1),
                "train_loss": np.nan,
                "val_loss": float(np.mean((local_val_pred - y_val_true) ** 2)),
                "best_val_loss_so_far": float(np.nanmin([row.get("best_val_loss_so_far", np.nan) for row in history] + [np.mean((local_val_pred - y_val_true) ** 2)])),
            }
        )
    summary = dict(getattr(local_bundle.fitted_model, "_training_summary", {}) or {})
    summary["duration_sec"] = float(summary.get("duration_sec", 0.0) + sum(bundle.fit_duration_sec for bundle in source_bundles) + calibration_duration)
    summary["source_fit_duration_sec"] = float(sum(bundle.fit_duration_sec for bundle in source_bundles))
    summary["calibration_duration_sec"] = float(calibration_duration)
    summary["reused_local_bundle"] = bool(reused_local_bundle)

    return SafePatchTSTBundle(
        local_bundle=local_bundle,
        source_bundles=source_bundles,
        source_ids=source_ids,
        source_weights=source_weights.astype(np.float32),
        transfer_strength_base=transfer_strength_base.astype(np.float32),
        calibration_alpha=calibration_alpha.astype(np.float32),
        residual_cap=residual_cap.astype(np.float32),
        dispersion_scale=dispersion_scale.astype(np.float32),
        fit_duration_sec=float(total_duration),
        calibration_duration_sec=float(calibration_duration),
        reused_local_bundle=reused_local_bundle,
        _training_history=history,
        _training_summary=summary,
    )


def predict_safe_patchtst(
    bundle: SafePatchTSTBundle,
    *,
    test_frame: pd.DataFrame,
    spec: DatasetSpec,
    feature_cols: Sequence[str],
    window_indices: Sequence[int],
    seq_len: int,
    pred_len: int,
) -> dict[str, np.ndarray]:
    y_true, local_pred = predict_nf_windows(
        bundle.local_bundle,
        test_frame=test_frame,
        spec=spec,
        feature_cols=feature_cols,
        window_indices=window_indices,
        seq_len=seq_len,
        pred_len=pred_len,
    )
    if not bundle.source_bundles:
        zeros = np.zeros_like(local_pred)
        return {
            "truths": y_true,
            "local": local_pred,
            "final": local_pred,
            "source_preds": np.zeros((local_pred.shape[0], 0, local_pred.shape[1]), dtype=np.float32),
            "source_weights": np.zeros((local_pred.shape[0], 0, local_pred.shape[1]), dtype=np.float32),
            "source_gates": np.zeros((local_pred.shape[0], 0, local_pred.shape[1]), dtype=np.float32),
            "transfer_strength": zeros,
            "raw_transfer": zeros,
            "bounded_transfer": zeros,
            "transfer_delta": zeros,
            "residual_budget": zeros,
            "calibration_alpha": np.broadcast_to(bundle.calibration_alpha[None, :], local_pred.shape).astype(np.float32),
        }

    source_preds = []
    for source_bundle in bundle.source_bundles:
        _, pred = predict_nf_windows(
            source_bundle,
            test_frame=test_frame,
            spec=spec,
            feature_cols=feature_cols,
            window_indices=window_indices,
            seq_len=seq_len,
            pred_len=pred_len,
        )
        source_preds.append(pred)
    source_preds_arr = np.stack(source_preds, axis=1).astype(np.float32)
    source_residuals = source_preds_arr - local_pred[:, None, :]
    raw_transfer = np.sum(bundle.source_weights[None, :, :] * source_residuals, axis=1)
    dispersion = _weighted_std(source_residuals, bundle.source_weights)
    gate = 1.0 / (1.0 + dispersion / np.maximum(bundle.dispersion_scale[None, :], 1e-6))
    bounded_transfer = _bounded_transfer(raw_transfer, bundle.residual_cap)
    transfer_strength = bundle.transfer_strength_base[None, :] * gate
    calibrated_strength = transfer_strength * bundle.calibration_alpha[None, :]
    transfer_delta = calibrated_strength * bounded_transfer
    final_pred = local_pred + transfer_delta
    source_weights = np.broadcast_to(bundle.source_weights[None, :, :], source_preds_arr.shape).astype(np.float32)
    source_gates = source_weights * transfer_strength[:, None, :]
    return {
        "truths": y_true.astype(np.float32),
        "local": local_pred.astype(np.float32),
        "final": final_pred.astype(np.float32),
        "source_preds": source_preds_arr,
        "source_weights": source_weights,
        "source_gates": source_gates.astype(np.float32),
        "transfer_strength": transfer_strength.astype(np.float32),
        "raw_transfer": raw_transfer.astype(np.float32),
        "bounded_transfer": bounded_transfer.astype(np.float32),
        "transfer_delta": transfer_delta.astype(np.float32),
        "residual_budget": np.broadcast_to(bundle.residual_cap[None, :], local_pred.shape).astype(np.float32),
        "calibration_alpha": np.broadcast_to(bundle.calibration_alpha[None, :], local_pred.shape).astype(np.float32),
    }
