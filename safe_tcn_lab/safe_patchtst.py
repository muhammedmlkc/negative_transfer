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
    backbone_method: str
    local_bundle: NFModelBundle
    source_bundles: list[NFModelBundle]
    source_ids: list[int]
    source_weights: np.ndarray
    transfer_strength_base: np.ndarray
    calibration_alpha: np.ndarray
    residual_cap: np.ndarray
    dispersion_scale: np.ndarray
    consensus_scale: np.ndarray
    transfer_scale: np.ndarray
    horizon_blocks: np.ndarray
    regime_thresholds: np.ndarray
    regime_alpha_table: np.ndarray
    fit_duration_sec: float
    calibration_duration_sec: float
    reused_local_bundle: bool = False
    _training_history: list[dict[str, float]] = field(default_factory=list)
    _training_summary: dict[str, float] = field(default_factory=dict)


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if weights.ndim == 2:
        mean = np.sum(values * weights[None, :, :], axis=1)
        centered = values - mean[:, None, :]
        var = np.sum(weights[None, :, :] * centered * centered, axis=1)
        return np.sqrt(np.maximum(var, 0.0))
    if weights.ndim == 3:
        mean = np.sum(values * weights, axis=1)
        centered = values - mean[:, None, :]
        var = np.sum(weights * centered * centered, axis=1)
        return np.sqrt(np.maximum(var, 0.0))
    raise ValueError(f"Unsupported weights shape: {weights.shape}")


def _bounded_transfer(raw_transfer: np.ndarray, residual_cap: np.ndarray) -> np.ndarray:
    cap = residual_cap[None, :]
    return cap * np.tanh(raw_transfer / np.maximum(cap, 1e-6))


def _normalize_static_source_scores(score: np.ndarray) -> np.ndarray:
    return score / np.maximum(score.sum(axis=0, keepdims=True), 1e-6)


def _normalize_dynamic_source_scores(score: np.ndarray) -> np.ndarray:
    return score / np.maximum(score.sum(axis=1, keepdims=True), 1e-6)


def _make_horizon_blocks(h: int, num_blocks: int) -> np.ndarray:
    num_blocks = max(1, min(int(num_blocks), h))
    edges = np.linspace(0, h, num_blocks + 1, dtype=int)
    edges[-1] = h
    block_index = np.zeros(h, dtype=np.int32)
    for block in range(num_blocks):
        start = edges[block]
        stop = edges[block + 1]
        if stop <= start:
            stop = min(h, start + 1)
        block_index[start:stop] = block
    return block_index


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
    return _normalize_static_source_scores(score)


def _compute_consensus_scale(source_residuals: np.ndarray, base_weights: np.ndarray, temperature: float) -> np.ndarray:
    center = np.sum(base_weights[None, :, :] * source_residuals, axis=1)
    deviation = np.abs(source_residuals - center[:, None, :])
    scale = np.quantile(deviation, 0.65, axis=(0, 1)) + 1e-6
    return np.maximum(scale * max(float(temperature), 1e-3), 1e-6)


def _compute_dynamic_source_weights(
    source_residuals: np.ndarray,
    base_weights: np.ndarray,
    consensus_scale: np.ndarray,
) -> np.ndarray:
    if source_residuals.shape[1] == 0:
        return np.zeros_like(source_residuals, dtype=np.float32)
    center = np.sum(base_weights[None, :, :] * source_residuals, axis=1)
    deviation = np.abs(source_residuals - center[:, None, :])
    agreement = np.exp(-deviation / np.maximum(consensus_scale[None, None, :], 1e-6))
    score = np.maximum(base_weights[None, :, :], 1e-4) * agreement
    score += 1e-4 * np.maximum(base_weights[None, :, :], 1e-4)
    return _normalize_dynamic_source_scores(score)


def _compute_regime_score(
    raw_transfer: np.ndarray,
    dispersion: np.ndarray,
    transfer_scale: np.ndarray,
    dispersion_scale: np.ndarray,
) -> np.ndarray:
    transfer_norm = np.clip(np.abs(raw_transfer) / np.maximum(transfer_scale[None, :], 1e-6), 0.0, 4.0)
    dispersion_norm = np.clip(dispersion / np.maximum(dispersion_scale[None, :], 1e-6), 0.0, 4.0)
    return 0.6 * dispersion_norm + 0.4 * transfer_norm


def _compute_regime_thresholds(regime_score: np.ndarray, horizon_blocks: np.ndarray, num_bins: int) -> np.ndarray:
    num_bins = max(1, int(num_bins))
    num_blocks = int(horizon_blocks.max()) + 1
    if num_bins == 1:
        return np.zeros((num_blocks, 0), dtype=np.float32)
    quantiles = [idx / num_bins for idx in range(1, num_bins)]
    thresholds = np.zeros((num_blocks, num_bins - 1), dtype=np.float64)
    global_values = regime_score.reshape(-1)
    for block in range(num_blocks):
        block_values = regime_score[:, horizon_blocks == block].reshape(-1)
        if block_values.size < max(16, num_bins * 8):
            block_values = global_values
        thresholds[block] = np.quantile(block_values, quantiles)
    return thresholds.astype(np.float32)


def _assign_regime_bins(regime_score: np.ndarray, horizon_blocks: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    bins = np.zeros_like(regime_score, dtype=np.int32)
    for block in range(int(horizon_blocks.max()) + 1):
        block_mask = horizon_blocks == block
        if thresholds.shape[1] == 0:
            bins[:, block_mask] = 0
            continue
        bins[:, block_mask] = np.digitize(regime_score[:, block_mask], thresholds[block], right=False)
    return bins


def _search_best_alpha(
    local_pred: np.ndarray,
    y_true: np.ndarray,
    local_se: np.ndarray,
    transfer_signal: np.ndarray,
    grid: np.ndarray,
    harm_limit: float,
    penalty_scale: float,
) -> float:
    if local_pred.size == 0:
        return 0.0
    best_alpha = 0.0
    best_objective = float(np.mean(local_se))
    local_loss = float(np.mean(local_se))
    for alpha in grid:
        pred = local_pred + alpha * transfer_signal
        se = (pred - y_true) ** 2
        harm = float(np.mean(se > local_se))
        regret = float(np.max(se - local_se))
        objective = float(np.mean(se))
        if harm > harm_limit:
            objective += local_loss * (harm - harm_limit) * (2.0 + penalty_scale)
        if regret > 0.0:
            objective += 0.05 * penalty_scale * regret
        if objective < best_objective:
            best_objective = objective
            best_alpha = float(alpha)
    return best_alpha


def _calibrate_parameters(
    local_pred: np.ndarray,
    source_preds: np.ndarray,
    y_true: np.ndarray,
    source_weights: np.ndarray,
    grid_size: int,
    harm_limit: float,
    residual_cap_scale: float,
    residual_cap_floor: float,
    num_regime_bins: int,
    num_horizon_blocks: int,
    min_bin_samples: int,
    tail_penalty: float,
    agreement_temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    local_se = (local_pred - y_true) ** 2
    source_se = (source_preds - y_true[:, None, :]) ** 2
    source_residuals = source_preds - local_pred[:, None, :]

    consensus_scale = _compute_consensus_scale(source_residuals, source_weights, agreement_temperature)
    dynamic_weights = _compute_dynamic_source_weights(source_residuals, source_weights, consensus_scale)
    weighted_source_se = np.sum(dynamic_weights * source_se, axis=1)
    transfer_strength_base = np.clip(
        np.mean(np.maximum(local_se - weighted_source_se, 0.0), axis=0) / np.maximum(local_se.mean(axis=0), 1e-6),
        0.0,
        1.0,
    )

    raw_transfer = np.sum(dynamic_weights * source_residuals, axis=1)
    dispersion = _weighted_std(source_residuals, dynamic_weights)
    dispersion_scale = np.quantile(dispersion, 0.6, axis=0) + 1e-6
    transfer_scale = np.quantile(np.abs(raw_transfer), 0.7, axis=0) + 1e-6
    gate = 1.0 / (1.0 + dispersion / dispersion_scale[None, :])

    target_scale = np.std(y_true, axis=0) + 1e-6
    residual_cap = np.maximum(
        residual_cap_floor * target_scale,
        residual_cap_scale * np.quantile(np.abs(raw_transfer), 0.75, axis=0),
    )
    bounded = _bounded_transfer(raw_transfer, residual_cap)
    transfer_signal = transfer_strength_base[None, :] * gate * bounded

    horizon_blocks = _make_horizon_blocks(local_pred.shape[1], num_horizon_blocks)
    regime_score = _compute_regime_score(raw_transfer, dispersion, transfer_scale, dispersion_scale)
    regime_thresholds = _compute_regime_thresholds(regime_score, horizon_blocks, num_regime_bins)
    regime_bins = _assign_regime_bins(regime_score, horizon_blocks, regime_thresholds)

    grid = np.linspace(0.0, 1.0, max(int(grid_size), 2), dtype=np.float64)
    num_blocks = int(horizon_blocks.max()) + 1
    num_regimes = max(1, int(num_regime_bins))
    regime_alpha_table = np.zeros((num_blocks, num_regimes), dtype=np.float64)
    calibration_alpha = np.zeros(local_pred.shape[1], dtype=np.float64)

    for block in range(num_blocks):
        block_mask = horizon_blocks == block
        block_local_pred = local_pred[:, block_mask]
        block_truth = y_true[:, block_mask]
        block_local_se = local_se[:, block_mask]
        block_transfer = transfer_signal[:, block_mask]
        penalty_scale = 1.0 + float(tail_penalty) * (block / max(1, num_blocks - 1))
        pooled_alpha = _search_best_alpha(
            block_local_pred.reshape(-1),
            block_truth.reshape(-1),
            block_local_se.reshape(-1),
            block_transfer.reshape(-1),
            grid,
            harm_limit,
            penalty_scale,
        )
        for regime in range(num_regimes):
            selected = regime_bins[:, block_mask] == regime
            if int(selected.sum()) < max(8, int(min_bin_samples)):
                regime_alpha_table[block, regime] = pooled_alpha
                continue
            regime_alpha_table[block, regime] = _search_best_alpha(
                block_local_pred[selected],
                block_truth[selected],
                block_local_se[selected],
                block_transfer[selected],
                grid,
                harm_limit,
                penalty_scale,
            )
        calibration_alpha[block_mask] = regime_alpha_table[block, min(num_regimes - 1, num_regimes // 2)]

    return (
        transfer_strength_base.astype(np.float32),
        calibration_alpha.astype(np.float32),
        residual_cap.astype(np.float32),
        dispersion_scale.astype(np.float32),
        consensus_scale.astype(np.float32),
        transfer_scale.astype(np.float32),
        horizon_blocks.astype(np.int32),
        regime_thresholds.astype(np.float32),
        regime_alpha_table.astype(np.float32),
    )


def _select_regime_alpha(bundle: SafePatchTSTBundle, regime_bins: np.ndarray, n_windows: int) -> np.ndarray:
    alpha_by_horizon = bundle.regime_alpha_table[bundle.horizon_blocks]
    alpha_cube = np.broadcast_to(alpha_by_horizon[None, :, :], (n_windows, alpha_by_horizon.shape[0], alpha_by_horizon.shape[1]))
    return np.take_along_axis(alpha_cube, regime_bins[:, :, None], axis=2).squeeze(2)


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
    return _fit_safe_nf_transfer(
        backbone_method="patchtst",
        spec=spec,
        feature_cols=feature_cols,
        input_size=input_size,
        h=h,
        target_train_frame=target_train_frame,
        target_val_frame=target_val_frame,
        target_val_indices=target_val_indices,
        source_frames=source_frames,
        seed=seed,
        device=device,
        args=args,
        local_bundle=local_bundle,
    )


def fit_safe_fedformer(
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
    return _fit_safe_nf_transfer(
        backbone_method="fedformer",
        spec=spec,
        feature_cols=feature_cols,
        input_size=input_size,
        h=h,
        target_train_frame=target_train_frame,
        target_val_frame=target_val_frame,
        target_val_indices=target_val_indices,
        source_frames=source_frames,
        seed=seed,
        device=device,
        args=args,
        local_bundle=local_bundle,
    )


def fit_safe_gru(
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
    return _fit_safe_nf_transfer(
        backbone_method="gru",
        spec=spec,
        feature_cols=feature_cols,
        input_size=input_size,
        h=h,
        target_train_frame=target_train_frame,
        target_val_frame=target_val_frame,
        target_val_indices=target_val_indices,
        source_frames=source_frames,
        seed=seed,
        device=device,
        args=args,
        local_bundle=local_bundle,
    )


def _fit_safe_nf_transfer(
    *,
    backbone_method: str,
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
            backbone_method,
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
            backbone_method,
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
        (
            transfer_strength_base,
            calibration_alpha,
            residual_cap,
            dispersion_scale,
            consensus_scale,
            transfer_scale,
            horizon_blocks,
            regime_thresholds,
            regime_alpha_table,
        ) = _calibrate_parameters(
            local_val_pred,
            source_val_preds_arr,
            y_val_true,
            source_weights,
            grid_size=args.calibration_grid_size,
            harm_limit=args.calibration_harm_limit,
            residual_cap_scale=args.residual_cap_scale,
            residual_cap_floor=args.residual_cap_floor,
            num_regime_bins=args.safe_patch_regime_bins,
            num_horizon_blocks=args.safe_patch_horizon_blocks,
            min_bin_samples=args.safe_patch_min_bin_samples,
            tail_penalty=args.safe_patch_tail_penalty,
            agreement_temperature=args.safe_patch_agreement_temperature,
        )
    else:
        source_weights = np.zeros((0, h), dtype=np.float32)
        transfer_strength_base = np.zeros(h, dtype=np.float32)
        calibration_alpha = np.zeros(h, dtype=np.float32)
        residual_cap = np.zeros(h, dtype=np.float32)
        dispersion_scale = np.ones(h, dtype=np.float32)
        consensus_scale = np.ones(h, dtype=np.float32)
        transfer_scale = np.ones(h, dtype=np.float32)
        horizon_blocks = _make_horizon_blocks(h, args.safe_patch_horizon_blocks)
        regime_thresholds = np.zeros((int(horizon_blocks.max()) + 1, max(0, args.safe_patch_regime_bins - 1)), dtype=np.float32)
        regime_alpha_table = np.zeros((int(horizon_blocks.max()) + 1, max(1, args.safe_patch_regime_bins)), dtype=np.float32)

    calibration_duration = time.perf_counter() - calibration_start
    total_duration = time.perf_counter() - fit_start

    history = list(getattr(local_bundle.fitted_model, "_training_history", []) or [])
    if history:
        history.append(
            {
                "epoch": float(len(history) + 1),
                "train_loss": np.nan,
                "val_loss": float(np.mean((local_val_pred - y_val_true) ** 2)),
                "best_val_loss_so_far": float(
                    np.nanmin([row.get("best_val_loss_so_far", np.nan) for row in history] + [np.mean((local_val_pred - y_val_true) ** 2)])
                ),
            }
        )
    summary = dict(getattr(local_bundle.fitted_model, "_training_summary", {}) or {})
    summary["duration_sec"] = float(summary.get("duration_sec", 0.0) + sum(bundle.fit_duration_sec for bundle in source_bundles) + calibration_duration)
    summary["source_fit_duration_sec"] = float(sum(bundle.fit_duration_sec for bundle in source_bundles))
    summary["calibration_duration_sec"] = float(calibration_duration)
    summary["reused_local_bundle"] = bool(reused_local_bundle)
    summary["safe_patch_regime_bins"] = int(args.safe_patch_regime_bins)
    summary["safe_patch_horizon_blocks"] = int(args.safe_patch_horizon_blocks)

    return SafePatchTSTBundle(
        backbone_method=backbone_method,
        local_bundle=local_bundle,
        source_bundles=source_bundles,
        source_ids=source_ids,
        source_weights=source_weights.astype(np.float32),
        transfer_strength_base=transfer_strength_base.astype(np.float32),
        calibration_alpha=calibration_alpha.astype(np.float32),
        residual_cap=residual_cap.astype(np.float32),
        dispersion_scale=dispersion_scale.astype(np.float32),
        consensus_scale=consensus_scale.astype(np.float32),
        transfer_scale=transfer_scale.astype(np.float32),
        horizon_blocks=horizon_blocks.astype(np.int32),
        regime_thresholds=regime_thresholds.astype(np.float32),
        regime_alpha_table=regime_alpha_table.astype(np.float32),
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
    horizon_block = np.broadcast_to(bundle.horizon_blocks[None, :], local_pred.shape).astype(np.float32)
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
            "regime_score": zeros,
            "regime_bin": zeros,
            "horizon_block": horizon_block,
            "source_dispersion": zeros,
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
    source_weights = _compute_dynamic_source_weights(source_residuals, bundle.source_weights, bundle.consensus_scale)
    raw_transfer = np.sum(source_weights * source_residuals, axis=1)
    dispersion = _weighted_std(source_residuals, source_weights)
    gate = 1.0 / (1.0 + dispersion / np.maximum(bundle.dispersion_scale[None, :], 1e-6))
    bounded_transfer = _bounded_transfer(raw_transfer, bundle.residual_cap)
    transfer_strength = bundle.transfer_strength_base[None, :] * gate
    regime_score = _compute_regime_score(raw_transfer, dispersion, bundle.transfer_scale, bundle.dispersion_scale)
    regime_bins = _assign_regime_bins(regime_score, bundle.horizon_blocks, bundle.regime_thresholds)
    calibration_alpha = _select_regime_alpha(bundle, regime_bins, local_pred.shape[0]).astype(np.float32)
    calibrated_strength = transfer_strength * calibration_alpha
    transfer_delta = calibrated_strength * bounded_transfer
    final_pred = local_pred + transfer_delta
    source_gates = source_weights * calibrated_strength[:, None, :]
    return {
        "truths": y_true.astype(np.float32),
        "local": local_pred.astype(np.float32),
        "final": final_pred.astype(np.float32),
        "source_preds": source_preds_arr,
        "source_weights": source_weights.astype(np.float32),
        "source_gates": source_gates.astype(np.float32),
        "transfer_strength": transfer_strength.astype(np.float32),
        "raw_transfer": raw_transfer.astype(np.float32),
        "bounded_transfer": bounded_transfer.astype(np.float32),
        "transfer_delta": transfer_delta.astype(np.float32),
        "residual_budget": np.broadcast_to(bundle.residual_cap[None, :], local_pred.shape).astype(np.float32),
        "calibration_alpha": calibration_alpha,
        "regime_score": regime_score.astype(np.float32),
        "regime_bin": regime_bins.astype(np.float32),
        "horizon_block": horizon_block,
        "source_dispersion": dispersion.astype(np.float32),
    }


def predict_safe_fedformer(
    bundle: SafePatchTSTBundle,
    *,
    test_frame: pd.DataFrame,
    spec: DatasetSpec,
    feature_cols: Sequence[str],
    window_indices: Sequence[int],
    seq_len: int,
    pred_len: int,
) -> dict[str, np.ndarray]:
    return predict_safe_patchtst(
        bundle,
        test_frame=test_frame,
        spec=spec,
        feature_cols=feature_cols,
        window_indices=window_indices,
        seq_len=seq_len,
        pred_len=pred_len,
    )


def predict_safe_gru(
    bundle: SafePatchTSTBundle,
    *,
    test_frame: pd.DataFrame,
    spec: DatasetSpec,
    feature_cols: Sequence[str],
    window_indices: Sequence[int],
    seq_len: int,
    pred_len: int,
) -> dict[str, np.ndarray]:
    return predict_safe_patchtst(
        bundle,
        test_frame=test_frame,
        spec=spec,
        feature_cols=feature_cols,
        window_indices=window_indices,
        seq_len=seq_len,
        pred_len=pred_len,
    )
