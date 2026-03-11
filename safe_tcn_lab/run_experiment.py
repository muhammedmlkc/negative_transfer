from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safe_tcn_lab.artifacts import (
    build_per_target_metrics_frame,
    build_prediction_frame,
    build_runtime_frame,
    build_safe_source_frame,
    build_source_selection_frame,
    build_training_history_frame,
    build_window_metric_frame,
    save_parquet,
)
from safe_tcn_lab.baselines import (
    build_persistence_predictions,
    build_tabular_matrix,
    fit_lgbm_multioutput,
    fit_ridge_multioutput,
    predict_lgbm,
)
from safe_tcn_lab.data import MultiTaskForecastData, get_dataset_spec
from safe_tcn_lab.metrics import add_method_relative_safety, add_transfer_safety, evaluate_predictions, summarize_results
from safe_tcn_lab.models import SafeTCNForecaster, TaskConditionedTCN
from safe_tcn_lab.nf_baselines import NF_METHODS, fit_nf_model, predict_nf_windows
from safe_tcn_lab.safe_patchtst import (
    fit_safe_fedformer,
    fit_safe_patchtst,
    predict_safe_fedformer,
    predict_safe_patchtst,
)
from safe_tcn_lab.train import (
    calibrate_safe_tcn,
    collect_local_predictions,
    collect_safe_outputs,
    set_seed,
    train_local_model,
    train_multitask_pretrain,
    train_multitask_target_model,
    train_safe_tcn,
)


DATASET_DEFAULTS = {
    "sdwpf": {
        "parquet_path": "data/processed/sdwpf_long.parquet",
        "seq_len": 144,
        "pred_len": 288,
        "target_ids": [1, 10, 25, 50, 75, 100, 120, 134],
        "max_sources": 1,
        "min_similarity": 0.0,
    },
    "gefcom": {
        "parquet_path": "data/processed/gefcom_wind_long.parquet",
        "seq_len": 168,
        "pred_len": 24,
        "target_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "max_sources": 2,
        "min_similarity": 0.0,
    },
}


CORE_METHODS = [
    "persistence",
    "ridge",
    "lightgbm",
    "tcn",
    "tcn_fine_tune",
    "safe_tcn",
]

PAPER_METHODS = [
    "persistence",
    "ridge",
    "lightgbm",
    "lstm",
    "gru",
    "tcn",
    "dlinear",
    "nbeats",
    "informer",
    "fedformer",
    "patchtst",
    "timesnet",
    "itransformer",
    "tcn_fine_tune",
    "tcn_multi_task",
    "safe_tcn",
]

EXTRA_METHODS = ["lgbm_transfer", "safe_patchtst", "safe_fedformer"]

ALL_METHODS = list(dict.fromkeys(CORE_METHODS + list(NF_METHODS) + PAPER_METHODS + EXTRA_METHODS))

METHOD_ALIASES = {
    "ridge_local": "ridge",
    "lgbm_local": "lightgbm",
    "lgbm": "lightgbm",
    "tcn_local": "tcn",
    "fine_tune": "tcn_fine_tune",
    "multi_task": "tcn_multi_task",
}


def save_json(payload: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=float)


def denormalize_with_stats(data: MultiTaskForecastData, task_id: int, values: np.ndarray, normalization_train_days: int | None = None) -> np.ndarray:
    stats = data.get_normalization_stats(task_id, train_days_limit=normalization_train_days)
    return data.denormalize_target(task_id, values, stats=stats)


def target_std_with_stats(data: MultiTaskForecastData, task_id: int, normalization_train_days: int | None = None) -> float:
    _, std = data.get_normalization_stats(task_id, train_days_limit=normalization_train_days)
    return float(std[-1])


def artifact_path(run_dir: str, *parts: str) -> str:
    return os.path.join(run_dir, "artifacts", *parts)


def get_device(device_name: str | None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_torch_runtime(device: torch.device, matmul_precision: str, allow_tf32: bool) -> None:
    if device.type != "cuda":
        return
    torch.set_float32_matmul_precision(matmul_precision)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SAFE-TCN wind power forecasting experiments")
    parser.add_argument("--dataset", choices=["sdwpf", "gefcom"], required=True)
    parser.add_argument("--parquet_path", default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--pred_len", type=int, default=None)
    parser.add_argument("--target_ids", nargs="*", type=int, default=None)
    parser.add_argument("--max_sources", type=int, default=None)
    parser.add_argument("--min_similarity", type=float, default=None)
    parser.add_argument("--target_train_days", type=int, default=None)
    parser.add_argument("--train_stride", type=int, default=1)
    parser.add_argument("--eval_stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--disable_pin_memory", action="store_true")
    parser.add_argument("--disable_persistent_workers", action="store_true")
    parser.add_argument("--matmul_precision", choices=["highest", "high", "medium"], default="high")
    parser.add_argument("--disable_tf32", action="store_true")
    parser.add_argument("--n_pc_bins", type=int, default=20)
    parser.add_argument("--max_rows_per_task", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--model_dim", type=int, default=64)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gate_hidden_dim", type=int, default=64)
    parser.add_argument("--fine_tune_mode", choices=["all", "head"], default="all")
    parser.add_argument("--safe_target_mode", choices=["all", "head"], default="head")
    parser.add_argument("--pretrain_epochs", type=int, default=20)
    parser.add_argument("--finetune_epochs", type=int, default=18)
    parser.add_argument("--safe_epochs", type=int, default=18)
    parser.add_argument("--lr_pretrain", type=float, default=1e-3)
    parser.add_argument("--lr_finetune", type=float, default=7e-4)
    parser.add_argument("--lr_safe", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--gate_lambda", type=float, default=0.25)
    parser.add_argument("--safety_lambda", type=float, default=1.0)
    parser.add_argument("--sparsity_lambda", type=float, default=0.02)
    parser.add_argument("--relation_lambda", type=float, default=0.05)
    parser.add_argument("--budget_lambda", type=float, default=0.02)
    parser.add_argument("--harm_margin", type=float, default=0.0)
    parser.add_argument("--residual_cap_scale", type=float, default=0.75)
    parser.add_argument("--residual_cap_floor", type=float, default=0.02)
    parser.add_argument("--calibration_harm_limit", type=float, default=0.45)
    parser.add_argument("--calibration_grid_size", type=int, default=11)
    parser.add_argument("--safe_patch_regime_bins", type=int, default=3)
    parser.add_argument("--safe_patch_horizon_blocks", type=int, default=3)
    parser.add_argument("--safe_patch_min_bin_samples", type=int, default=64)
    parser.add_argument("--safe_patch_tail_penalty", type=float, default=0.35)
    parser.add_argument("--safe_patch_agreement_temperature", type=float, default=1.0)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--lgbm_estimators", type=int, default=120)
    parser.add_argument("--nf_max_steps", type=int, default=300)
    parser.add_argument("--nf_learning_rate", type=float, default=1e-3)
    parser.add_argument("--nf_early_stop_patience_steps", type=int, default=30)
    parser.add_argument("--nf_val_check_steps", type=int, default=20)
    parser.add_argument("--nf_batch_size", type=int, default=32)
    parser.add_argument("--nf_windows_batch_size", type=int, default=128)
    parser.add_argument("--nf_inference_windows_batch_size", type=int, default=256)
    parser.add_argument("--nf_hidden_size", type=int, default=128)
    parser.add_argument("--nf_num_layers", type=int, default=2)
    parser.add_argument("--nf_dropout", type=float, default=0.1)
    parser.add_argument("--nf_n_heads", type=int, default=4)
    parser.add_argument("--nf_patch_len", type=int, default=16)
    parser.add_argument("--output_root", default="safe_tcn_lab/outputs")
    parser.add_argument("--disable_artifacts", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def resolve_methods(methods: List[str] | None) -> List[str]:
    if not methods:
        return list(CORE_METHODS)
    expanded: List[str] = []
    for method in methods:
        normalized = METHOD_ALIASES.get(method.lower(), method.lower())
        if normalized == "core":
            expanded.extend(CORE_METHODS)
        elif normalized == "paper_all":
            expanded.extend(PAPER_METHODS)
        elif normalized == "extended_all":
            expanded.extend(PAPER_METHODS + EXTRA_METHODS)
        else:
            if normalized not in ALL_METHODS:
                raise ValueError(f"Unknown method '{method}'. Available methods: {', '.join(ALL_METHODS)}")
            expanded.append(normalized)
    seen = set()
    ordered = []
    for method in expanded:
        if method not in seen:
            ordered.append(method)
            seen.add(method)
    return ordered


def make_model(args: argparse.Namespace, input_dim: int, profile_dim: int) -> TaskConditionedTCN:
    return TaskConditionedTCN(
        input_dim=input_dim,
        profile_dim=profile_dim,
        pred_len=args.pred_len,
        model_dim=args.model_dim,
        levels=args.levels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    )


def loader_for(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    *,
    prefetch_factor: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory and device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def evaluate_task_predictions(
    data: MultiTaskForecastData,
    task_id: int,
    dataset,
    y_true_norm: np.ndarray,
    y_pred_norm: np.ndarray,
    normalization_train_days: int | None = None,
) -> Dict[str, float]:
    stats = data.get_normalization_stats(task_id, train_days_limit=normalization_train_days)
    y_true = data.denormalize_target(task_id, y_true_norm, stats=stats)
    y_pred = data.denormalize_target(task_id, y_pred_norm, stats=stats)
    future_frames = [dataset.get_future_frame(idx) for idx in range(len(dataset))]
    metrics, window_mae, window_rmse = evaluate_predictions(
        dataset_name=data.spec.name,
        y_true=y_true,
        y_pred=y_pred,
        future_frames=future_frames,
        rated_capacity=data.spec.rated_capacity,
    )
    metrics["_WINDOW_MAE"] = window_mae.tolist()
    metrics["_WINDOW_RMSE"] = window_rmse.tolist()
    return metrics


def evaluate_task_predictions_raw(
    data: MultiTaskForecastData,
    dataset,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    future_frames = [dataset.get_future_frame(idx) for idx in range(len(dataset))]
    metrics, window_mae, window_rmse = evaluate_predictions(
        dataset_name=data.spec.name,
        y_true=y_true,
        y_pred=y_pred,
        future_frames=future_frames,
        rated_capacity=data.spec.rated_capacity,
    )
    metrics["_WINDOW_MAE"] = window_mae.tolist()
    metrics["_WINDOW_RMSE"] = window_rmse.tolist()
    return metrics


def save_prediction_artifact(
    run_dir: str,
    data: MultiTaskForecastData,
    task_id: int,
    dataset,
    seed: int,
    method: str,
    split: str,
    y_true_norm: np.ndarray,
    y_pred_norm: np.ndarray,
    normalization_train_days: int | None = None,
    extras: Dict[str, np.ndarray] | None = None,
) -> None:
    y_true = denormalize_with_stats(data, task_id, y_true_norm, normalization_train_days)
    y_pred = denormalize_with_stats(data, task_id, y_pred_norm, normalization_train_days)
    frame = build_prediction_frame(
        spec=data.spec,
        dataset=dataset,
        dataset_name=data.spec.name,
        seed=seed,
        target_id=task_id,
        method=method,
        split=split,
        y_true=y_true,
        y_pred=y_pred,
        extras=extras,
    )
    save_parquet(frame, artifact_path(run_dir, "predictions", method, f"target_{task_id}_{split}.parquet"))


def save_prediction_artifact_raw(
    run_dir: str,
    data: MultiTaskForecastData,
    task_id: int,
    dataset,
    seed: int,
    method: str,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    extras: Dict[str, np.ndarray] | None = None,
) -> None:
    frame = build_prediction_frame(
        spec=data.spec,
        dataset=dataset,
        dataset_name=data.spec.name,
        seed=seed,
        target_id=task_id,
        method=method,
        split=split,
        y_true=y_true,
        y_pred=y_pred,
        extras=extras,
    )
    save_parquet(frame, artifact_path(run_dir, "predictions", method, f"target_{task_id}_{split}.parquet"))


def save_window_metrics_artifact(
    run_dir: str,
    data: MultiTaskForecastData,
    task_id: int,
    dataset,
    seed: int,
    method: str,
    split: str,
    metrics: Dict[str, float],
) -> None:
    window_mae = metrics.get("_WINDOW_MAE")
    window_rmse = metrics.get("_WINDOW_RMSE")
    if window_mae is None or window_rmse is None:
        return
    frame = build_window_metric_frame(
        spec=data.spec,
        dataset=dataset,
        dataset_name=data.spec.name,
        seed=seed,
        target_id=task_id,
        method=method,
        split=split,
        window_mae=window_mae,
        window_rmse=window_rmse,
    )
    if not frame.empty:
        save_parquet(frame, artifact_path(run_dir, "window_metrics", method, f"target_{task_id}_{split}.parquet"))


def save_training_history_artifact(run_dir: str, dataset_name: str, seed: int, target_id: int, method: str, model) -> None:
    frame = build_training_history_frame(dataset_name, seed, target_id, method, model)
    if frame.empty:
        return
    save_parquet(frame, artifact_path(run_dir, "training_history", method, f"target_{target_id}.parquet"))


def drop_hidden(metrics: Dict[str, float]) -> Dict[str, float]:
    return {key: value for key, value in metrics.items() if not key.startswith("_")}


def record_runtime(
    runtime_rows: list[dict[str, object]],
    target_id: int,
    method: str,
    stage: str,
    duration_sec: float,
    device: torch.device,
) -> None:
    runtime_rows.append(
        {
            "target_id": int(target_id),
            "method": method,
            "stage": stage,
            "duration_sec": float(duration_sec),
            "device": str(device),
        }
    )
def run_experiment(args: argparse.Namespace) -> Dict[str, object]:
    defaults = DATASET_DEFAULTS[args.dataset]
    args.parquet_path = args.parquet_path or defaults["parquet_path"]
    args.seq_len = args.seq_len or defaults["seq_len"]
    args.pred_len = args.pred_len or defaults["pred_len"]
    args.target_ids = args.target_ids or defaults["target_ids"]
    args.max_sources = args.max_sources if args.max_sources is not None else defaults["max_sources"]
    args.min_similarity = args.min_similarity if args.min_similarity is not None else defaults["min_similarity"]
    args.methods = resolve_methods(args.methods)
    args.nf_val_check_steps = min(args.nf_val_check_steps, args.nf_max_steps)

    if args.smoke:
        args.target_ids = args.target_ids[:1]
        args.pretrain_epochs = min(args.pretrain_epochs, 2)
        args.finetune_epochs = min(args.finetune_epochs, 2)
        args.safe_epochs = min(args.safe_epochs, 2)
        args.nf_max_steps = min(args.nf_max_steps, 2)
        args.nf_val_check_steps = min(args.nf_val_check_steps, args.nf_max_steps)
        args.train_stride = max(args.train_stride, 24)
        args.eval_stride = max(args.eval_stride, 24)

    set_seed(args.seed)
    device = get_device(args.device)
    configure_torch_runtime(
        device,
        matmul_precision=args.matmul_precision,
        allow_tf32=not args.disable_tf32,
    )
    spec = get_dataset_spec(args.dataset)
    data = MultiTaskForecastData(
        spec=spec,
        parquet_path=args.parquet_path,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        n_pc_bins=args.n_pc_bins,
        max_rows_per_task=args.max_rows_per_task,
    ).load()
    profile_bank_tensor = torch.from_numpy(data.get_profiles(data.task_ids))

    per_target: Dict[int, Dict[str, Dict[str, float]]] = {}
    run_dir = os.path.join(args.output_root, f"{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    save_json(vars(args), os.path.join(run_dir, "config.json"))
    source_selection_frames: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, object]] = []

    for target_id in args.target_ids:
        print(f"\n=== Target {target_id} ===")
        target_results: Dict[str, Dict[str, float]] = {}
        source_pairs = data.select_sources(
            target_id,
            max_sources=args.max_sources,
            min_similarity=args.min_similarity,
            target_train_days_limit=args.target_train_days,
        )
        source_ids = [task_id for task_id, _ in source_pairs]
        target_results["_sources"] = [{"task_id": task_id, "similarity": similarity} for task_id, similarity in source_pairs]
        if not args.disable_artifacts:
            source_selection_frames.append(
                build_source_selection_frame(
                    dataset_name=args.dataset,
                    seed=args.seed,
                    target_id=target_id,
                    target_train_days=args.target_train_days,
                    source_pairs=source_pairs,
                )
            )

        target_train = data.get_dataset(
            target_id,
            "train",
            train_days_limit=args.target_train_days,
            normalization_train_days=args.target_train_days,
            stride=args.train_stride,
        )
        target_val = data.get_dataset(
            target_id,
            "val",
            normalization_train_days=args.target_train_days,
            stride=args.eval_stride,
        )
        target_test = data.get_dataset(
            target_id,
            "test",
            normalization_train_days=args.target_train_days,
            stride=args.eval_stride,
        )
        if len(target_train) == 0 or len(target_val) == 0 or len(target_test) == 0:
            print(f"Target {target_id} skipped due to empty split.")
            continue

        train_loader = loader_for(
            target_train,
            args.batch_size,
            True,
            args.num_workers,
            device,
            prefetch_factor=args.prefetch_factor,
            pin_memory=not args.disable_pin_memory,
            persistent_workers=not args.disable_persistent_workers,
        )
        val_loader = loader_for(
            target_val,
            args.batch_size,
            False,
            args.num_workers,
            device,
            prefetch_factor=args.prefetch_factor,
            pin_memory=not args.disable_pin_memory,
            persistent_workers=not args.disable_persistent_workers,
        )
        test_loader = loader_for(
            target_test,
            args.batch_size,
            False,
            args.num_workers,
            device,
            prefetch_factor=args.prefetch_factor,
            pin_memory=not args.disable_pin_memory,
            persistent_workers=not args.disable_persistent_workers,
        )

        target_profile_np = data.get_profile(target_id, train_days_limit=args.target_train_days).astype(np.float32)
        target_profile = torch.from_numpy(target_profile_np)
        source_profiles_np = data.get_profiles(source_ids) if source_ids else np.zeros((0, data.profile_dim), dtype=np.float32)
        relation_np = data.build_relation_matrix(
            target_id,
            source_ids,
            target_train_days_limit=args.target_train_days,
        )
        relation_np = relation_np if relation_np.size else np.zeros((0, 8), dtype=np.float32)
        target_std = target_std_with_stats(data, target_id, normalization_train_days=args.target_train_days)

        y_true_test = np.stack([target_test[idx][2].numpy() for idx in range(len(target_test))], axis=0)
        y_true_test_raw = denormalize_with_stats(
            data,
            target_id,
            y_true_test,
            normalization_train_days=args.target_train_days,
        )
        x_train_local, y_train_local = build_tabular_matrix(target_train)
        x_test, _ = build_tabular_matrix(target_test)
        local_model = None
        pretrain_model = None
        nf_bundles: Dict[str, object] = {}

        def finalize_method(
            method_name: str,
            metrics: Dict[str, float],
            *,
            normalized_outputs: tuple[np.ndarray, np.ndarray] | None = None,
            raw_outputs: tuple[np.ndarray, np.ndarray] | None = None,
            extras: Dict[str, np.ndarray] | None = None,
            model=None,
        ) -> None:
            target_results[method_name] = metrics
            if args.disable_artifacts:
                return
            if model is not None:
                save_training_history_artifact(run_dir, args.dataset, args.seed, target_id, method_name, model)
            save_window_metrics_artifact(run_dir, data, target_id, target_test, args.seed, method_name, "test", metrics)
            if normalized_outputs is not None:
                y_true_norm, y_pred_norm = normalized_outputs
                save_prediction_artifact(
                    run_dir,
                    data,
                    target_id,
                    target_test,
                    args.seed,
                    method_name,
                    "test",
                    y_true_norm,
                    y_pred_norm,
                    normalization_train_days=args.target_train_days,
                    extras=extras,
                )
            elif raw_outputs is not None:
                y_true_raw, y_pred_raw = raw_outputs
                save_prediction_artifact_raw(
                    run_dir,
                    data,
                    target_id,
                    target_test,
                    args.seed,
                    method_name,
                    "test",
                    y_true_raw,
                    y_pred_raw,
                    extras=extras,
                )

        if "persistence" in args.methods:
            print(f"Target {target_id} | persistence")
            method_start = time.perf_counter()
            y_true, y_pred = build_persistence_predictions(target_test)
            metrics = evaluate_task_predictions(
                data,
                target_id,
                target_test,
                y_true,
                y_pred,
                normalization_train_days=args.target_train_days,
            )
            finalize_method("persistence", metrics, normalized_outputs=(y_true, y_pred))
            record_runtime(runtime_rows, target_id, "persistence", "total", time.perf_counter() - method_start, device)

        if "ridge" in args.methods:
            print(f"Target {target_id} | ridge")
            fit_start = time.perf_counter()
            ridge_model = fit_ridge_multioutput(x_train_local, y_train_local, alpha=args.ridge_alpha)
            fit_duration = time.perf_counter() - fit_start
            pred_start = time.perf_counter()
            y_pred = ridge_model.predict(x_test).astype(np.float32)
            predict_duration = time.perf_counter() - pred_start
            metrics = evaluate_task_predictions(
                data,
                target_id,
                target_test,
                y_true_test,
                y_pred,
                normalization_train_days=args.target_train_days,
            )
            finalize_method("ridge", metrics, normalized_outputs=(y_true_test, y_pred))
            record_runtime(runtime_rows, target_id, "ridge", "fit", fit_duration, device)
            record_runtime(runtime_rows, target_id, "ridge", "predict", predict_duration, device)
            record_runtime(runtime_rows, target_id, "ridge", "total", fit_duration + predict_duration, device)

        if "lightgbm" in args.methods:
            print(f"Target {target_id} | lightgbm")
            fit_start = time.perf_counter()
            lightgbm_model = fit_lgbm_multioutput(
                x_train_local,
                y_train_local,
                n_estimators=args.lgbm_estimators,
                random_state=args.seed,
            )
            fit_duration = time.perf_counter() - fit_start
            pred_start = time.perf_counter()
            y_pred = predict_lgbm(lightgbm_model, x_test)
            predict_duration = time.perf_counter() - pred_start
            metrics = evaluate_task_predictions(
                data,
                target_id,
                target_test,
                y_true_test,
                y_pred,
                normalization_train_days=args.target_train_days,
            )
            finalize_method("lightgbm", metrics, normalized_outputs=(y_true_test, y_pred))
            record_runtime(runtime_rows, target_id, "lightgbm", "fit", fit_duration, device)
            record_runtime(runtime_rows, target_id, "lightgbm", "predict", predict_duration, device)
            record_runtime(runtime_rows, target_id, "lightgbm", "total", fit_duration + predict_duration, device)

        if "lgbm_transfer" in args.methods and source_ids:
            print(f"Target {target_id} | lgbm_transfer")
            x_parts = [x_train_local]
            y_parts = [y_train_local]
            for source_id in source_ids:
                source_train = data.get_dataset(source_id, "train")
                if len(source_train) == 0:
                    continue
                x_source, y_source = build_tabular_matrix(source_train)
                x_parts.append(x_source)
                y_parts.append(y_source)
            fit_start = time.perf_counter()
            lgbm_transfer = fit_lgbm_multioutput(
                np.concatenate(x_parts, axis=0),
                np.concatenate(y_parts, axis=0),
                n_estimators=args.lgbm_estimators,
                random_state=args.seed + 11,
            )
            fit_duration = time.perf_counter() - fit_start
            pred_start = time.perf_counter()
            y_pred = predict_lgbm(lgbm_transfer, x_test)
            predict_duration = time.perf_counter() - pred_start
            metrics = evaluate_task_predictions(
                data,
                target_id,
                target_test,
                y_true_test,
                y_pred,
                normalization_train_days=args.target_train_days,
            )
            finalize_method("lgbm_transfer", metrics, normalized_outputs=(y_true_test, y_pred))
            record_runtime(runtime_rows, target_id, "lgbm_transfer", "fit", fit_duration, device)
            record_runtime(runtime_rows, target_id, "lgbm_transfer", "predict", predict_duration, device)
            record_runtime(runtime_rows, target_id, "lgbm_transfer", "total", fit_duration + predict_duration, device)

        nf_methods = [method for method in args.methods if method in NF_METHODS]
        if nf_methods:
            _, train_model_frame = data.get_frame(target_id, "train", train_days_limit=args.target_train_days)
            _, val_model_frame = data.get_frame(target_id, "val")
            _, test_model_frame = data.get_frame(target_id, "test")
            for method_name in nf_methods:
                print(f"Target {target_id} | {method_name}")
                bundle = fit_nf_model(
                    method_name,
                    train_frame=train_model_frame,
                    val_frame=val_model_frame,
                    spec=data.spec,
                    feature_cols=data.feature_cols,
                    input_size=args.seq_len,
                    h=args.pred_len,
                    seed=args.seed,
                    device=str(device),
                    args=args,
                )
                pred_start = time.perf_counter()
                _, y_pred_raw = predict_nf_windows(
                    bundle,
                    test_frame=test_model_frame,
                    spec=data.spec,
                    feature_cols=data.feature_cols,
                    window_indices=target_test.indices,
                    seq_len=args.seq_len,
                    pred_len=args.pred_len,
                )
                predict_duration = time.perf_counter() - pred_start
                metrics = evaluate_task_predictions_raw(
                    data,
                    target_test,
                    y_true_test_raw,
                    y_pred_raw,
                )
                finalize_method(
                    method_name,
                    metrics,
                    raw_outputs=(y_true_test_raw, y_pred_raw),
                    model=bundle.fitted_model,
                )
                nf_bundles[method_name] = bundle
                record_runtime(runtime_rows, target_id, method_name, "fit", bundle.fit_duration_sec, device)
                record_runtime(runtime_rows, target_id, method_name, "predict", predict_duration, device)
                record_runtime(runtime_rows, target_id, method_name, "total", bundle.fit_duration_sec + predict_duration, device)

        if "safe_patchtst" in args.methods:
            print(f"Target {target_id} | safe_patchtst")
            _, train_model_frame = data.get_frame(target_id, "train", train_days_limit=args.target_train_days)
            _, val_model_frame = data.get_frame(target_id, "val")
            _, test_model_frame = data.get_frame(target_id, "test")
            source_frames = []
            for source_id, similarity in source_pairs:
                _, source_train_frame = data.get_frame(source_id, "train")
                _, source_val_frame = data.get_frame(source_id, "val")
                source_frames.append((source_id, similarity, source_train_frame, source_val_frame))
            safe_patch_fit_start = time.perf_counter()
            safe_patch_bundle = fit_safe_patchtst(
                spec=data.spec,
                feature_cols=data.feature_cols,
                input_size=args.seq_len,
                h=args.pred_len,
                target_train_frame=train_model_frame,
                target_val_frame=val_model_frame,
                target_val_indices=target_val.indices,
                source_frames=source_frames,
                seed=args.seed,
                device=str(device),
                args=args,
                local_bundle=nf_bundles.get("patchtst"),
            )
            fit_duration = time.perf_counter() - safe_patch_fit_start
            pred_start = time.perf_counter()
            safe_patch_outputs = predict_safe_patchtst(
                safe_patch_bundle,
                test_frame=test_model_frame,
                spec=data.spec,
                feature_cols=data.feature_cols,
                window_indices=target_test.indices,
                seq_len=args.seq_len,
                pred_len=args.pred_len,
            )
            predict_duration = time.perf_counter() - pred_start
            y_true = safe_patch_outputs["truths"]
            y_pred = safe_patch_outputs["final"]
            metrics = evaluate_task_predictions_raw(
                data,
                target_test,
                y_true,
                y_pred,
            )
            prediction_extras = {
                "y_local": safe_patch_outputs["local"],
                "transfer_strength": safe_patch_outputs["transfer_strength"],
                "raw_transfer": safe_patch_outputs["raw_transfer"],
                "bounded_transfer": safe_patch_outputs["bounded_transfer"],
                "transfer_delta": safe_patch_outputs["transfer_delta"],
                "residual_budget": safe_patch_outputs["residual_budget"],
                "calibration_alpha": safe_patch_outputs["calibration_alpha"],
                "regime_score": safe_patch_outputs["regime_score"],
                "regime_bin": safe_patch_outputs["regime_bin"],
                "horizon_block": safe_patch_outputs["horizon_block"],
                "source_dispersion": safe_patch_outputs["source_dispersion"],
            }
            finalize_method(
                "safe_patchtst",
                metrics,
                raw_outputs=(y_true, y_pred),
                extras=prediction_extras,
                model=safe_patch_bundle,
            )
            if not args.disable_artifacts:
                safe_source_frame = build_safe_source_frame(
                    spec=data.spec,
                    dataset=target_test,
                    dataset_name=args.dataset,
                    seed=args.seed,
                    target_id=target_id,
                    split="test",
                    source_ids=safe_patch_bundle.source_ids,
                    source_preds=safe_patch_outputs["source_preds"],
                    source_weights=safe_patch_outputs["source_weights"],
                    source_gates=safe_patch_outputs["source_gates"],
                )
                if not safe_source_frame.empty:
                    save_parquet(
                        safe_source_frame,
                        artifact_path(run_dir, "safe_sources", "safe_patchtst", f"target_{target_id}_test.parquet"),
                    )
            fit_duration_total = float(getattr(safe_patch_bundle, "_training_summary", {}).get("duration_sec", fit_duration))
            calibration_duration = float(getattr(safe_patch_bundle, "_training_summary", {}).get("calibration_duration_sec", 0.0))
            source_fit_duration = float(getattr(safe_patch_bundle, "_training_summary", {}).get("source_fit_duration_sec", 0.0))
            if getattr(safe_patch_bundle, "reused_local_bundle", False):
                record_runtime(runtime_rows, target_id, "safe_patchtst", "local_fit_reused", 0.0, device)
            else:
                local_fit_duration = float(getattr(safe_patch_bundle.local_bundle, "fit_duration_sec", 0.0))
                record_runtime(runtime_rows, target_id, "safe_patchtst", "local_fit", local_fit_duration, device)
            record_runtime(runtime_rows, target_id, "safe_patchtst", "source_fit", source_fit_duration, device)
            record_runtime(runtime_rows, target_id, "safe_patchtst", "calibrate", calibration_duration, device)
            record_runtime(runtime_rows, target_id, "safe_patchtst", "predict", predict_duration, device)
            record_runtime(runtime_rows, target_id, "safe_patchtst", "total", fit_duration_total + predict_duration, device)

        if "safe_fedformer" in args.methods:
            print(f"Target {target_id} | safe_fedformer")
            _, train_model_frame = data.get_frame(target_id, "train", train_days_limit=args.target_train_days)
            _, val_model_frame = data.get_frame(target_id, "val")
            _, test_model_frame = data.get_frame(target_id, "test")
            source_frames = []
            for source_id, similarity in source_pairs:
                _, source_train_frame = data.get_frame(source_id, "train")
                _, source_val_frame = data.get_frame(source_id, "val")
                source_frames.append((source_id, similarity, source_train_frame, source_val_frame))
            safe_fed_fit_start = time.perf_counter()
            safe_fed_bundle = fit_safe_fedformer(
                spec=data.spec,
                feature_cols=data.feature_cols,
                input_size=args.seq_len,
                h=args.pred_len,
                target_train_frame=train_model_frame,
                target_val_frame=val_model_frame,
                target_val_indices=target_val.indices,
                source_frames=source_frames,
                seed=args.seed,
                device=str(device),
                args=args,
                local_bundle=nf_bundles.get("fedformer"),
            )
            fit_duration = time.perf_counter() - safe_fed_fit_start
            pred_start = time.perf_counter()
            safe_fed_outputs = predict_safe_fedformer(
                safe_fed_bundle,
                test_frame=test_model_frame,
                spec=data.spec,
                feature_cols=data.feature_cols,
                window_indices=target_test.indices,
                seq_len=args.seq_len,
                pred_len=args.pred_len,
            )
            predict_duration = time.perf_counter() - pred_start
            y_true = safe_fed_outputs["truths"]
            y_pred = safe_fed_outputs["final"]
            metrics = evaluate_task_predictions_raw(
                data,
                target_test,
                y_true,
                y_pred,
            )
            prediction_extras = {
                "y_local": safe_fed_outputs["local"],
                "transfer_strength": safe_fed_outputs["transfer_strength"],
                "raw_transfer": safe_fed_outputs["raw_transfer"],
                "bounded_transfer": safe_fed_outputs["bounded_transfer"],
                "transfer_delta": safe_fed_outputs["transfer_delta"],
                "residual_budget": safe_fed_outputs["residual_budget"],
                "calibration_alpha": safe_fed_outputs["calibration_alpha"],
                "regime_score": safe_fed_outputs["regime_score"],
                "regime_bin": safe_fed_outputs["regime_bin"],
                "horizon_block": safe_fed_outputs["horizon_block"],
                "source_dispersion": safe_fed_outputs["source_dispersion"],
            }
            finalize_method(
                "safe_fedformer",
                metrics,
                raw_outputs=(y_true, y_pred),
                extras=prediction_extras,
                model=safe_fed_bundle,
            )
            if not args.disable_artifacts:
                safe_source_frame = build_safe_source_frame(
                    spec=data.spec,
                    dataset=target_test,
                    dataset_name=args.dataset,
                    seed=args.seed,
                    target_id=target_id,
                    split="test",
                    source_ids=safe_fed_bundle.source_ids,
                    source_preds=safe_fed_outputs["source_preds"],
                    source_weights=safe_fed_outputs["source_weights"],
                    source_gates=safe_fed_outputs["source_gates"],
                )
                if not safe_source_frame.empty:
                    save_parquet(
                        safe_source_frame,
                        artifact_path(run_dir, "safe_sources", "safe_fedformer", f"target_{target_id}_test.parquet"),
                    )
            fit_duration_total = float(getattr(safe_fed_bundle, "_training_summary", {}).get("duration_sec", fit_duration))
            calibration_duration = float(getattr(safe_fed_bundle, "_training_summary", {}).get("calibration_duration_sec", 0.0))
            source_fit_duration = float(getattr(safe_fed_bundle, "_training_summary", {}).get("source_fit_duration_sec", 0.0))
            if getattr(safe_fed_bundle, "reused_local_bundle", False):
                record_runtime(runtime_rows, target_id, "safe_fedformer", "local_fit_reused", 0.0, device)
            else:
                local_fit_duration = float(getattr(safe_fed_bundle.local_bundle, "fit_duration_sec", 0.0))
                record_runtime(runtime_rows, target_id, "safe_fedformer", "local_fit", local_fit_duration, device)
            record_runtime(runtime_rows, target_id, "safe_fedformer", "source_fit", source_fit_duration, device)
            record_runtime(runtime_rows, target_id, "safe_fedformer", "calibrate", calibration_duration, device)
            record_runtime(runtime_rows, target_id, "safe_fedformer", "predict", predict_duration, device)
            record_runtime(runtime_rows, target_id, "safe_fedformer", "total", fit_duration_total + predict_duration, device)

        if any(method in args.methods for method in ("tcn", "safe_tcn")):
            print(f"Target {target_id} | tcn")
            local_model = train_local_model(
                make_model(args, input_dim=len(data.feature_cols) + 1, profile_dim=data.profile_dim),
                train_loader,
                val_loader,
                target_profile,
                device,
                trainable_parts="all",
                epochs=args.finetune_epochs,
                lr=args.lr_finetune,
                weight_decay=args.weight_decay,
                patience=args.patience,
            )
            pred_start = time.perf_counter()
            y_true, y_pred = collect_local_predictions(local_model, test_loader, target_profile, device)
            predict_duration = time.perf_counter() - pred_start
            metrics = evaluate_task_predictions(
                data,
                target_id,
                target_test,
                y_true,
                y_pred,
                normalization_train_days=args.target_train_days,
            )
            finalize_method("tcn", metrics, normalized_outputs=(y_true, y_pred), model=local_model)
            tcn_fit_duration = float(getattr(local_model, "_training_summary", {}).get("duration_sec", 0.0))
            record_runtime(runtime_rows, target_id, "tcn", "fit", tcn_fit_duration, device)
            record_runtime(runtime_rows, target_id, "tcn", "predict", predict_duration, device)
            record_runtime(runtime_rows, target_id, "tcn", "total", tcn_fit_duration + predict_duration, device)

        if "tcn_multi_task" in args.methods:
            print(f"Target {target_id} | tcn_multi_task")
            multi_task_ids = source_ids + [target_id]
            multi_train = data.make_multitask_dataset(multi_task_ids, "train", stride=args.train_stride)
            multi_model = train_multitask_target_model(
                make_model(args, input_dim=len(data.feature_cols) + 1, profile_dim=data.profile_dim),
                loader_for(
                    multi_train,
                    args.batch_size,
                    True,
                    args.num_workers,
                    device,
                    prefetch_factor=args.prefetch_factor,
                    pin_memory=not args.disable_pin_memory,
                    persistent_workers=not args.disable_persistent_workers,
                ),
                val_loader,
                profile_bank_tensor.to(device),
                target_profile,
                device,
                epochs=args.pretrain_epochs,
                lr=args.lr_pretrain,
                weight_decay=args.weight_decay,
                patience=args.patience,
            )
            pred_start = time.perf_counter()
            y_true, y_pred = collect_local_predictions(multi_model, test_loader, target_profile, device)
            predict_duration = time.perf_counter() - pred_start
            metrics = evaluate_task_predictions(
                data,
                target_id,
                target_test,
                y_true,
                y_pred,
                normalization_train_days=args.target_train_days,
            )
            finalize_method("tcn_multi_task", metrics, normalized_outputs=(y_true, y_pred), model=multi_model)
            multi_fit_duration = float(getattr(multi_model, "_training_summary", {}).get("duration_sec", 0.0))
            record_runtime(runtime_rows, target_id, "tcn_multi_task", "fit", multi_fit_duration, device)
            record_runtime(runtime_rows, target_id, "tcn_multi_task", "predict", predict_duration, device)
            record_runtime(runtime_rows, target_id, "tcn_multi_task", "total", multi_fit_duration + predict_duration, device)

        if any(method in args.methods for method in ("tcn_fine_tune", "safe_tcn")) and source_ids:
            print(f"Target {target_id} | multitask_pretrain")
            pretrain_train = data.make_multitask_dataset(source_ids, "train", stride=args.train_stride)
            pretrain_val = data.make_multitask_dataset(source_ids, "val", stride=args.eval_stride)
            pretrain_model = train_multitask_pretrain(
                make_model(args, input_dim=len(data.feature_cols) + 1, profile_dim=data.profile_dim),
                loader_for(
                    pretrain_train,
                    args.batch_size,
                    True,
                    args.num_workers,
                    device,
                    prefetch_factor=args.prefetch_factor,
                    pin_memory=not args.disable_pin_memory,
                    persistent_workers=not args.disable_persistent_workers,
                ),
                loader_for(
                    pretrain_val,
                    args.batch_size,
                    False,
                    args.num_workers,
                    device,
                    prefetch_factor=args.prefetch_factor,
                    pin_memory=not args.disable_pin_memory,
                    persistent_workers=not args.disable_persistent_workers,
                ),
                profile_bank_tensor.to(device),
                device,
                epochs=args.pretrain_epochs,
                lr=args.lr_pretrain,
                weight_decay=args.weight_decay,
                patience=args.patience,
            )
            pretrain_duration = float(getattr(pretrain_model, "_training_summary", {}).get("duration_sec", 0.0))
            if not args.disable_artifacts:
                save_training_history_artifact(run_dir, args.dataset, args.seed, target_id, "multitask_pretrain", pretrain_model)

            if "tcn_fine_tune" in args.methods:
                print(f"Target {target_id} | tcn_fine_tune")
                fine_tune_model = make_model(args, input_dim=len(data.feature_cols) + 1, profile_dim=data.profile_dim)
                fine_tune_model.load_state_dict(pretrain_model.state_dict())
                fine_tune_model = train_local_model(
                    fine_tune_model,
                    train_loader,
                    val_loader,
                    target_profile,
                    device,
                    trainable_parts=args.fine_tune_mode,
                    epochs=args.finetune_epochs,
                    lr=args.lr_finetune,
                    weight_decay=args.weight_decay,
                    patience=args.patience,
                )
                pred_start = time.perf_counter()
                y_true, y_pred = collect_local_predictions(fine_tune_model, test_loader, target_profile, device)
                predict_duration = time.perf_counter() - pred_start
                metrics = evaluate_task_predictions(
                    data,
                    target_id,
                    target_test,
                    y_true,
                    y_pred,
                    normalization_train_days=args.target_train_days,
                )
                finalize_method("tcn_fine_tune", metrics, normalized_outputs=(y_true, y_pred), model=fine_tune_model)
                fine_tune_duration = float(getattr(fine_tune_model, "_training_summary", {}).get("duration_sec", 0.0))
                record_runtime(runtime_rows, target_id, "tcn_fine_tune", "pretrain", pretrain_duration, device)
                record_runtime(runtime_rows, target_id, "tcn_fine_tune", "fit", fine_tune_duration, device)
                record_runtime(runtime_rows, target_id, "tcn_fine_tune", "predict", predict_duration, device)
                record_runtime(runtime_rows, target_id, "tcn_fine_tune", "total", pretrain_duration + fine_tune_duration + predict_duration, device)

            if "safe_tcn" in args.methods:
                print(f"Target {target_id} | safe_tcn")
                if local_model is None:
                    raise RuntimeError("safe_tcn requires a trained local backbone.")
                safe_target_model = make_model(args, input_dim=len(data.feature_cols) + 1, profile_dim=data.profile_dim)
                safe_target_model.load_state_dict(local_model.state_dict())
                safe_model = train_safe_tcn(
                    SafeTCNForecaster(
                        target_model=safe_target_model,
                        source_model=pretrain_model,
                        relation_dim=relation_np.shape[1] if relation_np.size else 8,
                        num_sources=len(source_ids),
                        gate_hidden_dim=args.gate_hidden_dim,
                        dropout=args.dropout,
                        residual_cap_scale=args.residual_cap_scale,
                        residual_cap_floor=args.residual_cap_floor,
                    ),
                    train_loader,
                    val_loader,
                    target_profile=torch.from_numpy(target_profile_np),
                    source_profiles=torch.from_numpy(source_profiles_np),
                    relation_features=torch.from_numpy(relation_np),
                    device=device,
                    epochs=args.safe_epochs,
                    lr=args.lr_safe,
                    weight_decay=args.weight_decay,
                    gate_lambda=args.gate_lambda,
                    safety_lambda=args.safety_lambda,
                    sparsity_lambda=args.sparsity_lambda,
                    relation_lambda=args.relation_lambda,
                    budget_lambda=args.budget_lambda,
                    harm_margin=args.harm_margin,
                    patience=args.patience,
                )
                calibrate_start = time.perf_counter()
                safe_model = calibrate_safe_tcn(
                    safe_model,
                    val_loader,
                    target_profile=torch.from_numpy(target_profile_np),
                    source_profiles=torch.from_numpy(source_profiles_np),
                    relation_features=torch.from_numpy(relation_np),
                    device=device,
                    harm_limit=args.calibration_harm_limit,
                    grid_size=args.calibration_grid_size,
                )
                calibrate_duration = time.perf_counter() - calibrate_start
                pred_start = time.perf_counter()
                safe_outputs = collect_safe_outputs(
                    safe_model,
                    test_loader,
                    target_profile=torch.from_numpy(target_profile_np),
                    source_profiles=torch.from_numpy(source_profiles_np),
                    relation_features=torch.from_numpy(relation_np),
                    device=device,
                )
                predict_duration = time.perf_counter() - pred_start
                y_true = safe_outputs["truths"]
                y_pred = safe_outputs["final"]
                metrics = evaluate_task_predictions(
                    data,
                    target_id,
                    target_test,
                    y_true,
                    y_pred,
                    normalization_train_days=args.target_train_days,
                )
                target_pred_denorm = denormalize_with_stats(data, target_id, safe_outputs["target"], normalization_train_days=args.target_train_days)
                final_pred_denorm = denormalize_with_stats(data, target_id, safe_outputs["final"], normalization_train_days=args.target_train_days)
                source_pred_denorm = denormalize_with_stats(
                    data,
                    target_id,
                    safe_outputs["source_preds"],
                    normalization_train_days=args.target_train_days,
                )
                prediction_extras = {
                    "y_local": target_pred_denorm,
                    "transfer_strength": safe_outputs["transfer_strength"],
                    "raw_transfer": safe_outputs["raw_transfer"] * target_std,
                    "bounded_transfer": safe_outputs["bounded_transfer"] * target_std,
                    "transfer_delta": final_pred_denorm - target_pred_denorm,
                    "residual_budget": safe_outputs["residual_budget"] * target_std,
                    "calibration_alpha": safe_outputs["calibration_alpha"],
                }
                finalize_method("safe_tcn", metrics, normalized_outputs=(y_true, y_pred), extras=prediction_extras, model=safe_model)
                if not args.disable_artifacts:
                    safe_source_frame = build_safe_source_frame(
                        spec=data.spec,
                        dataset=target_test,
                        dataset_name=args.dataset,
                        seed=args.seed,
                        target_id=target_id,
                        split="test",
                        source_ids=source_ids,
                        source_preds=source_pred_denorm,
                        source_weights=safe_outputs["source_weights"],
                        source_gates=safe_outputs["source_gates"],
                    )
                    if not safe_source_frame.empty:
                        save_parquet(
                            safe_source_frame,
                            artifact_path(run_dir, "safe_sources", "safe_tcn", f"target_{target_id}_test.parquet"),
                        )
                safe_fit_duration = float(getattr(safe_model, "_training_summary", {}).get("duration_sec", 0.0))
                record_runtime(runtime_rows, target_id, "safe_tcn", "pretrain", pretrain_duration, device)
                record_runtime(runtime_rows, target_id, "safe_tcn", "fit", safe_fit_duration, device)
                record_runtime(runtime_rows, target_id, "safe_tcn", "calibrate", calibrate_duration, device)
                record_runtime(runtime_rows, target_id, "safe_tcn", "predict", predict_duration, device)
                record_runtime(
                    runtime_rows,
                    target_id,
                    "safe_tcn",
                    "total",
                    pretrain_duration + safe_fit_duration + calibrate_duration + predict_duration,
                    device,
                )

        add_transfer_safety(target_results, baseline_method="tcn")
        add_method_relative_safety(
            target_results,
            method="safe_patchtst",
            baseline_method="patchtst",
            prefix="LOCAL_",
        )
        add_method_relative_safety(
            target_results,
            method="safe_fedformer",
            baseline_method="fedformer",
            prefix="LOCAL_",
        )
        per_target[target_id] = target_results

    summary = summarize_results(per_target)
    payload = {
        "config": vars(args),
        "per_target": {
            int(task_id): {method: drop_hidden(metrics) if isinstance(metrics, dict) else metrics for method, metrics in rows.items()}
            for task_id, rows in per_target.items()
        },
        "summary": summary,
    }
    save_json(payload, os.path.join(run_dir, "report.json"))
    if not args.disable_artifacts:
        metrics_frame = build_per_target_metrics_frame(args.dataset, args.seed, per_target)
        if not metrics_frame.empty:
            save_parquet(metrics_frame, artifact_path(run_dir, "per_target_metrics.parquet"))
        if source_selection_frames:
            source_selection = pd.concat(source_selection_frames, ignore_index=True)
            save_parquet(source_selection, artifact_path(run_dir, "source_selection.parquet"))
        runtime_frame = build_runtime_frame(args.dataset, args.seed, runtime_rows)
        if not runtime_frame.empty:
            save_parquet(runtime_frame, artifact_path(run_dir, "method_runtime.parquet"))

    print("\nSummary")
    for method, metrics in summary.items():
        print(f"{method:<14} RMSE={metrics.get('RMSE', math.nan):.4f} Harm={metrics.get('WINDOW_HARM_RATE', math.nan):.4f}")
    return payload


def main() -> None:
    run_experiment(build_parser().parse_args())


if __name__ == "__main__":
    main()
