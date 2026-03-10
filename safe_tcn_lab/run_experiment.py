from __future__ import annotations

import argparse
import json
import math
import os
import sys
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
    build_safe_source_frame,
    build_source_selection_frame,
    build_training_history_frame,
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
from safe_tcn_lab.metrics import add_transfer_safety, evaluate_predictions, summarize_results
from safe_tcn_lab.models import SafeTCNForecaster, TaskConditionedTCN
from safe_tcn_lab.train import (
    calibrate_safe_tcn,
    collect_local_predictions,
    collect_safe_outputs,
    set_seed,
    train_local_model,
    train_multitask_pretrain,
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
    "ridge_local",
    "lgbm_local",
    "lgbm_transfer",
    "tcn_local",
    "fine_tune",
    "safe_tcn",
]


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
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--lgbm_estimators", type=int, default=120)
    parser.add_argument("--output_root", default="safe_tcn_lab/outputs")
    parser.add_argument("--disable_artifacts", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def resolve_methods(methods: List[str] | None) -> List[str]:
    if not methods:
        return list(CORE_METHODS)
    expanded: List[str] = []
    for method in methods:
        if method == "core":
            expanded.extend(CORE_METHODS)
        else:
            expanded.append(method)
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


def loader_for(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


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


def save_training_history_artifact(run_dir: str, dataset_name: str, seed: int, target_id: int, method: str, model) -> None:
    frame = build_training_history_frame(dataset_name, seed, target_id, method, model)
    if frame.empty:
        return
    save_parquet(frame, artifact_path(run_dir, "training_history", method, f"target_{target_id}.parquet"))


def drop_hidden(metrics: Dict[str, float]) -> Dict[str, float]:
    return {key: value for key, value in metrics.items() if not key.startswith("_")}


def run_experiment(args: argparse.Namespace) -> Dict[str, object]:
    defaults = DATASET_DEFAULTS[args.dataset]
    args.parquet_path = args.parquet_path or defaults["parquet_path"]
    args.seq_len = args.seq_len or defaults["seq_len"]
    args.pred_len = args.pred_len or defaults["pred_len"]
    args.target_ids = args.target_ids or defaults["target_ids"]
    args.max_sources = args.max_sources if args.max_sources is not None else defaults["max_sources"]
    args.min_similarity = args.min_similarity if args.min_similarity is not None else defaults["min_similarity"]
    args.methods = resolve_methods(args.methods)

    if args.smoke:
        args.target_ids = args.target_ids[:1]
        args.pretrain_epochs = min(args.pretrain_epochs, 2)
        args.finetune_epochs = min(args.finetune_epochs, 2)
        args.safe_epochs = min(args.safe_epochs, 2)
        args.train_stride = max(args.train_stride, 24)
        args.eval_stride = max(args.eval_stride, 24)

    set_seed(args.seed)
    device = get_device(args.device)
    spec = get_dataset_spec(args.dataset)
    data = MultiTaskForecastData(
        spec=spec,
        parquet_path=args.parquet_path,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        n_pc_bins=args.n_pc_bins,
        max_rows_per_task=args.max_rows_per_task,
    ).load()

    per_target: Dict[int, Dict[str, Dict[str, float]]] = {}
    run_dir = os.path.join(args.output_root, f"{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    save_json(vars(args), os.path.join(run_dir, "config.json"))
    source_selection_frames: list[pd.DataFrame] = []

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

        train_loader = loader_for(target_train, args.batch_size, True, args.num_workers)
        val_loader = loader_for(target_val, args.batch_size, False, args.num_workers)
        test_loader = loader_for(target_test, args.batch_size, False, args.num_workers)

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
        x_train_local, y_train_local = build_tabular_matrix(target_train)
        x_test, _ = build_tabular_matrix(target_test)
        local_model = None

        if "persistence" in args.methods:
            print(f"Target {target_id} | persistence")
            y_true, y_pred = build_persistence_predictions(target_test)
            target_results["persistence"] = evaluate_task_predictions(
                data,
                target_id,
                target_test,
                y_true,
                y_pred,
                normalization_train_days=args.target_train_days,
            )
            if not args.disable_artifacts:
                save_prediction_artifact(
                    run_dir,
                    data,
                    target_id,
                    target_test,
                    args.seed,
                    "persistence",
                    "test",
                    y_true,
                    y_pred,
                    normalization_train_days=args.target_train_days,
                )

        if "ridge_local" in args.methods:
            print(f"Target {target_id} | ridge_local")
            ridge_local = fit_ridge_multioutput(x_train_local, y_train_local, alpha=args.ridge_alpha)
            y_pred = ridge_local.predict(x_test).astype(np.float32)
            target_results["ridge_local"] = evaluate_task_predictions(
                data,
                target_id,
                target_test,
                y_true_test,
                y_pred,
                normalization_train_days=args.target_train_days,
            )
            if not args.disable_artifacts:
                save_prediction_artifact(
                    run_dir,
                    data,
                    target_id,
                    target_test,
                    args.seed,
                    "ridge_local",
                    "test",
                    y_true_test,
                    y_pred,
                    normalization_train_days=args.target_train_days,
                )

        if "lgbm_local" in args.methods:
            print(f"Target {target_id} | lgbm_local")
            lgbm_local = fit_lgbm_multioutput(
                x_train_local,
                y_train_local,
                n_estimators=args.lgbm_estimators,
                random_state=args.seed,
            )
            y_pred = predict_lgbm(lgbm_local, x_test)
            target_results["lgbm_local"] = evaluate_task_predictions(
                data,
                target_id,
                target_test,
                y_true_test,
                y_pred,
                normalization_train_days=args.target_train_days,
            )
            if not args.disable_artifacts:
                save_prediction_artifact(
                    run_dir,
                    data,
                    target_id,
                    target_test,
                    args.seed,
                    "lgbm_local",
                    "test",
                    y_true_test,
                    y_pred,
                    normalization_train_days=args.target_train_days,
                )

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
            lgbm_transfer = fit_lgbm_multioutput(
                np.concatenate(x_parts, axis=0),
                np.concatenate(y_parts, axis=0),
                n_estimators=args.lgbm_estimators,
                random_state=args.seed + 11,
            )
            y_pred = predict_lgbm(lgbm_transfer, x_test)
            target_results["lgbm_transfer"] = evaluate_task_predictions(
                data,
                target_id,
                target_test,
                y_true_test,
                y_pred,
                normalization_train_days=args.target_train_days,
            )
            if not args.disable_artifacts:
                save_prediction_artifact(
                    run_dir,
                    data,
                    target_id,
                    target_test,
                    args.seed,
                    "lgbm_transfer",
                    "test",
                    y_true_test,
                    y_pred,
                    normalization_train_days=args.target_train_days,
                )

        if any(method in args.methods for method in ("tcn_local", "safe_tcn")):
            print(f"Target {target_id} | tcn_local train")
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
            y_true, y_pred = collect_local_predictions(local_model, test_loader, target_profile, device)
            target_results["tcn_local"] = evaluate_task_predictions(
                data,
                target_id,
                target_test,
                y_true,
                y_pred,
                normalization_train_days=args.target_train_days,
            )
            if not args.disable_artifacts:
                save_training_history_artifact(run_dir, args.dataset, args.seed, target_id, "tcn_local", local_model)
                save_prediction_artifact(
                    run_dir,
                    data,
                    target_id,
                    target_test,
                    args.seed,
                    "tcn_local",
                    "test",
                    y_true,
                    y_pred,
                    normalization_train_days=args.target_train_days,
                )

        if any(method in args.methods for method in ("fine_tune", "safe_tcn")) and source_ids:
            print(f"Target {target_id} | multitask_pretrain")
            pretrain_train = data.make_multitask_dataset(source_ids, "train", stride=args.train_stride)
            pretrain_val = data.make_multitask_dataset(source_ids, "val", stride=args.eval_stride)
            pretrain_model = train_multitask_pretrain(
                make_model(args, input_dim=len(data.feature_cols) + 1, profile_dim=data.profile_dim),
                loader_for(pretrain_train, args.batch_size, True, args.num_workers),
                loader_for(pretrain_val, args.batch_size, False, args.num_workers),
                torch.from_numpy(data.get_profiles(data.task_ids)).to(device),
                device,
                epochs=args.pretrain_epochs,
                lr=args.lr_pretrain,
                weight_decay=args.weight_decay,
                patience=args.patience,
            )
            if not args.disable_artifacts:
                save_training_history_artifact(run_dir, args.dataset, args.seed, target_id, "multitask_pretrain", pretrain_model)

            if "fine_tune" in args.methods:
                print(f"Target {target_id} | fine_tune")
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
                y_true, y_pred = collect_local_predictions(fine_tune_model, test_loader, target_profile, device)
                target_results["fine_tune"] = evaluate_task_predictions(
                    data,
                    target_id,
                    target_test,
                    y_true,
                    y_pred,
                    normalization_train_days=args.target_train_days,
                )
                if not args.disable_artifacts:
                    save_training_history_artifact(run_dir, args.dataset, args.seed, target_id, "fine_tune", fine_tune_model)
                    save_prediction_artifact(
                        run_dir,
                        data,
                        target_id,
                        target_test,
                        args.seed,
                        "fine_tune",
                        "test",
                        y_true,
                        y_pred,
                        normalization_train_days=args.target_train_days,
                    )

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
                safe_outputs = collect_safe_outputs(
                    safe_model,
                    test_loader,
                    target_profile=torch.from_numpy(target_profile_np),
                    source_profiles=torch.from_numpy(source_profiles_np),
                    relation_features=torch.from_numpy(relation_np),
                    device=device,
                )
                y_true = safe_outputs["truths"]
                y_pred = safe_outputs["final"]
                target_results["safe_tcn"] = evaluate_task_predictions(
                    data,
                    target_id,
                    target_test,
                    y_true,
                    y_pred,
                    normalization_train_days=args.target_train_days,
                )
                if not args.disable_artifacts:
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
                    save_training_history_artifact(run_dir, args.dataset, args.seed, target_id, "safe_tcn", safe_model)
                    save_prediction_artifact(
                        run_dir,
                        data,
                        target_id,
                        target_test,
                        args.seed,
                        "safe_tcn",
                        "test",
                        y_true,
                        y_pred,
                        normalization_train_days=args.target_train_days,
                        extras=prediction_extras,
                    )
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

        add_transfer_safety(target_results, baseline_method="tcn_local")
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

    print("\nSummary")
    for method, metrics in summary.items():
        print(f"{method:<14} RMSE={metrics.get('RMSE', math.nan):.4f} Harm={metrics.get('WINDOW_HARM_RATE', math.nan):.4f}")
    return payload


def main() -> None:
    run_experiment(build_parser().parse_args())


if __name__ == "__main__":
    main()
