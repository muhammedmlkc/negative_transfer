from __future__ import annotations

import time
import logging
import warnings
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from safe_tcn_lab.data import DatasetSpec

try:
    from neuralforecast import NeuralForecast
    from neuralforecast import models as nf_models
except Exception:  # pragma: no cover - handled at runtime when methods are requested
    NeuralForecast = None
    nf_models = None


NF_METHODS = {
    "lstm",
    "gru",
    "dlinear",
    "nbeats",
    "informer",
    "fedformer",
    "patchtst",
    "timesnet",
    "itransformer",
}

FUTR_EXOG_METHODS = {"lstm", "gru", "informer", "fedformer", "timesnet"}
HIST_EXOG_METHODS = set()


@dataclass
class NFModelBundle:
    method: str
    nf: NeuralForecast
    fitted_model: object
    fit_duration_sec: float
    supports_future_exog: bool
    supports_hist_exog: bool
    prediction_column: str


def require_neuralforecast() -> None:
    if NeuralForecast is None or nf_models is None:
        raise ImportError(
            "NeuralForecast baselines requested but neuralforecast is unavailable. "
            "Install neuralforecast and compatible torch/lightning packages first."
        )
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*val_check_steps is greater than max_steps.*")
    warnings.filterwarnings("ignore", message=".*LeafSpec.*deprecated.*")


def dataset_freq(spec: DatasetSpec) -> str:
    if spec.name == "gefcom":
        return "h"
    if spec.name == "sdwpf":
        return "10min"
    raise ValueError(f"Unsupported dataset '{spec.name}'.")


def build_nf_frame(frame: pd.DataFrame, spec: DatasetSpec, feature_cols: Sequence[str], unique_id: str) -> pd.DataFrame:
    cols = [spec.time_col, spec.target_col, *feature_cols]
    available = [col for col in cols if col in frame.columns]
    out = frame[available].copy()
    out = out.rename(columns={spec.time_col: "ds", spec.target_col: "y"})
    out.insert(0, "unique_id", unique_id)
    return out.reset_index(drop=True)


def _history_summary_rows(model, duration_sec: float) -> list[dict[str, float]]:
    train_traj = list(getattr(model, "train_trajectories", []) or [])
    valid_traj = list(getattr(model, "valid_trajectories", []) or [])
    max_len = max(len(train_traj), len(valid_traj))
    rows = []
    best_val = float("inf")
    best_epoch = 0
    for idx in range(max_len):
        train_step, train_loss = train_traj[idx] if idx < len(train_traj) else (idx, np.nan)
        valid_step, valid_loss = valid_traj[idx] if idx < len(valid_traj) else (idx, np.nan)
        if np.isfinite(valid_loss) and valid_loss < best_val:
            best_val = float(valid_loss)
            best_epoch = idx + 1
        rows.append(
            {
                "epoch": float(idx + 1),
                "train_loss": float(train_loss) if np.isfinite(train_loss) else np.nan,
                "val_loss": float(valid_loss) if np.isfinite(valid_loss) else np.nan,
                "best_val_loss_so_far": float(best_val) if np.isfinite(best_val) else np.nan,
            }
        )
    if not rows:
        metrics = getattr(model, "metrics", {}) or {}
        best_val = float(metrics.get("valid_loss", np.nan))
        rows.append(
            {
                "epoch": 1.0,
                "train_loss": float(metrics.get("train_loss_epoch", np.nan)),
                "val_loss": best_val,
                "best_val_loss_so_far": best_val,
            }
        )
        best_epoch = 1
    model._training_history = rows
    model._training_summary = {
        "best_epoch": int(best_epoch or 1),
        "best_val_loss": float(best_val if np.isfinite(best_val) else np.nan),
        "duration_sec": float(duration_sec),
        "epochs_ran": int(len(rows)),
    }
    return rows


def make_nf_model(method: str, h: int, input_size: int, feature_cols: Sequence[str], seed: int, device: str, args) -> object:
    require_neuralforecast()
    alias = method
    common = {
        "h": h,
        "input_size": input_size,
        "max_steps": args.nf_max_steps,
        "learning_rate": args.nf_learning_rate,
        "early_stop_patience_steps": args.nf_early_stop_patience_steps,
        "val_check_steps": args.nf_val_check_steps,
        "batch_size": args.nf_batch_size,
        "valid_batch_size": args.nf_batch_size,
        "windows_batch_size": args.nf_windows_batch_size,
        "inference_windows_batch_size": args.nf_inference_windows_batch_size,
        "random_seed": seed,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
        "enable_checkpointing": False,
        "num_sanity_val_steps": 0,
        "log_every_n_steps": max(1, min(50, args.nf_val_check_steps)),
        "deterministic": True,
        "accelerator": "gpu" if device.startswith("cuda") else "cpu",
        "devices": 1,
        "alias": alias,
    }
    if method in FUTR_EXOG_METHODS:
        common["futr_exog_list"] = list(feature_cols)
    if method in HIST_EXOG_METHODS:
        common["hist_exog_list"] = list(feature_cols)

    if method == "lstm":
        return nf_models.LSTM(
            encoder_hidden_size=args.nf_hidden_size,
            decoder_hidden_size=args.nf_hidden_size,
            encoder_n_layers=args.nf_num_layers,
            decoder_layers=max(1, args.nf_num_layers // 2),
            encoder_dropout=args.nf_dropout,
            **common,
        )
    if method == "gru":
        return nf_models.GRU(
            encoder_hidden_size=args.nf_hidden_size,
            decoder_hidden_size=args.nf_hidden_size,
            encoder_n_layers=args.nf_num_layers,
            decoder_layers=max(1, args.nf_num_layers // 2),
            encoder_dropout=args.nf_dropout,
            **common,
        )
    if method == "dlinear":
        moving_avg_window = max(5, min(25, h))
        if moving_avg_window % 2 == 0:
            moving_avg_window += 1
        return nf_models.DLinear(
            moving_avg_window=moving_avg_window,
            **common,
        )
    if method == "nbeats":
        return nf_models.NBEATS(
            mlp_units=[[args.nf_hidden_size, args.nf_hidden_size]] * 3,
            n_blocks=[1, 1, 1],
            **common,
        )
    if method == "informer":
        return nf_models.Informer(
            hidden_size=args.nf_hidden_size,
            encoder_layers=args.nf_num_layers,
            dropout=args.nf_dropout,
            **common,
        )
    if method == "fedformer":
        return nf_models.FEDformer(
            hidden_size=args.nf_hidden_size,
            encoder_layers=args.nf_num_layers,
            dropout=args.nf_dropout,
            **common,
        )
    if method == "patchtst":
        return nf_models.PatchTST(
            hidden_size=args.nf_hidden_size,
            encoder_layers=args.nf_num_layers,
            n_heads=args.nf_n_heads,
            dropout=args.nf_dropout,
            patch_len=args.nf_patch_len,
            stride=max(1, args.nf_patch_len // 2),
            **common,
        )
    if method == "timesnet":
        return nf_models.TimesNet(
            hidden_size=args.nf_hidden_size,
            encoder_layers=args.nf_num_layers,
            dropout=args.nf_dropout,
            **common,
        )
    if method == "itransformer":
        return nf_models.iTransformer(
            n_series=1,
            hidden_size=max(64, args.nf_hidden_size),
            e_layers=args.nf_num_layers,
            n_heads=args.nf_n_heads,
            dropout=args.nf_dropout,
            **common,
        )
    raise ValueError(f"Unsupported NeuralForecast method '{method}'.")


def fit_nf_model(
    method: str,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    spec: DatasetSpec,
    feature_cols: Sequence[str],
    input_size: int,
    h: int,
    seed: int,
    device: str,
    args,
) -> NFModelBundle:
    fit_frame = pd.concat(
        [
            build_nf_frame(train_frame, spec, feature_cols, "target"),
            build_nf_frame(val_frame, spec, feature_cols, "target"),
        ],
        ignore_index=True,
    )
    model = make_nf_model(method, h=h, input_size=input_size, feature_cols=feature_cols, seed=seed, device=device, args=args)
    nf = NeuralForecast(models=[model], freq=dataset_freq(spec))
    start = time.perf_counter()
    nf.fit(fit_frame, val_size=len(val_frame), verbose=False)
    duration = time.perf_counter() - start
    fitted_model = nf.models[0]
    _history_summary_rows(fitted_model, duration)
    return NFModelBundle(
        method=method,
        nf=nf,
        fitted_model=fitted_model,
        fit_duration_sec=duration,
        supports_future_exog=method in FUTR_EXOG_METHODS,
        supports_hist_exog=method in HIST_EXOG_METHODS,
        prediction_column=method,
    )


def predict_nf_windows(
    bundle: NFModelBundle,
    test_frame: pd.DataFrame,
    spec: DatasetSpec,
    feature_cols: Sequence[str],
    window_indices: Sequence[int],
    seq_len: int,
    pred_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    obs_parts = []
    futr_parts = []
    truths = []
    for window_id, start in enumerate(window_indices):
        history = test_frame.iloc[start : start + seq_len].copy()
        future = test_frame.iloc[start + seq_len : start + seq_len + pred_len].copy()
        uid = f"window_{window_id}"

        obs = pd.DataFrame(
            {
                "unique_id": uid,
                "ds": pd.to_datetime(history[spec.time_col]).to_numpy(),
                "y": history[spec.target_col].to_numpy(dtype=np.float32),
            }
        )
        if bundle.supports_future_exog or bundle.supports_hist_exog:
            for col in feature_cols:
                obs[col] = history[col].to_numpy(dtype=np.float32)
        obs_parts.append(obs)

        futr = pd.DataFrame(
            {
                "unique_id": uid,
                "ds": pd.to_datetime(future[spec.time_col]).to_numpy(),
            }
        )
        if bundle.supports_future_exog:
            for col in feature_cols:
                futr[col] = future[col].to_numpy(dtype=np.float32)
        futr_parts.append(futr)
        truths.append(future[spec.target_col].to_numpy(dtype=np.float32))

    def _extract_predictions(pred_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        pred_col = bundle.prediction_column
        if pred_col not in pred_df.columns:
            fallback = [col for col in pred_df.columns if col not in {"unique_id", "ds"}]
            if not fallback:
                raise RuntimeError(f"No prediction column found for method '{bundle.method}'.")
            pred_col = fallback[0]
        pred_map: Dict[str, np.ndarray] = {}
        for uid, group in pred_df.groupby("unique_id"):
            ordered = group.sort_values("ds")
            pred_map[str(uid)] = ordered[pred_col].to_numpy(dtype=np.float32)
        return pred_map

    obs_df = pd.concat(obs_parts, ignore_index=True)
    futr_df = pd.concat(futr_parts, ignore_index=True)
    try:
        pred_df = bundle.nf.predict(df=obs_df, futr_df=futr_df, verbose=False)
        pred_map = _extract_predictions(pred_df)
    except RuntimeError:
        pred_map = {}
        for window_id in range(len(window_indices)):
            uid = f"window_{window_id}"
            obs_chunk = obs_df.loc[obs_df["unique_id"] == uid].copy()
            futr_chunk = futr_df.loc[futr_df["unique_id"] == uid].copy()
            pred_df = bundle.nf.predict(df=obs_chunk, futr_df=futr_chunk, verbose=False)
            pred_map.update(_extract_predictions(pred_df))

    y_pred = np.stack([pred_map[f"window_{window_id}"] for window_id in range(len(window_indices))], axis=0)
    y_true = np.stack(truths, axis=0)
    return y_true, y_pred
