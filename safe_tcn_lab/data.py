from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset


EPS = 1e-6


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    id_col: str
    time_col: str
    target_col: str
    feature_cols: Tuple[str, ...]
    wind_col: str
    steps_per_day: int
    rated_capacity: float | None
    split_mode: str


SDWPF_SPEC = DatasetSpec(
    name="sdwpf",
    id_col="turbine_id",
    time_col="timestamp",
    target_col="Patv",
    feature_cols=("Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Pab2", "Pab3", "Prtv"),
    wind_col="Wspd",
    steps_per_day=144,
    rated_capacity=1500.0,
    split_mode="official",
)


GEFCOM_SPEC = DatasetSpec(
    name="gefcom",
    id_col="zone_id",
    time_col="timestamp",
    target_col="power",
    feature_cols=("U10", "V10", "U100", "V100", "Wspd10", "Wdir10", "Wspd100", "Wdir100"),
    wind_col="Wspd100",
    steps_per_day=24,
    rated_capacity=1.0,
    split_mode="calendar",
)


def get_dataset_spec(name: str) -> DatasetSpec:
    lowered = name.lower()
    if lowered == "sdwpf":
        return SDWPF_SPEC
    if lowered in {"gefcom", "gefcom2014"}:
        return GEFCOM_SPEC
    raise ValueError(f"Unknown dataset '{name}'.")


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) <= EPS or np.std(b) <= EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _safe_autocorr(x: np.ndarray, lag: int) -> float:
    if x.size <= lag + 1:
        return 0.0
    return _safe_corr(x[:-lag], x[lag:])


def _build_power_curve(
    frame: pd.DataFrame,
    wind_col: str,
    target_col: str,
    wind_bins: np.ndarray,
    target_scale: float,
) -> np.ndarray:
    work = frame[[wind_col, target_col]].dropna().copy()
    if work.empty:
        return np.zeros(len(wind_bins) - 1, dtype=np.float32)
    work["bin"] = pd.cut(work[wind_col], bins=wind_bins, labels=False, include_lowest=True)
    curve = work.groupby("bin")[target_col].mean().reindex(range(len(wind_bins) - 1))
    curve = curve.interpolate(limit_direction="both").fillna(0.0)
    return (curve.to_numpy(dtype=np.float32) / max(target_scale, EPS)).clip(0.0, 1.5)


def _build_task_profile(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    wind_bins: np.ndarray,
    feature_cols: Sequence[str],
) -> np.ndarray:
    target = frame[spec.target_col].to_numpy(dtype=np.float32)
    target_scale = spec.rated_capacity or float(np.quantile(target, 0.99) + EPS)
    wind = frame[spec.wind_col].to_numpy(dtype=np.float32)
    power_curve = _build_power_curve(frame, spec.wind_col, spec.target_col, wind_bins, target_scale)
    diffs = np.diff(target, prepend=target[:1])
    feature_corrs = np.array(
        [_safe_corr(frame[col].to_numpy(dtype=np.float32), target) for col in feature_cols],
        dtype=np.float32,
    )
    summary = np.array(
        [
            float(np.mean(target) / max(target_scale, EPS)),
            float(np.std(target) / max(target_scale, EPS)),
            float(np.mean(np.abs(diffs)) / max(target_scale, EPS)),
            float(np.std(diffs) / max(target_scale, EPS)),
            float(np.mean(wind) / max(float(wind_bins[-1]), EPS)),
            float(np.std(wind) / max(float(wind_bins[-1]), EPS)),
            _safe_autocorr(target, 1),
            _safe_autocorr(target, 6),
            _safe_autocorr(target, 24),
            _safe_corr(wind, target),
        ],
        dtype=np.float32,
    )
    return np.concatenate([power_curve, summary, feature_corrs], axis=0)


class WindowDataset(Dataset):
    def __init__(
        self,
        model_frame: pd.DataFrame,
        raw_frame: pd.DataFrame,
        feature_cols: Sequence[str],
        target_col: str,
        seq_len: int,
        pred_len: int,
        mean: np.ndarray,
        std: np.ndarray,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.feature_cols = tuple(feature_cols)
        self.target_col = target_col
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.raw_frame = raw_frame.reset_index(drop=True).copy()
        values = model_frame[list(self.feature_cols) + [self.target_col]].to_numpy(dtype=np.float32)
        self.values = (values - mean) / std
        max_start = len(self.values) - (self.seq_len + self.pred_len)
        self.indices = list(range(0, max_start + 1, stride)) if max_start >= 0 else []

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = self.indices[index]
        window = self.values[start : start + self.seq_len]
        target = self.values[start + self.seq_len : start + self.seq_len + self.pred_len, -1]
        return (
            torch.from_numpy(window[:, :-1]),
            torch.from_numpy(window[:, -1:]),
            torch.from_numpy(target),
        )

    def get_future_frame(self, index: int) -> pd.DataFrame:
        start = self.indices[index] + self.seq_len
        stop = start + self.pred_len
        return self.raw_frame.iloc[start:stop].reset_index(drop=True)


class TaskAnnotatedDataset(Dataset):
    def __init__(self, dataset: Dataset, task_index: int) -> None:
        self.dataset = dataset
        self.task_index = task_index

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_feat, x_tgt, y = self.dataset[index]
        return x_feat, x_tgt, y, torch.tensor(self.task_index, dtype=torch.long)


class MultiTaskForecastData:
    def __init__(
        self,
        spec: DatasetSpec,
        parquet_path: str,
        seq_len: int,
        pred_len: int,
        n_pc_bins: int = 20,
        max_rows_per_task: int | None = None,
    ) -> None:
        self.spec = spec
        self.parquet_path = parquet_path
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_pc_bins = n_pc_bins
        self.max_rows_per_task = max_rows_per_task
        self.feature_cols: List[str] = list(spec.feature_cols) + ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
        self.task_ids: List[int] = []
        self.task_to_index: Dict[int, int] = {}
        self._splits_raw: Dict[int, Dict[str, pd.DataFrame]] = {}
        self._splits_model: Dict[int, Dict[str, pd.DataFrame]] = {}
        self._stats: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._profiles: Dict[int, np.ndarray] = {}
        self._wind_bins: np.ndarray | None = None

    @property
    def profile_dim(self) -> int:
        sample = next(iter(self._profiles.values()))
        return int(sample.shape[0])

    def _frame_stats(self, frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        stats_frame = frame[list(self.feature_cols) + [self.spec.target_col]]
        mean = stats_frame.mean().to_numpy(dtype=np.float32)
        std = stats_frame.std().to_numpy(dtype=np.float32)
        std[std < 1e-6] = 1.0
        return mean, std

    def _profile_from_train_frame(self, frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return self._profiles[self.task_ids[0]].copy()
        return _build_task_profile(frame, self.spec, self._wind_bins, self.feature_cols)

    def get_profile(self, task_id: int, train_days_limit: int | None = None) -> np.ndarray:
        if train_days_limit is None:
            return self._profiles[task_id]
        _, model_frame = self.get_frame(task_id, "train", train_days_limit=train_days_limit)
        return self._profile_from_train_frame(model_frame)

    def get_profiles(self, task_ids: Sequence[int]) -> np.ndarray:
        return np.stack([self._profiles[task_id] for task_id in task_ids], axis=0).astype(np.float32)

    def get_normalization_stats(self, task_id: int, train_days_limit: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        if train_days_limit is None:
            return self._stats[task_id]
        _, model_frame = self.get_frame(task_id, "train", train_days_limit=train_days_limit)
        if model_frame.empty:
            return self._stats[task_id]
        return self._frame_stats(model_frame)

    def _split_task_frame(self, frame: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        if self.spec.name == "sdwpf":
            total_rows = 184 * self.spec.steps_per_day
            work = frame.iloc[:total_rows].copy().reset_index(drop=True)
            train_rows = 153 * self.spec.steps_per_day
            val_rows = 16 * self.spec.steps_per_day
            return {
                "train": work.iloc[:train_rows].copy(),
                "val": work.iloc[train_rows : train_rows + val_rows].copy(),
                "test": work.iloc[train_rows + val_rows :].copy(),
            }

        if self.spec.name == "gefcom":
            ts = pd.to_datetime(frame[self.spec.time_col])
            return {
                "train": frame.loc[ts < pd.Timestamp("2013-01-01")].copy(),
                "val": frame.loc[(ts >= pd.Timestamp("2013-01-01")) & (ts < pd.Timestamp("2013-07-01"))].copy(),
                "test": frame.loc[ts >= pd.Timestamp("2013-07-01")].copy(),
            }

        raise ValueError(f"Unsupported dataset '{self.spec.name}'.")

    def load(self) -> "MultiTaskForecastData":
        raw = pd.read_parquet(self.parquet_path)
        all_cols = list(self.feature_cols) + [self.spec.target_col]
        raw[self.spec.time_col] = pd.to_datetime(raw[self.spec.time_col])
        task_raw_frames: Dict[int, pd.DataFrame] = {}
        task_model_frames: Dict[int, pd.DataFrame] = {}

        for task_id, task_frame in raw.groupby(self.spec.id_col):
            frame = task_frame.sort_values(self.spec.time_col).reset_index(drop=True).copy()
            hour = frame[self.spec.time_col].dt.hour + frame[self.spec.time_col].dt.minute / 60.0
            dayofyear = frame[self.spec.time_col].dt.dayofyear.astype(float)
            frame["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
            frame["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
            frame["doy_sin"] = np.sin(2.0 * np.pi * dayofyear / 366.0)
            frame["doy_cos"] = np.cos(2.0 * np.pi * dayofyear / 366.0)
            if self.max_rows_per_task is not None and len(frame) > self.max_rows_per_task:
                frame = frame.iloc[: self.max_rows_per_task].copy()
            raw_frame = frame.copy()
            model_frame = frame.copy()
            model_frame[all_cols] = model_frame[all_cols].ffill().bfill().fillna(0.0)
            model_frame[self.spec.target_col] = model_frame[self.spec.target_col].clip(lower=0.0)
            task_raw_frames[int(task_id)] = raw_frame
            task_model_frames[int(task_id)] = model_frame

        self.task_ids = sorted(task_raw_frames)
        self.task_to_index = {task_id: idx for idx, task_id in enumerate(self.task_ids)}

        for task_id in self.task_ids:
            self._splits_raw[task_id] = self._split_task_frame(task_raw_frames[task_id])
            self._splits_model[task_id] = self._split_task_frame(task_model_frames[task_id])

        train_wind = np.concatenate(
            [
                self._splits_model[task_id]["train"][self.spec.wind_col].to_numpy(dtype=np.float32)
                for task_id in self.task_ids
                if len(self._splits_model[task_id]["train"]) > 0
            ],
            axis=0,
        )
        wind_upper = float(np.quantile(train_wind, 0.99)) if train_wind.size else 1.0
        self._wind_bins = np.linspace(0.0, max(wind_upper, 1.0), self.n_pc_bins + 1, dtype=np.float32)

        for task_id in self.task_ids:
            train_frame = self._splits_model[task_id]["train"]
            self._stats[task_id] = self._frame_stats(train_frame)
            self._profiles[task_id] = _build_task_profile(train_frame, self.spec, self._wind_bins, self.feature_cols)

        return self

    def get_frame(self, task_id: int, split: str, train_days_limit: int | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raw_frame = self._splits_raw[task_id][split].copy()
        model_frame = self._splits_model[task_id][split].copy()
        if split == "train" and train_days_limit is not None:
            max_rows = min(len(model_frame), int(train_days_limit * self.spec.steps_per_day))
            raw_frame = raw_frame.iloc[:max_rows].copy()
            model_frame = model_frame.iloc[:max_rows].copy()
        return raw_frame.reset_index(drop=True), model_frame.reset_index(drop=True)

    def get_dataset(
        self,
        task_id: int,
        split: str,
        train_days_limit: int | None = None,
        normalization_train_days: int | None = None,
        stride: int = 1,
    ) -> WindowDataset:
        raw_frame, model_frame = self.get_frame(task_id, split, train_days_limit=train_days_limit)
        mean, std = self.get_normalization_stats(task_id, train_days_limit=normalization_train_days)
        return WindowDataset(
            model_frame=model_frame,
            raw_frame=raw_frame,
            feature_cols=self.feature_cols,
            target_col=self.spec.target_col,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            mean=mean,
            std=std,
            stride=stride,
        )

    def make_multitask_dataset(self, task_ids: Iterable[int], split: str, stride: int = 1) -> ConcatDataset:
        parts: List[Dataset] = []
        for task_id in task_ids:
            dataset = self.get_dataset(task_id, split, stride=stride)
            if len(dataset) == 0:
                continue
            parts.append(TaskAnnotatedDataset(dataset, self.task_to_index[task_id]))
        if not parts:
            raise ValueError(f"No non-empty datasets found for split '{split}'.")
        return ConcatDataset(parts)

    def denormalize_target(
        self,
        task_id: int,
        values: np.ndarray,
        stats: Tuple[np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        mean, std = stats or self._stats[task_id]
        return np.clip(np.asarray(values, dtype=np.float64) * float(std[-1]) + float(mean[-1]), 0.0, None)

    def cosine_similarity(
        self,
        source_id: int,
        target_id: int,
        target_train_days_limit: int | None = None,
    ) -> float:
        source = self._profiles[source_id]
        target = self.get_profile(target_id, train_days_limit=target_train_days_limit)
        denom = np.linalg.norm(source) * np.linalg.norm(target) + EPS
        return float(np.dot(source, target) / denom)

    def select_sources(
        self,
        target_id: int,
        max_sources: int,
        min_similarity: float = -1.0,
        candidate_ids: Sequence[int] | None = None,
        target_train_days_limit: int | None = None,
    ) -> List[Tuple[int, float]]:
        candidate_ids = candidate_ids or [task_id for task_id in self.task_ids if task_id != target_id]
        scored: List[Tuple[int, float]] = []
        for source_id in candidate_ids:
            if source_id == target_id:
                continue
            similarity = self.cosine_similarity(
                source_id,
                target_id,
                target_train_days_limit=target_train_days_limit,
            )
            if similarity >= min_similarity:
                scored.append((source_id, similarity))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:max_sources]

    def build_relation_matrix(
        self,
        target_id: int,
        source_ids: Sequence[int],
        target_train_days_limit: int | None = None,
    ) -> np.ndarray:
        target_profile = self.get_profile(target_id, train_days_limit=target_train_days_limit)
        pc_len = self.n_pc_bins
        target_pc = target_profile[:pc_len]
        target_cf = target_profile[pc_len]
        target_wind_corr = target_profile[pc_len + 9]
        rows = []
        for source_id in source_ids:
            source_profile = self._profiles[source_id]
            source_pc = source_profile[:pc_len]
            diff = source_profile - target_profile
            rows.append(
                np.array(
                    [
                        self.cosine_similarity(
                            source_id,
                            target_id,
                            target_train_days_limit=target_train_days_limit,
                        ),
                        float(np.mean(np.abs(diff))),
                        float(np.linalg.norm(diff)),
                        _safe_corr(source_pc, target_pc),
                        float(source_profile[pc_len]),
                        float(target_cf),
                        float(source_profile[pc_len + 9]),
                        float(target_wind_corr),
                    ],
                    dtype=np.float32,
                )
            )
        return np.stack(rows, axis=0) if rows else np.zeros((0, 8), dtype=np.float32)
