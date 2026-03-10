from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from safe_tcn_lab.data import WindowDataset


def build_persistence_predictions(dataset: WindowDataset) -> Tuple[np.ndarray, np.ndarray]:
    preds = []
    truths = []
    for idx in range(len(dataset)):
        _, x_tgt, y = dataset[idx]
        preds.append(np.full(dataset.pred_len, float(x_tgt[-1, 0]), dtype=np.float32))
        truths.append(y.numpy())
    return np.asarray(truths, dtype=np.float32), np.asarray(preds, dtype=np.float32)


def build_tabular_matrix(dataset: WindowDataset) -> Tuple[np.ndarray, np.ndarray]:
    x_rows = []
    y_rows = []
    for idx in range(len(dataset)):
        x_feat, x_tgt, y = dataset[idx]
        x_rows.append(np.concatenate([x_feat.numpy().reshape(-1), x_tgt.numpy().reshape(-1)], axis=0))
        y_rows.append(y.numpy())
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32)


def _lgbm_feature_frame(x: np.ndarray) -> pd.DataFrame:
    columns = [f"f_{idx}" for idx in range(x.shape[1])]
    return pd.DataFrame(np.asarray(x, dtype=np.float32), columns=columns)


def fit_ridge_multioutput(x_train: np.ndarray, y_train: np.ndarray, alpha: float = 1.0) -> MultiOutputRegressor:
    model = MultiOutputRegressor(
        Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
    )
    model.fit(x_train, y_train)
    return model


def fit_lgbm_multioutput(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 120,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    random_state: int = 42,
) -> List[LGBMRegressor]:
    models: List[LGBMRegressor] = []
    x_train_frame = _lgbm_feature_frame(x_train)
    for horizon in range(y_train.shape[1]):
        target = np.asarray(y_train[:, horizon], dtype=np.float32).copy()
        model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state + horizon,
            verbose=-1,
        )
        model.fit(x_train_frame, target)
        models.append(model)
    return models


def predict_lgbm(models: Iterable[LGBMRegressor], x_test: np.ndarray) -> np.ndarray:
    x_test_frame = _lgbm_feature_frame(x_test)
    return np.stack([model.predict(x_test_frame) for model in models], axis=1).astype(np.float32)
