"""Cải tiến B: thay "max theo chunk + ngưỡng 0.5" bằng một bộ phân loại nhỏ học trên dev.

Mục tiêu đánh giá là BAcc trung bình macro theo dataset, nên mỗi ô (dataset, nhãn) được gán
tổng trọng số bằng nhau khi huấn luyện. Nhờ vậy RAGTruth (dataset lớn nhất) không lấn át và
ngưỡng 0.5 trên xác suất dự đoán tương ứng với việc cân bằng TPR/TNR.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import DATASETS_11
from .metrics import macro_bacc_arrays

PARAM_GRID = {
    "lr": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "hgb": [{"max_depth": d, "learning_rate": lr, "max_iter": 300}
            for d, lr in itertools.product((2, 3), (0.03, 0.1))],
}


def balanced_weights(datasets: Sequence[str], labels: Sequence[int]) -> np.ndarray:
    """Trọng số để mỗi ô (dataset, nhãn) có tổng trọng số bằng nhau; trung bình = 1."""
    key = pd.Series(list(zip(datasets, labels)))
    counts = key.map(key.value_counts())
    w = 1.0 / counts.values.astype(float)
    return w * len(w) / w.sum()


def make_model(kind: str, params: Dict, seed: int = 0):
    if kind == "lr":
        return Pipeline([("scale", StandardScaler()),
                         ("clf", LogisticRegression(C=params["C"], max_iter=5000))])
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_depth=params["max_depth"],
                                              learning_rate=params["learning_rate"],
                                              max_iter=params["max_iter"], l2_regularization=1.0,
                                              early_stopping=False, random_state=seed)
    raise ValueError(kind)


def _fit(model, X, y, w):
    if isinstance(model, Pipeline):
        return model.fit(X, y, clf__sample_weight=w)
    return model.fit(X, y, sample_weight=w)


def fit_predict(kind: str, params: Dict, train: pd.DataFrame, test: pd.DataFrame,
                features: List[str], seed: int = 0) -> np.ndarray:
    w = balanced_weights(train["dataset"].values, train["label"].values)
    model = _fit(make_model(kind, params, seed), train[features].values, train["label"].values, w)
    return model.predict_proba(test[features].values)[:, 1]


def cv_select(kind: str, dev: pd.DataFrame, features: List[str], n_splits: int = 5,
              seed: int = 0, datasets: Sequence[str] = DATASETS_11):
    """Chọn siêu tham số bằng cross-validation 5-fold trên dev (phân tầng theo dataset x nhãn).

    Trả về (params tốt nhất, bảng điểm CV của mọi cấu hình, xác suất out-of-fold)."""
    strata = dev["dataset"].astype(str) + "|" + dev["label"].astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(dev, strata))
    rows, oof_best, best = [], None, (None, -1.0)
    for params in PARAM_GRID[kind]:
        oof = np.zeros(len(dev))
        for tr, va in folds:
            oof[va] = fit_predict(kind, params, dev.iloc[tr], dev.iloc[va], features, seed)
        score = macro_bacc_arrays(dev["label"].values, (oof > 0.5).astype(int),
                                  dev["dataset"].values, datasets)
        rows.append({"kind": kind, **params, "cv_macro_bacc": 100 * score})
        if score > best[1]:
            best, oof_best = (params, score), oof
    return best[0], pd.DataFrame(rows), oof_best


def leave_one_dataset_out(kind: str, params: Dict, dev: pd.DataFrame, test: pd.DataFrame,
                          features: List[str], seed: int = 0) -> np.ndarray:
    """Với mỗi dataset d: huấn luyện trên dev của 10 dataset còn lại, dự đoán test của d.
    Mô phỏng việc triển khai sang một domain chưa từng thấy nhãn."""
    out = np.full(len(test), np.nan)
    for name in test["dataset"].unique():
        tr = dev[dev["dataset"] != name]
        mask = (test["dataset"] == name).values
        out[mask] = fit_predict(kind, params, tr, test[mask], features, seed)
    return out


def lr_coefficients(dev: pd.DataFrame, features: List[str], params: Dict) -> pd.Series:
    """Hệ số LR trên đặc trưng đã chuẩn hóa (để diễn giải)."""
    w = balanced_weights(dev["dataset"].values, dev["label"].values)
    model = _fit(make_model("lr", params), dev[features].values, dev["label"].values, w)
    return pd.Series(model.named_steps["clf"].coef_[0], index=features).sort_values()
