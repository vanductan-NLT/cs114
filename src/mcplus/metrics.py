"""Metric: Balanced Accuracy theo dataset, trung bình macro, hiệu chỉnh xác suất, kiểm định."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from . import DATASETS_11, DATASETS_PAPER_10


def rates(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    y, yhat = np.asarray(y).astype(int), np.asarray(yhat).astype(int)
    pos, neg = y == 1, y == 0
    tpr = float((yhat[pos] == 1).mean()) if pos.any() else np.nan
    tnr = float((yhat[neg] == 0).mean()) if neg.any() else np.nan
    return {"tpr": tpr, "tnr": tnr, "bacc": (tpr + tnr) / 2}


def per_dataset(frame: pd.DataFrame, pred_col: str, label_col: str = "label",
                datasets: Sequence[str] = DATASETS_11) -> pd.DataFrame:
    """BAcc, TPR, TNR (đơn vị %) cho từng dataset."""
    rows = []
    for name in datasets:
        g = frame[frame["dataset"] == name]
        if g.empty:
            continue
        r = rates(g[label_col].values, g[pred_col].values)
        rows.append({"dataset": name, "n": len(g), "n_pos": int(g[label_col].sum()),
                     "bacc": 100 * r["bacc"], "tpr": 100 * r["tpr"], "tnr": 100 * r["tnr"]})
    return pd.DataFrame(rows).set_index("dataset")


def macro(table: pd.DataFrame, datasets: Iterable[str], col: str = "bacc") -> float:
    names = [d for d in datasets if d in table.index]
    return float(table.loc[names, col].mean()) if names else np.nan


def summary_row(frame: pd.DataFrame, pred_col: str) -> Dict[str, float]:
    table = per_dataset(frame, pred_col)
    return {"macro10": macro(table, DATASETS_PAPER_10), "macro11": macro(table, DATASETS_11),
            **{d: table.loc[d, "bacc"] for d in table.index}}


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error cho xác suất lớp 1 (bin đều trên [0, 1])."""
    probs, y = np.asarray(probs, float), np.asarray(y, float)
    bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.mean() * abs(probs[mask].mean() - y[mask].mean())
    return float(total)


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(probs, float) - np.asarray(y, float)) ** 2))


def macro_bacc_arrays(y: np.ndarray, yhat: np.ndarray, groups: np.ndarray,
                      datasets: Sequence[str]) -> float:
    vals = [rates(y[groups == d], yhat[groups == d])["bacc"] for d in datasets if (groups == d).any()]
    return float(np.mean(vals))


def tune_threshold(scores: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None,
                   datasets: Sequence[str] = DATASETS_11,
                   grid: np.ndarray = np.round(np.arange(0.01, 1.0, 0.01), 2)) -> float:
    """Ngưỡng tối đa hóa BAcc (macro theo dataset nếu có `groups`). Chỉ dùng trên dev."""
    best_t, best_v = 0.5, -1.0
    for t in grid:
        yhat = (scores > t).astype(int)
        v = macro_bacc_arrays(y, yhat, groups, datasets) if groups is not None else rates(y, yhat)["bacc"]
        if v > best_v + 1e-12:
            best_t, best_v = float(t), v
    return best_t


def paired_bootstrap(frame: pd.DataFrame, pred_a: str, pred_b: str,
                     datasets: Sequence[str] = DATASETS_PAPER_10, n_boot: int = 10000,
                     seed: int = 0, label_col: str = "label") -> Dict[str, float]:
    """Bootstrap có cặp cho hiệu macro-BAcc (b - a).

    Lấy mẫu lại có hoàn lại *trong từng ô (dataset, nhãn)* nên số mẫu dương/âm của mỗi
    dataset giữ nguyên, đúng với định nghĩa BAcc. Trong một ô, việc lấy lại n mẫu tương
    đương rút số đếm của 4 tổ hợp (a đúng?, b đúng?) từ phân phối đa thức — tính rất nhanh.
    """
    rng = np.random.default_rng(seed)
    diffs = np.zeros(n_boot)
    used = [d for d in datasets
            if frame.loc[frame["dataset"] == d, label_col].nunique() == 2]
    for name in used:
        g = frame[frame["dataset"] == name]
        recall = {}
        for label in (0, 1):
            cell = g[g[label_col] == label]
            ca = (cell[pred_a].values == label)
            cb = (cell[pred_b].values == label)
            n = len(cell)
            p = np.array([np.mean(ca & cb), np.mean(ca & ~cb), np.mean(~ca & cb), np.mean(~ca & ~cb)])
            draws = rng.multinomial(n, p / p.sum(), size=n_boot)
            recall[label] = ((draws[:, 0] + draws[:, 1]) / n, (draws[:, 0] + draws[:, 2]) / n)
        bacc_a = (recall[0][0] + recall[1][0]) / 2
        bacc_b = (recall[0][1] + recall[1][1]) / 2
        diffs += (bacc_b - bacc_a) / len(used)
    diffs *= 100
    point = macro(per_dataset(frame, pred_b), used) - macro(per_dataset(frame, pred_a), used)
    p_two = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": point, "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)), "p_value": float(min(1.0, p_two))}


def mcnemar(frame: pd.DataFrame, pred_a: str, pred_b: str, label_col: str = "label") -> Dict[str, float]:
    """Kiểm định McNemar chính xác (binomial) trên các cặp bất đồng."""
    from scipy.stats import binomtest

    ca = frame[pred_a].values == frame[label_col].values
    cb = frame[pred_b].values == frame[label_col].values
    only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n = only_a + only_b
    p = binomtest(only_b, n, 0.5).pvalue if n else 1.0
    return {"a_right_b_wrong": only_a, "b_right_a_wrong": only_b, "p_value": float(p)}
