"""Đặc trưng cho cải tiến B (tổng hợp điểm có học).

Mỗi ví dụ có một danh sách xác suất theo chunk từ baseline, điểm của ngữ cảnh bằng chứng
(cải tiến A) và vài đặc trưng độ dài. MiniCheck gốc chỉ dùng max(chunk_probs) > 0.5; mô hình
học có thể dùng thêm hình dạng phân bố điểm, số chunk, độ dài...
"""

from __future__ import annotations

import json
from typing import List

import numpy as np
import pandas as pd

EPS = 1e-4


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def parse_probs(value) -> List[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(v) for v in value]
    return [float(v) for v in json.loads(value)]


def lexical_overlap(doc: str, claim: str) -> float:
    """Tỷ lệ từ nội dung của claim (bỏ stopword) xuất hiện trong văn bản."""
    from .evidence import tokenize_for_bm25

    claim_words = set(tokenize_for_bm25(claim))
    if not claim_words:
        return 1.0
    return len(claim_words & set(tokenize_for_bm25(doc))) / len(claim_words)


def chunk_stats(probs: List[float]) -> dict:
    arr = np.asarray(probs, dtype=float)
    top = np.sort(arr)[::-1]
    second = top[1] if len(top) > 1 else top[0]
    return {
        "n_chunks": len(arr),
        "p_max": top[0],
        "p_second": second,
        "p_mean": arr.mean(),
        "p_min": arr.min(),
        "p_std": arr.std(),
        "frac_above_05": float((arr > 0.5).mean()),
        "argmax_pos": float(np.argmax(arr) / max(len(arr) - 1, 1)),
    }


BASE_FEATURES = [
    "logit_p_max", "logit_p_second", "logit_p_mean", "logit_p_min", "p_std", "frac_above_05",
    "argmax_pos", "log_n_chunks", "log_doc_len", "log_claim_len", "claim_sents", "overlap_ratio",
]
EVIDENCE_FEATURES = ["logit_e_score", "e_minus_max"]
SPLIT_FEATURES = ["logit_e_split"]


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Nhận DataFrame đã nối baseline (+ evidence nếu có); trả về bảng đặc trưng số."""
    stats = pd.DataFrame([chunk_stats(parse_probs(v)) for v in frame["chunk_probs"]],
                         index=frame.index)
    feats = pd.DataFrame(index=frame.index)
    for col in ("p_max", "p_second", "p_mean", "p_min"):
        feats[f"logit_{col}"] = logit(stats[col])
    feats["p_std"] = stats["p_std"]
    feats["frac_above_05"] = stats["frac_above_05"]
    feats["argmax_pos"] = stats["argmax_pos"]
    feats["log_n_chunks"] = np.log(stats["n_chunks"])
    feats["log_doc_len"] = np.log1p(frame["doc_len"].astype(float))
    feats["log_claim_len"] = np.log1p(frame["claim_len"].astype(float))
    feats["claim_sents"] = frame["n_claim_sents"].astype(float)
    feats["overlap_ratio"] = frame["overlap"].astype(float)
    if "e_score" in frame.columns:
        feats["logit_e_score"] = logit(frame["e_score"])
        feats["e_minus_max"] = frame["e_score"] - stats["p_max"]
    if "e_split" in frame.columns:
        feats["logit_e_split"] = logit(frame["e_split"])
    return feats


def feature_sets(columns) -> dict:
    """Các nhóm đặc trưng dùng cho ablation của B."""
    cols = set(columns)
    sets = {"chunk_only": ["logit_p_max", "logit_p_second", "logit_p_mean", "logit_p_min", "p_std",
                           "frac_above_05", "argmax_pos", "log_n_chunks"]}
    sets["chunk+len"] = sets["chunk_only"] + ["log_doc_len", "log_claim_len", "claim_sents", "overlap_ratio"]
    if all(c in cols for c in EVIDENCE_FEATURES):
        sets["all"] = sets["chunk+len"] + EVIDENCE_FEATURES
        if all(c in cols for c in SPLIT_FEATURES):
            sets["all+split"] = sets["all"] + SPLIT_FEATURES
    return sets
