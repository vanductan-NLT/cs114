"""Tải LLM-AggreFact và tạo tập con phân tầng."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import DATASETS_11, HF_DATASET

# Revision của dataset trên Hugging Face được pin để số liệu không đổi nếu tác giả cập nhật.
HF_REVISION = "981dfd0"
EXPECTED_ROWS = {"dev": 30420, "test": 29320}


def get_hf_token() -> str | None:
    """Đọc HF token từ biến môi trường, Kaggle Secrets hoặc Colab Secrets. Không bao giờ in token."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
    try:
        from google.colab import userdata  # type: ignore

        return userdata.get("HF_TOKEN")
    except Exception:
        return None


def load_split(split: str, revision: str | None = HF_REVISION, token: str | None = None,
               check_size: bool = True) -> pd.DataFrame:
    """Trả về DataFrame với cột: uid, split, idx, dataset, doc, claim, label.

    idx = vị trí dòng trong split gốc; uid = "<split>:<idx>" là khóa nối giữa các notebook.
    Nếu đặt biến môi trường MCPLUS_LOCAL_DATA (file .csv/.parquet có cột split) thì đọc từ
    file đó; chỉ dùng cho smoke test trên máy cá nhân.
    """
    assert split in ("dev", "test"), split
    local = os.environ.get("MCPLUS_LOCAL_DATA")
    if local:
        frame = pd.read_parquet(local) if local.endswith(".parquet") else pd.read_csv(local)
        frame = frame[frame["split"] == split].reset_index(drop=True)
        if "idx" not in frame.columns:
            frame["idx"] = np.arange(len(frame))
        check_size = False
    else:
        from datasets import load_dataset

        ds = load_dataset(HF_DATASET, split=split, revision=revision, token=token or get_hf_token())
        frame = ds.to_pandas()
        frame["idx"] = np.arange(len(frame))
    frame["split"] = split
    frame["uid"] = split + ":" + frame["idx"].astype(str)
    frame["label"] = frame["label"].astype(int)
    if check_size:
        assert len(frame) == EXPECTED_ROWS[split], f"{split}: {len(frame)} rows"
        assert sorted(frame["dataset"].unique()) == sorted(DATASETS_11), frame["dataset"].unique()
    return frame[["uid", "split", "idx", "dataset", "doc", "claim", "label"]]


def stratified_subset(frame: pd.DataFrame, per_dataset: int, seed: int = 42,
                      min_per_label: int = 30) -> pd.DataFrame:
    """Lấy tối đa `per_dataset` dòng mỗi dataset, giữ tỷ lệ nhãn nhưng đảm bảo
    mỗi nhãn có ít nhất `min_per_label` dòng (nếu đủ) để BAcc không quá nhiễu."""
    rng = np.random.RandomState(seed)
    parts = []
    for _, group in frame.groupby("dataset", sort=True):
        if len(group) <= per_dataset:
            parts.append(group)
            continue
        quota = {}
        for label, sub in group.groupby("label"):
            share = int(round(per_dataset * len(sub) / len(group)))
            quota[label] = min(len(sub), max(share, min(min_per_label, len(sub))))
        for label, sub in group.groupby("label"):
            idx = rng.choice(len(sub), size=quota[label], replace=False)
            parts.append(sub.iloc[np.sort(idx)])
    return pd.concat(parts).sort_index().reset_index(drop=True)  # giữ thứ tự dòng gốc


def dataset_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Số mẫu, số nhãn 0/1, tỷ lệ nhãn 1, số văn bản khác nhau và độ dài theo từ."""
    rows = []
    for name in DATASETS_11:
        g = frame[frame["dataset"] == name]
        if g.empty:
            continue
        doc_words = g["doc"].str.split().str.len()
        claim_words = g["claim"].str.split().str.len()
        rows.append({
            "dataset": name,
            "n": len(g),
            "n_unsupported": int((g["label"] == 0).sum()),
            "n_supported": int((g["label"] == 1).sum()),
            "pct_supported": round(100 * g["label"].mean(), 1),
            "unique_docs": g["doc"].nunique(),
            "doc_words_median": int(doc_words.median()),
            "doc_words_p90": int(doc_words.quantile(0.9)),
            "claim_words_median": int(claim_words.median()),
        })
    return pd.DataFrame(rows)
