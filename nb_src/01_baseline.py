# %% [markdown]
# # 01 · Baseline MiniCheck (Tuần 1)
#
# **Mục tiêu:** chạy lại MiniCheck gốc (checkpoint công khai của tác giả, không train lại) trên
# **dev + test** của LLM-AggreFact, lưu xác suất *của từng chunk* cho mọi ví dụ. Đây là đầu vào
# cho notebook 02–05.
#
# **Cấu hình Kaggle:** Accelerator = GPU T4 (1 GPU là đủ) · Internet = On ·
# Add-ons → Secrets → `HF_TOKEN` (token của tài khoản đã bấm *Agree* ở trang dataset
# `lytang/LLM-AggreFact`).
#
# **Chạy:** đổi `MODEL_NAME` nếu cần → *Save Version → Save & Run All*. Mỗi model chạy một lần.
# Có thể chạy lại khi bị ngắt: notebook bỏ qua các shard đã xong (xem ghi chú cuối).
#
# **Output** (`/kaggle/working/results/baseline/`):
# `<model>__dev.csv.gz`, `<model>__test.csv.gz`, `<model>__metrics.csv`,
# `<model>__fidelity.json`, `<model>__manifest.json`.

# %%
# --- Bootstrap: lấy code của repo (src/mcplus) ---
import os, sys, subprocess
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # dùng 1 GPU, tránh device_map="auto" chia model ra 2 GPU
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
REPO_URL = "https://github.com/vanductan-NLT/cs114.git"


def find_repo():
    for base in [Path.cwd(), *Path.cwd().parents]:
        if (base / "src" / "mcplus").exists():
            return base
    target = Path("/kaggle/working/cs114") if Path("/kaggle/working").exists() else Path.cwd() / "cs114"
    if not target.exists():
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(target)], check=True)
    return target


REPO = find_repo()
sys.path.insert(0, str(REPO / "src"))
print("repo:", REPO)

# %%
from mcplus.nb import ensure_minicheck, ensure_nltk, hf_login, env_summary, smoke_mode, display

ensure_minicheck()
ensure_nltk()
hf_login()
env_summary()

# %% [markdown]
# ## 1. Cấu hình

# %%
MODEL_NAME = os.environ.get("MCPLUS_MODEL", "deberta-v3-large")  # "flan-t5-large" | "roberta-large"
SPLITS = ["dev", "test"]
FIDELITY_PER_DATASET = 20   # 20 x 11 = 220 ví dụ để so khớp với MiniCheck.score gốc
FIDELITY_MIN_PER_LABEL = 5
SHARD_SIZE = 3000           # lưu tạm mỗi 3000 ví dụ để chạy tiếp được khi bị ngắt
SMOKE = smoke_mode()        # True = chỉ chạy vài chục ví dụ để thử code
if SMOKE:
    FIDELITY_PER_DATASET, FIDELITY_MIN_PER_LABEL, SHARD_SIZE = 4, 2, 40
print(MODEL_NAME, SPLITS, "smoke" if SMOKE else "full")

# %%
import json
import shutil
import time

import numpy as np
import pandas as pd

from mcplus import DATASETS_11, DATASETS_PAPER_10, MODEL_CHECKPOINTS, PAPER_TABLE2, MINICHECK_COMMIT
from mcplus.chunking import split_claim
from mcplus.data import load_split, stratified_subset
from mcplus.features import lexical_overlap
from mcplus.io_utils import model_revision, output_root, write_manifest
from mcplus.metrics import macro, per_dataset
from mcplus.scorer import FastScorer

OUT = output_root() / "baseline"
OUT.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 2. Dữ liệu

# %%
frames = {s: load_split(s) for s in SPLITS}
if SMOKE:
    frames = {s: stratified_subset(f, per_dataset=12, min_per_label=4) for s, f in frames.items()}
for s, f in frames.items():
    print(s, len(f), "rows,", f["dataset"].nunique(), "datasets")

# %% [markdown]
# ## 3. Tải model

# %%
import torch

t0 = time.perf_counter()
scorer = FastScorer(MODEL_NAME, batch_size=16)  # cache mặc định của HF, không lẫn vào output
print(f"Loaded {MODEL_CHECKPOINTS[MODEL_NAME]} in {time.perf_counter() - t0:.0f}s on {scorer.device}, "
      f"dtype={next(scorer.model.parameters()).dtype}, chunk_size={scorer.chunk_size}")

# %% [markdown]
# ## 4. Kiểm tra độ khớp với `MiniCheck.score` gốc
#
# `FastScorer` gom batch qua nhiều ví dụ để chạy nhanh. Trước khi chạy full, so với hàm gốc
# trên một mẫu phân tầng: số chunk phải trùng 100%, nhãn dự đoán trùng ≥ 99.5% và chênh lệch
# xác suất chỉ ở mức sai số dấu phẩy động do padding khác nhau.

# %%
sample = stratified_subset(frames["test"], per_dataset=FIDELITY_PER_DATASET, min_per_label=FIDELITY_MIN_PER_LABEL, seed=1)
t0 = time.perf_counter()
off_pred, off_prob, off_chunks, _ = scorer.official_score(sample["doc"].tolist(), sample["claim"].tolist())
t_official = time.perf_counter() - t0
t0 = time.perf_counter()
fast_prob, _, fast_nchunks, _ = scorer.score_examples(sample["doc"].tolist(), sample["claim"].tolist(),
                                                      progress=False)
t_fast = time.perf_counter() - t0

fidelity = {
    "n": len(sample),
    "chunk_count_match": float(np.mean([len(c) == n for c, n in zip(off_chunks, fast_nchunks)])),
    "label_agreement": float(np.mean(np.array(off_pred) == (fast_prob > 0.5).astype(int))),
    "max_abs_prob_diff": float(np.max(np.abs(np.array(off_prob) - fast_prob))),
    "seconds_official": round(t_official, 1),
    "seconds_fast": round(t_fast, 1),
}
print(json.dumps(fidelity, indent=2))
(OUT / f"{MODEL_NAME}__fidelity.json").write_text(json.dumps(fidelity, indent=2))
assert fidelity["chunk_count_match"] == 1.0, "Cách cắt chunk khác MiniCheck gốc!"
assert fidelity["label_agreement"] >= 0.995, "Nhãn lệch MiniCheck gốc quá nhiều, kiểm tra lại trước khi chạy full."

# %% [markdown]
# ## 5. Chạy full dev + test (có resume theo shard)

# %%
def run_split(split: str, frame: pd.DataFrame) -> pd.DataFrame:
    final = OUT / f"{MODEL_NAME}__{split}.csv.gz"
    if final.exists():
        print(f"[skip] {final.name} đã có")
        return pd.read_csv(final)
    shard_dir = OUT / f"_shards_{MODEL_NAME}_{split}"
    shard_dir.mkdir(exist_ok=True)
    # Resume trên Kaggle: nếu output lần chạy trước được Add Input thì chép các shard đã xong.
    for old in Path("/kaggle/input").glob(f"**/_shards_{MODEL_NAME}_{split}/*.csv.gz"):
        if not (shard_dir / old.name).exists():
            shutil.copy2(old, shard_dir / old.name)
    started, forward_before = time.perf_counter(), scorer.n_forward
    for start in range(0, len(frame), SHARD_SIZE):
        path = shard_dir / f"{start:06d}.csv.gz"
        if path.exists():
            continue
        sub = frame.iloc[start:start + SHARD_SIZE]
        docs, claims = sub["doc"].tolist(), sub["claim"].tolist()
        p_max, chunk_probs, n_chunks, doc_len = scorer.score_examples(docs, claims)
        pd.DataFrame({
            "uid": sub["uid"].values, "dataset": sub["dataset"].values, "label": sub["label"].values,
            "n_chunks": n_chunks, "chunk_probs": [json.dumps([round(p, 6) for p in ps]) for ps in chunk_probs],
            "p_max": np.round(p_max, 6), "pred": (p_max > 0.5).astype(int),
            "doc_len": doc_len, "claim_len": scorer.sentence_lengths(claims),
            "n_claim_sents": [len(split_claim(c)) for c in claims],
            "overlap": np.round([lexical_overlap(d, c) for d, c in zip(docs, claims)], 4),
        }).to_csv(path, index=False)
        print(f"{split}: {min(start + SHARD_SIZE, len(frame))}/{len(frame)} "
              f"({time.perf_counter() - started:.0f}s)")
    result = pd.concat([pd.read_csv(p) for p in sorted(shard_dir.glob("*.csv.gz"))], ignore_index=True)
    assert len(result) == len(frame) and result["uid"].is_unique
    result.to_csv(final, index=False)
    timing[split] = {"seconds": round(time.perf_counter() - started, 1),
                     "pairs_scored": scorer.n_forward - forward_before, "rows": len(frame)}
    return result


timing = {}
results = {s: run_split(s, frames[s]) for s in SPLITS}

# %% [markdown]
# ## 6. Kết quả baseline so với bài báo
#
# Bài báo (EMNLP 2024, Table 2) báo cáo 10 dataset, chưa có RAGTruth. Ta báo cáo cả
# trung bình trên 10 dataset đó (**so thẳng với bài báo**) và trên đủ 11 dataset.

# %%
paper = PAPER_TABLE2.get(MODEL_NAME, {})
test_tab = per_dataset(results["test"], "pred")
dev_tab = per_dataset(results["dev"], "pred")
metrics = pd.DataFrame({
    "paper_bacc": [paper.get(d, np.nan) for d in DATASETS_11],
    "test_bacc": [test_tab["bacc"].get(d, np.nan) for d in DATASETS_11],
    "test_tpr": [test_tab["tpr"].get(d, np.nan) for d in DATASETS_11],
    "test_tnr": [test_tab["tnr"].get(d, np.nan) for d in DATASETS_11],
    "dev_bacc": [dev_tab["bacc"].get(d, np.nan) for d in DATASETS_11],
}, index=DATASETS_11)
metrics["diff_vs_paper"] = metrics["test_bacc"] - metrics["paper_bacc"]
metrics.loc["macro10"] = [paper.get("avg", np.nan), macro(test_tab, DATASETS_PAPER_10),
                          macro(test_tab, DATASETS_PAPER_10, "tpr"), macro(test_tab, DATASETS_PAPER_10, "tnr"),
                          macro(dev_tab, DATASETS_PAPER_10), np.nan]
metrics.loc["macro10", "diff_vs_paper"] = metrics.loc["macro10", "test_bacc"] - metrics.loc["macro10", "paper_bacc"]
metrics.loc["macro11"] = [np.nan, macro(test_tab, DATASETS_11), macro(test_tab, DATASETS_11, "tpr"),
                          macro(test_tab, DATASETS_11, "tnr"), macro(dev_tab, DATASETS_11), np.nan]
metrics = metrics.round(2)
metrics.to_csv(OUT / f"{MODEL_NAME}__metrics.csv", index_label="dataset")
display(metrics)

# %%
chunks_share = results["test"].groupby("dataset")["n_chunks"].agg(
    multi_chunk_pct=lambda s: round(100 * (s > 1).mean(), 1), mean_chunks="mean").round(2)
display(chunks_share)

# %%
write_manifest(f"baseline/{MODEL_NAME}__manifest.json",
               notebook="01_baseline", model=MODEL_NAME, checkpoint=MODEL_CHECKPOINTS[MODEL_NAME],
               checkpoint_revision=model_revision(MODEL_CHECKPOINTS[MODEL_NAME]),
               minicheck_commit=MINICHECK_COMMIT, smoke=SMOKE, fidelity=fidelity, timing=timing,
               chunk_size=scorer.chunk_size, token_budget=scorer.token_budget)
print("Done. Tải thư mục results/baseline về và commit vào repo (trừ thư mục _shards_*).")

# %% [markdown]
# **Ghi chú**
# - Nếu Kaggle ngắt giữa chừng: mở lại notebook, *Add Input* → chọn output của lần chạy trước,
#   rồi Run All. Notebook tự chép các shard đã xong từ `/kaggle/input` và chỉ chạy phần còn lại.
# - Thời gian ước tính trên 1×T4 (fp32, giống MiniCheck gốc): vài giờ cho dev + test mỗi model.
#   Thời gian thực được ghi trong `manifest.json` (mục `timing`) để đưa vào phần chi phí.
