# %% [markdown]
# # 03 · Cải tiến A: chọn lọc bằng chứng theo claim (+ ablation C: tách claim) — Tuần 3
#
# **Ý tưởng.** MiniCheck cắt văn bản dài thành các chunk liền nhau (~400 token) và lấy max.
# Nếu bằng chứng nằm rải ở nhiều chunk, không chunk nào đủ để kết luận "supported".
# Cải tiến A xếp hạng từng câu của văn bản theo độ liên quan với claim (BM25 / embedding / lai),
# chọn các câu tốt nhất kèm câu lân cận cho tới khi đầy **đúng một chunk**, giữ thứ tự gốc,
# rồi chấm một lần bằng **cùng checkpoint MiniCheck**. Văn bản chỉ có 1 chunk giữ nguyên.
#
# **Hai cách dùng điểm mới** (`mode`): `replace` = dùng điểm ngữ cảnh chọn lọc thay cho max
# theo chunk; `union` = max(điểm baseline, điểm chọn lọc).
#
# **Ablation C** (hướng mở mà chính bài báo nêu): tách claim nhiều câu, chấm từng câu, lấy min.
#
# **Quy tắc chống rò rỉ:** siêu tham số (bộ chọn, k, cửa sổ, mode) chỉ chọn trên *một tập con
# của dev*. Ngưỡng luôn cố định 0.5 như bài báo (không dùng nhãn để tune ngưỡng) → vẫn là
# thiết lập zero-shot, so thẳng được với bài báo. Test chỉ chạy một lần với cấu hình đã chốt.
#
# **Kaggle:** GPU T4 · Internet On · Secret `HF_TOKEN` · Add Input = output của notebook 01.
#
# **Output** (`results/evidence/`): `<model>__tuning.csv`, `<model>__dev.csv.gz`,
# `<model>__test.csv.gz`, `<model>__test_metrics.csv`, `<model>__manifest.json`.

# %%
# --- Bootstrap: lấy code của repo (src/mcplus) ---
import os, sys, subprocess
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
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

# %%
from mcplus.nb import ensure_minicheck, ensure_nltk, ensure_packages, hf_login, env_summary, smoke_mode, display

ensure_minicheck()
ensure_packages("rank_bm25", "sentence-transformers")
ensure_nltk()
hf_login()
env_summary()

# %% [markdown]
# ## 1. Cấu hình

# %%
MODEL_NAME = os.environ.get("MCPLUS_MODEL", "deberta-v3-large")
TUNE_PER_DATASET = 500          # tập con dev để chọn siêu tham số (~5K dòng)
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SELECTORS = ["bm25", "dense", "hybrid"]
TOP_K_GRID = [3, 5, None]        # None = lấy theo thứ hạng tới khi đầy chunk
WINDOW_GRID = [0, 1]
SMOKE = smoke_mode()
if SMOKE:
    TUNE_PER_DATASET, TOP_K_GRID, WINDOW_GRID = 12, [3, None], [1]

# %%
import json
import time

import numpy as np
import pandas as pd
import torch

from mcplus import DATASETS_11, DATASETS_PAPER_10
from mcplus.chunking import split_claim
from mcplus.data import load_split, stratified_subset
from mcplus.evidence import EvidenceConfig, EvidenceSelector, SentenceRanker
from mcplus.io_utils import load_table, output_root, save_table, write_manifest
from mcplus.metrics import macro, macro_bacc_arrays, per_dataset
from mcplus.scorer import FastScorer

OUT = output_root() / "evidence"
OUT.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 2. Dữ liệu + điểm baseline từ notebook 01

# %%
data = {}
for split in ("dev", "test"):
    base = load_table(f"baseline/{MODEL_NAME}__{split}.csv.gz")
    text = load_split(split)[["uid", "doc", "claim"]]
    data[split] = base.merge(text, on="uid", how="inner", validate="one_to_one")
    print(split, len(data[split]), "rows;", round(100 * (data[split]["n_chunks"] > 1).mean(), 1), "% nhiều chunk")

# %%
scorer = FastScorer(MODEL_NAME, batch_size=16)
device = "cuda" if torch.cuda.is_available() else "cpu"
selector = EvidenceSelector(scorer, SentenceRanker(DENSE_MODEL, device=device))

# %% [markdown]
# ## 3. Hàm chạy một cấu hình

# %%
def needs_scoring(frame: pd.DataFrame, cfg: EvidenceConfig) -> np.ndarray:
    multi = (frame["n_chunks"] > 1).to_numpy()
    if cfg.split_claim:
        return multi | (frame["n_claim_sents"] > 1).to_numpy()
    return multi


def run_config(frame: pd.DataFrame, cfg: EvidenceConfig) -> np.ndarray:
    """Điểm e cho mọi dòng: dòng không bị ảnh hưởng giữ nguyên p_max của baseline."""
    e = frame["p_max"].to_numpy(dtype=float).copy()
    mask = needs_scoring(frame, cfg)
    if mask.any():
        sub = frame[mask]
        e[mask] = selector.score(sub["doc"].tolist(), sub["claim"].tolist(), cfg, split_fn=split_claim)
    return e


def run_c_only(frame: pd.DataFrame) -> np.ndarray:
    """Ablation C thuần: giữ cách cắt chunk của baseline, chỉ tách claim nhiều câu."""
    c = frame["p_max"].to_numpy(dtype=float).copy()
    mask = (frame["n_claim_sents"] > 1).to_numpy()
    if mask.any():
        sub = frame[mask]
        c[mask] = scorer.score_split_claims(sub["doc"].tolist(), sub["claim"].tolist(), split_claim)
    return c


def evaluate(frame: pd.DataFrame, e: np.ndarray) -> dict:
    y, groups = frame["label"].to_numpy(), frame["dataset"].to_numpy()
    out = {}
    for mode, score in (("replace", e), ("union", np.maximum(frame["p_max"].to_numpy(), e))):
        pred = (score > 0.5).astype(int)
        out[f"{mode}_macro11"] = 100 * macro_bacc_arrays(y, pred, groups, DATASETS_11)
        out[f"{mode}_macro10"] = 100 * macro_bacc_arrays(y, pred, groups, DATASETS_PAPER_10)
    return out

# %% [markdown]
# ## 4. Chọn siêu tham số trên tập con dev (3 giai đoạn)

# %%
tune = stratified_subset(data["dev"], per_dataset=TUNE_PER_DATASET, min_per_label=30)
print("Tập con dev:", len(tune), "dòng;", int((tune["n_chunks"] > 1).sum()), "dòng nhiều chunk")
t0 = time.perf_counter()
selector.precompute_embeddings(tune["doc"].tolist(), tune["claim"].tolist())
print(f"Embedding xong trong {time.perf_counter() - t0:.0f}s")

base_eval = evaluate(tune, tune["p_max"].to_numpy())
tuning = [{"config": "baseline", "stage": 0, "replace_macro11": base_eval["replace_macro11"],
           "replace_macro10": base_eval["replace_macro10"], "union_macro11": np.nan, "union_macro10": np.nan}]
scores_cache = {}


def try_config(cfg: EvidenceConfig, stage: int):
    if cfg.name() in scores_cache:
        return
    t = time.perf_counter()
    e = run_config(tune, cfg)
    scores_cache[cfg.name()] = e
    row = {"config": cfg.name(), "stage": stage, **cfg.as_dict(), **evaluate(tune, e),
           "seconds": round(time.perf_counter() - t, 1)}
    tuning.append(row)
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()})


def best_row(stage_max: int):
    table = pd.DataFrame(tuning)
    table = table[(table["stage"] >= 1) & (table["stage"] <= stage_max)]
    long = pd.concat([table.assign(mode=m, score=table[f"{m}_macro11"]) for m in ("replace", "union")])
    return long.sort_values("score", ascending=False, kind="stable").iloc[0]  # hòa điểm: giữ cấu hình thử trước


# Giai đoạn 1: so sánh bộ chọn câu (k = lấp đầy, cửa sổ 1)
for sel in SELECTORS:
    try_config(EvidenceConfig(selector=sel, top_k=None, window=1), stage=1)
best1 = best_row(1)
print("-> bộ chọn tốt nhất:", best1["selector"], best1["mode"])

# Giai đoạn 2: k và cửa sổ với bộ chọn tốt nhất
for k in TOP_K_GRID:
    for w in WINDOW_GRID:
        try_config(EvidenceConfig(selector=best1["selector"], top_k=k, window=w), stage=2)
best2 = best_row(2)
BEST_CFG = EvidenceConfig(selector=best2["selector"],
                          top_k=None if pd.isna(best2["top_k"]) else int(best2["top_k"]),
                          window=int(best2["window"]))
BEST_MODE = best2["mode"]
print("-> cấu hình A chốt:", BEST_CFG.name(), "| mode:", BEST_MODE)

# Giai đoạn 3: tách claim (C) trên cấu hình đã chốt, và C thuần trên baseline
split_cfg = EvidenceConfig(BEST_CFG.selector, BEST_CFG.top_k, BEST_CFG.window, split_claim=True)
try_config(split_cfg, stage=3)
c_only = run_c_only(tune)
tuning.append({"config": "C_only(baseline_chunks+split)", "stage": 3,
               "replace_macro11": evaluate(tune, c_only)["replace_macro11"],
               "replace_macro10": evaluate(tune, c_only)["replace_macro10"]})

tuning_table = pd.DataFrame(tuning).round(2)
save_table(tuning_table, f"evidence/{MODEL_NAME}__tuning.csv")
display(tuning_table)

# %% [markdown]
# ## 5. Chạy cấu hình đã chốt trên toàn bộ dev và test
#
# Mỗi split lưu 3 điểm: `e_score` (A), `e_split` (A + C), `c_only` (C trên baseline). Notebook 04
# dùng các điểm này làm đặc trưng cho cải tiến B.

# %%
def apply_mode(p_max, e, mode):
    return np.maximum(p_max, e) if mode == "union" else e


timing = {}
for split in ("dev", "test"):
    path = OUT / f"{MODEL_NAME}__{split}.csv.gz"
    if path.exists():
        print("[skip]", path.name)
        continue
    frame = data[split]
    t, f0 = time.perf_counter(), scorer.n_forward
    if BEST_CFG.selector in ("dense", "hybrid"):
        selector.precompute_embeddings(frame.loc[frame["n_chunks"] > 1, "doc"].tolist(),
                                       frame["claim"].tolist())
    e_score = run_config(frame, BEST_CFG)
    t_a, f_a = time.perf_counter() - t, scorer.n_forward - f0
    e_split = e_score.copy()
    mask = (frame["n_claim_sents"] > 1).to_numpy()
    if mask.any():  # chỉ claim nhiều câu mới khác e_score
        sub = frame[mask]
        e_split[mask] = selector.score(sub["doc"].tolist(), sub["claim"].tolist(), split_cfg,
                                       split_fn=split_claim)
    c_score = run_c_only(frame)
    p_max = frame["p_max"].to_numpy()
    out = frame[["uid", "dataset", "label", "n_chunks", "n_claim_sents", "p_max"]].copy()
    out["e_score"], out["e_split"], out["c_only"] = e_score.round(6), e_split.round(6), c_score.round(6)
    out["pred_A"] = (apply_mode(p_max, e_score, BEST_MODE) > 0.5).astype(int)
    out["pred_AC"] = (apply_mode(p_max, e_split, BEST_MODE) > 0.5).astype(int)
    out["pred_C"] = (c_score > 0.5).astype(int)
    out.to_csv(path, index=False)
    timing[split] = {"seconds_A": round(t_a, 1), "forward_passes_A": int(f_a),
                     "forward_passes_baseline": int(frame["n_chunks"].sum()), "rows": len(frame)}
    print(split, timing[split])

# %% [markdown]
# ## 6. Kết quả trên test (zero-shot, ngưỡng 0.5)

# %%
test = pd.read_csv(OUT / f"{MODEL_NAME}__test.csv.gz")
cols = {"Baseline": "pred", "A": "pred_A", "A+C": "pred_AC", "C": "pred_C"}
test["pred"] = (test["p_max"] > 0.5).astype(int)
table = pd.DataFrame({name: per_dataset(test, col)["bacc"] for name, col in cols.items()}).reindex(DATASETS_11)
table.loc["macro10"] = [macro(per_dataset(test, c), DATASETS_PAPER_10) for c in cols.values()]
table.loc["macro11"] = [macro(per_dataset(test, c), DATASETS_11) for c in cols.values()]
table = table.round(2)
save_table(table.reset_index(names="dataset"), f"evidence/{MODEL_NAME}__test_metrics.csv")
display(table)

# %%
multi = test[test["n_chunks"] > 1]
print("Chỉ xét văn bản nhiều chunk (test):")
display(pd.DataFrame({name: per_dataset(multi, col)["bacc"] for name, col in cols.items()}).round(2))

# %%
write_manifest(f"evidence/{MODEL_NAME}__manifest.json", notebook="03_evidence_selection",
               model=MODEL_NAME, dense_model=DENSE_MODEL, tune_rows=len(tune),
               best_config=BEST_CFG.as_dict(), best_mode=BEST_MODE, split_config=split_cfg.as_dict(),
               timing=timing, smoke=SMOKE)
print("Done. Tải results/evidence về và commit vào repo.")
