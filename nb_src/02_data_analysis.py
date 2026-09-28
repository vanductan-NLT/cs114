# %% [markdown]
# # 02 · Phân tích dữ liệu và lỗi của baseline (Tuần 2)
#
# **Mục tiêu:** (1) mô tả LLM-AggreFact cho phần *Dataset* của báo cáo; (2) kiểm tra giả thuyết
# "MiniCheck bỏ sót claim đúng khi bằng chứng nằm rải ở nhiều chunk" **trên tập dev** (không
# nhìn test để khỏi "học lỏm" khi thiết kế cải tiến); (3) xuất phiếu để tự đọc 50 lỗi.
#
# **Cần:** `HF_TOKEN` (để đọc văn bản) và output của notebook 01
# (`results/baseline/<model>__dev.csv.gz`, `__test.csv.gz`). Chạy trên CPU là đủ.
#
# **Output** (`results/analysis/`): bảng thống kê, hình, `<model>__diagnostics.json`,
# `<model>__fn_annotation_sheet.csv`.

# %%
# --- Bootstrap: lấy code của repo (src/mcplus) ---
import os, sys, subprocess
from pathlib import Path

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
from mcplus.nb import ensure_nltk, hf_login, env_summary, smoke_mode, display, setup_plotting, savefig

ensure_nltk()
hf_login()
env_summary()
plt = setup_plotting()

# %%
import json

import numpy as np
import pandas as pd

from mcplus import DATASETS_11, DATASETS_PAPER_10
from mcplus.data import dataset_summary, load_split
from mcplus.evidence import tokenize_for_bm25
from mcplus.io_utils import load_table, output_root, save_table
from mcplus.metrics import rates
from mcplus.scorer import Chunker

MODEL_NAME = os.environ.get("MCPLUS_MODEL", "deberta-v3-large")
OUT = output_root() / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Tổng quan LLM-AggreFact

# %%
frames = {s: load_split(s) for s in ("dev", "test")}
base = {s: load_table(f"baseline/{MODEL_NAME}__{s}.csv.gz") for s in ("dev", "test")}
if smoke_mode():  # smoke test: chỉ giữ các dòng đã chạy ở notebook 01
    frames = {s: f[f["uid"].isin(base[s]["uid"])].reset_index(drop=True) for s, f in frames.items()}

summary = dataset_summary(frames["test"]).merge(
    dataset_summary(frames["dev"])[["dataset", "n", "pct_supported"]].rename(
        columns={"n": "dev_n", "pct_supported": "dev_pct_supported"}), on="dataset", how="left")
summary.insert(1, "in_paper_table2", summary["dataset"].isin(DATASETS_PAPER_10))
save_table(summary, "analysis/dataset_summary.csv")
display(summary)

# %%
fig, ax = plt.subplots(figsize=(8, 3.6))
order = summary.sort_values("pct_supported")
ax.barh(order["dataset"], order["pct_supported"], color="#4C78A8", label="test")
ax.scatter(order["dev_pct_supported"], order["dataset"], color="#F58518", zorder=3, label="dev")
ax.axvline(50, color="grey", lw=0.8, ls="--")
ax.set_xlim(0, 100)
ax.set_xlabel("% claim có nhãn supported")
ax.set_title("Mất cân bằng nhãn rất khác nhau giữa các dataset → dùng Balanced Accuracy")
ax.legend(loc="lower right")
savefig(fig, "figures/label_balance.png")
plt.show()

# %%
test_words = frames["test"].assign(doc_words=frames["test"]["doc"].str.split().str.len())
fig, ax = plt.subplots(figsize=(8, 3.8))
data = [test_words.loc[test_words["dataset"] == d, "doc_words"] for d in DATASETS_11]
try:
    ax.boxplot(data, orientation="horizontal", showfliers=False)  # matplotlib >= 3.10
except TypeError:
    ax.boxplot(data, vert=False, showfliers=False)
ax.set_yticks(range(1, 12), DATASETS_11)
ax.set_xscale("log")
ax.set_xlabel("Số từ của văn bản (thang log)")
ax.set_title("Độ dài văn bản theo dataset (test)")
savefig(fig, "figures/doc_length.png")
plt.show()

# %% [markdown]
# ## 2. Số chunk mà MiniCheck phải cắt

# %%
chunk_share = pd.DataFrame({
    s: base[s].groupby("dataset")["n_chunks"].apply(lambda x: round(100 * (x > 1).mean(), 1))
    for s in ("dev", "test")
}).reindex(DATASETS_11)
chunk_share.columns = ["dev_%multi_chunk", "test_%multi_chunk"]
chunk_share["test_mean_chunks"] = base["test"].groupby("dataset")["n_chunks"].mean().reindex(DATASETS_11).round(2)
save_table(chunk_share.reset_index(), f"analysis/{MODEL_NAME}__chunk_share.csv")
display(chunk_share)

# %% [markdown]
# ## 3. Lỗi của baseline trên DEV theo số chunk
#
# Nếu giả thuyết "bằng chứng bị cắt rời" đúng, TPR (tỷ lệ bắt được claim đúng) phải thấp hơn ở
# văn bản nhiều chunk. Vì mỗi dataset có độ dài khác nhau, so sánh *bên trong từng dataset*
# (1 chunk vs ≥2 chunk) rồi mới gộp, để tránh nhầm hiệu ứng dataset thành hiệu ứng chunk.

# %%
dev = base["dev"].copy()
dev["bucket"] = pd.cut(dev["n_chunks"], [0, 1, 2, 3, np.inf], labels=["1", "2", "3", "4+"])
rows = []
for bucket, g in dev.groupby("bucket", observed=True):
    r = rates(g["label"].values, g["pred"].values)
    errors = g[g["label"] != g["pred"]]
    rows.append({"n_chunks": bucket, "n": len(g), "n_pos": int(g["label"].sum()),
                 "tpr": 100 * r["tpr"], "tnr": 100 * r["tnr"], "bacc": 100 * r["bacc"],
                 "fn_share_of_errors": 100 * (errors["label"] == 1).mean() if len(errors) else np.nan})
by_bucket = pd.DataFrame(rows).round(1)
save_table(by_bucket, f"analysis/{MODEL_NAME}__dev_by_chunks.csv")
display(by_bucket)

# %%
MIN_CELL = 20  # chỉ so sánh khi mỗi nhóm có đủ mẫu dương/âm
rows = []
for name, g in dev.groupby("dataset"):
    single, multi = g[g["n_chunks"] == 1], g[g["n_chunks"] > 1]
    row = {"dataset": name, "n_single": len(single), "n_multi": len(multi)}
    for label, key in ((1, "tpr"), (0, "tnr")):
        s, m = single[single["label"] == label], multi[multi["label"] == label]
        if len(s) >= MIN_CELL and len(m) >= MIN_CELL:
            row[f"{key}_single"] = 100 * (s["pred"] == label).mean()
            row[f"{key}_multi"] = 100 * (m["pred"] == label).mean()
            row[f"delta_{key}"] = row[f"{key}_multi"] - row[f"{key}_single"]
    rows.append(row)
within = pd.DataFrame(rows).set_index("dataset").reindex(DATASETS_11).round(1)
save_table(within.reset_index(), f"analysis/{MODEL_NAME}__dev_within_dataset.csv")
display(within)
adj = {k: float(within[k].mean()) for k in ("delta_tpr", "delta_tnr") if k in within}
print("Trung bình trong-dataset (multi - single):", {k: round(v, 2) for k, v in adj.items()})

# %% [markdown]
# ## 4. Đo "bằng chứng rải rác" bằng một proxy tự động
#
# Với claim đúng (nhãn 1) nằm trong văn bản nhiều chunk: lấy các từ nội dung của claim có xuất
# hiện trong văn bản; `best_chunk_cov` = tỷ lệ các từ đó nằm trọn trong *chunk tốt nhất*.
# Nếu < 1 thì không chunk nào chứa đủ từ của claim → dấu hiệu bằng chứng bị cắt rời.
# So sánh tỷ lệ này giữa các ca MiniCheck bỏ sót (FN) và bắt đúng (TP).

# %%
chunker = Chunker(MODEL_NAME)
pos_multi = dev[(dev["label"] == 1) & (dev["n_chunks"] > 1)].merge(
    frames["dev"][["uid", "doc", "claim"]], on="uid")


def coverage(doc, claim):
    words = set(tokenize_for_bm25(claim)) & set(tokenize_for_bm25(doc))
    if not words:
        return np.nan
    chunks = chunker.chunk_doc(doc)["chunks"]
    return max(len(words & set(tokenize_for_bm25(c))) / len(words) for c in chunks)


pos_multi["best_chunk_cov"] = [coverage(d, c) for d, c in zip(pos_multi["doc"], pos_multi["claim"])]
pos_multi["outcome"] = np.where(pos_multi["pred"] == 1, "TP", "FN")
pos_multi["spread"] = pos_multi["best_chunk_cov"] < 1.0
spread = pos_multi.groupby("outcome").agg(n=("uid", "size"), spread_pct=("spread", "mean"),
                                          median_best_chunk_cov=("best_chunk_cov", "median"))
spread["spread_pct"] *= 100
spread = spread.round(2)
save_table(spread.reset_index(), f"analysis/{MODEL_NAME}__dev_spread_proxy.csv")
display(spread)

# %% [markdown]
# ## 5. Phiếu tự đọc 50 lỗi FN (văn bản nhiều chunk)
#
# Mở file CSV, đọc từng dòng và điền cột `evidence_location` bằng một trong các giá trị:
# `one_chunk` (bằng chứng nằm gọn trong 1 chunk nhưng model vẫn sai), `spread` (bằng chứng nằm
# ở ≥2 chunk), `not_found` (không tìm thấy bằng chứng rõ ràng), `label_noise` (nhãn có vẻ sai).

# %%
fn = pos_multi[pos_multi["outcome"] == "FN"]
per = max(1, int(np.ceil(50 / max(fn["dataset"].nunique(), 1))))
picked = [g.sample(min(len(g), per), random_state=0) for _, g in fn.groupby("dataset")]
sheet = (pd.concat(picked) if picked else fn).head(50).copy()


def render_chunks(doc, probs):
    chunks = chunker.chunk_doc(doc)["chunks"]
    probs = json.loads(probs)
    return "\n\n".join(f"=== CHUNK {i + 1}/{len(chunks)} (p={p:.3f}) ===\n{c}"
                       for i, (c, p) in enumerate(zip(chunks, probs)))


sheet["chunks_text"] = [render_chunks(d, p) for d, p in zip(sheet["doc"], sheet["chunk_probs"])]
sheet["evidence_location"] = ""
sheet["note"] = ""
cols = ["uid", "dataset", "claim", "n_chunks", "p_max", "best_chunk_cov", "chunks_text",
        "evidence_location", "note"]
save_table(sheet[cols], f"analysis/{MODEL_NAME}__fn_annotation_sheet.csv")
print(len(sheet), "dòng ->", OUT / f"{MODEL_NAME}__fn_annotation_sheet.csv")

# %% [markdown]
# ## 6. Tổng hợp và quyết định go/no-go cho cải tiến A

# %%
fn_spread = spread["spread_pct"].get("FN", np.nan)
tp_spread = spread["spread_pct"].get("TP", np.nan)
diag = {
    "model": MODEL_NAME,
    "dev_rows": int(len(dev)),
    "dev_pct_multi_chunk": round(100 * float((dev["n_chunks"] > 1).mean()), 2),
    "dev_fn_share_of_errors": round(100 * float((dev.loc[dev["label"] != dev["pred"], "label"] == 1).mean()), 2),
    "within_dataset_delta_tpr": round(adj.get("delta_tpr", np.nan), 2),
    "within_dataset_delta_tnr": round(adj.get("delta_tnr", np.nan), 2),
    "spread_pct_fn": fn_spread, "spread_pct_tp": tp_spread,
}
criteria = {
    "recall_drops_on_multi_chunk": bool(diag["within_dataset_delta_tpr"] < 0),
    "fn_more_spread_than_tp_by_10pts": bool((fn_spread - tp_spread) >= 10),
}
diag["criteria"] = criteria
diag["go_for_A"] = bool(any(criteria.values()))
(OUT / f"{MODEL_NAME}__diagnostics.json").write_text(json.dumps(diag, indent=2, default=float))
print(json.dumps(diag, indent=2, default=float))
print("\n=> Cải tiến A:", "TIẾP TỤC" if diag["go_for_A"] else
      "tín hiệu yếu - vẫn chạy A như một thí nghiệm, nhưng dồn trọng tâm vào B và C")

# %% [markdown]
# **Việc cần làm sau notebook này**
# 1. Điền phiếu `fn_annotation_sheet.csv` (khoảng 1–2 giờ), đếm tỷ lệ `spread` và ghi vào
#    `docs/progress_log.md`. Nếu ≥ 30% lỗi FN là `spread` thì đây là bằng chứng định tính mạnh cho A.
# 2. Chép bảng `dataset_summary.csv` và các hình trong `results/figures/` vào phần Dataset của báo cáo.
