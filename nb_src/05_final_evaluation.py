# %% [markdown]
# # 05 · Đánh giá cuối cùng + tự điền báo cáo (Tuần 5)
#
# Gom kết quả của notebook 01–04, so với bài báo gốc, kiểm định thống kê, đo chi phí và phân
# tích lỗi, rồi **tự điền các bảng số vào `docs/report.md`** (các khối `<!--AUTO:...-->`).
#
# **Cần:** các thư mục `results/baseline`, `results/analysis`, `results/evidence`,
# `results/aggregation` (commit vào repo, hoặc Add Input trên Kaggle). **Chạy trên CPU.**
#
# **Output** (`results/final/`): bảng CSV, `tables.md`, `report_filled.md`, danh sách ca đổi nhãn.

# %%
# --- Bootstrap: lấy code của repo (src/mcplus) ---
import os, sys, subprocess
from pathlib import Path

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
import json

import numpy as np
import pandas as pd

from mcplus import DATASETS_11, DATASETS_PAPER_10, MODEL_CHECKPOINTS, PAPER_TABLE2
from mcplus.io_utils import find_input, load_table, output_root, save_table
from mcplus.metrics import macro, mcnemar, paired_bootstrap, per_dataset
from mcplus.nb import display, env_summary, savefig, setup_plotting, smoke_mode
from mcplus.report import fill_template, md_table

OUT = output_root() / "final"
OUT.mkdir(parents=True, exist_ok=True)
plt = setup_plotting()
env_summary()
N_BOOT = 2000 if smoke_mode() else 10000

# %% [markdown]
# ## 1. Nạp dự đoán test của mọi phương pháp

# %%
MODELS = []
preds = {}
for model in MODEL_CHECKPOINTS:
    try:
        preds[model] = load_table(f"aggregation/{model}__test_predictions.csv.gz")
        MODELS.append(model)
    except FileNotFoundError:
        pass
assert MODELS, "Chưa có kết quả notebook 04 cho model nào."
PRIMARY = "deberta-v3-large" if "deberta-v3-large" in MODELS else MODELS[0]
print("Models:", MODELS, "| model chính:", PRIMARY)

# Nhóm 1: zero-shot, ngưỡng cố định 0.5, không dùng nhãn để tune ngưỡng hay huấn luyện.
# Nhóm 2: có dùng nhãn dev (tune ngưỡng hoặc huấn luyện bộ tổng hợp).
ZERO_SHOT = {"Baseline": "pred_base", "A": "pred_A", "A+C": "pred_AC", "C": "pred_C"}
USE_DEV = {"Thr-global": "pred_thr_global", "Thr-per-dataset": "pred_thr_dataset",
           "Thr-LODO": "pred_thr_lodo", "B in-domain": "pred_B_in", "B LODO": "pred_B_lodo"}
METHODS = {**ZERO_SHOT, **USE_DEV}

# %% [markdown]
# ## 2. Bảng chính: BAcc (%) theo dataset, so với bài báo

# %%
ORDER = DATASETS_PAPER_10 + ["RAGTruth"]
main_tables = {}
for model in MODELS:
    df = preds[model]
    table = pd.DataFrame({m: per_dataset(df, c)["bacc"] for m, c in METHODS.items()}).reindex(ORDER)
    table.insert(0, "Paper", [PAPER_TABLE2.get(model, {}).get(d, np.nan) for d in ORDER])
    table.loc["Avg (10, như bài báo)"] = [PAPER_TABLE2.get(model, {}).get("avg", np.nan)] + \
        [macro(per_dataset(df, c), DATASETS_PAPER_10) for c in METHODS.values()]
    table.loc["Avg (11, có RAGTruth)"] = [np.nan] + [macro(per_dataset(df, c), DATASETS_11) for c in METHODS.values()]
    main_tables[model] = table.round(2)
    save_table(table.round(2).reset_index(names="dataset"), f"final/{model}__main_table.csv")
    print(model)
    display(main_tables[model])

# %% [markdown]
# ## 3. Kiểm định thống kê so với baseline của chính mình
#
# Bootstrap có cặp (10.000 lần, lấy mẫu lại trong từng ô dataset × nhãn) cho hiệu BAcc trung bình;
# McNemar chính xác cho từng dataset.

# %%
sig_rows, mcn_rows = [], []
for model in MODELS:
    df = preds[model]
    for name, col in METHODS.items():
        if name == "Baseline":
            continue
        for label, dsets in (("10", DATASETS_PAPER_10), ("11", DATASETS_11)):
            r = paired_bootstrap(df, "pred_base", col, datasets=dsets, n_boot=N_BOOT, seed=0)
            sig_rows.append({"model": model, "method": name, "avg_over": label, **r})
    for name in ("A", "B LODO"):
        for d in ORDER:
            sub = df[df["dataset"] == d]
            if len(sub):
                mcn_rows.append({"model": model, "method": name, "dataset": d,
                                 **mcnemar(sub, "pred_base", METHODS[name])})
significance = pd.DataFrame(sig_rows).round(4)
mcnemar_table = pd.DataFrame(mcn_rows).round(4)
save_table(significance, "final/significance_bootstrap.csv")
save_table(mcnemar_table, "final/significance_mcnemar.csv")
display(significance[significance["model"] == PRIMARY])

# %% [markdown]
# ## 4. Tác động theo số chunk (văn bản ngắn vs dài)

# %%
chunk_rows = []
for model in MODELS:
    df = preds[model]
    for part, sub in (("1 chunk", df[df["n_chunks"] == 1]), ("≥2 chunk", df[df["n_chunks"] > 1])):
        row = {"model": model, "subset": part, "n": len(sub)}
        for name in ("Baseline", "A", "B in-domain", "B LODO"):
            row[name] = macro(per_dataset(sub, METHODS[name]), DATASETS_11)
        chunk_rows.append(row)
chunk_effect = pd.DataFrame(chunk_rows).round(2)
save_table(chunk_effect, "final/chunk_effect.csv")
display(chunk_effect)

# %% [markdown]
# ## 5. Chi phí tính toán

# %%
eff_rows = []
for model in MODELS:
    base_m = json.loads(find_input(f"baseline/{model}__manifest.json").read_text(encoding="utf-8"))
    evid_m = json.loads(find_input(f"evidence/{model}__manifest.json").read_text(encoding="utf-8"))
    for split in ("test",):
        bt, et = base_m.get("timing", {}).get(split, {}), evid_m.get("timing", {}).get(split, {})
        rows = bt.get("rows") or et.get("rows")
        if not rows:
            continue
        mode = evid_m.get("best_mode", "union")
        passes_base = bt.get("pairs_scored", et.get("forward_passes_baseline"))
        passes_a_extra = et.get("forward_passes_A", np.nan)
        df = preds[model]
        single = int((df["n_chunks"] == 1).sum())
        passes_a = passes_base + passes_a_extra if mode == "union" else single + passes_a_extra
        eff_rows.append({
            "model": model, "split": split, "rows": rows,
            "baseline_passes_per_example": passes_base / rows,
            "A_passes_per_example": passes_a / rows,
            "baseline_seconds_per_1k": 1000 * bt.get("seconds", np.nan) / rows,
            "A_extra_seconds_per_1k": 1000 * et.get("seconds_A", np.nan) / rows,
            "A_mode": mode,
        })
efficiency = pd.DataFrame(eff_rows).round(3)
save_table(efficiency, "final/efficiency.csv")
display(efficiency)

# %% [markdown]
# ## 6. Các ca đổi nhãn (để đọc tay và lấy ví dụ minh họa cho slide)

# %%
df = preds[PRIMARY]
for name in ("A", "B LODO"):
    col = METHODS[name]
    fixed = df[(df["pred_base"] != df["label"]) & (df[col] == df["label"])]
    broken = df[(df["pred_base"] == df["label"]) & (df[col] != df["label"])]
    print(f"{name}: sửa đúng {len(fixed)} ca, làm sai {len(broken)} ca (so với baseline)")
    flips = pd.concat([fixed.assign(change="fixed"), broken.assign(change="broken")])
    save_table(flips[["uid", "dataset", "label", "n_chunks", "p_max", "score_A", "prob_B_lodo", "change"]],
               f"final/{PRIMARY}__flips_{name.replace(' ', '_')}.csv")

# %% [markdown]
# ## 7. Hình tổng hợp

# %%
t = main_tables[PRIMARY].loc[ORDER, ["Paper", "Baseline", "A", "B LODO"]]
fig, ax = plt.subplots(figsize=(9, 3.8))
x = np.arange(len(ORDER))
for i, col in enumerate(t.columns):
    ax.bar(x + (i - 1.5) * 0.2, t[col], width=0.2, label=col)
ax.set_xticks(x, [d.replace("AggreFact-", "").replace("TofuEval-", "") for d in ORDER], rotation=30, ha="right")
ax.set_ylim(50, 95)
ax.set_ylabel("BAcc (%)")
ax.set_title(f"{PRIMARY}: bài báo vs chạy lại vs cải tiến (test)")
ax.legend(ncol=4, fontsize=8)
savefig(fig, f"figures/{PRIMARY}__main_comparison.png")
plt.show()

# %% [markdown]
# ## 8. Tự điền báo cáo

# %%
def safe_table(relative, **kwargs):
    try:
        return load_table(relative, **kwargs)
    except FileNotFoundError:
        return None


blocks = {}
summary = safe_table("analysis/dataset_summary.csv")
if summary is not None:
    blocks["dataset_table"] = md_table(summary.set_index("dataset"), floatfmt=".1f")
diag_path = None
try:
    diag_path = find_input(f"analysis/{PRIMARY}__diagnostics.json")
except FileNotFoundError:
    pass
by_chunks = safe_table(f"analysis/{PRIMARY}__dev_by_chunks.csv")
if by_chunks is not None:
    text = md_table(by_chunks.set_index("n_chunks"), floatfmt=".1f")
    if diag_path:
        d = json.loads(diag_path.read_text(encoding="utf-8"))
        text += (f"\n\nTrên dev: {d['dev_pct_multi_chunk']}% ví dụ có ≥2 chunk; FN chiếm "
                 f"{d['dev_fn_share_of_errors']}% số lỗi; chênh TPR trong-dataset (nhiều chunk − 1 chunk) = "
                 f"{d['within_dataset_delta_tpr']} điểm; tỷ lệ 'bằng chứng rải rác' ở FN = {d['spread_pct_fn']}%, "
                 f"ở TP = {d['spread_pct_tp']}%.")
    blocks["diagnostics"] = text
base_metrics = safe_table(f"baseline/{PRIMARY}__metrics.csv")
if base_metrics is not None:
    blocks["baseline_table"] = md_table(base_metrics.set_index("dataset"))
tuning = safe_table(f"evidence/{PRIMARY}__tuning.csv")
if tuning is not None:
    keep = [c for c in ("config", "stage", "replace_macro11", "union_macro11", "replace_macro10", "union_macro10")
            if c in tuning.columns]
    blocks["tuning_A"] = md_table(tuning[keep], index=False)
blocks["main_table"] = "\n\n".join(
    f"**{m}** (`{MODEL_CHECKPOINTS[m]}`)\n\n" + md_table(
        main_tables[m], bold_max_rows="all", bold_cols=list(METHODS)) for m in MODELS)
sig = significance[significance["model"] == PRIMARY].copy()
sig["CI95"] = sig.apply(lambda r: f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]", axis=1)
blocks["significance"] = md_table(sig[["method", "avg_over", "diff", "CI95", "p_value"]]
                                  .rename(columns={"diff": "Δ BAcc"}), index=False, floatfmt=".3f")
blocks["chunks_effect"] = md_table(chunk_effect, index=False)
cv = safe_table(f"aggregation/{PRIMARY}__cv.csv")
abl = safe_table(f"aggregation/{PRIMARY}__ablation_test.csv")
if cv is not None:
    blocks["ablation_B"] = md_table(cv, index=False) + ("\n\n" + md_table(abl, index=False) if abl is not None else "")
calib = safe_table(f"aggregation/{PRIMARY}__calibration.csv")
if calib is not None:
    blocks["calibration"] = md_table(calib, index=False, floatfmt=".4f")
blocks["efficiency"] = md_table(efficiency, index=False, floatfmt=".3f") if len(efficiency) else "(chưa có số liệu thời gian)"

(OUT / "tables.md").write_text("\n\n".join(f"## {k}\n\n{v}" for k, v in blocks.items()), encoding="utf-8")
template = (REPO / "docs" / "report.md").read_text(encoding="utf-8")
(OUT / "report_filled.md").write_text(fill_template(template, blocks), encoding="utf-8")
print("Đã ghi", OUT / "report_filled.md", "— chép đè lên docs/report.md sau khi kiểm tra.")
