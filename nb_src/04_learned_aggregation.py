# %% [markdown]
# # 04 · Cải tiến B: tổng hợp điểm bằng mô hình học + hiệu chỉnh (Tuần 4)
#
# **Ý tưởng.** MiniCheck quyết định bằng hai luật cố định: *lấy max theo chunk* rồi *so với 0.5*.
# Max theo chunk bỏ qua hình dạng phân bố điểm (ví dụ 1 chunk 0.55 và 9 chunk 0.01) và
# văn bản càng nhiều chunk thì càng dễ có một chunk "lỡ" vượt 0.5. Ta thay hai luật này bằng một
# bộ phân loại nhỏ (Logistic Regression / Gradient Boosting) học trên **dev** từ các đặc trưng:
# thống kê điểm theo chunk, điểm của ngữ cảnh chọn lọc (A), độ dài, độ trùng từ.
#
# **Ba cách đánh giá** (đều không đụng nhãn test):
# - *Tune ngưỡng trên dev* (đối chứng): 1 ngưỡng chung, hoặc 1 ngưỡng/dataset (kiểu FactCG).
# - *B in-domain*: huấn luyện trên toàn bộ dev → dự đoán test.
# - *B leave-one-dataset-out (LODO)*: muốn dự đoán dataset d thì huấn luyện trên dev của 10 dataset
#   còn lại → đo khả năng dùng cho domain chưa từng có nhãn (gần với tinh thần zero-shot của bài báo).
#
# Chọn loại mô hình và nhóm đặc trưng bằng **cross-validation 5-fold trên dev**, không nhìn test.
#
# **Cần:** output notebook 01 và 03. **Chạy trên CPU** (vài phút).
#
# **Output** (`results/aggregation/`): dự đoán test của mọi phương pháp, bảng CV, bảng ablation,
# bảng calibration, hình reliability.

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

from mcplus import DATASETS_11, DATASETS_PAPER_10
from mcplus.aggregator import cv_select, fit_predict, leave_one_dataset_out, lr_coefficients
from mcplus.features import build_features, feature_sets
from mcplus.io_utils import find_input, load_table, output_root, save_table, write_manifest
from mcplus.metrics import brier, ece, macro, per_dataset, tune_threshold
from mcplus.nb import display, env_summary, savefig, setup_plotting, smoke_mode

MODEL_NAME = os.environ.get("MCPLUS_MODEL", "deberta-v3-large")
KINDS = ["lr", "hgb"]
SEED = 0
OUT = output_root() / "aggregation"
OUT.mkdir(parents=True, exist_ok=True)
plt = setup_plotting()
env_summary()

# %% [markdown]
# ## 1. Nạp điểm từ notebook 01 (baseline) và 03 (A, C) rồi tạo đặc trưng

# %%
manifest_a = json.loads(find_input(f"evidence/{MODEL_NAME}__manifest.json").read_text(encoding="utf-8"))
A_MODE = manifest_a["best_mode"]
print("Mode của A chọn ở notebook 03:", A_MODE)


def load(split):
    base = load_table(f"baseline/{MODEL_NAME}__{split}.csv.gz")
    evid = load_table(f"evidence/{MODEL_NAME}__{split}.csv.gz")[
        ["uid", "e_score", "e_split", "c_only", "pred_A", "pred_AC", "pred_C"]]
    frame = base.merge(evid, on="uid", how="inner", validate="one_to_one").reset_index(drop=True)
    score_a = np.maximum(frame["p_max"], frame["e_score"]) if A_MODE == "union" else frame["e_score"]
    frame["score_A"] = score_a
    return pd.concat([frame, build_features(frame)], axis=1)


dev, test = load("dev"), load("test")
SETS = feature_sets(dev.columns)
print({k: len(v) for k, v in SETS.items()}, "| dev", len(dev), "| test", len(test))

# %% [markdown]
# ## 2. Đối chứng: tune ngưỡng trên dev

# %%
y_dev, g_dev = dev["label"].to_numpy(), dev["dataset"].to_numpy()
thr = {"global_pmax": tune_threshold(dev["p_max"].to_numpy(), y_dev, g_dev),
       "global_A": tune_threshold(dev["score_A"].to_numpy(), y_dev, g_dev),
       "per_dataset_pmax": {d: tune_threshold(dev.loc[g_dev == d, "p_max"].to_numpy(), y_dev[g_dev == d])
                            for d in DATASETS_11 if (g_dev == d).any()}}
test["pred_base"] = (test["p_max"] > 0.5).astype(int)
test["pred_thr_global"] = (test["p_max"] > thr["global_pmax"]).astype(int)
test["pred_A_thr_global"] = (test["score_A"] > thr["global_A"]).astype(int)
test["pred_thr_dataset"] = [int(p > thr["per_dataset_pmax"].get(d, 0.5)) for p, d in zip(test["p_max"], test["dataset"])]
lodo_thr = {}
for d in test["dataset"].unique():
    keep = g_dev != d
    lodo_thr[d] = tune_threshold(dev.loc[keep, "p_max"].to_numpy(), y_dev[keep], g_dev[keep])
test["pred_thr_lodo"] = [int(p > lodo_thr[d]) for p, d in zip(test["p_max"], test["dataset"])]
print(json.dumps({"global_pmax": thr["global_pmax"], "global_A": thr["global_A"]}, indent=1))

# %% [markdown]
# ## 3. Chọn mô hình B bằng cross-validation trên dev

# %%
cv_rows, oof_store = [], {}
for set_name, cols in SETS.items():
    for kind in KINDS:
        params, table, oof = cv_select(kind, dev, cols, seed=SEED)
        best = table["cv_macro_bacc"].max()
        cv_rows.append({"features": set_name, "kind": kind, "params": json.dumps(params), "cv_macro_bacc": best})
        oof_store[(set_name, kind)] = (params, oof)
cv_table = pd.DataFrame(cv_rows).sort_values("cv_macro_bacc", ascending=False).round(2)
save_table(cv_table, f"aggregation/{MODEL_NAME}__cv.csv")
display(cv_table)
BEST_SET, BEST_KIND = cv_table.iloc[0]["features"], cv_table.iloc[0]["kind"]
BEST_PARAMS = json.loads(cv_table.iloc[0]["params"])
print("-> B chốt:", BEST_KIND, BEST_SET, BEST_PARAMS)

# %% [markdown]
# ## 4. Dự đoán test: B in-domain và B LODO

# %%
cols = SETS[BEST_SET]
test["prob_B_in"] = fit_predict(BEST_KIND, BEST_PARAMS, dev, test, cols, seed=SEED)
test["prob_B_lodo"] = leave_one_dataset_out(BEST_KIND, BEST_PARAMS, dev, test, cols, seed=SEED)
test["pred_B_in"] = (test["prob_B_in"] > 0.5).astype(int)
test["pred_B_lodo"] = (test["prob_B_lodo"] > 0.5).astype(int)

methods = {"Baseline": "pred_base", "A": "pred_A", "Thr-global": "pred_thr_global",
           "Thr-per-dataset": "pred_thr_dataset", "Thr-LODO": "pred_thr_lodo",
           "A+Thr-global": "pred_A_thr_global", "B in-domain": "pred_B_in", "B LODO": "pred_B_lodo"}
summary = pd.DataFrame({m: per_dataset(test, c)["bacc"] for m, c in methods.items()}).reindex(DATASETS_11)
summary.loc["macro10"] = [macro(per_dataset(test, c), DATASETS_PAPER_10) for c in methods.values()]
summary.loc["macro11"] = [macro(per_dataset(test, c), DATASETS_11) for c in methods.values()]
display(summary.round(2))

# %% [markdown]
# ## 5. Ablation của B (chỉ để báo cáo — lựa chọn đã chốt bằng CV ở mục 3)

# %%
abl = []
for (set_name, kind), (params, _) in oof_store.items():
    prob = leave_one_dataset_out(kind, params, dev, test, SETS[set_name], seed=SEED)
    tmp = test.assign(_p=(prob > 0.5).astype(int))
    abl.append({"features": set_name, "kind": kind, "setting": "LODO",
                "test_macro10": macro(per_dataset(tmp, "_p"), DATASETS_PAPER_10),
                "test_macro11": macro(per_dataset(tmp, "_p"), DATASETS_11)})
ablation = pd.DataFrame(abl).round(2)
save_table(ablation, f"aggregation/{MODEL_NAME}__ablation_test.csv")
display(ablation)

# %% [markdown]
# ## 6. Hiệu chỉnh xác suất (calibration)
#
# ECE thấp nghĩa là xác suất dự đoán khớp tần suất thật. Max theo chunk có xu hướng quá tự tin.

# %%
calib_cols = {"MiniCheck max": "p_max", "A score": "score_A", "B in-domain": "prob_B_in", "B LODO": "prob_B_lodo"}
calib = pd.DataFrame([{"score": k, "ece": ece(test[c], test["label"]), "brier": brier(test[c], test["label"])}
                      for k, c in calib_cols.items()]).round(4)
save_table(calib, f"aggregation/{MODEL_NAME}__calibration.csv")
display(calib)

fig, ax = plt.subplots(figsize=(4.6, 4.2))
bins = np.linspace(0, 1, 11)
for name, col in calib_cols.items():
    idx = np.clip(np.digitize(test[col], bins) - 1, 0, 9)
    frame = pd.DataFrame({"b": idx, "p": test[col], "y": test["label"]}).groupby("b").mean()
    ax.plot(frame["p"], frame["y"], marker="o", ms=3, label=name)
ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=0.8)
ax.set_xlabel("Xác suất dự đoán (trung bình bin)")
ax.set_ylabel("Tỷ lệ supported thực tế")
ax.set_title("Reliability diagram (test)")
ax.legend(fontsize=8)
savefig(fig, f"figures/{MODEL_NAME}__reliability.png")
plt.show()

# %% [markdown]
# ## 7. Diễn giải: hệ số Logistic Regression (đặc trưng đã chuẩn hóa)

# %%
coef_params = BEST_PARAMS if BEST_KIND == "lr" else json.loads(
    cv_table[cv_table["kind"] == "lr"].iloc[0]["params"])
coefs = lr_coefficients(dev, cols, coef_params)
save_table(coefs.rename_axis("feature").reset_index(name="coef"), f"aggregation/{MODEL_NAME}__lr_coefficients.csv")
display(coefs.round(3))

# %%
keep = ["uid", "dataset", "label", "n_chunks", "n_claim_sents", "p_max", "score_A", "e_split", "c_only",
        "prob_B_in", "prob_B_lodo", "pred_base", "pred_A", "pred_AC", "pred_C", "pred_thr_global",
        "pred_thr_dataset", "pred_thr_lodo", "pred_A_thr_global", "pred_B_in", "pred_B_lodo"]
test[keep].to_csv(OUT / f"{MODEL_NAME}__test_predictions.csv.gz", index=False)
write_manifest(f"aggregation/{MODEL_NAME}__manifest.json", notebook="04_learned_aggregation",
               model=MODEL_NAME, a_mode=A_MODE, thresholds={"global_pmax": thr["global_pmax"],
               "global_A": thr["global_A"], "per_dataset_pmax": thr["per_dataset_pmax"], "lodo": lodo_thr},
               best_kind=BEST_KIND, best_features=BEST_SET, best_params=BEST_PARAMS, feature_columns=cols,
               smoke=smoke_mode())
print("Done. Tải results/aggregation về và commit vào repo.")
