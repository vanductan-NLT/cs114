# results/

Kết quả do các notebook ghi ra (trên Kaggle là `/kaggle/working/results/`). Sau mỗi lần chạy, tải về
và chép vào đây rồi commit. **Không commit** các thư mục `_shards_*` (file tạm để resume).

| Thư mục | Notebook | File chính |
|---|---|---|
| `baseline/` | 01 | `<model>__dev.csv.gz`, `<model>__test.csv.gz` (điểm từng chunk, mỗi dòng một ví dụ), `<model>__metrics.csv`, `<model>__fidelity.json`, `<model>__manifest.json` |
| `analysis/` | 02 | `dataset_summary.csv`, `<model>__dev_by_chunks.csv`, `<model>__dev_within_dataset.csv`, `<model>__dev_spread_proxy.csv`, `<model>__diagnostics.json`, `<model>__fn_annotation_sheet.csv` |
| `evidence/` | 03 | `<model>__tuning.csv`, `<model>__dev.csv.gz`, `<model>__test.csv.gz`, `<model>__test_metrics.csv`, `<model>__manifest.json` |
| `aggregation/` | 04 | `<model>__test_predictions.csv.gz`, `<model>__cv.csv`, `<model>__ablation_test.csv`, `<model>__calibration.csv`, `<model>__lr_coefficients.csv`, `<model>__manifest.json` |
| `final/` | 05 | `<model>__main_table.csv`, `significance_*.csv`, `chunk_effect.csv`, `efficiency.csv`, `<model>__flips_*.csv`, `tables.md`, `report_filled.md` |
| `figures/` | 02, 04, 05 | hình dùng cho báo cáo và slide |

Khóa nối giữa các file là cột `uid` = `"<split>:<vị trí dòng trong split gốc>"`.
Các file `.csv.gz` không chứa văn bản gốc (nhẹ, và không phát tán lại dataset); cần văn bản
thì nối với LLM-AggreFact theo `uid`.
