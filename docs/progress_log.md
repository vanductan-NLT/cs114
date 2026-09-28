# Kế hoạch và nhật ký tiến độ (Time managing — 10%)

**Hạn nộp:** khoảng cuối tháng 12/2026. **Mục tiêu:** xong trong 6 tuần (28/09 → 08/11/2026), phần
còn lại để dự phòng. Nhóm 1 người, khoảng 10–12 giờ/tuần.

Mỗi tuần: làm các ô `[ ]`, ghi *Nhật ký* (ngày, số giờ, vấn đề gặp phải, cách xử lý), commit và
push kết quả. Tạo 6 **GitHub Milestones** (Tuần 1 … Tuần 6) và gắn issue tương ứng; đây là minh
chứng quản lý thời gian khi thuyết trình.

| Tuần | Thời gian | Trọng tâm | Sản phẩm | Phần điểm |
|---|---|---|---|---|
| 1 | 28/09 – 04/10 | Chốt đề, dựng repo, chạy baseline | `01_baseline`, `docs/proposal.md` | Intro, Time |
| 2 | 05/10 – 11/10 | Phân tích dữ liệu và lỗi baseline, quyết định go/no-go | `02_data_analysis`, phiếu 50 lỗi | Dataset (20%) |
| 3 | 12/10 – 18/10 | Cải tiến A (+ ablation C) | `03_evidence_selection` | Method |
| 4 | 19/10 – 25/10 | Cải tiến B + calibration | `04_learned_aggregation` | Method (25%) |
| 5 | 26/10 – 01/11 | Đánh giá cuối, kiểm định, chi phí, phân tích lỗi | `05_final_evaluation`, bảng/hình | Evaluation (25%) |
| 6 | 02/11 – 08/11 | Viết báo cáo, slide, tập thuyết trình | `docs/report.md`, slide | Conclusion, Time |
| 7+ | 09/11 → hạn | Dự phòng; làm thêm nếu dư (Flan-T5, hướng ModernBERT) | – | – |

---

## Tuần 1 (28/09 – 04/10): Chốt đề + baseline

- [ ] Gửi email xin duyệt đề tài (bài báo MiniCheck, EMNLP 2024) và lưu phản hồi.
- [ ] Push repo lên GitHub (cấu trúc sẵn: `src/`, `nb_src/`, `notebooks/`, `docs/`, `results/`).
- [ ] Trên Hugging Face: đăng nhập, vào trang `lytang/LLM-AggreFact` bấm *Agree*, tạo token read.
- [ ] Trên Kaggle: *Add-ons → Secrets* thêm `HF_TOKEN`; bật Internet; chọn GPU T4.
- [ ] Chạy `notebooks/01_baseline.ipynb` với `MODEL_NAME = "deberta-v3-large"`.
  - [ ] Kiểm tra ô *fidelity*: `chunk_count_match = 1.0`, `label_agreement ≥ 0.995`.
  - [ ] Kiểm tra macro 10 dataset trên test gần 72.6 (Table 2 của bài báo). Lệch khoảng ±1 điểm là bình thường.
- [ ] (Nên làm) Chạy lại 01 với `MODEL_NAME = "flan-t5-large"` (bài báo: 74.7).
- [ ] Tải `results/baseline/` về, commit (không commit thư mục `_shards_*`).
- [ ] Đọc lại `docs/proposal.md`, sửa nếu cô góp ý.

**Nhật ký:**
- …

## Tuần 2 (05/10 – 11/10): Dữ liệu + chẩn đoán

- [ ] Chạy `02_data_analysis.ipynb` (CPU, cần `HF_TOKEN` và output tuần 1).
- [ ] Đọc bảng *TPR theo số chunk trong từng dataset* và *proxy bằng chứng rải rác*.
- [ ] Điền phiếu `results/analysis/deberta-v3-large__fn_annotation_sheet.csv` (50 dòng, khoảng 1–2 giờ).
- [ ] Ghi kết luận go/no-go cho A vào nhật ký, kèm số liệu.
- [ ] Viết nháp mục 2 (Dataset) trong `docs/report.md`, chèn hình từ `results/figures/`.

**Nhật ký:**
- …

## Tuần 3 (12/10 – 18/10): Cải tiến A (+ C)

- [ ] Chạy `03_evidence_selection.ipynb` (GPU; Add Input = output notebook 01).
- [ ] Ghi lại cấu hình được chọn (bộ chọn, k, cửa sổ, mode) và bảng tuning.
- [ ] Xem kết quả test của A, A+C, C. **Không** quay lại sửa cấu hình theo kết quả test.
- [ ] Viết nháp mục 3.2–3.3 của báo cáo.

**Nhật ký:**
- …

## Tuần 4 (19/10 – 25/10): Cải tiến B

- [ ] Chạy `04_learned_aggregation.ipynb` (CPU).
- [ ] Ghi mô hình và nhóm đặc trưng được chọn theo CV, hệ số LR, bảng calibration.
- [ ] Viết nháp mục 3.4 của báo cáo.

**Nhật ký:**
- …

## Tuần 5 (26/10 – 01/11): Đánh giá cuối

- [ ] Chạy `05_final_evaluation.ipynb` (CPU). Kiểm tra `results/final/report_filled.md`.
- [ ] Chép `report_filled.md` đè lên `docs/report.md`.
- [ ] Đọc 10–20 ca trong `flips_A.csv` và `flips_B_LODO.csv`, chọn 2–3 ví dụ cho mục 4.8 và slide.
- [ ] Viết các ô ✍️ ĐIỀN của mục 4.

**Nhật ký:**
- …

## Tuần 6 (02/11 – 08/11): Báo cáo + slide

- [ ] Hoàn thiện mục 1, 5, 6 của báo cáo. Soát lại mọi con số với file CSV.
- [ ] Làm slide theo `docs/slides_outline.md` (khoảng 12 slide, 10–12 phút).
- [ ] Tập thuyết trình 2 lần, bấm giờ. Chuẩn bị trả lời các câu hỏi dự kiến trong outline.
- [ ] Tag release `v1.0` trên GitHub.

**Nhật ký:**
- …

---

## Rủi ro và phương án

| Rủi ro | Dấu hiệu | Phương án |
|---|---|---|
| Hết quota GPU Kaggle | Tuần 1 hoặc 3 chưa chạy xong | Notebook tự resume theo shard. Chuyển sang Colab (cùng notebook, đặt `HF_TOKEN` bằng biến môi trường) |
| A không cải thiện | Tuần 2 go/no-go âm tính, hoặc tuning A ≤ baseline | Vẫn báo cáo A như một kết quả âm có phân tích; trọng tâm chuyển sang B và C |
| Lệch nhiều so với bài báo | Macro-10 lệch > 1.5 điểm | Kiểm tra lại fidelity, phiên bản transformers và revision dataset; ghi rõ trong báo cáo |
| Trễ tiến độ | Một tuần chưa xong | Dùng 1–2 tuần dự phòng (tuần 7+). Bỏ Flan-T5, giữ DeBERTa làm kết quả chính |
