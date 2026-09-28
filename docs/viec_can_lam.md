# Việc bạn cần làm (checklist)

Code, notebook và khung báo cáo đã có sẵn. Phần máy không làm thay được là **chạy trên GPU
Kaggle/Colab** (dataset cần token của bạn), **đọc và kiểm tra kết quả**, và **viết nhận xét**.
Chi tiết từng tuần nằm ở [`progress_log.md`](progress_log.md).

## 0. Làm một lần (khoảng 20 phút)

- [ ] Gửi email xin duyệt đề cho cô. Lưu phản hồi (chụp màn hình) để đưa vào phần Time managing.
- [ ] Review và **merge Pull Request** vào `main`. Notebook tự clone `main` nên phải merge trước khi chạy.
- [ ] Hugging Face:
  - [ ] Đăng nhập, mở https://huggingface.co/datasets/lytang/LLM-AggreFact, bấm **Agree** (dataset cần xin quyền).
  - [ ] Tạo token loại *read* tại https://huggingface.co/settings/tokens.
- [ ] Kaggle:
  - [ ] Xác minh số điện thoại (để dùng GPU và Internet).
  - [ ] Vào *Add-ons → Secrets*, thêm secret tên `HF_TOKEN`.
- [ ] Tạo 6 Milestones trên GitHub (Tuần 1 … Tuần 6) để làm minh chứng quản lý thời gian.

**Cách chạy một notebook trên Kaggle:**
1. *Create → New Notebook → File → Import Notebook*, chọn file trong `notebooks/`.
2. *Settings*: Accelerator = **GPU T4** (chỉ notebook 01, 03), Internet = **On**. Bật secret `HF_TOKEN` cho notebook.
3. Notebook cần output của notebook trước thì *Add Input* → *Your Work* → chọn notebook đó.
4. *Save Version → Save & Run All*.
5. Xong thì tải `results/` trong tab Output về, chép vào `results/` của repo và commit. **Bỏ** các thư mục `_shards_*`.

## Tuần 1 (28/09 – 04/10): Baseline

- [ ] Chạy `01_baseline.ipynb` (GPU), giữ `MODEL_NAME = "deberta-v3-large"`.
- [ ] **Kiểm tra:**
  - [ ] Ô fidelity in ra `chunk_count_match: 1.0` và `label_agreement ≥ 0.995`. Nếu không, dừng lại và báo mình.
  - [ ] Bảng cuối: dòng `macro10`, cột `test_bacc` gần **72.6** (số của bài báo). Lệch khoảng ±1 là bình thường.
- [ ] (Nên làm) Chạy lại 01 với `MODEL_NAME = "flan-t5-large"`. Bài báo: **74.7**.
- [ ] Commit `results/baseline/`. Ghi nhật ký tuần 1 trong `progress_log.md` (ngày, số giờ, thời gian chạy lấy từ `manifest.json`).

## Tuần 2 (05/10 – 11/10): Dữ liệu và chẩn đoán

- [ ] Chạy `02_data_analysis.ipynb` (CPU là đủ; Add Input = output notebook 01).
- [ ] Đọc 3 bảng: *TPR theo số chunk*, *so sánh trong từng dataset*, *proxy bằng chứng rải rác*.
- [ ] **Tự làm:** mở `results/analysis/deberta-v3-large__fn_annotation_sheet.csv`, đọc 50 dòng, điền cột `evidence_location` (`one_chunk` / `spread` / `not_found` / `label_noise`). Mất khoảng 1–2 giờ.
- [ ] Ghi kết luận go/no-go và tỷ lệ `spread` vào nhật ký tuần 2.
- [ ] Viết mục 2 của `docs/report.md` (chỗ ✍️ ĐIỀN ở mục 2.4).

## Tuần 3 (12/10 – 18/10): Cải tiến A

- [ ] Chạy `03_evidence_selection.ipynb` (GPU; Add Input = output notebook 01).
- [ ] **Kiểm tra:** bảng tuning đủ 3 giai đoạn; cấu hình chốt được in ra (`-> cấu hình A chốt: ...`).
- [ ] **Không** sửa cấu hình sau khi đã xem kết quả test. Đây là quy tắc chống rò rỉ, sẽ bị hỏi khi thuyết trình.
- [ ] Commit `results/evidence/`, ghi nhật ký.

## Tuần 4 (19/10 – 25/10): Cải tiến B

- [ ] Chạy `04_learned_aggregation.ipynb` (CPU; Add Input = output notebook 01 và 03).
- [ ] Ghi lại: mô hình và nhóm đặc trưng được chọn theo CV, 3 hệ số LR lớn nhất, ECE trước/sau.
- [ ] Commit `results/aggregation/`, ghi nhật ký.

## Tuần 5 (26/10 – 01/11): Đánh giá

- [ ] Chạy `05_final_evaluation.ipynb` (CPU; Add Input = output 01–04, hoặc chạy trên máy vì `results/` đã commit).
- [ ] Mở `results/final/report_filled.md`, kiểm tra các bảng, rồi chép đè lên `docs/report.md`.
- [ ] Đọc 10–20 ca trong `results/final/deberta-v3-large__flips_A.csv`, chọn 2–3 ví dụ cho mục 4.8.
- [ ] Viết mọi ô **✍️ ĐIỀN** ở mục 4.

## Tuần 6 (02/11 – 08/11): Báo cáo và slide

- [ ] Viết mục Tóm tắt, 5 (Kết luận), 6 (Time managing).
- [ ] Soát mọi con số trong báo cáo với file CSV tương ứng.
- [ ] Làm slide theo `docs/slides_outline.md`. Học thuộc 6 câu hỏi dự kiến ở cuối file đó.
- [ ] Tập nói 2 lần, bấm giờ 10–12 phút. Tag release `v1.0` trên GitHub.

## Khi gặp lỗi

- **Kaggle ngắt giữa chừng ở notebook 01:** chạy lại với *Add Input* là output lần trước. Notebook tự bỏ qua phần đã xong.
- **`FileNotFoundError: Không tìm thấy ...`:** thiếu output của notebook trước. Hãy *Add Input*, hoặc commit `results/` rồi chạy lại.
- **Lỗi tải dataset (401/403):** chưa bấm Agree trên Hugging Face, hoặc secret `HF_TOKEN` chưa bật cho notebook.
