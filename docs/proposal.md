# Đề xuất đồ án CS114 (Hướng 2)

**Tên đề tài:** Cải tiến MiniCheck trên văn bản dài: chọn lọc bằng chứng theo claim và tổng hợp điểm có học
**Thực hiện:** 1 sinh viên · **Thời gian:** 6 tuần (28/09 – 08/11/2026)

## Bài báo gốc

Tang, Laban, Durrett. *MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents.*
**EMNLP 2024** ([arXiv:2404.10774](https://arxiv.org/abs/2404.10774)).

- **Bài toán:** cho văn bản nguồn và một câu (claim) do LLM sinh ra, xác định claim có được văn bản hỗ trợ không, tức phát hiện hallucination.
- **Đóng góp của bài báo:**
  - Các mô hình kiểm tra nhỏ (0.4–0.8B tham số) huấn luyện trên dữ liệu tổng hợp do GPT-4 tạo.
  - Đạt độ chính xác ngang GPT-4 với chi phí thấp hơn khoảng 400 lần.
  - Benchmark LLM-AggreFact gộp nhiều dataset.
- **Dữ liệu gốc:**
  - Đánh giá: LLM-AggreFact (dev 30,420 / test 29,320 mẫu), metric Balanced Accuracy.
  - Huấn luyện: khoảng 21K mẫu ANLI + 14K mẫu tổng hợp C2D/D2C.

## Điểm yếu nhắm tới

Với văn bản dài, MiniCheck cắt thành các đoạn liền nhau (~400 token), chấm từng đoạn, lấy **điểm
lớn nhất**, rồi so với **ngưỡng cố định 0.5**. Hệ quả:

- Bằng chứng nằm ở nhiều đoạn thì bị cắt rời.
- Luật max dễ bị một đoạn nhiễu làm sai.
- Một ngưỡng chung không hợp với các dataset có tỷ lệ nhãn khác nhau.

Bài báo không có thí nghiệm nào về các lựa chọn này.

## Cải tiến đề xuất

| | Thay đổi | Có huấn luyện? | So với bài báo |
|---|---|---|---|
| **A** | Xếp hạng câu theo độ liên quan với claim (BM25 / embedding / lai), ghép các câu tốt nhất thành **một** đoạn rồi mới chấm | Không | Cùng thiết lập zero-shot (ngưỡng 0.5), so thẳng được |
| **B** | Thay "max + 0.5" bằng Logistic Regression / Gradient Boosting học trên dev, với trọng số theo mục tiêu macro-BAcc | Có (mô hình nhỏ, CPU) | So với việc chỉ tune ngưỡng; kiểm tra khả năng dùng cho domain mới bằng leave-one-dataset-out |
| C (ablation) | Tách claim nhiều câu, lấy min (bài báo gợi ý nhưng không làm) | Không | Zero-shot |

## Thí nghiệm và đánh giá

- **Mô hình:** MiniCheck-DeBERTa-v3-Large (chính), MiniCheck-Flan-T5-Large (phụ). Checkpoint công khai, không huấn luyện lại.
- **Metric:** BAcc từng dataset, trung bình trên 10 dataset (như Table 2 của bài báo) và trên 11 dataset (thêm RAGTruth).
- **Kiểm định:** bootstrap có cặp và McNemar.
- **Phân tích thêm:** theo số đoạn văn bản, calibration (ECE), chi phí tính toán.
- **Quy tắc:** dev dùng để chọn siêu tham số và huấn luyện B; test chỉ chạy một lần.
- **Tài nguyên:** Kaggle / Colab GPU T4.

## Kế hoạch

Chi tiết theo tuần trong [`progress_log.md`](progress_log.md):

| Tuần | Việc |
|---|---|
| 1 | Chạy lại baseline |
| 2 | Phân tích dữ liệu và lỗi |
| 3 | Cải tiến A |
| 4 | Cải tiến B |
| 5 | Đánh giá |
| 6 | Báo cáo |
