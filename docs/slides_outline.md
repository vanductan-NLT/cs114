# Dàn ý slide (10–12 phút, khoảng 12 slide)

Mỗi slide ghi phần điểm nó phục vụ. Số liệu lấy từ `docs/report.md` sau khi chạy notebook 05.

| # | Slide | Nội dung chính | Hình/bảng | Điểm |
|---|---|---|---|---|
| 1 | Tiêu đề | Tên đề tài, bài báo gốc (MiniCheck, EMNLP 2024), tên SV | – | – |
| 2 | Bài toán | Hallucination; cho (văn bản, claim) → supported? Ví dụ 1 claim đúng / 1 claim sai | Ví dụ ngắn | Intro |
| 3 | Bài báo gốc làm gì | Mô hình nhỏ + dữ liệu tổng hợp C2D/D2C, ngang GPT-4, rẻ hơn khoảng 400 lần | Sơ đồ pipeline MiniCheck | Intro |
| 4 | Dataset | LLM-AggreFact 11 dataset, dev/test, mất cân bằng nhãn → BAcc; 10 vs 11 dataset | `label_balance.png`, bảng thống kê | Dataset |
| 5 | Điểm yếu tìm ra | Cắt chunk + max + ngưỡng 0.5. Số liệu chẩn đoán trên dev (TPR theo số chunk, proxy rải rác, 50 lỗi đọc tay) | Bảng/hình notebook 02 | Dataset / Method |
| 6 | Cải tiến A | Xếp hạng câu → chọn tới khi đầy 1 chunk → chấm 1 lần; replace/union | Sơ đồ thuật toán | Method |
| 7 | Cải tiến B | Đặc trưng, LR/HGB, trọng số theo macro-BAcc, LODO | Sơ đồ + công thức trọng số | Method |
| 8 | Thiết lập thí nghiệm | Dev/test dùng vào đâu, fidelity check, kiểm định | Bảng quy tắc dữ liệu | Method / Eval |
| 9 | Kết quả chính | Bài báo vs chạy lại vs A vs B; macro 10 và 11 | `main_comparison.png` + bảng | Evaluation |
| 10 | Vì sao tốt hơn / không tốt hơn | Theo số chunk, theo dataset, ý nghĩa thống kê, calibration | Bảng chunks_effect, reliability | Evaluation |
| 11 | Chi phí + ví dụ lỗi | Lượt forward / thời gian; 1 ca A sửa đúng, 1 ca làm sai | Bảng efficiency | Evaluation |
| 12 | Kết luận + hướng phát triển + tiến độ | Trả lời RQ1–3; hạn chế; ModernBERT / dữ liệu multi-hop; timeline thực tế | Timeline từ `progress_log.md` | Conclusion, Time |

## Câu hỏi có thể bị hỏi — chuẩn bị trước

1. **"Dùng dev để train B có phải gian lận so với bài báo không?"** → Không đụng test. Bài báo cố ý không dùng dev. Vì vậy B được so với "chỉ tune ngưỡng trên dev" (cùng điều kiện), và có thêm LODO để chứng minh dùng được cho domain chưa có nhãn. A thì hoàn toàn zero-shot như bài báo.
2. **"Siêu tham số của A chọn thế nào?"** → Trên tập con dev (tối đa 500 mẫu mỗi dataset), trước khi chạy test. Test chỉ chạy một lần.
3. **"Kết quả chạy lại lệch bài báo vì sao?"** → Cùng checkpoint và mã nguồn (pin commit), đã kiểm tra khớp 100% với hàm gốc. Lệch nhỏ đến từ phần cứng (T4 vs A6000) và phiên bản thư viện.
4. **"Cải thiện có thật hay do may?"** → Bootstrap có cặp 10,000 lần và McNemar từng dataset. Nêu rõ dataset nhỏ (CNN chỉ 57 mẫu âm) có khoảng tin cậy rộng.
5. **"Tốn thêm bao nhiêu?"** → Bảng chi phí: lượt forward mỗi ví dụ và giây trên 1K ví dụ.
6. **"Sao không train lại model?"** → Nằm ngoài quy mô 1 người / 6 tuần. Ghi thành hướng phát triển (ModernBERT ngữ cảnh dài).
