# cs114

## Đồ án cuối kỳ CS114: Learned Adaptive Retrieval

Môn **CS114 Máy học** (lớp CS114.R11, GV ThS. Huỳnh Tân Bối), UIT. Làm một mình: Văn Đức Tân, MSSV 24521586. Hướng 2: chọn một paper ở hội nghị lớn rồi cải tiến phương pháp của nó.

- **Paper gốc:** *When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories* (Mallen et al., ACL 2023).
- **Dữ liệu gốc:** PopQA, 14.267 câu hỏi về thực thể, có độ phổ biến `s_pop` (lượt xem Wikipedia) và 16 loại quan hệ `prop`.
- **Bảng theo dõi:** [GitHub Project "CS114 · Learned Adaptive Retrieval"](https://github.com/users/vanductan-NLT/projects/7)

### Ý tưởng

Paper dùng *Adaptive Retrieval*: với mỗi câu hỏi, nếu thực thể đủ nổi tiếng thì để mô hình tự trả lời, còn không thì tra cứu (BM25) rồi mới trả lời. Ranh giới "đủ nổi tiếng" là một **ngưỡng `s_pop` chọn tay theo từng loại quan hệ**.

Đồ án này thay ngưỡng đó bằng **mô hình phân loại ML cổ điển (scikit-learn)** học từ các đặc trưng của câu hỏi: độ phổ biến, loại quan hệ, độ tự tin của mô hình khi tự trả lời, độ dài câu... để dự đoán câu nào nên tra cứu. Sau đó so với ngưỡng của paper trên hai con số: độ chính xác cuối cùng và tỉ lệ câu phải tra cứu.

Mô hình dự kiến: logistic regression, SVM, cây quyết định / Random Forest, MLP (đúng phạm vi đề cương CS114).

### Ranh giới với đồ án CS221

Paper này cũng là đồ án nhóm môn CS221 Nhập môn NLP ([vanductan-NLT/CS221](https://github.com/vanductan-NLT/CS221)). Hai đồ án tách như sau:

| | CS221 (nhóm) | CS114 (cá nhân) |
|---|---|---|
| Làm gì | Chạy lại paper, thử mô hình khác (Qwen...) | Bộ phân loại học từ đặc trưng thay cho ngưỡng |
| Code phân loại | Không có | Chỉ ở repo này |

Phần dùng lại từ CS221 (notebook chạy mô hình, dữ liệu BM25, kết quả vanilla/BM25) luôn ghi nguồn rõ trong code, dữ liệu và slide. Đề tài còn có thể đổi trước 15/11 tuỳ ý kiến của cô.

### Dữ liệu dự kiến

Chạy **cả 14.267 câu** PopQA với **GPT-Neo 1.3B** (fp16, 15 ví dụ mẫu) trên Kaggle: một lượt vanilla (có lưu độ tự tin) và một lượt BM25. Tập 300 câu của CS221 quá ít để huấn luyện (chỉ 43 câu "nên tra").

### Nguyên tắc

- Nhãn "nên tra" = tra BM25 thì đúng còn không tra thì sai.
- Chia train/test theo câu hỏi, cố định seed. Ngưỡng của paper cũng chỉ chọn trên train hoặc dev.
- Báo cáo hai con số: độ chính xác cuối cùng và tỉ lệ câu phải tra.
- Đặc trưng độ tự tin cần mô hình trả lời trước một lần; chi phí này được ghi vào phần phân tích.

### Lịch

| Mốc | Thời gian |
|---|---|
| M0 Chốt đề | 27/10 – 15/11 |
| M1 Học nền | 27/10 – 9/11 |
| M2 Dữ liệu và EDA | 3/11 – 16/11 |
| M3 Baseline của paper | 10/11 – 20/11 |
| M4 Huấn luyện mô hình | 17/11 – 7/12 (24–30/11 chạy nhẹ) |
| M5 Đánh giá và phân tích | 1/12 – 10/12 |
| M6 Slide và tập nói | 8/12 – 14/12 (ngày thuyết trình tạm đặt 14/12) |

### Nhãn issue

`hoc` học nền · `tay` tự làm tay · `chay-gpu` chạy trên Kaggle · `claude-code` nhờ Claude Code · `quyet-dinh` cần chọn · `phu-thuoc-cs221` cần kết quả từ CS221
