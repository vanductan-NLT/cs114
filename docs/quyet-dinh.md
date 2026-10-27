# Nhật ký quyết định

## Đã chốt

| Ngày | Quyết định | Lý do |
|---|---|---|
| 27/10/2026 | Đề tài: Learned Adaptive Retrieval. Thay ngưỡng độ phổ biến chọn tay của paper "When Not to Trust Language Models" (Mallen et al., ACL 2023) bằng bộ phân loại ML cổ điển (scikit-learn) | Hướng 2, một mình |
| 27/10/2026 | Chạy cả 14.267 câu PopQA với GPT-Neo 1.3B (fp16, 15 ví dụ mẫu) trên Kaggle; vanilla có lưu độ tự tin (`conf`) | 300 câu của CS221 chỉ có 43 câu "nên tra", quá ít để huấn luyện. Ước tính: vanilla khoảng 26 phút, BM25 khoảng 3,4 giờ trên Kaggle T4 |
| 27/10/2026 | Đóng PR #1 (khung đồ án MiniCheck cũ), giữ nhánh `feat/minicheck-evidence-aggregation` | Đề tài đã chuyển sang PopQA |
| 27/10/2026 | **Cô đồng ý dùng chung paper với môn khác**, với điều kiện ghi rõ phần nào làm cho môn nào, tách biệt. Không cần phương án dự phòng | Xem câu trả lời của cô bên dưới |

## Cô trả lời (27/10/2026, 10:19, thư trả lời email "[CS114.R11] Xin ý kiến về đề tài đồ án cuối kỳ")

Chép lại từ ảnh chụp màn hình thư, giữ nguyên chữ của cô:

> Em hoàn toàn có thể sử dụng chung paper với các môn học khác miễn là em note phần nào em làm cho môn học nào 1 cách tách biệt. Và nếu đc em có thể viết proposal hoặc skeleton cho report đẻ cô có thể hiểu thêm về chính xác em đinh làm gì nhé

Việc rút ra:
- Ranh giới CS221/CS114 đã ghi ở `README.md`; phải giữ ghi chú tách biệt này trong code, dữ liệu, báo cáo và slide.
- Cần viết proposal hoặc skeleton báo cáo gửi cô (issue "Viết proposal và skeleton báo cáo gửi cô").

## Đang chờ

| Việc | Câu hỏi | Chốt dựa vào |
|---|---|---|
| Cỡ dữ liệu, người chạy | Ai chạy 14.267 câu, dùng chung với CS221 không | Nhóm CS221 (issue CS221#13, CS221#14) |
| Hướng C của CS221 | Phân định với phần phân loại của CS114 | Nhóm CS221 (issue CS221#32, CS221#34) |
| Nhãn khi cùng đúng hoặc cùng sai | Cách A, B hay C | Bạn chọn |
