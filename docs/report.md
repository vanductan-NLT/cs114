# Cải tiến MiniCheck trên văn bản dài: chọn lọc bằng chứng theo claim và tổng hợp điểm có học

**Môn:** CS114 – Máy học · **Hướng 2:** Tối ưu lại một nghiên cứu từ bài báo khoa học
**Bài báo gốc:** Liyan Tang, Philippe Laban, Greg Durrett. *MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents.* EMNLP 2024 ([arXiv:2404.10774](https://arxiv.org/abs/2404.10774))
**Mã nguồn:** repo này (`src/mcplus`, `notebooks/01–05`)

> **Cách dùng file này.** Các khối `<!--AUTO:...-->` được notebook 05 tự điền bảng số
> (file `results/final/report_filled.md`). Các chỗ ghi **✍️ ĐIỀN** là phần nhận xét phải tự viết
> sau khi đọc kết quả. Không xóa các thẻ AUTO nếu muốn điền lại tự động.

---

## Tóm tắt

MiniCheck kiểm tra một câu (claim) có được văn bản nguồn hỗ trợ hay không bằng các mô hình nhỏ
(0.4–0.8B tham số) đạt độ chính xác gần GPT-4. Với văn bản dài, MiniCheck cắt thành các đoạn
(chunk) liền nhau khoảng 400 token, chấm từng đoạn và lấy **điểm lớn nhất**, rồi so với **ngưỡng
cố định 0.5**. Đồ án thay đổi hai luật này mà **không huấn luyện lại** mô hình MiniCheck:

- **(A) Chọn lọc bằng chứng theo claim:** ghép các câu liên quan nhất với claim thành một đoạn duy nhất trước khi chấm.
- **(B) Tổng hợp điểm có học:** một bộ phân loại nhỏ thay cho "max + 0.5", học trên tập dev, đánh giá cả trên dataset chưa từng thấy (leave-one-dataset-out).
- **(C) Tách claim nhiều câu** (hướng mà chính bài báo gợi ý).

✍️ ĐIỀN (2–3 câu): kết quả chính, ví dụ "A tăng … điểm BAcc (zero-shot), B LODO tăng … điểm; cải thiện tập trung ở văn bản nhiều chunk …".

---

## 1. Giới thiệu (Introduction)

### 1.1 Bài toán

Mô hình ngôn ngữ lớn (LLM) hay sinh ra nội dung không có trong tài liệu nguồn (*hallucination*).
Bài toán **grounded fact-checking**: cho văn bản nguồn $D$ và một claim $c$ (thường là một câu do
LLM sinh ra), dự đoán $y \in \{0, 1\}$: $c$ có được $D$ hỗ trợ đầy đủ hay không. Đây là bước kiểm
tra cốt lõi trong RAG, tóm tắt, hỏi đáp dựa trên tài liệu.

### 1.2 Bài báo gốc: MiniCheck (EMNLP 2024)

- **Vấn đề bài báo giải quyết:** dùng LLM (như GPT-4) để kiểm tra từng câu thì chính xác nhưng rất tốn kém.
- **Phương pháp:** huấn luyện các bộ kiểm tra nhỏ (RoBERTa-L, DeBERTa-v3-L, Flan-T5-L) trên dữ liệu tổng hợp do GPT-4 tạo:
  - C2D (claim → document) và D2C (document → claim), tổng khoảng 14K mẫu, cộng khoảng 21K mẫu ANLI.
  - Dữ liệu được thiết kế để mô hình phải *kiểm tra từng ý của claim* và *tổng hợp thông tin qua nhiều câu*.
- **Benchmark:** tác giả gộp các dataset kiểm tra tính đúng đắn gần đây thành **LLM-AggreFact**.
- **Kết quả:** theo bài báo, MiniCheck-FT5 (770M) đạt độ chính xác ngang GPT-4 với chi phí thấp hơn khoảng 400 lần.
- **Không tune ngưỡng:** bài báo cố ý dùng ngưỡng 0.5 cho mọi dataset (mục tiêu dùng zero-shot).

### 1.3 Động lực cải tiến

Ở khâu suy luận, MiniCheck xử lý văn bản dài như sau:

$$p_j = f_\theta(c_j, c), \qquad s = \max_j p_j, \qquad \hat y = \mathbb{1}[s > 0.5],$$

trong đó $c_1, \dots, c_m$ là các chunk liền nhau (~400 token với DeBERTa, 500 từ với Flan-T5).
Cách này có ba điểm yếu:

1. **Bằng chứng bị cắt rời.** Nếu các ý của claim nằm ở hai chunk khác nhau thì không chunk nào hỗ trợ đầy đủ, dù toàn văn bản có hỗ trợ. Mô hình được huấn luyện để tổng hợp thông tin *qua nhiều câu*, nhưng không thể tổng hợp *qua nhiều chunk*.
2. **Max nhạy với nhiễu.** Văn bản càng nhiều chunk thì càng dễ có một chunk "lỡ" vượt 0.5. Max cũng bỏ qua hình dạng phân bố điểm giữa các chunk.
3. **Ngưỡng 0.5 cố định** không tính tới việc các dataset có tỷ lệ nhãn rất khác nhau.

Bài báo không có thí nghiệm nào về kích thước chunk, cách chọn bằng chứng hay cách tổng hợp điểm.
Phần hạn chế của bài báo cũng thừa nhận benchmark chưa kiểm tra kỹ trường hợp bằng chứng nằm rải
rác. Đó là khoảng trống mà đồ án nhắm vào.

### 1.4 Mục tiêu và câu hỏi nghiên cứu

- **RQ1:** Chọn lọc bằng chứng theo claim (A) có cải thiện MiniCheck *mà vẫn giữ thiết lập zero-shot* (không tune ngưỡng, không huấn luyện) không?
- **RQ2:** Một bộ tổng hợp điểm có học (B) cải thiện bao nhiêu so với "max + 0.5" và so với việc chỉ tune ngưỡng? Cải thiện có giữ được trên dataset chưa từng thấy nhãn (LODO) không?
- **RQ3:** Cải thiện đến từ đâu (văn bản nhiều chunk hay ít chunk, dataset nào), và đổi lại tốn thêm bao nhiêu tính toán?

---

## 2. Dữ liệu (Dataset)

### 2.1 Dữ liệu đánh giá: LLM-AggreFact

LLM-AggreFact (`lytang/LLM-AggreFact` trên Hugging Face) gồm các cặp (văn bản, claim, nhãn) từ
nhiều nguồn:

| Dataset | Loại văn bản / claim (theo mô tả của bài báo) |
|---|---|
| AggreFact-CNN, AggreFact-XSum | Tóm tắt tin tức do các mô hình tóm tắt sinh ra |
| TofuEval-MediaS, TofuEval-MeetB | Tóm tắt theo chủ đề của hội thoại phỏng vấn (MediaSum) và biên bản họp (MeetingBank) do LLM sinh |
| Wice | Claim trên Wikipedia kèm các nguồn được trích dẫn |
| Reveal | Các bước suy luận (chain-of-thought) đối chiếu với bằng chứng |
| ClaimVerify | Câu trả lời của công cụ tìm kiếm tạo sinh kèm nguồn trích dẫn |
| FactCheck-GPT | Câu trả lời của LLM được kiểm chứng thủ công |
| ExpertQA | Câu trả lời cho câu hỏi của chuyên gia nhiều lĩnh vực |
| Lfqa | Hỏi đáp dạng dài có truy xuất tài liệu |
| RAGTruth | Câu trả lời của hệ thống RAG có gán nhãn hallucination (thêm vào benchmark sau bài báo) |

**Hai điểm cần lưu ý về giao thức:**

- **Số dataset.** Bảng kết quả chính của bài báo (Table 2, bản EMNLP 2024) chỉ có **10 dataset, chưa có RAGTruth**. Bản trên Hugging Face hiện có 11 dataset. Đồ án báo cáo cả trung bình 10 dataset (so thẳng với bài báo) và 11 dataset.
- **Cách dùng các split.** Mỗi dataset có split **dev** (30,420 mẫu tổng) và **test** (29,320 mẫu tổng). Dev dùng để chọn siêu tham số và huấn luyện bộ tổng hợp B. Test chỉ dùng một lần để báo cáo.

### 2.2 Thống kê

<!--AUTO:dataset_table-->
*(Bảng tự điền từ `results/analysis/dataset_summary.csv` — notebook 02.)*
<!--/AUTO:dataset_table-->

![Mất cân bằng nhãn](../results/figures/label_balance.png)
![Độ dài văn bản](../results/figures/doc_length.png)

Tỷ lệ nhãn *supported* chênh rất lớn giữa các dataset (ví dụ RAGTruth nghiêng mạnh về supported,
FactCheck-GPT và Reveal ngược lại). Vì vậy metric là **Balanced Accuracy**:
BAcc = (TPR + TNR)/2, tính riêng từng dataset rồi lấy trung bình không trọng số, giống bài báo.

### 2.3 Dữ liệu huấn luyện của MiniCheck (không huấn luyện lại, chỉ mô tả)

Các checkpoint MiniCheck được tác giả huấn luyện trên khoảng 35K mẫu:
- khoảng 21K mẫu ANLI mà các mô hình entailment trước đó hay sai;
- 7,076 mẫu C2D: sinh các "atomic fact" rồi viết văn bản hỗ trợ / không hỗ trợ;
- 7,319 mẫu D2C: tóm tắt đoạn văn thành claim rồi xóa câu để tạo mẫu không hỗ trợ.

Đồ án dùng nguyên checkpoint công khai. Mọi cải tiến nằm ở khâu suy luận và khâu tổng hợp.

### 2.4 Phân tích lỗi của baseline trên dev (động lực cho A)

<!--AUTO:diagnostics-->
*(Tự điền từ notebook 02: BAcc/TPR/TNR theo số chunk và các chỉ số chẩn đoán.)*
<!--/AUTO:diagnostics-->

**Đọc tay 50 lỗi FN** (`results/analysis/<model>__fn_annotation_sheet.csv`):
✍️ ĐIỀN: bằng chứng nằm gọn trong 1 chunk …%, rải ở ≥2 chunk …%, không tìm thấy …%, nhãn có vẻ sai …%.
Kèm 1 ví dụ minh họa ngắn.

---

## 3. Phương pháp (Methodology)

### 3.1 Baseline: MiniCheck gốc

- **Mô hình:** `lytang/MiniCheck-DeBERTa-v3-Large` (mô hình chính, 435M) và `lytang/MiniCheck-Flan-T5-Large` (770M).
- **Mã nguồn:** MiniCheck được pin ở commit `b58b9fa`. Không huấn luyện lại, chạy fp32 như mã gốc.
- **Cách chấm:** tách câu bằng NLTK, gom câu thành chunk ≤ 400 token (DeBERTa) hoặc ≤ 500 từ (Flan-T5), chấm từng cặp (chunk, claim), lấy max, so với 0.5.
- **Đường chạy nhanh.** Hàm `MiniCheck.score` gốc chạy tuần tự từng ví dụ. `FastScorer` dùng lại đúng model, tokenizer và định dạng input, nhưng gom batch qua nhiều ví dụ.
- **Kiểm tra độ khớp.** Notebook 01 so `FastScorer` với hàm gốc trên 220 ví dụ phân tầng của test. Notebook dừng lại nếu số chunk không trùng 100% hoặc nhãn trùng dưới 99.5%. ✍️ ĐIỀN từ `results/baseline/<model>__fidelity.json`: tỷ lệ trùng nhãn …, chênh lệch xác suất lớn nhất …

### 3.2 Cải tiến A — Chọn lọc bằng chứng theo claim

Áp dụng cho văn bản có từ 2 chunk trở lên (văn bản 1 chunk giữ nguyên như baseline):

1. Tách văn bản thành câu $s_1, \dots, s_n$.
2. Xếp hạng câu theo độ liên quan với claim bằng một trong ba bộ chọn:
   - **BM25** (từ vựng, bỏ stopword);
   - **dense**: cosine giữa embedding `all-MiniLM-L6-v2` của câu và của claim;
   - **hybrid**: gộp hai thứ hạng bằng Reciprocal Rank Fusion, $\text{RRF}(s) = \sum_r \frac{1}{60 + \text{rank}_r(s)}$.
3. Duyệt theo thứ hạng. Mỗi câu "hạt giống" kéo theo $w$ câu lân cận mỗi bên. Dừng khi tổng độ dài chạm **ngân sách = đúng một chunk** của MiniCheck, hoặc khi đã lấy $k$ hạt giống.
4. Giữ **thứ tự gốc** của các câu được chọn, ghép thành một ngữ cảnh $\tilde D$, chấm $e = f_\theta(\tilde D, c)$.
5. Hai cách dùng điểm $e$:
   - `replace`: $s = e$;
   - `union`: $s = \max(\max_j p_j,\; e)$.

**Siêu tham số** (bộ chọn, $k \in \{3, 5, \text{lấp đầy}\}$, $w \in \{0, 1\}$, mode) được chọn trên
một tập con phân tầng của dev (tối đa 500 mẫu mỗi dataset). Ngưỡng vẫn là 0.5, nên A **vẫn là
thiết lập zero-shot** và so thẳng được với bài báo.

**Chi phí thêm:** 1 lượt chấm mỗi claim ở văn bản nhiều chunk, cộng việc tính embedding câu (mô
hình 22M tham số, có cache theo văn bản).

### 3.3 Ablation C — Tách claim nhiều câu

MiniCheck là bộ kiểm tra *cấp câu*, nhưng một số claim có nhiều câu (đặc biệt AggreFact-CNN là
tóm tắt 3 câu). Chính mã nguồn MiniCheck khuyến nghị tách claim. Ta tách claim bằng NLTK, chấm
từng câu (theo baseline hoặc theo A) rồi lấy **min**: claim chỉ đúng khi mọi câu đều đúng.

### 3.4 Cải tiến B — Tổng hợp điểm có học

Thay "$\max_j p_j > 0.5$" bằng một bộ phân loại $g_\phi(\mathbf{x}) \in [0, 1]$.

**Đặc trưng $\mathbf{x}$ cho mỗi (văn bản, claim):**
- logit của max, điểm lớn thứ hai, trung bình và min của $\{p_j\}$;
- độ lệch chuẩn của $\{p_j\}$, tỷ lệ chunk vượt 0.5, vị trí chunk tốt nhất, $\log m$;
- điểm $e$ của A (dạng logit) và $e - \max_j p_j$; điểm A+C;
- log độ dài văn bản và claim, số câu của claim, tỷ lệ từ của claim xuất hiện trong văn bản.

**Mô hình:** Logistic Regression (chuẩn hóa đặc trưng) và Histogram Gradient Boosting.

**Hàm mất mát có trọng số theo mục tiêu macro-BAcc.** Mỗi ô (dataset $d$, nhãn $y$) được gán tổng
trọng số bằng nhau: $w_i = \frac{1}{n_{d_i, y_i}}$ (chuẩn hóa về trung bình 1). Khi đó log-loss có
trọng số xấp xỉ trung bình các loss cân bằng lớp của từng dataset, đúng với metric cần tối ưu. Nhờ
vậy dùng luôn ngưỡng 0.5 trên xác suất dự đoán.

**Chọn mô hình:** cross-validation 5-fold trên dev (phân tầng theo dataset × nhãn), lưới nhỏ cho
$C$ (LR) và depth/learning rate (HGB), cùng 3 nhóm đặc trưng. Chọn theo macro-BAcc out-of-fold.

**Ba thiết lập so sánh** (đều dùng nhãn dev, không dùng nhãn test):

| Thiết lập | Mô tả |
|---|---|
| Thr-global / Thr-per-dataset | Chỉ tune ngưỡng trên dev: 1 ngưỡng chung, hoặc 1 ngưỡng mỗi dataset (giống chế độ "dynamic threshold" của FactCG) |
| B in-domain | Huấn luyện $g_\phi$ trên toàn bộ dev, dự đoán test |
| **B LODO** | Muốn dự đoán dataset $d$ thì huấn luyện trên dev của 10 dataset còn lại. Mô phỏng việc dùng cho một domain mới chưa có nhãn |

### 3.5 Thiết lập thí nghiệm

| Thành phần | Chi tiết |
|---|---|
| Phần cứng | Kaggle, 1× NVIDIA T4 16GB (notebook 01, 03). CPU cho notebook 02, 04, 05 |
| Phần mềm | Phiên bản thư viện, commit MiniCheck và revision checkpoint ghi trong các file `manifest.json` |
| Tính tái lập | Seed cố định (0 / 42). Dataset pin revision `981dfd0`. Mọi dự đoán lưu theo `uid` |
| Quy tắc dữ liệu | Dev: chọn siêu tham số A, CV và huấn luyện B, tune ngưỡng đối chứng. Test: chạy một lần ở cuối |
| Kiểm định | Bootstrap có cặp 10,000 lần (lấy mẫu lại trong từng ô dataset × nhãn) cho hiệu macro-BAcc. McNemar chính xác cho từng dataset |

---

## 4. Đánh giá (Evaluation)

### 4.1 Tái lập bài báo

<!--AUTO:baseline_table-->
*(Tự điền từ `results/baseline/<model>__metrics.csv` — notebook 01.)*
<!--/AUTO:baseline_table-->

✍️ ĐIỀN: baseline chạy lại lệch bài báo bao nhiêu điểm trên trung bình 10 dataset; dataset nào
lệch nhiều nhất; vì sao chấp nhận được (cùng checkpoint, cùng mã nguồn, khác phần cứng).

### 4.2 Chọn cấu hình cho A (trên tập con dev)

<!--AUTO:tuning_A-->
*(Tự điền từ `results/evidence/<model>__tuning.csv` — notebook 03.)*
<!--/AUTO:tuning_A-->

### 4.3 Kết quả chính trên test

Hai nhóm phương pháp:
- **Zero-shot** (không dùng nhãn để tune ngưỡng hay huấn luyện): Baseline, A, A+C, C.
- **Có dùng nhãn dev:** Thr-*, B in-domain, B LODO.

<!--AUTO:main_table-->
*(Tự điền từ notebook 05.)*
<!--/AUTO:main_table-->

![So sánh chính](../results/figures/deberta-v3-large__main_comparison.png)

✍️ ĐIỀN (trả lời RQ1, RQ2): A thay đổi bao nhiêu so với baseline và bài báo; B so với chỉ tune
ngưỡng; B LODO giữ được bao nhiêu phần cải thiện của B in-domain.

### 4.4 Ý nghĩa thống kê (so với baseline chạy lại)

<!--AUTO:significance-->
*(Tự điền từ notebook 05.)*
<!--/AUTO:significance-->

### 4.5 Cải thiện đến từ đâu? (RQ3)

<!--AUTO:chunks_effect-->
*(Tự điền từ notebook 05: BAcc trên văn bản 1 chunk và ≥2 chunk.)*
<!--/AUTO:chunks_effect-->

✍️ ĐIỀN: nếu giả thuyết đúng, A phải tăng chủ yếu ở nhóm ≥2 chunk và gần như không đổi ở nhóm 1
chunk. Ghi dataset nào được lợi nhất và dataset nào bị hại.

### 4.6 Ablation của B và hiệu chỉnh xác suất

<!--AUTO:ablation_B-->
*(Tự điền: bảng CV trên dev, rồi macro-BAcc LODO trên test cho mọi tổ hợp mô hình × nhóm đặc trưng.)*
<!--/AUTO:ablation_B-->

<!--AUTO:calibration-->
*(Tự điền: ECE và Brier của điểm max của MiniCheck, điểm A và xác suất B.)*
<!--/AUTO:calibration-->

![Reliability diagram](../results/figures/deberta-v3-large__reliability.png)

✍️ ĐIỀN: đặc trưng nào có hệ số LR lớn nhất (`results/aggregation/<model>__lr_coefficients.csv`) và ý nghĩa của nó.

### 4.7 Chi phí

<!--AUTO:efficiency-->
*(Tự điền: số lượt forward mỗi ví dụ và thời gian trên 1K ví dụ.)*
<!--/AUTO:efficiency-->

### 4.8 Phân tích lỗi định tính

✍️ ĐIỀN: 2–3 ví dụ lấy từ `results/final/<model>__flips_A.csv`, gồm 1 ca A sửa đúng và 1 ca A làm
sai. Với mỗi ca: claim, đoạn bằng chứng được chọn, điểm trước/sau.

---

## 5. Kết luận (Conclusion)

✍️ ĐIỀN: 3–4 câu trả lời trực tiếp RQ1–RQ3 bằng số liệu.

**Hạn chế**
- Chỉ thay đổi khâu suy luận và tổng hợp, không huấn luyện lại mô hình kiểm tra.
- B dùng nhãn dev. Thiết lập LODO giảm bớt nhưng không loại bỏ hoàn toàn lợi thế này so với bài báo (hoàn toàn zero-shot).
- Siêu tham số của A chọn trên dev. Dataset nhỏ (CNN chỉ có 57 mẫu âm trên test) có khoảng tin cậy rộng.

**Hướng phát triển**
- Dùng encoder ngữ cảnh dài (ví dụ ModernBERT, 8K token) huấn luyện trên dữ liệu MiniCheck để bỏ hẳn việc cắt chunk.
- Sinh thêm dữ liệu huấn luyện có bằng chứng rải rác (multi-hop) như hướng của FactCG.
- Áp dụng A/B cho Bespoke-MiniCheck-7B.

---

## 6. Quản lý thời gian (Time managing)

Kế hoạch 6 tuần, nhật ký và các mốc nằm trong [`docs/progress_log.md`](progress_log.md). Lịch sử
commit trên GitHub là minh chứng tiến độ.

✍️ ĐIỀN: bảng tóm tắt dự kiến so với thực tế (ngày hoàn thành từng tuần, số giờ, việc trễ và cách xử lý).

---

## Tài liệu tham khảo

1. L. Tang, P. Laban, G. Durrett. *MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents.* EMNLP 2024.
2. D. Lei et al. *FactCG: Enhancing Fact Checks via Graph-Based Multi-Hop Data.* NAACL 2025. (tham chiếu cho thiết lập tune ngưỡng theo dataset)
3. S. Robertson, H. Zaragoza. *The Probabilistic Relevance Framework: BM25 and Beyond.* 2009.
4. N. Reimers, I. Gurevych. *Sentence-BERT.* EMNLP 2019. (mô hình `all-MiniLM-L6-v2`)
5. G. Cormack, C. Clarke, S. Büttcher. *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods.* SIGIR 2009.
6. B. Efron, R. Tibshirani. *An Introduction to the Bootstrap.* 1993. · Q. McNemar (1947), kiểm định cặp.
