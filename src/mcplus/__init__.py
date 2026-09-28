"""mcplus: cải tiến MiniCheck (EMNLP 2024) bằng chọn lọc bằng chứng + tổng hợp điểm có học.

Các hằng số dùng chung cho mọi notebook.
"""

__version__ = "0.1.0"

# Commit MiniCheck được pin để tái lập (Liyan06/MiniCheck).
MINICHECK_REPO = "https://github.com/Liyan06/MiniCheck.git"
MINICHECK_COMMIT = "b58b9fa69acbd1015ec970fa65dd752413a053d2"

HF_DATASET = "lytang/LLM-AggreFact"

# 11 dataset trên Hugging Face (thứ tự giống Table 2 của bài báo, RAGTruth cuối).
DATASETS_11 = [
    "AggreFact-CNN", "AggreFact-XSum", "TofuEval-MediaS", "TofuEval-MeetB",
    "Wice", "Reveal", "ClaimVerify", "FactCheck-GPT", "ExpertQA", "Lfqa", "RAGTruth",
]
# Bài báo (bản EMNLP 2024 = arXiv v2) chỉ báo cáo 10 dataset, chưa có RAGTruth.
DATASETS_PAPER_10 = [d for d in DATASETS_11 if d != "RAGTruth"]

# Table 2, MiniCheck (arXiv 2404.10774v2 / EMNLP 2024): BAcc (%) trên test,
# KHÔNG tune ngưỡng. "avg" là giá trị bài báo in (trung bình 10 dataset).
PAPER_TABLE2 = {
    "roberta-large": {
        "AggreFact-CNN": 63.7, "AggreFact-XSum": 70.8, "TofuEval-MediaS": 71.9,
        "TofuEval-MeetB": 75.9, "Wice": 67.6, "Reveal": 88.8, "ClaimVerify": 77.4,
        "FactCheck-GPT": 73.3, "ExpertQA": 57.4, "Lfqa": 84.4, "avg": 72.7,
    },
    "deberta-v3-large": {
        "AggreFact-CNN": 64.2, "AggreFact-XSum": 71.0, "TofuEval-MediaS": 69.3,
        "TofuEval-MeetB": 72.7, "Wice": 69.4, "Reveal": 87.3, "ClaimVerify": 75.6,
        "FactCheck-GPT": 73.0, "ExpertQA": 58.9, "Lfqa": 83.9, "avg": 72.6,
    },
    "flan-t5-large": {
        "AggreFact-CNN": 69.9, "AggreFact-XSum": 74.3, "TofuEval-MediaS": 73.6,
        "TofuEval-MeetB": 77.3, "Wice": 72.2, "Reveal": 86.2, "ClaimVerify": 74.6,
        "FactCheck-GPT": 74.7, "ExpertQA": 59.0, "Lfqa": 85.2, "avg": 74.7,
    },
}

MODEL_CHECKPOINTS = {
    "roberta-large": "lytang/MiniCheck-RoBERTa-Large",
    "deberta-v3-large": "lytang/MiniCheck-DeBERTa-v3-Large",
    "flan-t5-large": "lytang/MiniCheck-Flan-T5-Large",
}

# Chunk size mặc định của MiniCheck (minicheck.py): 500 từ cho Flan-T5, 400 token cho encoder.
DEFAULT_CHUNK_SIZE = {"flan-t5-large": 500, "deberta-v3-large": 400, "roberta-large": 400}
