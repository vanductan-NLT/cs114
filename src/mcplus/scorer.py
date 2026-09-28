"""Chấm điểm MiniCheck theo lô (batch) trên nhiều cặp (văn bản, claim).

`MiniCheck.score` gốc chạy tuần tự từng ví dụ; mỗi lần chỉ gom các chunk của một ví dụ
thành batch. Với ~60K ví dụ (dev + test) và nhiều cấu hình thí nghiệm thì quá chậm trên T4.
`FastScorer` dùng lại đúng model, tokenizer và cách định dạng input của MiniCheck nhưng gom
batch qua nhiều ví dụ, sắp theo độ dài để giảm padding. Notebook 01 có bước so khớp với
`MiniCheck.score` gốc trước khi chạy full.
"""

from __future__ import annotations

import hashlib
import time
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from . import DEFAULT_CHUNK_SIZE, MODEL_CHECKPOINTS
from .chunking import chunks_from_lengths, sent_tokenize_with_newlines


def _load_minicheck(model_name: str, cache_dir: str | None, batch_size: int):
    """Tải MiniCheck. Nếu transformers báo lỗi với device_map='auto' (một số phiên bản
    không hỗ trợ cho DeBERTa), tự dựng Inferencer với đúng cấu hình gốc rồi đưa lên GPU."""
    from minicheck.minicheck import MiniCheck

    try:
        return MiniCheck(model_name=model_name, batch_size=batch_size, cache_dir=cache_dir)
    except (ValueError, NotImplementedError) as exc:
        if "device_map" not in str(exc):
            raise
        print(f"[fallback] device_map='auto' lỗi ({exc}); tải model thủ công.")
    import torch.nn as nn
    from minicheck.inference import Inferencer
    from transformers import (AutoConfig, AutoModelForSeq2SeqLM,
                              AutoModelForSequenceClassification, AutoTokenizer)

    ckpt = MODEL_CHECKPOINTS[model_name]
    inf = Inferencer.__new__(Inferencer)
    inf.model_name = model_name
    if model_name == "flan-t5-large":
        inf.model = AutoModelForSeq2SeqLM.from_pretrained(ckpt, cache_dir=cache_dir)
        inf.tokenizer = AutoTokenizer.from_pretrained(ckpt, cache_dir=cache_dir)
        inf.max_model_len, inf.max_output_length = 2048, 256
    else:
        config = AutoConfig.from_pretrained(ckpt, num_labels=2, finetuning_task="text-classification",
                                            cache_dir=cache_dir)
        config.problem_type = "single_label_classification"
        inf.tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=True, cache_dir=cache_dir)
        inf.model = AutoModelForSequenceClassification.from_pretrained(ckpt, config=config,
                                                                       cache_dir=cache_dir)
        inf.max_model_len = 512 if model_name == "roberta-large" else 2048
    inf.model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    inf.batch_size, inf.softmax = batch_size, nn.Softmax(dim=-1)
    mc = MiniCheck.__new__(MiniCheck)
    mc.model = inf
    return mc


class Chunker:
    """Chỉ cần tokenizer: cắt chunk đúng như MiniCheck (dùng được trên CPU, notebook 02)."""

    def __init__(self, model_name: str, tokenizer=None, max_model_len: int | None = None,
                 cache_dir: str | None = None):
        assert model_name in MODEL_CHECKPOINTS, model_name
        self.model_name = model_name
        self.is_t5 = model_name == "flan-t5-large"
        self.chunk_size = DEFAULT_CHUNK_SIZE[model_name]
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINTS[model_name], use_fast=True,
                                                      cache_dir=cache_dir)
        self.tokenizer = tokenizer
        self.max_model_len = max_model_len or {"roberta-large": 512}.get(model_name, 2048)
        self._chunk_cache: Dict[str, dict] = {}

    # ----- độ dài và chunk theo đúng đơn vị MiniCheck -----
    def sentence_lengths(self, sentences: Sequence[str]) -> List[int]:
        """Flan-T5: đếm từ theo khoảng trắng; encoder: đếm token (không special token)."""
        if self.is_t5:
            return [len(s.split()) for s in sentences]
        if not sentences:
            return []
        enc = self.tokenizer(list(sentences), padding=False, add_special_tokens=False,
                             max_length=self.max_model_len, truncation=True)
        return [len(ids) for ids in enc["input_ids"]]

    def length(self, text: str) -> int:
        return self.sentence_lengths([text])[0]

    def chunk_doc(self, doc: str) -> dict:
        """{'chunks': [...], 'doc_len': tổng độ dài các câu} — có cache theo văn bản
        (nhiều claim dùng chung một văn bản, ví dụ RAGTruth ~450 văn bản cho 16K claim)."""
        key = hashlib.md5(doc.encode("utf-8")).hexdigest()
        hit = self._chunk_cache.get(key)
        if hit is None:
            sents = sent_tokenize_with_newlines(doc) or [""]
            lengths = self.sentence_lengths(sents)
            hit = {"chunks": chunks_from_lengths(sents, lengths, self.chunk_size),
                   "doc_len": int(sum(lengths))}
            self._chunk_cache[key] = hit
        return hit


class FastScorer(Chunker):
    def __init__(self, model_name: str, cache_dir: str | None = None, batch_size: int = 16,
                 token_budget: int = 12000, max_batch: int = 64):
        assert model_name in MODEL_CHECKPOINTS, model_name
        self.mc = _load_minicheck(model_name, cache_dir, batch_size)
        self.inf = self.mc.model
        super().__init__(model_name, tokenizer=self.inf.tokenizer, max_model_len=self.inf.max_model_len)
        self.model = self.inf.model
        self.token_budget, self.max_batch = token_budget, max_batch
        self.n_forward = 0  # số cặp (chunk, claim) đã chấm, để so chi phí giữa các phương pháp

    @property
    def device(self):
        return self.model.device

    # ----- chấm điểm -----
    def _format(self, docs: Sequence[str], claims: Sequence[str]) -> List[str]:
        eos = self.tokenizer.eos_token
        texts = [eos.join([d, c]) for d, c in zip(docs, claims)]
        return ["predict: " + t for t in texts] if self.is_t5 else texts

    @torch.no_grad()
    def _forward(self, texts: List[str]) -> np.ndarray:
        inputs = self.tokenizer(texts, max_length=self.max_model_len, truncation=True,
                                padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.is_t5:
            dec = torch.zeros((inputs["input_ids"].size(0), 1), dtype=torch.long, device=self.device)
            logits = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                decoder_input_ids=dec).logits.squeeze(1)
            # MiniCheck: token 3 = không hỗ trợ, token 209 = hỗ trợ
            probs = F.softmax(logits[:, torch.tensor([3, 209])].float().cpu(), dim=-1)
        else:
            probs = F.softmax(self.model(**inputs).logits.float(), dim=1).cpu()
        return probs[:, 1].numpy()

    def _forward_safe(self, texts: List[str]) -> np.ndarray:
        try:
            return self._forward(texts)
        except torch.cuda.OutOfMemoryError:
            if len(texts) == 1:
                raise
            torch.cuda.empty_cache()
            mid = len(texts) // 2
            return np.concatenate([self._forward_safe(texts[:mid]), self._forward_safe(texts[mid:])])

    def score_pairs(self, docs: Sequence[str], claims: Sequence[str], progress: bool = True,
                    desc: str = "scoring") -> np.ndarray:
        """P(supported) cho từng cặp (doc, claim); doc ở đây là một chunk/ngữ cảnh đã cắt."""
        assert len(docs) == len(claims)
        n = len(docs)
        out = np.empty(n, dtype=np.float64)
        if n == 0:
            return out
        texts = self._format(docs, claims)
        lens = np.array([min(len(ids), self.max_model_len) for ids in
                         self.tokenizer(texts, truncation=True, max_length=self.max_model_len)["input_ids"]])
        order = np.argsort(-lens, kind="stable")
        batches, i = [], 0
        while i < n:
            size = max(1, min(self.max_batch, self.token_budget // max(int(lens[order[i]]), 1)))
            batches.append(order[i:i + size])
            i += size
        iterator = batches
        if progress:
            from tqdm.auto import tqdm
            iterator = tqdm(batches, desc=desc, mininterval=10)
        for batch in iterator:
            out[batch] = self._forward_safe([texts[j] for j in batch])
        self.n_forward += n
        return out

    def score_examples(self, docs: Sequence[str], claims: Sequence[str], progress: bool = True):
        """Baseline MiniCheck: cắt chunk, chấm từng (chunk, claim), lấy max.

        Trả về (max_prob, chunk_probs [list], n_chunks, doc_len)."""
        pair_docs, pair_claims, owner, n_chunks, doc_len = [], [], [], [], []
        for i, (doc, claim) in enumerate(zip(docs, claims)):
            info = self.chunk_doc(doc)
            n_chunks.append(len(info["chunks"]))
            doc_len.append(info["doc_len"])
            for chunk in info["chunks"]:
                pair_docs.append(chunk)
                pair_claims.append(claim)
                owner.append(i)
        probs = self.score_pairs(pair_docs, pair_claims, progress=progress, desc="chunks")
        chunk_probs: List[List[float]] = [[] for _ in docs]
        for i, p in zip(owner, probs):
            chunk_probs[i].append(float(p))
        max_prob = np.array([max(ps) for ps in chunk_probs])
        return max_prob, chunk_probs, np.array(n_chunks), np.array(doc_len)

    def score_split_claims(self, docs: Sequence[str], claims: Sequence[str], split_fn,
                           progress: bool = True) -> np.ndarray:
        """Ablation C trên baseline: mỗi câu của claim chấm như MiniCheck (max theo chunk),
        điểm claim = min các câu."""
        rep_docs, parts, owner = [], [], []
        for i, (doc, claim) in enumerate(zip(docs, claims)):
            for part in split_fn(claim):
                rep_docs.append(doc)
                parts.append(part)
                owner.append(i)
        p_max, _, _, _ = self.score_examples(rep_docs, parts, progress=progress)
        out = np.full(len(docs), np.inf)
        for i, p in zip(owner, p_max):
            out[i] = min(out[i], p)
        return out

    def official_score(self, docs: Sequence[str], claims: Sequence[str]):
        """Gọi thẳng `MiniCheck.score` gốc (dùng để kiểm tra độ khớp)."""
        return self.mc.score(docs=list(docs), claims=list(claims))


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start
