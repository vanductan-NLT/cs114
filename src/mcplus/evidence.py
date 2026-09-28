"""Cải tiến A: chọn lọc bằng chứng theo claim (claim-aware evidence selection).

MiniCheck cắt văn bản dài thành các chunk liền nhau rồi lấy max. Nếu bằng chứng cho một
claim nằm rải ở nhiều chunk thì không chunk nào đủ để kết luận "supported". Ở đây ta xếp
hạng từng câu của văn bản theo độ liên quan với claim, chọn các câu tốt nhất (kèm câu lân
cận) cho tới khi đầy đúng một chunk, giữ nguyên thứ tự gốc, rồi chấm một lần bằng cùng model.
Văn bản chỉ có 1 chunk thì giữ nguyên như baseline.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .chunking import doc_sentences

_WORD = re.compile(r"\w+", re.UNICODE)
# Stopword tiếng Anh tối thiểu để BM25 không bị các từ chức năng chi phối.
_STOP = set("""a an the and or but if of to in on at for from by with as is are was were be been being
this that these those it its it's he she they them his her their we our you your i me my
not no do does did so than then there here which who whom what when where why how all any
both each few more most other some such only own same too very can will just should now""".split())


def tokenize_for_bm25(text: str) -> List[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


@dataclass(frozen=True)
class EvidenceConfig:
    selector: str = "hybrid"      # "bm25" | "dense" | "hybrid"
    top_k: Optional[int] = None   # None = lấy theo thứ hạng tới khi đầy ngân sách
    window: int = 1               # số câu lân cận mỗi bên của câu được chọn
    split_claim: bool = False     # ablation C: chấm từng câu của claim, lấy min

    def name(self) -> str:
        k = "fill" if self.top_k is None else f"k{self.top_k}"
        return f"{self.selector}_{k}_w{self.window}" + ("_split" if self.split_claim else "")

    def as_dict(self) -> dict:
        return asdict(self)


class SentenceRanker:
    """Xếp hạng câu theo BM25, embedding (cosine) hoặc lai (Reciprocal Rank Fusion)."""

    def __init__(self, dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None, rrf_k: int = 60):
        self.dense_model_name = dense_model
        self.device = device
        self.rrf_k = rrf_k
        self._encoder = None
        self._emb_cache: Dict[str, np.ndarray] = {}

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.dense_model_name, device=self.device)
        return self._encoder

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._emb_cache]
        if missing:
            vecs = self.encoder.encode(missing, batch_size=256, normalize_embeddings=True,
                                       show_progress_bar=False, convert_to_numpy=True)
            self._emb_cache.update(zip(missing, vecs))
        return np.stack([self._emb_cache[t] for t in texts])

    def precompute(self, texts: Sequence[str]) -> None:
        """Mã hóa trước toàn bộ câu/claim theo lô lớn (nhanh hơn nhiều so với từng ví dụ)."""
        uniq = [t for t in dict.fromkeys(texts) if t not in self._emb_cache]
        if uniq:
            vecs = self.encoder.encode(uniq, batch_size=512, normalize_embeddings=True,
                                       show_progress_bar=True, convert_to_numpy=True)
            self._emb_cache.update(zip(uniq, vecs))

    @staticmethod
    def bm25_scores(sents: Sequence[str], query: str) -> np.ndarray:
        from rank_bm25 import BM25Okapi

        corpus = [tokenize_for_bm25(s) or ["_"] for s in sents]
        q = tokenize_for_bm25(query)
        if not q:
            return np.zeros(len(sents))
        return np.asarray(BM25Okapi(corpus).get_scores(q), dtype=float)

    def dense_scores(self, sents: Sequence[str], query: str) -> np.ndarray:
        emb = self._embed(list(sents))
        q = self._embed([query])[0]
        return emb @ q

    def rank(self, sents: Sequence[str], query: str, selector: str) -> np.ndarray:
        """Chỉ số câu theo thứ tự liên quan giảm dần (hòa điểm thì câu đứng trước ưu tiên)."""
        n = len(sents)
        if selector == "bm25":
            return np.lexsort((np.arange(n), -self.bm25_scores(sents, query)))
        if selector == "dense":
            return np.lexsort((np.arange(n), -self.dense_scores(sents, query)))
        if selector == "hybrid":
            fused = np.zeros(n)
            for scores in (self.bm25_scores(sents, query), self.dense_scores(sents, query)):
                order = np.lexsort((np.arange(n), -scores))
                ranks = np.empty(n)
                ranks[order] = np.arange(1, n + 1)
                fused += 1.0 / (self.rrf_k + ranks)
            return np.lexsort((np.arange(n), -fused))
        raise ValueError(selector)


def select_evidence(sents: Sequence[str], ranking: Sequence[int], lengths: Sequence[int],
                    budget: int, top_k: Optional[int], window: int) -> List[int]:
    """Chọn chỉ số câu (thứ tự gốc) sao cho tổng độ dài <= budget.

    Duyệt các câu theo thứ hạng; mỗi câu "hạt giống" kéo theo `window` câu lân cận mỗi bên
    (câu hạt giống trước, rồi lân cận gần trước xa sau). Câu nào làm vượt ngân sách thì bỏ
    qua câu đó và thử câu kế tiếp. Nếu `top_k` được đặt thì chỉ dùng k hạt giống đầu tiên.
    Câu hạt giống đầu tiên luôn được giữ kể cả khi dài hơn ngân sách (model sẽ tự cắt).
    """
    n = len(sents)
    seeds = list(ranking) if top_k is None else list(ranking)[:top_k]
    chosen: set = set()
    used = 0
    for rank_pos, seed in enumerate(seeds):
        group = [seed]
        for d in range(1, window + 1):
            group += [j for j in (seed - d, seed + d) if 0 <= j < n]
        for j in group:
            if j in chosen:
                continue
            if used + lengths[j] <= budget or (rank_pos == 0 and j == seed and not chosen):
                chosen.add(j)
                used += lengths[j]
    return sorted(chosen)


class EvidenceSelector:
    """Kết hợp ranker + FastScorer để tạo ngữ cảnh bằng chứng và chấm điểm."""

    def __init__(self, scorer, ranker: SentenceRanker):
        self.scorer, self.ranker = scorer, ranker
        self._sent_cache: Dict[str, dict] = {}

    def doc_info(self, doc: str) -> dict:
        key = hashlib.md5(doc.encode("utf-8")).hexdigest()
        info = self._sent_cache.get(key)
        if info is None:
            sents = doc_sentences(doc) or [doc]
            info = {"sents": sents, "lengths": self.scorer.sentence_lengths(sents)}
            self._sent_cache[key] = info
        return info

    def build_context(self, doc: str, query: str, cfg: EvidenceConfig) -> str:
        info = self.doc_info(doc)
        ranking = self.ranker.rank(info["sents"], query, cfg.selector)
        keep = select_evidence(info["sents"], ranking, info["lengths"], self.scorer.chunk_size,
                               cfg.top_k, cfg.window)
        return " ".join(info["sents"][j] for j in keep)

    def precompute_embeddings(self, docs: Sequence[str], queries: Sequence[str]) -> None:
        texts: List[str] = []
        for doc in dict.fromkeys(docs):
            texts.extend(self.doc_info(doc)["sents"])
        self.ranker.precompute(texts + list(queries))

    def context(self, doc: str, query: str, cfg: EvidenceConfig) -> str:
        """Văn bản vừa một chunk thì dùng đúng chunk đó (giống baseline); dài hơn thì chọn lọc."""
        chunks = self.scorer.chunk_doc(doc)["chunks"]
        return chunks[0] if len(chunks) == 1 else self.build_context(doc, query, cfg)

    def score(self, docs: Sequence[str], claims: Sequence[str], cfg: EvidenceConfig,
              split_fn: Optional[Callable[[str], List[str]]] = None, progress: bool = True) -> np.ndarray:
        """Điểm P(supported) theo cấu hình. Với split_claim: chọn bằng chứng và chấm cho
        từng câu của claim, điểm claim = min các câu (claim đúng khi mọi câu đều đúng)."""
        owners, ctx, queries = [], [], []
        for i, (doc, claim) in enumerate(zip(docs, claims)):
            parts = split_fn(claim) if (cfg.split_claim and split_fn) else [claim]
            for part in parts:
                owners.append(i)
                ctx.append(self.context(doc, part, cfg))
                queries.append(part)
        probs = self.scorer.score_pairs(ctx, queries, progress=progress, desc=cfg.name())
        out = np.full(len(docs), np.inf)
        for i, p in zip(owners, probs):
            out[i] = min(out[i], p)
        return out
