"""Tách câu và cắt chunk giống hệt MiniCheck (minicheck/inference.py @ b58b9fa).

Hàm `chunks` trong MiniCheck là hàm lồng bên trong `Inferencer.inference_per_example`
nên không gọi lại được; file này chép lại đúng logic đó. Notebook 01 có bước kiểm tra
số liệu của đường chạy nhanh khớp với `MiniCheck.score` gốc.
"""

from __future__ import annotations

from typing import Callable, Iterable, List

from nltk.tokenize import sent_tokenize


def sent_tokenize_with_newlines(text: str) -> List[str]:
    """Giống MiniCheck: tách câu theo từng dòng, chèn '\\n' giữa các dòng."""
    blocks = text.split("\n")
    tokenized_text: List[str] = []
    for block in blocks:
        tokenized_text.extend(sent_tokenize(block))
        tokenized_text.append("\n")
    return tokenized_text[:-1]


def _chunk_sentences(sentences: Iterable[str], n: int, length_fn: Callable[[str], int]):
    current, count = [], 0
    for sentence in sentences:
        length = length_fn(sentence)
        if count + length > n:
            yield " ".join(current)
            current, count = [sentence], length
        else:
            current.append(sentence)
            count += length
    if current:
        yield " ".join(current)


def make_chunks(doc: str, chunk_size: int, length_fn: Callable[[str], int]) -> List[str]:
    """Danh sách chunk của `doc`, đúng như `used_chunks` mà MiniCheck trả về."""
    sents = sent_tokenize_with_newlines(doc) or [""]
    return chunks_from_lengths(sents, [length_fn(s) for s in sents], chunk_size)


def chunks_from_lengths(sents: List[str], lengths: List[int], chunk_size: int) -> List[str]:
    """Như `make_chunks` nhưng nhận sẵn độ dài từng câu (để tokenize cả văn bản một lần)."""
    lookup = iter(lengths)
    raw = _chunk_sentences(sents, chunk_size, lambda _s: next(lookup))
    chunks = [c.replace(" \n ", "\n").strip() for c in raw]
    chunks = [c for c in chunks if c != ""]
    return chunks or [" ".join(sents).strip()]


def doc_sentences(doc: str) -> List[str]:
    """Câu của văn bản (bỏ các phần tử '\\n' phân cách dòng) để xếp hạng bằng chứng."""
    return [s for s in sent_tokenize_with_newlines(doc) if s.strip() and s != "\n"]


def split_claim(claim: str) -> List[str]:
    """Tách claim nhiều câu thành từng câu (dùng cho ablation C)."""
    parts = [s.strip() for s in sent_tokenize(claim) if s.strip()]
    return parts or [claim]
