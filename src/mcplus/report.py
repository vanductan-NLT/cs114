"""Sinh bảng Markdown và tự điền vào các khối <!--AUTO:ten-->...<!--/AUTO:ten--> của báo cáo."""

from __future__ import annotations

import math
import re
from typing import Dict

import pandas as pd


def _fmt(value, floatfmt: str) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "–"
        return format(value, floatfmt)
    return str(value)


def md_table(frame: pd.DataFrame, floatfmt: str = ".2f", index: bool = True, bold_max_rows=None,
             bold_cols=None) -> str:
    """Bảng Markdown không cần thư viện tabulate. `bold_max_rows`: danh sách nhãn dòng (hoặc
    "all") mà giá trị lớn nhất trên dòng được in đậm, chỉ so giữa `bold_cols` (mặc định mọi cột số)."""
    df = frame.reset_index() if index else frame.copy()
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" if i == 0 and index else ":---:" for i in range(len(df.columns))) + "|"
    lines = [header, sep]
    numeric_cols = bold_cols or [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    for (label, row), (_, raw) in zip(df.iterrows(), frame.iterrows()):
        best = None
        if bold_max_rows == "all" or (bold_max_rows is not None and raw.name in bold_max_rows):
            vals = pd.to_numeric(raw[numeric_cols], errors="coerce")
            best = vals.max() if vals.notna().any() else None
        cells = []
        for col in df.columns:
            text = _fmt(row[col], floatfmt)
            if best is not None and col in numeric_cols and isinstance(row[col], float) \
                    and not math.isnan(row[col]) and abs(row[col] - best) < 1e-9:
                text = f"**{text}**"
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fill_template(text: str, blocks: Dict[str, str]) -> str:
    """Thay nội dung giữa <!--AUTO:ten--> và <!--/AUTO:ten--> bằng blocks[ten]."""
    for name, content in blocks.items():
        pattern = re.compile(rf"(<!--AUTO:{re.escape(name)}-->)(.*?)(<!--/AUTO:{re.escape(name)}-->)", re.S)
        text = pattern.sub(lambda m: f"{m.group(1)}\n{content}\n{m.group(3)}", text)
    return text
