"""Chuyển nb_src/*.py (định dạng "percent": các ô bắt đầu bằng `# %%`) thành notebooks/*.ipynb.

Viết notebook dưới dạng .py giúp đọc diff dễ và chạy thử được bằng `python nb_src/xx.py`.
Chạy lại lệnh này mỗi khi sửa file trong nb_src/:

    python tools/build_notebooks.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELL = re.compile(r"^# %%(.*)$")


def parse(text: str):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        lines = buf[:]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if kind == "markdown":
            lines = [re.sub(r"^# ?", "", ln) for ln in lines]
        if lines:
            cells.append((kind, "\n".join(lines)))

    for line in text.splitlines():
        m = CELL.match(line)
        if m:
            flush()
            kind = "markdown" if "[markdown]" in m.group(1) else "code"
            buf = []
        elif kind is not None:
            buf.append(line)
    flush()
    return cells


def to_notebook(cells):
    nb_cells = []
    for kind, src in cells:
        lines = src.split("\n")
        source = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": kind, "metadata": {}, "source": source}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    src_dir, out_dir = ROOT / "nb_src", ROOT / "notebooks"
    out_dir.mkdir(exist_ok=True)
    for path in sorted(src_dir.glob("*.py")):
        nb = to_notebook(parse(path.read_text(encoding="utf-8")))
        target = out_dir / (path.stem + ".ipynb")
        target.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{path.relative_to(ROOT)} -> {target.relative_to(ROOT)} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    sys.exit(main())
