"""Hàm dùng chung cho notebook: cài đặt môi trường, chế độ smoke test, hiển thị, lưu hình."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from . import MINICHECK_COMMIT, MINICHECK_REPO
from .io_utils import IN_KAGGLE, output_root, pip_install


def smoke_mode() -> bool:
    """MCPLUS_SMOKE=1 -> chạy trên vài chục mẫu để kiểm tra code (không phải kết quả)."""
    return os.environ.get("MCPLUS_SMOKE", "0") == "1"


def ensure_nltk() -> None:
    import nltk
    for res in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{res}")
        except LookupError:
            nltk.download(res, quiet=True)


def ensure_minicheck() -> None:
    """Cài MiniCheck đúng commit đã pin, không đụng tới torch/transformers có sẵn."""
    try:
        import minicheck  # noqa: F401
        return
    except ImportError:
        pass
    pip_install("--no-deps", f"git+{MINICHECK_REPO}@{MINICHECK_COMMIT}")


def ensure_packages(*names: str) -> None:
    import importlib
    missing = []
    for name in names:
        module = {"rank_bm25": "rank_bm25", "sentence-transformers": "sentence_transformers"}.get(name, name)
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(name)
    if missing:
        pip_install(*missing)


def hf_login() -> bool:
    from .data import get_hf_token
    token = get_hf_token()
    if token:
        os.environ["HF_TOKEN"] = token
    print("HF_TOKEN:", "có (không in ra)" if token else "KHÔNG có -> không tải được LLM-AggreFact (dataset gated)")
    return bool(token)


def display(obj) -> None:
    try:
        from IPython.display import display as _display
        if "ipykernel" in sys.modules:
            _display(obj)
            return
    except ImportError:
        pass
    print(obj.to_string() if hasattr(obj, "to_string") else obj)


def setup_plotting():
    import matplotlib
    if "ipykernel" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def savefig(fig, relative: str) -> Path:
    path = output_root() / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    return path


def env_summary() -> None:
    print("Kaggle:", IN_KAGGLE, "| smoke:", smoke_mode(), "| results ->", output_root())
