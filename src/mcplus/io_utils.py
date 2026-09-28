"""Tiện ích đọc/ghi kết quả, tìm file đầu vào và ghi manifest (để tái lập)."""

from __future__ import annotations

import glob
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

IN_KAGGLE = Path("/kaggle/working").exists()


def output_root() -> Path:
    """Nơi ghi kết quả: /kaggle/working/results trên Kaggle, ./results ở chỗ khác
    (có thể đổi bằng biến môi trường MCPLUS_RESULTS)."""
    env = os.environ.get("MCPLUS_RESULTS")
    root = Path(env) if env else (Path("/kaggle/working/results") if IN_KAGGLE else Path("results"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def find_input(relative: str, extra_dirs: Iterable[str] = ()) -> Path:
    """Tìm file kết quả của notebook trước, theo thứ tự: thư mục output hiện tại,
    các thư mục thêm vào, thư mục results/ của repo, rồi mọi thứ trong /kaggle/input."""
    candidates = [output_root() / relative]
    candidates += [Path(d) / relative for d in extra_dirs]
    here = Path(__file__).resolve().parents[2]  # thư mục gốc repo
    candidates.append(here / "results" / relative)
    for cand in candidates:
        if cand.exists():
            return cand
    name = Path(relative).name
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    hits = [h for h in hits if h.replace("\\", "/").endswith(relative.replace("\\", "/"))] or hits
    if hits:
        return Path(hits[0])
    raise FileNotFoundError(
        f"Không tìm thấy '{relative}'. Hãy chạy notebook tạo ra file này, rồi trên Kaggle "
        "bấm Add Input -> chọn output của notebook đó, hoặc copy file vào results/ trong repo.")


def save_table(frame: pd.DataFrame, relative: str) -> Path:
    path = output_root() / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def load_table(relative: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(find_input(relative), **kwargs)


def _pkg_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def write_manifest(relative: str, **info) -> Path:
    """Ghi thông tin môi trường + cấu hình chạy vào JSON."""
    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "gpu": gpu,
        "packages": {p: _pkg_version(p) for p in ("torch", "transformers", "datasets", "minicheck",
                                                  "sentence-transformers", "rank-bm25",
                                                  "scikit-learn", "pandas", "numpy")},
        **info,
    }
    path = output_root() / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def pip_install(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)


def model_revision(repo_id: str) -> Optional[str]:
    """Commit hash hiện tại của checkpoint trên Hugging Face (ghi vào manifest)."""
    try:
        from huggingface_hub import model_info
        return model_info(repo_id).sha
    except Exception:
        return None
