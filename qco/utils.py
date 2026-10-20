"""通用小工具：随机种子、设备选择、JSON 序列化与目录管理。"""

from __future__ import annotations

import dataclasses
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    """设置 ``random`` / ``numpy`` / ``torch`` 的全局随机种子，保证可复现。"""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch 可能尚未安装（例如只跑纯模拟器测试）
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - 依赖本机 GPU
            torch.cuda.manual_seed_all(seed)
    except Exception:  # noqa: BLE001 - 可选依赖，失败时不影响 numpy 部分
        pass


def resolve_device(name: str = "cpu"):
    """把字符串转换为 ``torch.device``；不可用时回退到 CPU。"""
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def ensure_dir(path: str | Path) -> Path:
    """创建目录（含父目录）并返回 ``Path``。"""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def to_jsonable(obj: Any) -> Any:
    """把 dataclass / numpy / Path / set 等递归转换为可 JSON 序列化对象。"""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {field.name: to_jsonable(getattr(obj, field.name)) for field in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(item) for item in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(to_jsonable(item) for item in obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj: Any, path: str | Path) -> None:
    """保存 JSON 文件（自动创建父目录，支持 dataclass/numpy）。"""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(obj), handle, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    """读取 JSON 文件。"""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def config_to_dict(config: Any) -> dict[str, Any]:
    """把（可能嵌套的）配置 dataclass 转换为字典。"""
    return to_jsonable(config)


def format_seconds(seconds: float) -> str:
    """把秒数格式化为易读字符串。"""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    minutes, rest = divmod(seconds, 60.0)
    return f"{int(minutes)} min {rest:.1f} s"


def process_memory_mb() -> float:
    """返回当前进程常驻内存（MB）；非 Windows 平台回退为 0。"""
    try:
        import psutil  # type: ignore

        return psutil.Process(os.getpid()).memory_info().rss / 1024**2
    except Exception:  # noqa: BLE001 - psutil 可选
        return 0.0
