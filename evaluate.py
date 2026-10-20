"""根目录评估入口（薄封装）。

等价于 ``python scripts/evaluate.py``，保留原仓库的根目录文件名习惯。
参数与行为完全由 :mod:`scripts.evaluate` 提供，见::

    python evaluate.py --help
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    """以 ``__main__`` 方式执行 ``scripts/evaluate.py`` 并返回其退出码。"""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        runpy.run_path(str(ROOT / "scripts" / "evaluate.py"), run_name="__main__")
    except SystemExit as exit_signal:  # argparse 的 --help / 参数错误走 SystemExit
        code = exit_signal.code
        return 0 if code is None else int(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
