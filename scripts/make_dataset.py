"""生成评估数据集：一批固定种子的随机量子电路，保存为 JSON。

用法（在项目根目录运行）::

    python scripts/make_dataset.py --out data/eval.json --count 100 \\
        --n-qubits 5 --depth 12 --seed 10000

数据集与训练集必须使用不同种子，避免评估泄漏。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):  # 以 `python scripts/make_dataset.py` 方式直接运行
    sys.path.insert(0, str(ROOT))

LOGGER = logging.getLogger("qco.scripts.make_dataset")


def _resolve(path: str) -> Path:
    """把命令行路径解析为绝对路径：相对路径统一相对项目根目录。"""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="生成固定种子的随机量子电路评估集（JSON）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", default="data/eval.json", help="输出 JSON 路径")
    parser.add_argument("--count", type=int, default=100, help="电路数量")
    parser.add_argument("--n-qubits", type=int, default=5, help="量子比特数")
    parser.add_argument("--depth", type=int, default=12, help="电路深度（层数）")
    parser.add_argument("--seed", type=int, default=10000, help="随机种子")
    parser.add_argument("--p-two", type=float, default=0.15, help="放置二比特门的概率")
    parser.add_argument("--log-level", default="INFO", help="日志级别")
    return parser


def main(argv: list[str] | None = None) -> int:
    """脚本入口：生成并保存数据集，返回进程退出码。"""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 延迟导入：--help 不必付出 numpy 等依赖的导入开销。
    from qco.generator import random_circuits, save_circuits

    out_path = _resolve(args.out)
    try:
        circuits = random_circuits(
            args.count,
            args.n_qubits,
            args.depth,
            seed=args.seed,
            p_two=args.p_two,
        )
        save_circuits(circuits, out_path)
    except ValueError as error:
        LOGGER.error("生成数据集失败：%s", error)
        return 1

    gate_counts = [circuit.num_ops for circuit in circuits]
    depths = [circuit.depth for circuit in circuits]
    LOGGER.info("已生成 %d 个随机电路：比特数=%d，深度=%d", len(circuits), args.n_qubits, args.depth)
    LOGGER.info(
        "平均门数=%.2f（%.1f ~ %.1f），平均深度=%.2f",
        sum(gate_counts) / len(gate_counts),
        min(gate_counts),
        max(gate_counts),
        sum(depths) / len(depths),
    )
    LOGGER.info("数据集已写入 %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
