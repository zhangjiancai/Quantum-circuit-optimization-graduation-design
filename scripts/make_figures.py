"""论文插图脚本：根据训练产物生成训练曲线、方法对比、规则使用占比与电路图。

用法（在项目根目录运行）::

    python scripts/make_figures.py --run results/default --out results/figures

生成的图（缺数据时自动跳过并给出中文警告，不会中断）::

    training_curves.png   训练回报 + 评估集压缩率 2×2 曲线
    method_comparison.png noop/random/greedy/agent 压缩率分组柱状图
    rule_usage.png        各方法的规则使用占比堆叠柱状图
    circuit_before.png    示例电路（优化前）
    circuit_after.png     同一电路经贪心基线优化后的结果
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):  # 以 `python scripts/make_figures.py` 方式直接运行
    sys.path.insert(0, str(ROOT))

LOGGER = logging.getLogger("qco.scripts.make_figures")


def _resolve(path: str) -> Path:
    """把命令行路径解析为绝对路径：相对路径统一相对项目根目录。"""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="根据训练产物生成论文插图（训练曲线 / 方法对比 / 规则占比 / 电路图）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run", default="results/default", help="训练产物目录（含 CSV / summary.json）")
    parser.add_argument("--out", default="results/figures", help="插图输出目录")
    parser.add_argument("--dataset", default="data/eval.json", help="评估数据集（用于绘制示例电路）")
    parser.add_argument("--n-qubits", type=int, default=5, help="数据集缺失时示例电路的比特数")
    parser.add_argument("--depth", type=int, default=12, help="数据集缺失时示例电路的深度")
    parser.add_argument("--seed", type=int, default=0, help="数据集缺失时的随机种子")
    parser.add_argument("--log-level", default="INFO", help="日志级别")
    return parser


def _example_circuit(args: argparse.Namespace):
    """返回一个示例电路：优先取数据集第一个，缺失时现场生成。"""
    from qco.generator import load_circuits, random_circuits

    dataset_path = _resolve(args.dataset)
    if dataset_path.exists():
        circuits = load_circuits(dataset_path)
        if circuits:
            return circuits[0]
    return random_circuits(1, args.n_qubits, args.depth, seed=args.seed)[0]


def main(argv: list[str] | None = None) -> int:
    """脚本入口：生成全部插图，返回进程退出码。"""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 延迟导入：--help 不必付出 torch / matplotlib 等重依赖的导入开销。
    from qco.baselines import optimize_greedy
    from qco.environment import EnvConfig
    from qco.evaluate import EvalResult
    from qco.utils import ensure_dir, load_json
    from qco.visualizer import (
        plot_circuit,
        plot_method_comparison,
        plot_rule_usage,
        plot_training_curves,
    )

    run_dir = _resolve(args.run)
    out_dir = ensure_dir(_resolve(args.out))
    generated: list[Path] = []

    train_csv = run_dir / "train_log.csv"
    eval_csv = run_dir / "eval_log.csv"
    if train_csv.exists() and eval_csv.exists():
        path = out_dir / "training_curves.png"
        plot_training_curves(train_csv, eval_csv, path)
        generated.append(path)
    else:
        LOGGER.warning("缺少训练日志（%s / %s），跳过训练曲线", train_csv, eval_csv)

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        results = {
            method: EvalResult(**payload)
            for method, payload in dict(summary.get("results", {})).items()
        }
        if results:
            path = out_dir / "method_comparison.png"
            plot_method_comparison(results, path)
            generated.append(path)
            path = out_dir / "rule_usage.png"
            plot_rule_usage(results, path)
            generated.append(path)
        else:
            LOGGER.warning("%s 中没有评估结果，跳过方法对比图", summary_path)
    else:
        LOGGER.warning("缺少训练摘要 %s，跳过方法对比图", summary_path)

    try:
        circuit = _example_circuit(args)
        env_config = EnvConfig(
            n_qubits=circuit.n_qubits,
            n_moments=max(circuit.depth, 1),
            max_steps=48,
            initial_depth=max(circuit.depth, 1),
            p_two=0.15,
            seed=0,
        )
        optimized = optimize_greedy(circuit, env_config)
        before_path = out_dir / "circuit_before.png"
        after_path = out_dir / "circuit_after.png"
        plot_circuit(circuit, before_path, title=f"优化前电路（门数 {circuit.num_ops}，深度 {circuit.depth}）")
        plot_circuit(
            optimized.final_circuit,
            after_path,
            title=(
                f"优化后电路（门数 {optimized.final_circuit.num_ops}，"
                f"深度 {optimized.final_circuit.depth}）"
            ),
        )
        generated.extend([before_path, after_path])
    except ValueError as error:  # pragma: no cover - 配置错误时给出明确提示
        LOGGER.error("生成示例电路图失败：%s", error)

    if not generated:
        LOGGER.error("没有生成任何插图：请先运行 scripts/train.py 或检查 --run / --out 参数")
        return 1
    for path in generated:
        LOGGER.info("已生成插图 %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
