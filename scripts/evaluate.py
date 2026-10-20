"""评估入口脚本：加载训练好的检查点，在评估集上对比智能体与基线。

用法（在项目根目录运行）::

    python scripts/evaluate.py --checkpoint results/default/best.pt \\
        --dataset data/eval.json --out results/default/eval --baselines random,greedy

输出：``<out>/evaluation.json``（机读指标）与 stdout 上的 Markdown 对比表；
每个电路都用 ``qco.simulator.circuits_equivalent`` 验证语义等价，等价率应为 1.0。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):  # 以 `python scripts/evaluate.py` 方式直接运行
    sys.path.insert(0, str(ROOT))

LOGGER = logging.getLogger("qco.scripts.evaluate")

#: ``--baselines`` 允许的取值。
BASELINE_CHOICES: tuple[str, ...] = ("noop", "random", "greedy")


def _resolve(path: str) -> Path:
    """把命令行路径解析为绝对路径：相对路径统一相对项目根目录。"""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="在评估集上对比 PPO 智能体与 noop/random/greedy 基线",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default="results/default/best.pt", help="模型检查点路径")
    parser.add_argument("--dataset", default="data/eval.json", help="评估数据集 JSON；缺失时现场生成")
    parser.add_argument("--out", default="results/default/eval", help="输出目录（或 .json 文件路径）")
    parser.add_argument(
        "--baselines",
        default="random,greedy",
        help="要评估的基线，逗号分隔，可选值：noop,random,greedy",
    )
    parser.add_argument("--device", default="cpu", help="推理设备，如 cpu / cuda")
    parser.add_argument("--count", type=int, default=100, help="数据集缺失时现场生成的电路数")
    parser.add_argument("--seed", type=int, default=10000, help="数据集缺失时现场生成的随机种子")
    parser.add_argument("--log-level", default="INFO", help="日志级别")
    return parser


def _parse_baselines(text: str) -> list[str]:
    """解析 ``--baselines`` 逗号列表并校验取值。"""
    names: list[str] = []
    for piece in str(text).split(","):
        name = piece.strip().lower()
        if not name:
            continue
        if name not in BASELINE_CHOICES:
            raise ValueError(f"未知基线 {name!r}，可选值：{', '.join(BASELINE_CHOICES)}")
        if name not in names:
            names.append(name)
    return names


def _resolve_output_path(args: argparse.Namespace) -> Path:
    """把 ``--out`` 解析为最终的 ``evaluation.json`` 文件路径。"""
    out = _resolve(args.out)
    if out.suffix.lower() == ".json":
        return out
    return out / "evaluation.json"


def _load_eval_circuits(args: argparse.Namespace, env_config: EnvConfig) -> list:
    """加载评估集：优先 ``--dataset``，缺失时按配置现场生成。"""
    from qco.generator import load_circuits, random_circuits

    dataset_path = _resolve(args.dataset)
    if dataset_path.exists():
        circuits = load_circuits(dataset_path)
        for circuit in circuits:
            if circuit.n_qubits != env_config.n_qubits:
                raise ValueError(
                    f"数据集电路比特数 {circuit.n_qubits} 与模型配置 {env_config.n_qubits} 不一致"
                )
            if circuit.depth > env_config.n_moments:
                raise ValueError(
                    f"数据集电路深度 {circuit.depth} 超过模型的 n_moments={env_config.n_moments}"
                )
        LOGGER.info("从 %s 加载 %d 个评估电路", dataset_path, len(circuits))
        return circuits
    circuits = random_circuits(
        args.count,
        env_config.n_qubits,
        env_config.initial_depth,
        seed=args.seed,
        p_two=env_config.p_two,
    )
    LOGGER.info("未找到 %s，现场生成 %d 个评估电路", dataset_path, len(circuits))
    return circuits


def main(argv: list[str] | None = None) -> int:
    """脚本入口：评估并输出结果，返回进程退出码。"""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 延迟导入：--help 不必付出 torch 等重依赖的导入开销。
    from qco.evaluate import (
        EvalResult,
        evaluate_agent,
        evaluate_greedy,
        evaluate_noop,
        evaluate_random,
        format_comparison,
        save_eval_results,
    )
    from qco.environment import EnvConfig
    from qco.trainer import build_policy_from_checkpoint, load_checkpoint
    from qco.utils import resolve_device

    checkpoint_path = _resolve(args.checkpoint)
    if not checkpoint_path.exists():
        LOGGER.error("检查点不存在：%s（请先运行 scripts/train.py）", checkpoint_path)
        return 1

    try:
        baseline_names = _parse_baselines(args.baselines)
        device = resolve_device(args.device)
        checkpoint = load_checkpoint(checkpoint_path)
        policy, env_config, _raw_config = build_policy_from_checkpoint(
            checkpoint, device=device
        )
        circuits = _load_eval_circuits(args, env_config)
    except (ValueError, KeyError) as error:
        LOGGER.error("评估配置非法：%s", error)
        return 1

    LOGGER.info(
        "评估配置：%d 个电路，设备=%s，检查点迭代轮次=%s",
        len(circuits),
        device,
        checkpoint.get("iteration", "?"),
    )
    results: dict[str, EvalResult] = {"noop": evaluate_noop(circuits, env_config)}
    for name in baseline_names:
        if name == "noop":
            continue
        if name == "random":
            results["random"] = evaluate_random(circuits, env_config, seed=args.seed)
        elif name == "greedy":
            results["greedy"] = evaluate_greedy(circuits, env_config)
    results["agent"] = evaluate_agent(
        policy, circuits, env_config, device=device, deterministic=True
    )

    output_path = _resolve_output_path(args)
    try:
        save_eval_results(results, output_path)
    except OSError as error:
        LOGGER.error("保存评估结果失败：%s", error)
        return 1

    for result in results.values():
        LOGGER.info(
            "%s：代价压缩 %.2f%%，门压缩 %.2f%%，深度压缩 %.2f%%，等价率 %.2f，平均步数 %.2f",
            result.method,
            result.cost_reduction_pct,
            result.gate_reduction_pct,
            result.depth_reduction_pct,
            result.equivalence_rate,
            result.mean_steps,
        )
    LOGGER.info("评估结果已保存到 %s", output_path)

    table = format_comparison(results)
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
