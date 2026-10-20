"""训练入口脚本：按契约约定的环境与 PPO 配置训练策略网络。

用法（在项目根目录运行）::

    python scripts/train.py --out results/default --iterations 150 \\
        --num-envs 8 --steps-per-iter 16 --seed 0 --device cpu

扩展规模实验时用 ``--dataset`` 指定与 ``--n-qubits`` 匹配的评估数据集，
例如 7 比特：``python scripts/train.py --n-qubits 7 --dataset data/eval7.json``。

训练过程用中文日志打印每次迭代的回报、代价压缩率与 PPO 指标；
训练结束后打印 noop / random / greedy / agent 四种方法的汇总对比表。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):  # 以 `python scripts/train.py` 方式直接运行
    sys.path.insert(0, str(ROOT))

LOGGER = logging.getLogger("qco.scripts.train")


def _resolve(path: str) -> Path:
    """把命令行路径解析为绝对路径：相对路径统一相对项目根目录。"""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="训练 PPO 策略优化量子电路（CNN Actor-Critic）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", default="results/default", help="产物输出目录")
    parser.add_argument("--iterations", type=int, default=150, help="训练迭代次数")
    parser.add_argument("--num-envs", type=int, default=8, help="并行环境数")
    parser.add_argument("--steps-per-iter", type=int, default=16, help="每次迭代每个环境采样步数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--device", default="cpu", help="训练设备，如 cpu / cuda")
    parser.add_argument("--max-steps", type=int, default=48, help="单回合最大步数")
    parser.add_argument("--lambda-depth", type=float, default=0.5, help="代价函数中深度的权重")
    parser.add_argument("--n-qubits", type=int, default=5, help="量子比特数")
    parser.add_argument("--n-moments", type=int, default=15, help="观测张量时隙窗口")
    parser.add_argument("--initial-depth", type=int, default=12, help="随机初始电路深度")
    parser.add_argument("--p-two", type=float, default=0.15, help="放置二比特门的概率")
    parser.add_argument("--hidden-channels", type=int, default=64, help="CNN 隐藏通道数")
    parser.add_argument("--eval-every", type=int, default=10, help="每隔多少轮评估一次")
    parser.add_argument("--save-every", type=int, default=25, help="每隔多少轮保存一次周期检查点")
    parser.add_argument("--eval-circuits", type=int, default=100, help="评估电路数量（无数据集时现场生成）")
    parser.add_argument("--train-circuits", type=int, default=200, help="训练电路池大小")
    parser.add_argument("--eval-seed", type=int, default=10000, help="评估集随机种子")
    parser.add_argument(
        "--dataset",
        default="",
        help="评估数据集 JSON；默认使用 data/eval.json，不存在则按 eval-seed 现场生成；"
        "相对路径按项目根目录解析",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="PPO 学习率")
    parser.add_argument("--log-level", default="INFO", help="日志级别")
    return parser


def build_config(args: argparse.Namespace) -> TrainConfig:
    """把命令行参数转换为 :class:`qco.trainer.TrainConfig`。

    单独抽成函数便于测试“参数是否真的接线到配置”，也避免在 ``--help`` 路径上
    导入 torch 等重依赖。
    """
    from qco.environment import EnvConfig
    from qco.ppo import PPOConfig
    from qco.trainer import TrainConfig

    env_config = EnvConfig(
        n_qubits=args.n_qubits,
        n_moments=args.n_moments,
        max_steps=args.max_steps,
        lambda_depth=args.lambda_depth,
        initial_depth=args.initial_depth,
        p_two=args.p_two,
        seed=args.seed,
    )
    return TrainConfig(
        env=env_config,
        ppo=PPOConfig(lr=args.lr),
        iterations=args.iterations,
        steps_per_iter=args.steps_per_iter,
        num_envs=args.num_envs,
        hidden_channels=args.hidden_channels,
        eval_every=args.eval_every,
        save_every=args.save_every,
        eval_circuits=args.eval_circuits,
        train_circuits=args.train_circuits,
        eval_seed=args.eval_seed,
        out_dir=str(_resolve(args.out)),
        device=args.device,
        eval_dataset=str(_resolve(args.dataset)) if str(args.dataset).strip() else None,
    )


def main(argv: list[str] | None = None) -> int:
    """脚本入口：构造配置、训练并打印汇总表，返回进程退出码。"""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 延迟导入：--help 不必付出 torch 等重依赖的导入开销。
    from qco.evaluate import EvalResult, format_comparison
    from qco.trainer import Trainer

    try:
        config = build_config(args)
    except ValueError as error:
        LOGGER.error("配置非法：%s", error)
        return 1

    trainer = Trainer(config)
    summary = trainer.train()

    results = {
        method: EvalResult(**payload) for method, payload in summary["results"].items()
    }
    print("=" * 72)
    print("训练完成，四种方法在评估集上的对比：")
    print(format_comparison(results))
    print(f"产物目录：{summary['out_dir']}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
