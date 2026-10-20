"""训练器：把环境、策略网络、rollout 与 PPO 串成完整的训练流水线。

训练流程
--------
1. 固定随机种子并构造 ``num_envs`` 个并行环境；
2. 每个 iteration 采样 ``steps_per_iter`` 步的轨迹（:class:`qco.rollout.RolloutCollector`），
   用 :class:`qco.ppo.PPO` 做多轮小批量更新；
3. 记录 ``train_log.csv``（每个 iteration 一行）；
4. 每 ``eval_every`` 轮在固定评估集上评估智能体并记录 ``eval_log.csv``，
   以评估集 ``mean_final_cost`` 最小为准更新 ``best.pt``；
5. 训练结束后在评估集上评估 noop / random / greedy / agent 四种方法，
   写出 ``summary.json``。

所有产物（``config.json``、``train_log.csv``、``eval_log.csv``、``best.pt``、
``last.pt``、``summary.json``）都落在 ``TrainConfig.out_dir`` 下，便于复现实验。
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from qco.agent import CircuitPolicyNet
from qco.circuit import Circuit
from qco.environment import CircuitOptimizationEnv, EnvConfig
from qco.evaluate import (
    EvalResult,
    evaluate_agent,
    evaluate_greedy,
    evaluate_noop,
    evaluate_random,
    format_comparison,
)
from qco.gates import N_CHANNELS
from qco.generator import load_circuits, random_circuits
from qco.ppo import PPO, PPOConfig
from qco.rollout import EpisodeStats, RolloutBatch, RolloutCollector
from qco.rules import RULES
from qco.utils import (
    config_to_dict,
    ensure_dir,
    resolve_device,
    save_json,
    set_seed,
    to_jsonable,
)

#: 训练日志列（顺序固定，便于外部脚本按列名读取）。
TRAIN_LOG_COLUMNS: tuple[str, ...] = (
    "iteration",
    "mean_return",
    "mean_episode_len",
    "mean_initial_cost",
    "mean_final_cost",
    "gate_reduction_pct",
    "depth_reduction_pct",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "explained_variance",
)

#: 评估日志列（顺序固定）。
EVAL_LOG_COLUMNS: tuple[str, ...] = (
    "iteration",
    "method",
    "cost_reduction_pct",
    "gate_reduction_pct",
    "depth_reduction_pct",
    "equivalence_rate",
    "mean_steps",
)

#: 项目根目录（本文件位于 ``<root>/qco/trainer.py``）。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: 数据集中评估电路的默认路径。
#: 相对**项目根目录**解析（而不是当前工作目录），因此从任意目录启动脚本都能命中
#: 同一份 ``data/eval.json``，避免“换一个 cwd 就静默改用现场生成的评估集”。
DEFAULT_EVAL_DATASET = PROJECT_ROOT / "data" / "eval.json"


@dataclass(frozen=True)
class TrainConfig:
    """训练超参数与产物配置。

    Attributes
    ----------
    env
        环境配置（比特数、时隙窗口、步数上限、代价深度权重、随机种子等）。
    ppo
        PPO 超参数。
    iterations
        训练迭代（rollout + 更新）的次数。
    steps_per_iter
        每次迭代每个环境连续采样的步数（horizon）。
    num_envs
        并行环境数。
    hidden_channels
        策略网络卷积主干的隐藏通道数。
    eval_every
        每隔多少轮在评估集上评估一次智能体。
    save_every
        每隔多少轮额外保存一次周期性检查点；``0`` 表示只保留 best/last。
    eval_circuits
        评估集电路数（仅在需要现场生成评估集时生效）。
    train_circuits
        训练电路池大小；每个回合结束后从池中随机取新电路。
    eval_seed
        评估集随机种子。
    out_dir
        产物输出目录。
    device
        训练设备，``"cpu"`` 或 ``"cuda"`` 等。
    eval_dataset
        评估数据集 JSON 路径；``None`` 表示使用项目根目录下的 ``data/eval.json``，
        该文件不存在时用 ``eval_seed`` 现场生成。相对路径按**项目根目录**解析，
        因此从任意工作目录启动都能命中同一份数据集；显式给出的路径必须存在。
    """

    env: EnvConfig = EnvConfig()
    ppo: PPOConfig = PPOConfig()
    iterations: int = 150
    steps_per_iter: int = 16
    num_envs: int = 8
    hidden_channels: int = 64
    eval_every: int = 10
    save_every: int = 25
    eval_circuits: int = 100
    train_circuits: int = 200
    eval_seed: int = 10000
    out_dir: str = "results/default"
    device: str = "cpu"
    eval_dataset: str | None = None

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations 至少为 1")
        if self.steps_per_iter < 1:
            raise ValueError("steps_per_iter 至少为 1")
        if self.num_envs < 1:
            raise ValueError("num_envs 至少为 1")
        if self.hidden_channels < 1:
            raise ValueError("hidden_channels 至少为 1")
        if self.eval_every < 1:
            raise ValueError("eval_every 至少为 1")
        if self.save_every < 0:
            raise ValueError("save_every 不能为负")
        if self.eval_circuits < 1 or self.train_circuits < 1:
            raise ValueError("eval_circuits / train_circuits 至少为 1")


def _mean(values: Sequence[float]) -> float:
    """求算术平均；空序列返回 ``0.0``。"""
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _reduction_pct(initial: float, final: float) -> float:
    """计算压缩百分比 ``(initial - final) / initial × 100``。"""
    if abs(initial) < 1e-12:
        return 0.0
    return float((initial - final) / initial * 100.0)


class Trainer:
    """量子电路优化策略的训练器。

    Parameters
    ----------
    config
        训练配置 :class:`TrainConfig`。

    Attributes
    ----------
    policy
        策略网络 :class:`qco.agent.CircuitPolicyNet`。
    ppo
        PPO 更新器 :class:`qco.ppo.PPO`。
    collector
        并行轨迹采样器 :class:`qco.rollout.RolloutCollector`。
    envs
        并行环境列表。
    train_circuits
        训练电路池。
    eval_circuits
        固定评估集。
    out_dir
        产物输出目录。
    logger
        本模块的 ``logging.Logger``（脚本负责配置 handler 与级别）。
    """

    POLICY_TRAIN_FILENAME = "train_log.csv"
    POLICY_EVAL_FILENAME = "eval_log.csv"
    SUMMARY_FILENAME = "summary.json"
    CONFIG_FILENAME = "config.json"
    BEST_FILENAME = "best.pt"
    LAST_FILENAME = "last.pt"

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.out_dir = ensure_dir(config.out_dir)

        self._seed = int(config.env.seed) if config.env.seed is not None else 0
        set_seed(self._seed)

        self.device = resolve_device(config.device)
        self.policy = CircuitPolicyNet(
            in_channels=N_CHANNELS,
            n_rules=len(RULES),
            n_qubits=config.env.n_qubits,
            n_moments=config.env.n_moments,
            hidden_channels=config.hidden_channels,
        ).to(self.device)
        self.ppo = PPO(self.policy, config.ppo, device=self.device)

        # 训练集：显式给定种子的随机电路池，保证实验可复现。
        self.train_circuits: list[Circuit] = random_circuits(
            config.train_circuits,
            config.env.n_qubits,
            config.env.initial_depth,
            seed=self._seed,
            p_two=config.env.p_two,
        )
        self.eval_circuits: list[Circuit] = self._load_eval_circuits()

        # 采样器使用的随机数流与策略训练的解耦，避免互相影响复现性。
        self._circuit_rng = np.random.default_rng(self._seed + 20240517)
        self._train_rng = np.random.default_rng(self._seed)

        self.envs: list[CircuitOptimizationEnv] = [
            CircuitOptimizationEnv(
                config.env,
                circuit=self.train_circuits[index % len(self.train_circuits)],
                rng=np.random.default_rng(self._seed + 1000 + index),
            )
            for index in range(config.num_envs)
        ]
        self.collector = RolloutCollector(self.policy, self.envs, device=self.device)

        self.best_iteration: int = 0
        self.best_eval: EvalResult | None = None
        self.final_eval: EvalResult | None = None

    # ------------------------------------------------------------------
    # 数据
    # ------------------------------------------------------------------
    def _resolve_eval_dataset(self) -> Path | None:
        """解析评估数据集路径；返回 ``None`` 表示没有可用数据集、需要现场生成。

        - 显式配置 ``eval_dataset``：相对路径按项目根目录解析，文件必须存在；
        - 未配置：使用项目根目录的 ``data/eval.json``，不存在则返回 ``None``。
        """
        raw = self.config.eval_dataset
        if raw is None or str(raw).strip() == "":
            return DEFAULT_EVAL_DATASET if DEFAULT_EVAL_DATASET.exists() else None
        path = Path(str(raw))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    def _load_eval_circuits(self) -> list[Circuit]:
        """加载评估集：优先显式给出的数据集，否则 ``data/eval.json``，都不存在时现场生成。"""
        config = self.config
        dataset_path = self._resolve_eval_dataset()
        if dataset_path is not None:
            if not dataset_path.exists():
                raise FileNotFoundError(
                    f"评估数据集不存在：{dataset_path}；"
                    "请先用 scripts/make_dataset.py 生成，或去掉 --dataset 让其自动生成"
                )
            circuits = load_circuits(dataset_path)
            for circuit in circuits:
                if circuit.n_qubits != config.env.n_qubits:
                    raise ValueError(
                        f"评估集电路比特数 {circuit.n_qubits} 与配置 {config.env.n_qubits} 不一致："
                        f"请检查 {dataset_path}（可用 scripts/make_dataset.py --n-qubits "
                        f"{config.env.n_qubits} 重新生成）"
                    )
                if circuit.depth > config.env.n_moments:
                    raise ValueError(
                        f"评估集电路深度 {circuit.depth} 超过 n_moments={config.env.n_moments}"
                    )
            self.logger.info("已从 %s 加载 %d 个评估电路", dataset_path, len(circuits))
            return circuits
        circuits = random_circuits(
            config.eval_circuits,
            config.env.n_qubits,
            config.env.initial_depth,
            seed=config.eval_seed,
            p_two=config.env.p_two,
        )
        self.logger.info(
            "未找到 %s，用种子 %d 现场生成 %d 个评估电路",
            DEFAULT_EVAL_DATASET,
            config.eval_seed,
            len(circuits),
        )
        return circuits

    def _sample_train_circuit(self) -> Circuit:
        """从训练电路池随机取一个电路（供回合结束后的 auto-reset 使用）。"""
        index = int(self._circuit_rng.integers(len(self.train_circuits)))
        return self.train_circuits[index]

    # ------------------------------------------------------------------
    # 训练主循环
    # ------------------------------------------------------------------
    def train(self) -> dict[str, Any]:
        """执行完整训练并写出全部产物，返回 JSON 可序列化的摘要字典。

        Returns
        -------
        dict
            摘要：训练耗时、最优迭代轮次与评估、``final_eval``（最后一轮指标）、
            四种方法的最终评估结果（其中 ``results["agent"]`` 取最优轮指标，
            与 ``best.pt`` 的权重一致）、Markdown 对比表等；
            与 ``summary.json`` 的内容一致。
        """
        config = self.config
        config_dict = config_to_dict(config)
        save_json(config_dict, self.out_dir / self.CONFIG_FILENAME)

        train_log_path = self.out_dir / self.POLICY_TRAIN_FILENAME
        eval_log_path = self.out_dir / self.POLICY_EVAL_FILENAME
        with train_log_path.open("w", encoding="utf-8", newline="") as train_handle, \
                eval_log_path.open("w", encoding="utf-8", newline="") as eval_handle:
            train_writer = csv.DictWriter(train_handle, fieldnames=TRAIN_LOG_COLUMNS)
            eval_writer = csv.DictWriter(eval_handle, fieldnames=EVAL_LOG_COLUMNS)
            train_writer.writeheader()
            eval_writer.writeheader()
            train_handle.flush()
            eval_handle.flush()

            self.logger.info("开始训练：设备=%s，迭代=%d，并行环境=%d", self.device, config.iterations, config.num_envs)
            start_time = time.perf_counter()
            final_results: dict[str, EvalResult] | None = None
            for iteration in range(1, config.iterations + 1):
                self.policy.train()
                batch, episode_stats = self.collector.collect(
                    horizon=config.steps_per_iter,
                    circuit_sampler=self._sample_train_circuit,
                )
                metrics = self.ppo.update(batch)
                row = self._train_row(iteration, batch, episode_stats, metrics)
                train_writer.writerow(row)
                train_handle.flush()
                self.logger.info(
                    "迭代 %d/%d | 回报=%.4f | 代价 %.2f→%.2f | 门压缩 %.2f%% | 深度压缩 %.2f%% | "
                    "策略损失=%.4f | 价值损失=%.4f | 熵=%.3f | KL=%.4f",
                    iteration,
                    config.iterations,
                    float(row["mean_return"]),
                    float(row["mean_initial_cost"]),
                    float(row["mean_final_cost"]),
                    float(row["gate_reduction_pct"]),
                    float(row["depth_reduction_pct"]),
                    float(row["policy_loss"]),
                    float(row["value_loss"]),
                    float(row["entropy"]),
                    float(row["approx_kl"]),
                )

                is_last = iteration == config.iterations
                if iteration % config.eval_every == 0 or is_last:
                    eval_results = self._periodic_evaluation(iteration)
                    for method, eval_result in eval_results.items():
                        # eval_log 记录“该轮实际评估”的数字（曲线分析用）；
                        # 最后一轮 summary 里的 agent 行取最优轮指标（见 _periodic_evaluation）。
                        row_result = eval_result
                        if is_last and method == "agent" and self.final_eval is not None:
                            row_result = self.final_eval
                        eval_writer.writerow(self._eval_row(iteration, row_result))
                    eval_handle.flush()
                    if is_last:
                        final_results = eval_results

                if config.save_every > 0 and iteration % config.save_every == 0:
                    self._save_checkpoint(
                        self.out_dir / f"checkpoint_{iteration:04d}.pt", iteration
                    )
                self._save_checkpoint(self.out_dir / self.LAST_FILENAME, iteration)

            train_time_s = time.perf_counter() - start_time

        if final_results is None:  # pragma: no cover - 循环内必然在最后一轮评估
            final_results = self._periodic_evaluation(config.iterations)
        summary: dict[str, Any] = {
            "config": config_dict,
            "iterations": config.iterations,
            "train_time_s": float(train_time_s),
            "best_iteration": int(self.best_iteration),
            "best_eval": to_jsonable(self.best_eval) if self.best_eval is not None else None,
            "final_eval": (
                to_jsonable(self.final_eval) if self.final_eval is not None else None
            ),
            # 主结果表与 best.pt 保持一致：agent 行取“最优轮”指标（见 _periodic_evaluation），
            # 最后一轮指标单独放在 final_eval，供训练曲线/过度训练分析使用。
            "results": {
                method: to_jsonable(result) for method, result in final_results.items()
            },
            "comparison_table": format_comparison(final_results),
            "out_dir": str(self.out_dir),
        }
        save_json(summary, self.out_dir / self.SUMMARY_FILENAME)
        self.logger.info("训练完成，用时 %.2f 秒，最优轮次=%d", train_time_s, self.best_iteration)
        if self.best_eval is not None and self.final_eval is not None:
            self.logger.info(
                "智能体代价压缩率：最后一轮 %.2f%%（第 %d 轮） vs 最优轮 %.2f%%（第 %d 轮）",
                self.final_eval.cost_reduction_pct,
                config.iterations,
                self.best_eval.cost_reduction_pct,
                self.best_iteration,
            )
        self.logger.info("最终评估（agent 行为最优轮指标）：\n%s", summary["comparison_table"])
        self.logger.info("产物已写入 %s", self.out_dir)
        return summary

    # ------------------------------------------------------------------
    # 日志行
    # ------------------------------------------------------------------
    def _train_row(
        self,
        iteration: int,
        batch: RolloutBatch,
        episode_stats: Sequence[EpisodeStats],
        metrics: dict[str, float],
    ) -> dict[str, Any]:
        """构造一行训练日志。

        若本次 rollout 期间没有任何回合结束（例如 horizon 小于单回合长度），
        则退化为“用当前环境状态 + 本批奖励”估计本迭代的指标，保证日志不为空。
        """
        horizon = int(batch.rewards.shape[1])
        if episode_stats:
            mean_initial_cost = _mean([item.initial_cost for item in episode_stats])
            mean_final_cost = _mean([item.final_cost for item in episode_stats])
            mean_initial_gate = _mean([float(item.initial_gate_count) for item in episode_stats])
            mean_final_gate = _mean([float(item.final_gate_count) for item in episode_stats])
            mean_initial_depth = _mean([float(item.initial_depth) for item in episode_stats])
            mean_final_depth = _mean([float(item.final_depth) for item in episode_stats])
            mean_return = _mean([item.return_ for item in episode_stats])
            mean_episode_len = _mean([float(item.steps) for item in episode_stats])
        else:
            mean_initial_cost = _mean([float(env.initial_cost) for env in self.envs])
            mean_final_cost = _mean([float(env.cost) for env in self.envs])
            mean_initial_gate = _mean([float(env.initial_circuit.num_ops) for env in self.envs])
            mean_final_gate = _mean([float(env.circuit.num_ops) for env in self.envs])
            mean_initial_depth = _mean([float(env.initial_circuit.depth) for env in self.envs])
            mean_final_depth = _mean([float(env.circuit.depth) for env in self.envs])
            mean_return = float(batch.rewards.sum(dim=1).mean().item())
            mean_episode_len = float(horizon)

        return {
            "iteration": iteration,
            "mean_return": f"{mean_return:.6f}",
            "mean_episode_len": f"{mean_episode_len:.6f}",
            "mean_initial_cost": f"{mean_initial_cost:.6f}",
            "mean_final_cost": f"{mean_final_cost:.6f}",
            "gate_reduction_pct": f"{_reduction_pct(mean_initial_gate, mean_final_gate):.6f}",
            "depth_reduction_pct": f"{_reduction_pct(mean_initial_depth, mean_final_depth):.6f}",
            "policy_loss": f"{float(metrics['policy_loss']):.6f}",
            "value_loss": f"{float(metrics['value_loss']):.6f}",
            "entropy": f"{float(metrics['entropy']):.6f}",
            "approx_kl": f"{float(metrics['approx_kl']):.6f}",
            "explained_variance": f"{float(metrics['explained_variance']):.6f}",
        }

    @staticmethod
    def _eval_row(iteration: int, result: EvalResult) -> dict[str, Any]:
        """构造一行评估日志。"""
        return {
            "iteration": iteration,
            "method": result.method,
            "cost_reduction_pct": f"{result.cost_reduction_pct:.6f}",
            "gate_reduction_pct": f"{result.gate_reduction_pct:.6f}",
            "depth_reduction_pct": f"{result.depth_reduction_pct:.6f}",
            "equivalence_rate": f"{result.equivalence_rate:.6f}",
            "mean_steps": f"{result.mean_steps:.6f}",
        }

    # ------------------------------------------------------------------
    # 评估与检查点
    # ------------------------------------------------------------------
    def _periodic_evaluation(self, iteration: int) -> dict[str, EvalResult]:
        """在评估集上评估智能体；最后一轮额外评估三种基线。

        返回值中 ``agent`` 行始终取“最优轮”指标（与 ``best.pt`` 的权重一致），
        最后一轮指标保存在 :attr:`final_eval` 中，供训练曲线与过度训练分析。
        """
        config = self.config
        agent_result = evaluate_agent(
            self.policy,
            self.eval_circuits,
            config.env,
            device=self.device,
            deterministic=True,
        )
        self.policy.train()
        self.logger.info(
            "评估（第 %d 轮）：智能体代价压缩 %.2f%%，门压缩 %.2f%%，深度压缩 %.2f%%，等价率 %.2f",
            iteration,
            agent_result.cost_reduction_pct,
            agent_result.gate_reduction_pct,
            agent_result.depth_reduction_pct,
            agent_result.equivalence_rate,
        )

        if agent_result.equivalence_rate < 1.0:  # pragma: no cover - 规则不变量保证不会发生
            self.logger.error("检测到语义等价的电路比例小于 1.0，训练结果不可信！")

        # best.pt 以评估集 mean_final_cost 最小为准。权重必须在“提升发生的当下”
        # 就落盘：最后一轮再保存会把最后一轮的权重错标成历史最优（best/last 相等）。
        if self.best_eval is None or agent_result.mean_final_cost < self.best_eval.mean_final_cost:
            self.best_eval = agent_result
            self.best_iteration = iteration
            self._save_checkpoint(
                self.out_dir / self.BEST_FILENAME, iteration, agent_result
            )
            self.logger.info(
                "新的最优模型：第 %d 轮评估集 mean_final_cost=%.4f，已保存到 %s",
                iteration,
                agent_result.mean_final_cost,
                self.out_dir / self.BEST_FILENAME,
            )

        if iteration != config.iterations:
            return {"agent": agent_result}

        # 最后一轮：记录本轮（最后一轮）指标供曲线分析，但主结果表中的 agent 行
        # 与 best.pt 对齐，取最优轮指标；否则“重载 best.pt 的评估数字”会和
        # summary / README 主结果表对不上。
        self.final_eval = agent_result
        results: dict[str, EvalResult] = self._evaluate_baselines()
        results["agent"] = self.best_eval if self.best_eval is not None else agent_result
        self.logger.info(
            "智能体代价压缩率：最后一轮 %.2f%% vs 最优轮 %.2f%%（第 %d 轮）；"
            "summary.json 的 agent 行取最优轮指标",
            agent_result.cost_reduction_pct,
            results["agent"].cost_reduction_pct,
            self.best_iteration,
        )
        return results

    def _evaluate_baselines(self) -> dict[str, EvalResult]:
        """在评估集上评估 noop / random / greedy 三种非学习基线。"""
        config = self.config
        self.logger.info("评估基线（noop / random / greedy）……")
        return {
            "noop": evaluate_noop(self.eval_circuits, config.env),
            "random": evaluate_random(self.eval_circuits, config.env, seed=config.eval_seed),
            "greedy": evaluate_greedy(self.eval_circuits, config.env),
        }

    def _save_checkpoint(
        self,
        path: str | Path,
        iteration: int,
        eval_result: EvalResult | None = None,
    ) -> None:
        """保存策略网络 state_dict 与完整配置（CPU 张量，便于跨设备加载）。"""
        import torch

        payload: dict[str, Any] = {
            "iteration": int(iteration),
            "config": config_to_dict(self.config),
            "state_dict": {
                key: value.detach().cpu() for key, value in self.policy.state_dict().items()
            },
        }
        if eval_result is not None:
            payload["eval"] = to_jsonable(eval_result)
        torch.save(payload, Path(path))

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return (
            f"Trainer(iterations={self.config.iterations}, num_envs={self.config.num_envs}, "
            f"n_qubits={self.config.env.n_qubits}, device={self.device})"
        )


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """加载训练器保存的检查点文件，返回包含 ``config`` 与 ``state_dict`` 的字典。

    Parameters
    ----------
    path
        ``best.pt`` / ``last.pt`` / ``checkpoint_XXXX.pt`` 路径。

    Returns
    -------
    dict
        检查点内容；``state_dict`` 为 CPU 上的参数字典。
    """
    import torch

    return torch.load(Path(path), map_location="cpu", weights_only=False)


def build_policy_from_checkpoint(
    checkpoint: dict[str, Any], device: str = "cpu"
) -> tuple[CircuitPolicyNet, EnvConfig, dict[str, Any]]:
    """根据检查点重建策略网络与环境配置。

    Parameters
    ----------
    checkpoint
        :func:`load_checkpoint` 的返回值。
    device
        网络放置的设备。

    Returns
    -------
    tuple[CircuitPolicyNet, EnvConfig, dict[str, Any]]
        ``(policy, env_config, raw_config)``；``policy`` 已加载参数并切换到 ``eval`` 模式。
        ``device`` 支持 ``"auto"``（有 CUDA 时用 GPU，否则 CPU）。
    """
    import torch

    raw_config = dict(checkpoint.get("config", {}))
    env_data = dict(raw_config.get("env", {}))
    allowed = set(EnvConfig.__dataclass_fields__)
    env_config = EnvConfig(**{key: value for key, value in env_data.items() if key in allowed})
    policy = CircuitPolicyNet(
        in_channels=N_CHANNELS,
        n_rules=len(RULES),
        n_qubits=env_config.n_qubits,
        n_moments=env_config.n_moments,
        hidden_channels=int(raw_config.get("hidden_channels", 64)),
    )
    state_dict = checkpoint["state_dict"]
    policy.load_state_dict(state_dict)
    torch_device = device if isinstance(device, torch.device) else resolve_device(str(device))
    policy = policy.to(torch_device)
    policy.eval()
    return policy, env_config, raw_config
