"""统一评估：把 noop / random / greedy / agent 四种方法放在同一评估集上对比。

评估口径
--------
- 每个电路独立优化，统计初始 / 最终的门数、深度与代价；
- 用 ``qco.simulator.circuits_equivalent`` 验证优化前后语义等价（模全局相位），
  等价率必须恒为 ``1.0``——这是“优化不能改变电路功能”的硬性验收指标；
- 压缩率定义为 ``(初始均值 - 最终均值) / 初始均值 × 100``；
- 输出既包含可直接阅读的 Markdown 对比表，也包含可机读的 JSON 结果。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from qco.baselines import (
    OptimizationResult,
    optimize_greedy,
    optimize_noop,
    optimize_random,
)
from qco.circuit import Circuit
from qco.environment import CircuitOptimizationEnv, EnvConfig
from qco.rules import RULES
from qco.simulator import circuits_equivalent
from qco.utils import save_json, to_jsonable


@dataclass
class EvalResult:
    """一种方法在整个评估集上的汇总指标。

    Attributes
    ----------
    method
        方法名（``"noop"`` / ``"random"`` / ``"greedy"`` / ``"agent"``）。
    n_circuits
        参与评估的电路数量。
    mean_initial_gate_count / mean_final_gate_count
        优化前 / 后的平均门数。
    mean_initial_depth / mean_final_depth
        优化前 / 后的平均电路深度。
    mean_initial_cost / mean_final_cost
        优化前 / 后的平均代价。
    gate_reduction_pct / depth_reduction_pct / cost_reduction_pct
        门数 / 深度 / 代价的平均压缩百分比。
    equivalence_rate
        优化前后语义等价的比例，正常应严格等于 ``1.0``。
    mean_steps
        平均执行的改写步数。
    truncation_rate
        因达到步数上限而截断的比例。
    mean_rule_counts
        各规则的平均使用次数，键为规则名。
    mean_elapsed_ms
        每个电路的平均优化耗时（毫秒）。
    """

    method: str
    n_circuits: int
    mean_initial_gate_count: float
    mean_final_gate_count: float
    mean_initial_depth: float
    mean_final_depth: float
    mean_initial_cost: float
    mean_final_cost: float
    gate_reduction_pct: float
    depth_reduction_pct: float
    cost_reduction_pct: float
    equivalence_rate: float
    mean_steps: float
    truncation_rate: float
    mean_rule_counts: dict[str, float]
    mean_elapsed_ms: float


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


def _aggregate(method: str, results: Sequence[OptimizationResult]) -> EvalResult:
    """把逐个电路的结果汇总为 :class:`EvalResult`。"""
    results = list(results)
    if not results:
        raise ValueError("评估结果为空：至少需要一个电路")

    # 独立性复核：任何一步重写都不允许改变电路语义。
    for item in results:
        if not item.equivalent:
            raise AssertionError(
                f"方法 {method} 在电路上破坏了语义等价性："
                f"{item.initial_circuit!r} -> {item.final_circuit!r}"
            )

    initial_gates = [float(item.initial_gate_count) for item in results]
    final_gates = [float(item.final_gate_count) for item in results]
    initial_depths = [float(item.initial_depth) for item in results]
    final_depths = [float(item.final_depth) for item in results]
    initial_costs = [float(item.initial_cost) for item in results]
    final_costs = [float(item.final_cost) for item in results]

    mean_initial_gate_count = _mean(initial_gates)
    mean_final_gate_count = _mean(final_gates)
    mean_initial_depth = _mean(initial_depths)
    mean_final_depth = _mean(final_depths)
    mean_initial_cost = _mean(initial_costs)
    mean_final_cost = _mean(final_costs)

    rule_counts: dict[str, float] = {}
    for rule in RULES:
        rule_counts[rule.name] = _mean(
            [float(item.rule_counts.get(rule.name, 0)) for item in results]
        )

    return EvalResult(
        method=method,
        n_circuits=len(results),
        mean_initial_gate_count=mean_initial_gate_count,
        mean_final_gate_count=mean_final_gate_count,
        mean_initial_depth=mean_initial_depth,
        mean_final_depth=mean_final_depth,
        mean_initial_cost=mean_initial_cost,
        mean_final_cost=mean_final_cost,
        gate_reduction_pct=_reduction_pct(mean_initial_gate_count, mean_final_gate_count),
        depth_reduction_pct=_reduction_pct(mean_initial_depth, mean_final_depth),
        cost_reduction_pct=_reduction_pct(mean_initial_cost, mean_final_cost),
        equivalence_rate=_mean([1.0 if item.equivalent else 0.0 for item in results]),
        mean_steps=_mean([float(item.steps) for item in results]),
        truncation_rate=_mean([1.0 if item.truncated else 0.0 for item in results]),
        mean_rule_counts=rule_counts,
        mean_elapsed_ms=_mean([float(item.elapsed_ms) for item in results]),
    )


def _evaluate_agent_circuit(
    policy: Any,
    circuit: Circuit,
    config: EnvConfig,
    device: str,
    deterministic: bool,
) -> OptimizationResult:
    """用策略网络在单个电路上走完整回合，返回与基线同构的结果对象。"""
    import torch

    torch_device = device if isinstance(device, torch.device) else torch.device(device)
    env = CircuitOptimizationEnv(config, circuit=circuit)
    initial_circuit = env.circuit
    initial_cost = env.cost

    steps = 0
    truncated = False
    start = time.perf_counter()
    with torch.no_grad():
        while steps < config.max_steps:
            if not bool(env.action_mask.any()):
                break
            observation = torch.from_numpy(env.observation).float().unsqueeze(0).to(torch_device)
            mask = torch.from_numpy(env.action_mask).bool().unsqueeze(0).to(torch_device)
            action, _, _, _ = policy.act(observation, mask, deterministic=deterministic)
            action_value = int(action.reshape(-1)[0].item())
            _, _, done, info = env.step(action_value)
            steps += 1
            if done:
                truncated = bool(info["truncated"])
                break
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    final_circuit = env.circuit
    return OptimizationResult(
        method="agent",
        initial_circuit=initial_circuit,
        final_circuit=final_circuit,
        initial_cost=float(initial_cost),
        final_cost=float(env.cost),
        initial_gate_count=initial_circuit.num_ops,
        final_gate_count=final_circuit.num_ops,
        initial_depth=initial_circuit.depth,
        final_depth=final_circuit.depth,
        steps=steps,
        truncated=truncated,
        equivalent=circuits_equivalent(initial_circuit, final_circuit),
        rule_counts=env.rule_counts,
        elapsed_ms=elapsed_ms,
    )


def evaluate_agent(
    policy: Any,
    circuits: Sequence[Circuit],
    config: EnvConfig,
    device: str = "cpu",
    deterministic: bool = True,
) -> EvalResult:
    """在评估集上评估 PPO 策略网络（默认取确定性动作）。

    Parameters
    ----------
    policy
        :class:`qco.agent.CircuitPolicyNet` 实例。
    circuits
        评估电路列表。
    config
        环境配置；``n_qubits`` / ``n_moments`` / ``max_steps`` 必须与训练时一致。
    device
        ``"cpu"`` 或 ``"cuda"`` 等 ``torch.device`` 可接受的字符串。
    deterministic
        为 ``True`` 时取 ``argmax`` 动作，为 ``False`` 时按策略分布采样。

    Returns
    -------
    EvalResult
        汇总指标；``equivalence_rate`` 应为 ``1.0``。
    """
    circuits = list(circuits)
    if not circuits:
        raise ValueError("评估集不能为空")
    if not hasattr(policy, "act"):
        raise TypeError("policy 必须提供 act(observations, masks, deterministic) 方法")
    was_training = bool(getattr(policy, "training", False))
    if hasattr(policy, "eval"):
        policy.eval()
    try:
        results = [
            _evaluate_agent_circuit(policy, circuit, config, device, deterministic)
            for circuit in circuits
        ]
    finally:
        if was_training and hasattr(policy, "train"):
            policy.train()
    return _aggregate("agent", results)


def evaluate_noop(circuits: Sequence[Circuit], config: EnvConfig) -> EvalResult:
    """评估 noop 基线（不优化）。"""
    circuits = list(circuits)
    if not circuits:
        raise ValueError("评估集不能为空")
    return _aggregate("noop", [optimize_noop(circuit, config) for circuit in circuits])


def evaluate_random(circuits: Sequence[Circuit], config: EnvConfig, seed: int = 0) -> EvalResult:
    """评估随机基线；第 ``i`` 个电路使用 ``seed + i`` 作随机种子以保证可复现。"""
    circuits = list(circuits)
    if not circuits:
        raise ValueError("评估集不能为空")
    results = [
        optimize_random(circuit, config, seed=int(seed) + index)
        for index, circuit in enumerate(circuits)
    ]
    return _aggregate("random", results)


def evaluate_greedy(circuits: Sequence[Circuit], config: EnvConfig) -> EvalResult:
    """评估贪心基线（每步选择一步后代价最小的动作）。"""
    circuits = list(circuits)
    if not circuits:
        raise ValueError("评估集不能为空")
    return _aggregate("greedy", [optimize_greedy(circuit, config) for circuit in circuits])


def format_comparison(results: Mapping[str, EvalResult]) -> str:
    """把多种方法的评估结果格式化为 Markdown 对比表。

    Parameters
    ----------
    results
        方法名 -> :class:`EvalResult` 的映射；表格行顺序与字典插入顺序一致。

    Returns
    -------
    str
        Markdown 表格文本（含表头与对齐行）。
    """
    headers = (
        "方法",
        "门数(初始→最终)",
        "深度(初始→最终)",
        "门压缩率(%)",
        "深度压缩率(%)",
        "代价压缩率(%)",
        "等价率",
        "平均步数",
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for method, result in results.items():
        lines.append(
            "| "
            + " | ".join(
                (
                    str(method),
                    f"{result.mean_initial_gate_count:.2f} → {result.mean_final_gate_count:.2f}",
                    f"{result.mean_initial_depth:.2f} → {result.mean_final_depth:.2f}",
                    f"{result.gate_reduction_pct:.2f}",
                    f"{result.depth_reduction_pct:.2f}",
                    f"{result.cost_reduction_pct:.2f}",
                    f"{result.equivalence_rate:.2f}",
                    f"{result.mean_steps:.2f}",
                )
            )
            + " |"
        )
    return "\n".join(lines)


def save_eval_results(results: Mapping[str, EvalResult], path: str | Path) -> None:
    """把评估结果保存为 JSON 文件（自动创建父目录）。

    Parameters
    ----------
    results
        方法名 -> :class:`EvalResult` 的映射。
    path
        目标 JSON 路径，例如 ``results/default/eval/evaluation.json``。
    """
    payload = {str(method): to_jsonable(result) for method, result in results.items()}
    save_json(payload, path)
