"""非学习基线：不做优化（noop）、随机策略（random）与贪心策略（greedy）。

基线的价值在于给出**可对照的下界与上界**：

- ``noop``：完全不动作，代表“未优化”的原始电路；
- ``random``：每一步在所有合法动作中均匀随机采样，代表“不加学习的随机搜索”；
- ``greedy``：每一步在所有合法动作中挑选“一步之后代价最小”的动作，
  代表“有目标但短视的启发式搜索”。

三者与 PPO 智能体使用同一个环境、同一套重写规则与同一个代价函数，
因此评估结果可以直接比较。``optimize`` 是所有基线与智能体评估共用的通用优化循环。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from qco.actions import decode_action
from qco.circuit import Circuit
from qco.environment import CircuitOptimizationEnv, EnvConfig
from qco.rules import RULES
from qco.simulator import circuits_equivalent

#: 选择器：输入环境，返回一个**合法**动作的一维下标。
Chooser = Callable[[CircuitOptimizationEnv], int]


@dataclass
class OptimizationResult:
    """单个电路、单种方法的完整优化记录。

    Attributes
    ----------
    method
        方法名（``"noop"`` / ``"random"`` / ``"greedy"`` / ``"agent"``）。
    initial_circuit
        优化前的电路。
    final_circuit
        优化后的电路。
    initial_cost
        初始代价 ``C = 门数 + lambda_depth × 深度``。
    final_cost
        最终代价。
    initial_gate_count / final_gate_count
        优化前后的门总数。
    initial_depth / final_depth
        优化前后的电路深度。
    steps
        实际执行的改写步数。
    truncated
        是否因达到步数上限而提前结束（``False`` 表示因“没有可用动作”自然终止）。
    equivalent
        初始电路与最终电路是否**模全局相位语义等价**（由模拟器验证）。
    rule_counts
        各规则的使用次数，键为规则名。
    elapsed_ms
        本次优化耗时（毫秒）。
    """

    method: str
    initial_circuit: Circuit
    final_circuit: Circuit
    initial_cost: float
    final_cost: float
    initial_gate_count: int
    final_gate_count: int
    initial_depth: int
    final_depth: int
    steps: int
    truncated: bool
    equivalent: bool
    rule_counts: dict[str, int]
    elapsed_ms: float


def optimize(
    env: CircuitOptimizationEnv,
    chooser: Chooser,
    max_steps: int | None = None,
) -> OptimizationResult:
    """在给定环境上按 ``chooser`` 反复选择动作，直到自然终止或达到步数上限。

    Parameters
    ----------
    env
        已经 ``reset`` 到待优化电路的环境（即 ``env.circuit`` 为优化起点）。
    chooser
        动作选择函数，返回 ``env.valid_actions`` 中的一个一维下标。
    max_steps
        额外的步数上限；``None`` 表示使用 ``env.config.max_steps``。
        与配置冲突时取两者较小值，保证不会超过环境自身的截断上限。

    Returns
    -------
    OptimizationResult
        记录初始 / 最终电路、代价、规则使用次数与语义等价性。

    Notes
    -----
    每执行一步都用 ``qco.simulator.circuits_equivalent`` 复核初始与最终电路的
    语义等价性；重写规则的理论不变量保证结果恒为 ``True``，这里的验证用于
    在工程上兜底，避免评估报告出现“优化后语义已改变”的错误结论。
    """
    start = time.perf_counter()
    initial_circuit = env.circuit
    initial_cost = env.cost
    config = env.config
    limit = config.max_steps if max_steps is None else min(int(max_steps), config.max_steps)

    steps = 0
    truncated = False
    while steps < limit:
        if not bool(env.action_mask.any()):
            break
        action = int(chooser(env))
        _, _, done, info = env.step(action)
        steps += 1
        if done:
            truncated = bool(info["truncated"])
            break
    else:
        # 循环因外部步数上限退出：若仍存在可行动作则说明是截断。
        truncated = bool(env.action_mask.any())

    final_circuit = env.circuit
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return OptimizationResult(
        method="",
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


def choose_random(env: CircuitOptimizationEnv, rng: np.random.Generator) -> int:
    """在所有合法动作中均匀随机选择一个，返回一维动作下标。"""
    valid = env.valid_actions
    if valid.size == 0:  # pragma: no cover - 调用方保证有可行动作
        raise ValueError("当前电路没有可用的重写动作")
    return int(valid[int(rng.integers(valid.size))])


def choose_greedy(env: CircuitOptimizationEnv) -> int:
    """选择“执行一步后代价最小”的合法动作；并列时取动作下标最小者。

    实现上直接对每个合法动作调用对应重写规则并计算新代价，不修改环境状态；
    选定后再由调用方通过 ``env.step`` 真正执行。合法动作数量为
    ``规则数 × 比特数 × 时隙数`` 中的可用子集，单步扫描开销可接受。
    """
    valid = env.valid_actions
    if valid.size == 0:  # pragma: no cover - 调用方保证有可行动作
        raise ValueError("当前电路没有可用的重写动作")
    circuit = env.circuit
    lambda_depth = env.config.lambda_depth
    n_qubits = env.config.n_qubits
    n_moments = env.config.n_moments

    best_action = int(valid[0])
    best_cost = float("inf")
    for raw_action in valid:
        action = int(raw_action)
        rule_index, qubit, moment = decode_action(action, n_qubits, n_moments)
        candidate = RULES[rule_index].apply(circuit, qubit, moment)
        cost = candidate.cost(lambda_depth)
        # valid_actions 为升序，严格小于保证并列时保留最小下标。
        if cost < best_cost - 1e-12:
            best_cost = cost
            best_action = action
    return best_action


def optimize_noop(circuit: Circuit, config: EnvConfig) -> OptimizationResult:
    """不执行任何动作的基线：最终电路与初始电路完全相同。"""
    start = time.perf_counter()
    cost = float(circuit.cost(config.lambda_depth))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return OptimizationResult(
        method="noop",
        initial_circuit=circuit,
        final_circuit=circuit,
        initial_cost=cost,
        final_cost=cost,
        initial_gate_count=circuit.num_ops,
        final_gate_count=circuit.num_ops,
        initial_depth=circuit.depth,
        final_depth=circuit.depth,
        steps=0,
        truncated=False,
        equivalent=circuits_equivalent(circuit, circuit),
        rule_counts={rule.name: 0 for rule in RULES},
        elapsed_ms=elapsed_ms,
    )


def optimize_random(circuit: Circuit, config: EnvConfig, seed: int = 0) -> OptimizationResult:
    """随机基线：每次在合法动作中均匀随机采样，直到终止或截断。"""
    rng = np.random.default_rng(int(seed))
    env = CircuitOptimizationEnv(config, circuit=circuit, rng=np.random.default_rng(int(seed)))
    result = optimize(env, chooser=lambda current_env: choose_random(current_env, rng))
    result.method = "random"
    return result


def optimize_greedy(circuit: Circuit, config: EnvConfig) -> OptimizationResult:
    """贪心基线：每次选择一步后代价最小的合法动作。"""
    env = CircuitOptimizationEnv(config, circuit=circuit)
    result = optimize(env, chooser=choose_greedy)
    result.method = "greedy"
    return result
