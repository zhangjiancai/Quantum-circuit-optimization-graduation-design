"""强化学习环境：把“量子电路重写优化”建模为马尔可夫决策过程。

MDP 定义
--------
- 状态 ``s_t``：电路的门-比特-时隙张量 ``[N_CHANNELS, n_qubits, n_moments]``；
- 动作 ``a_t``：``(规则 r, 量子比特 q, 时隙 m)``，展平为 ``r*Q*M + q*M + m``；
- 奖励 ``r_t = (C(s_t) - C(s_{t+1})) / C(s_0)``，其中
  ``C = 门数量 + lambda_depth × 深度``。这是**势函数塑形**（potential-based shaping），
  回合总回报恰为 ``(C(s_0) - C(s_T)) / C(s_0)``，即“相对初始代价的压缩比例”；
- 终止：当前电路上已经没有任何可用的重写规则，或步数达到 ``max_steps``（截断）。

与原项目的区别：原环境既没有真正的“优化规则”（规则是在往电路上**加门**），
也没有奖励信号（奖励恒为负的门数/深度，且状态与电路不一致）。这里每一步都
严格降低代价，且语义正确性由重写规则的不变量保证。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qco.actions import action_space_size, decode_action
from qco.circuit import Circuit
from qco.generator import random_circuit
from qco.masking import build_action_mask, flatten_action_mask
from qco.observation import encode_circuit
from qco.rules import RULES


@dataclass(frozen=True)
class EnvConfig:
    """环境配置。

    Attributes
    ----------
    n_qubits
        量子比特数。
    n_moments
        观测张量的时隙窗口宽度；必须不小于初始电路深度。
    max_steps
        单个回合的最大步数（截断上限）。
    lambda_depth
        代价函数中深度的权重。
    initial_depth
        ``reset()`` 未指定电路时，随机生成的初始电路深度。
    p_two
        随机电路生成时放置二比特门的概率。
    seed
        随机数种子；``None`` 表示不固定。
    """

    n_qubits: int = 5
    n_moments: int = 15
    max_steps: int = 48
    lambda_depth: float = 0.5
    initial_depth: int = 12
    p_two: float = 0.15
    seed: int | None = 0

    def __post_init__(self) -> None:
        if self.n_qubits < 2:
            raise ValueError("n_qubits 至少为 2")
        if self.n_moments <= 0 or self.max_steps <= 0:
            raise ValueError("n_moments / max_steps 必须为正整数")
        if self.initial_depth <= 0:
            raise ValueError("initial_depth 必须为正整数")
        if self.initial_depth > self.n_moments:
            raise ValueError("initial_depth 不能大于 n_moments（观测窗口不足）")
        if not (0.0 <= self.p_two <= 1.0):
            raise ValueError("p_two 必须在 [0, 1] 内")
        if self.lambda_depth <= 0.0:
            # 深度权重必须严格为正：move_earlier / move_later 只改变深度、不改变门数，
            # λ=0 时它们不降低代价，会破坏“每个合法动作都严格进步”的不变量。
            raise ValueError("lambda_depth 必须为正数（move_* 规则依赖深度项提供奖励）")


class CircuitOptimizationEnv:
    """量子电路重写优化环境（Gym 风格接口）。

    Parameters
    ----------
    config
        环境配置。
    circuit
        初始电路；为空时在 :meth:`reset` 中随机生成。
    rng
        随机数生成器（用于随机初始电路与 ``sample_circuit``）；为空时按 ``config.seed`` 创建。
    """

    def __init__(
        self,
        config: EnvConfig | None = None,
        circuit: Circuit | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.config = config or EnvConfig()
        self._rng = rng if rng is not None else np.random.default_rng(self.config.seed)
        self._n_actions = action_space_size(
            len(RULES), self.config.n_qubits, self.config.n_moments
        )
        self._circuit: Circuit | None = None
        self._initial_circuit: Circuit | None = None
        self._initial_cost: float = 0.0
        self._cost: float = 0.0
        self._steps: int = 0
        self._return: float = 0.0
        self._rule_counts: dict[str, int] = {}
        self._mask: np.ndarray | None = None
        self.reset(circuit)

    # ------------------------------------------------------------------
    # 基本属性
    # ------------------------------------------------------------------
    @property
    def n_actions(self) -> int:
        """动作空间大小 ``len(RULES) * n_qubits * n_moments``。"""
        return self._n_actions

    @property
    def circuit(self) -> Circuit:
        """当前电路（不可变对象，重写后指向新对象）。"""
        assert self._circuit is not None
        return self._circuit

    @property
    def initial_circuit(self) -> Circuit:
        """本回合初始电路。"""
        assert self._initial_circuit is not None
        return self._initial_circuit

    @property
    def observation(self) -> np.ndarray:
        """当前观测张量 ``[N_CHANNELS, n_qubits, n_moments]``，float32。"""
        return encode_circuit(self.circuit, self.config.n_moments)

    @property
    def action_mask(self) -> np.ndarray:
        """当前动作掩码 ``[len(RULES), n_qubits, n_moments]``，bool。"""
        if self._mask is None:
            self._mask = build_action_mask(self.circuit, self.config.n_moments)
        return self._mask

    @property
    def valid_actions(self) -> np.ndarray:
        """当前可用动作的一维下标数组（int64）。"""
        return np.flatnonzero(flatten_action_mask(self.action_mask)).astype(np.int64)

    @property
    def has_valid_actions(self) -> bool:
        """当前是否还有可用的重写动作。"""
        return bool(self.action_mask.any())

    @property
    def cost(self) -> float:
        """当前电路代价 ``门数量 + lambda_depth × 深度``。"""
        return self._cost

    @property
    def initial_cost(self) -> float:
        """回合初始电路代价。"""
        return self._initial_cost

    @property
    def steps(self) -> int:
        """本回合已经执行的步数。"""
        return self._steps

    @property
    def rule_counts(self) -> dict[str, int]:
        """本回合各规则的使用次数（副本）。"""
        return dict(self._rule_counts)

    # ------------------------------------------------------------------
    # 回合接口
    # ------------------------------------------------------------------
    def reset(self, circuit: Circuit | None = None) -> np.ndarray:
        """重置环境并返回初始观测。"""
        if circuit is None:
            assert self.config is not None
            circuit = random_circuit(
                self.config.n_qubits,
                self.config.initial_depth,
                self._rng,
                p_two=self.config.p_two,
            )
        if circuit.n_qubits != self.config.n_qubits:
            raise ValueError(
                f"电路比特数 {circuit.n_qubits} 与环境配置 {self.config.n_qubits} 不一致"
            )
        if circuit.depth > self.config.n_moments:
            raise ValueError(
                f"电路深度 {circuit.depth} 超过观测窗口 n_moments={self.config.n_moments}"
            )
        self._circuit = circuit
        self._initial_circuit = circuit
        self._initial_cost = circuit.cost(self.config.lambda_depth)
        self._cost = self._initial_cost
        self._steps = 0
        self._return = 0.0
        self._rule_counts = {rule.name: 0 for rule in RULES}
        self._mask = None
        return self.observation

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """执行一个动作，返回 ``(obs, reward, done, info)``。

        Raises
        ------
        ValueError
            动作下标越界，或动作被掩码屏蔽（调用方必须先按掩码采样）。
        """
        action = int(action)
        if not (0 <= action < self._n_actions):
            raise ValueError(f"动作下标越界：{action}（动作空间大小 {self._n_actions}）")
        rule_index, qubit, moment = decode_action(
            action, self.config.n_qubits, self.config.n_moments
        )
        if not bool(self.action_mask[rule_index, qubit, moment]):
            raise ValueError(
                f"动作 {action} = (rule={rule_index}, qubit={qubit}, moment={moment}) 被掩码屏蔽，不能执行"
            )

        rule = RULES[rule_index]
        cost_before = self._cost
        new_circuit = rule.apply(self.circuit, qubit, moment)
        cost_after = new_circuit.cost(self.config.lambda_depth)
        if not cost_after < cost_before - 1e-12:  # 规则不变量，出现即为实现 bug
            raise AssertionError(
                f"规则 {rule.name} 未降低代价：{cost_before:.6f} -> {cost_after:.6f}"
            )

        self._circuit = new_circuit
        self._cost = cost_after
        self._steps += 1
        self._rule_counts[rule.name] += 1
        self._mask = None

        reward = (cost_before - cost_after) / self._initial_cost
        self._return += reward
        terminated = not self.has_valid_actions
        truncated = self._steps >= self.config.max_steps
        done = terminated or truncated
        info: dict[str, Any] = {
            "rule": rule_index,
            "rule_name": rule.name,
            "qubit": qubit,
            "moment": moment,
            "cost": cost_after,
            "initial_cost": self._initial_cost,
            "gate_count": new_circuit.num_ops,
            "num_two_qubit_ops": new_circuit.num_two_qubit_ops,
            "depth": new_circuit.depth,
            "valid_actions": int(self.valid_actions.size),
            "steps": self._steps,
            "episode_return": self._return,
            "return": self._return,
            "rule_counts": dict(self._rule_counts),
            "terminated": terminated,
            "truncated": truncated,
        }
        return self.observation, float(reward), bool(done), info

    def render(self) -> str:
        """返回当前电路的文本表示。"""
        return self.circuit.to_text()

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    def sample_circuit(self) -> Circuit:
        """按环境配置随机采样一个新电路（供训练时 auto-reset 使用）。"""
        return random_circuit(
            self.config.n_qubits,
            self.config.initial_depth,
            self._rng,
            p_two=self.config.p_two,
        )

    def __repr__(self) -> str:
        return (
            f"CircuitOptimizationEnv(n_qubits={self.config.n_qubits}, "
            f"n_moments={self.config.n_moments}, depth={self.circuit.depth}, "
            f"cost={self._cost:.3f}, steps={self._steps})"
        )
