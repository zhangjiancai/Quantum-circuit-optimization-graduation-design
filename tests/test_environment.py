"""强化学习环境的单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from qco.actions import encode_action
from qco.circuit import Circuit, Op
from qco.environment import CircuitOptimizationEnv, EnvConfig
from qco.masking import flatten_action_mask
from qco.observation import N_CHANNELS, observation_shape


def make(n_qubits: int, layers: list[list[tuple[str, tuple[int, ...]]]]) -> Circuit:
    return Circuit(n_qubits, [[Op(gate, qs) for gate, qs in layer] for layer in layers])


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        EnvConfig(n_qubits=1)
    with pytest.raises(ValueError):
        EnvConfig(n_moments=0)
    with pytest.raises(ValueError):
        EnvConfig(initial_depth=20, n_moments=15)
    with pytest.raises(ValueError):
        EnvConfig(p_two=2.0)
    with pytest.raises(ValueError):
        EnvConfig(lambda_depth=-1.0)
    # λ=0 会让 move_* 规则不再降低代价，破坏“每个合法动作严格进步”的不变量
    with pytest.raises(ValueError):
        EnvConfig(lambda_depth=0.0)


def test_reset_returns_observation() -> None:
    config = EnvConfig(n_qubits=2, n_moments=6, initial_depth=3, seed=0)
    env = CircuitOptimizationEnv(config)
    observation = env.reset()
    assert observation.shape == observation_shape(2, 6)
    assert observation.shape[0] == N_CHANNELS
    assert observation.dtype == np.float32
    assert env.cost == env.initial_cost
    assert env.steps == 0
    assert env.n_actions == 4 * 2 * 6
    assert env.has_valid_actions


def test_step_merge_reward_and_termination() -> None:
    """T·T -> S：门数 2->1、深度 2->1，代价 3->1.5，奖励 0.5。"""
    config = EnvConfig(n_qubits=2, n_moments=6, initial_depth=3, lambda_depth=0.5, seed=0)
    env = CircuitOptimizationEnv(config)
    circuit = make(2, [[("T", (0,))], [("T", (0,))]])
    env.reset(circuit)
    assert env.initial_cost == pytest.approx(3.0)

    action = encode_action(0, 0, 0, n_qubits=2, n_moments=6)
    observation, reward, done, info = env.step(action)

    assert observation.shape == observation_shape(2, 6)
    assert reward == pytest.approx(0.5)
    assert done is True
    assert info["rule_name"] == "merge_single_qubit"
    assert info["qubit"] == 0 and info["moment"] == 0
    assert info["gate_count"] == 1
    assert info["depth"] == 1
    assert info["cost"] == pytest.approx(1.5)
    assert info["terminated"] is True
    assert info["truncated"] is False
    assert info["valid_actions"] == 0
    assert info["rule_counts"]["merge_single_qubit"] == 1
    assert info["episode_return"] == pytest.approx(0.5)
    assert env.circuit.num_ops == 1
    assert env.circuit.layer(0)[0].gate == "S"


def test_step_rejects_masked_action() -> None:
    config = EnvConfig(n_qubits=2, n_moments=6, initial_depth=3, seed=0)
    env = CircuitOptimizationEnv(config)
    env.reset(make(2, [[("T", (0,))], [("T", (0,))]]))
    # 该电路上没有二比特门，cancel_two_qubit 必然被掩码
    action = encode_action(1, 0, 0, n_qubits=2, n_moments=6)
    assert not env.action_mask[1, 0, 0]
    with pytest.raises(ValueError):
        env.step(action)


def test_step_rejects_out_of_range_action() -> None:
    config = EnvConfig(n_qubits=2, n_moments=4, initial_depth=2, seed=0)
    env = CircuitOptimizationEnv(config)
    with pytest.raises(ValueError):
        env.step(env.n_actions)
    with pytest.raises(ValueError):
        env.step(-1)


def test_reward_equals_cost_reduction_ratio() -> None:
    config = EnvConfig(n_qubits=3, n_moments=8, initial_depth=4, lambda_depth=0.5, seed=0)
    env = CircuitOptimizationEnv(config)
    env.reset(make(3, [[("H", (0,))], [("H", (0,))], [("CNOT", (1, 2))]]))
    initial_cost = env.initial_cost
    while env.has_valid_actions:
        cost_before = env.cost
        action = int(env.valid_actions[0])
        _, reward, done, info = env.step(action)
        expected = (cost_before - env.cost) / initial_cost
        assert reward == pytest.approx(expected)
        assert info["cost"] == pytest.approx(env.cost)
        if done:
            break
    assert env.cost <= initial_cost


def test_truncation_at_max_steps() -> None:
    config = EnvConfig(
        n_qubits=2, n_moments=8, initial_depth=6, max_steps=1, seed=0
    )
    env = CircuitOptimizationEnv(config)
    env.reset(make(2, [[("T", (0,))], [("T", (0,))], [("T", (0,))], [("T", (0,))]]))
    action = int(env.valid_actions[0])
    _, _, done, info = env.step(action)
    assert done is True
    assert info["truncated"] is True
    assert env.steps == 1


def test_valid_actions_match_mask() -> None:
    config = EnvConfig(n_qubits=3, n_moments=6, initial_depth=4, seed=1)
    env = CircuitOptimizationEnv(config)
    env.reset()
    expected = np.flatnonzero(flatten_action_mask(env.action_mask))
    assert np.array_equal(env.valid_actions, expected)
    # 每个有效动作都能在独立环境里执行成功
    for action in env.valid_actions[:5]:
        probe = CircuitOptimizationEnv(config)
        probe.reset(env.initial_circuit)
        probe.step(int(action))


def test_reset_validation() -> None:
    config = EnvConfig(n_qubits=2, n_moments=4, initial_depth=2, seed=0)
    env = CircuitOptimizationEnv(config)
    with pytest.raises(ValueError):
        env.reset(make(3, [[("X", (0,))]]))
    with pytest.raises(ValueError):
        env.reset(make(2, [[("X", (0,))]] * 5))


def test_random_initial_circuit_is_reproducible() -> None:
    config = EnvConfig(n_qubits=3, n_moments=8, initial_depth=5, seed=123)
    first = CircuitOptimizationEnv(config)
    second = CircuitOptimizationEnv(config)
    assert first.initial_circuit == second.initial_circuit
    assert first.initial_circuit.depth == 5


def test_sample_circuit() -> None:
    config = EnvConfig(n_qubits=3, n_moments=8, initial_depth=5, seed=0)
    env = CircuitOptimizationEnv(config)
    circuit = env.sample_circuit()
    assert circuit.n_qubits == 3
    assert circuit.depth == 5


def test_render_and_repr() -> None:
    config = EnvConfig(n_qubits=2, n_moments=4, initial_depth=2, seed=0)
    env = CircuitOptimizationEnv(config)
    assert isinstance(env.render(), str) and env.render()
    assert "q0" in env.circuit.to_ascii()
    assert "CircuitOptimizationEnv" in repr(env)


def test_rule_counts_accumulate() -> None:
    config = EnvConfig(n_qubits=2, n_moments=8, initial_depth=4, seed=0)
    env = CircuitOptimizationEnv(config)
    env.reset(make(2, [[("T", (0,))], [("T", (0,))], [("H", (1,))], [("H", (1,))]]))
    while env.has_valid_actions:
        _, _, done, info = env.step(int(env.valid_actions[0]))
        if done:
            break
    counts = env.rule_counts
    assert counts["merge_single_qubit"] >= 1
    assert sum(counts.values()) == env.steps
