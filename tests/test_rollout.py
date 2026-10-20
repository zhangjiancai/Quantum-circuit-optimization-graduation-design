"""``qco.rollout`` 的单元测试。

覆盖：
1. 采样张量的形状 / 类型 / 设备；
2. 记录的 ``log_probs`` 与掩码、动作一致（旧策略的对数概率）；
3. 回合结束（done）时的 auto-reset：新回合从采样器给出的电路开始，
   ``EpisodeStats`` 统计量正确；
4. ``next_values`` 取 horizon 步之后观测的 bootstrap 价值；
5. ``RolloutBatch.flatten`` 的形状与展平顺序。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qco.agent import CircuitPolicyNet
from qco.circuit import Circuit, Op
from qco.environment import CircuitOptimizationEnv, EnvConfig
from qco.gates import N_CHANNELS
from qco.generator import random_circuit
from qco.masking import masked_log_softmax
from qco.rollout import EpisodeStats, RolloutBatch, RolloutCollector
from qco.rules import RULES

N_QUBITS = 3
N_MOMENTS = 6
N_RULES = len(RULES)
ACTION_SIZE = N_RULES * N_QUBITS * N_MOMENTS


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def make_policy() -> CircuitPolicyNet:
    """构造一个小型策略网络。"""
    torch.manual_seed(0)
    return CircuitPolicyNet(
        in_channels=N_CHANNELS,
        n_rules=N_RULES,
        n_qubits=N_QUBITS,
        n_moments=N_MOMENTS,
        hidden_channels=8,
    )


def make_envs(
    num_envs: int = 2,
    max_steps: int = 6,
    initial_depth: int = 4,
    seed: int = 0,
    circuits: list[Circuit] | None = None,
) -> list[CircuitOptimizationEnv]:
    """构造若干共享同一配置的并行环境（各自使用独立 RNG）。

    ``circuits`` 为空时每个环境在构造时随机生成初始电路；给定电路时按顺序使用，
    便于构造“可行动作足够多、horizon 内不会终止”的确定性测试场景。
    """
    config = EnvConfig(
        n_qubits=N_QUBITS,
        n_moments=N_MOMENTS,
        max_steps=max_steps,
        initial_depth=initial_depth,
        seed=seed,
    )
    return [
        CircuitOptimizationEnv(
            config=config,
            circuit=None if circuits is None else circuits[index],
            rng=np.random.default_rng(seed + index),
        )
        for index in range(num_envs)
    ]


def make_mergeable_circuit(depth: int = N_MOMENTS) -> Circuit:
    """构造一个“每层每个比特都放 H”的电路：始终有可合并的相邻 H·H，动作充足。"""
    layers = [[Op("H", (qubit,)) for qubit in range(N_QUBITS)] for _ in range(depth)]
    return Circuit(N_QUBITS, layers)


def make_mergeable_sampler(depth: int = N_MOMENTS):
    """auto-reset 用的确定性采样器：总是返回 :func:`make_mergeable_circuit`。"""

    def sampler() -> Circuit:
        return make_mergeable_circuit(depth)

    return sampler


def make_sampler(depth: int = 4):
    """构造 auto-reset 用的电路采样器。"""
    rng = np.random.default_rng(12345)

    def sampler():
        return random_circuit(N_QUBITS, depth, rng)

    return sampler


def recompute_log_probs(
    policy: CircuitPolicyNet, batch: RolloutBatch
) -> torch.Tensor:
    """用当前策略重新前向，返回每一步所选动作的 log 概率（展平）。"""
    flat = batch.flatten()
    with torch.no_grad():
        logits, _values = policy(flat.observations)
        log_probs = masked_log_softmax(logits.reshape(logits.shape[0], -1), flat.masks.reshape(flat.num_samples, -1))
        selected = log_probs.gather(1, flat.actions.unsqueeze(1)).squeeze(1)
    return selected


# ---------------------------------------------------------------------------
# 形状与一致性
# ---------------------------------------------------------------------------
def test_collect_shapes_dtypes_and_consistency() -> None:
    """采样批次的形状、类型正确；动作落在掩码内；log_probs 与策略一致。"""
    policy = make_policy()
    envs = make_envs(num_envs=2)
    collector = RolloutCollector(policy, envs, device="cpu")
    batch, stats = collector.collect(horizon=5, circuit_sampler=make_sampler())

    assert isinstance(batch, RolloutBatch)
    assert batch.observations.shape == (2, 5, N_CHANNELS, N_QUBITS, N_MOMENTS)
    assert batch.actions.shape == (2, 5) and batch.actions.dtype == torch.int64
    assert batch.log_probs.shape == (2, 5) and batch.log_probs.dtype == torch.float32
    assert batch.values.shape == (2, 5) and batch.values.dtype == torch.float32
    assert batch.rewards.shape == (2, 5) and batch.rewards.dtype == torch.float32
    assert batch.dones.shape == (2, 5) and batch.dones.dtype == torch.float32
    assert batch.masks.shape == (2, 5, N_RULES, N_QUBITS, N_MOMENTS)
    assert batch.masks.dtype == torch.bool
    assert batch.next_values.shape == (2,) and batch.next_values.dtype == torch.float32
    assert batch.observations.device.type == "cpu"
    assert isinstance(stats, list) and all(isinstance(item, EpisodeStats) for item in stats)

    # 每个样本至少有一个合法动作，采样动作必须合法。
    flat_mask = batch.masks.reshape(2 * 5, ACTION_SIZE)
    assert torch.all(flat_mask.any(dim=-1))
    flat_actions = batch.actions.reshape(-1)
    rows = torch.arange(2 * 5)
    assert torch.all(flat_mask[rows, flat_actions])

    # 采样时的 log_probs 与用同一策略重新前向的结果一致。
    recomputed = recompute_log_probs(policy, batch)
    assert torch.allclose(recomputed, batch.log_probs.reshape(-1), atol=1e-6)


def test_collect_without_episode_end_has_empty_stats() -> None:
    """horizon 较短、回合未结束时不会产生 EpisodeStats。

    使用每层都放 H 的确定性电路（始终存在可合并的相邻 H·H），保证 horizon
    内不会因“没有可用动作”而提前终止。
    """
    policy = make_policy()
    circuits = [make_mergeable_circuit() for _ in range(2)]
    envs = make_envs(num_envs=2, max_steps=100, circuits=circuits)
    collector = RolloutCollector(policy, envs, device="cpu")
    batch, stats = collector.collect(horizon=3, circuit_sampler=make_mergeable_sampler())

    assert stats == []
    assert torch.all(batch.dones == 0.0)
    # 未发生 auto-reset：环境的步数等于 horizon，电路确实被优化过。
    assert all(env.steps == 3 for env in envs)
    assert all(env.cost < env.initial_cost for env in envs)


# ---------------------------------------------------------------------------
# auto-reset 与 bootstrap 价值
# ---------------------------------------------------------------------------
def test_auto_reset_every_step_with_truncation() -> None:
    """``max_steps=1`` 时每一步都截断：dones 全 1，每步结束一个回合并 auto-reset。"""
    policy = make_policy()
    envs = make_envs(num_envs=2, max_steps=1)
    collector = RolloutCollector(policy, envs, device="cpu")
    horizon = 3
    batch, stats = collector.collect(horizon=horizon, circuit_sampler=make_sampler())

    assert torch.all(batch.dones == 1.0)
    assert len(stats) == 2 * horizon
    assert all(item.steps == 1 for item in stats)
    assert all(item.truncated for item in stats)
    # 每步结束即 reset，因此结束时所有环境都在新回合起点。
    assert sum(item.steps for item in stats) == 2 * horizon
    assert all(env.steps == 0 for env in envs)
    # 每个回合的回报 = 该步奖励 = 势函数塑形的代价压缩比例。
    for item in stats:
        expected_return = (item.initial_cost - item.final_cost) / item.initial_cost
        assert abs(item.return_ - expected_return) < 1e-6
        assert item.final_cost < item.initial_cost
        assert item.final_gate_count <= item.initial_gate_count
        assert item.final_depth <= item.initial_depth
        assert item.initial_gate_count > 0 and item.initial_depth == 4
        assert set(item.rule_counts) == {rule.name for rule in RULES}


def test_next_values_bootstrap_from_final_observation() -> None:
    """``next_values`` 等于 horizon 步之后观测的状态价值（含 auto-reset 后的新回合）。"""
    policy = make_policy()
    envs = make_envs(num_envs=2, max_steps=1)
    collector = RolloutCollector(policy, envs, device="cpu")
    batch, _stats = collector.collect(horizon=3, circuit_sampler=make_sampler())

    final_observations = torch.as_tensor(np.stack([env.observation for env in envs]))
    with torch.no_grad():
        _logits, expected_values = policy(final_observations)
    assert torch.allclose(batch.next_values, expected_values, atol=1e-6)

    # 非 reset 情形的 bootstrap：取当前未结束回合的状态价值。
    # 用每层都放 H 的确定性电路，确保 horizon 内不终止、不发生 auto-reset。
    envs2 = make_envs(
        num_envs=2,
        max_steps=100,
        seed=5,
        circuits=[make_mergeable_circuit() for _ in range(2)],
    )
    collector2 = RolloutCollector(make_policy(), envs2, device="cpu")
    batch2, stats2 = collector2.collect(
        horizon=4, circuit_sampler=make_mergeable_sampler()
    )
    assert stats2 == []
    assert all(env.steps == 4 for env in envs2)
    final_observations2 = torch.as_tensor(np.stack([env.observation for env in envs2]))
    with torch.no_grad():
        _logits2, expected_values2 = collector2.policy(final_observations2)
    assert torch.allclose(batch2.next_values, expected_values2, atol=1e-6)


def test_policy_mode_restored_after_collect() -> None:
    """采样时切到 eval，结束后恢复调用前的模式。"""
    policy = make_policy()
    envs = make_envs(num_envs=1)
    collector = RolloutCollector(policy, envs, device="cpu")
    policy.train()
    collector.collect(horizon=2, circuit_sampler=make_sampler())
    assert policy.training is True


def test_collector_resets_terminal_env_before_sampling() -> None:
    """传入已经终止（没有可用动作）的环境时，collect 先 auto-reset 再采样。"""
    policy = make_policy()
    config = EnvConfig(
        n_qubits=N_QUBITS,
        n_moments=N_MOMENTS,
        max_steps=100,
        initial_depth=N_MOMENTS,
    )
    env = CircuitOptimizationEnv(
        config=config, circuit=make_mergeable_circuit(), rng=np.random.default_rng(0)
    )
    # 一直执行动作，直到环境进入“无可用动作”的终态。
    while env.has_valid_actions:
        env.step(int(env.valid_actions[0]))
    assert not env.has_valid_actions

    collector = RolloutCollector(policy, [env], device="cpu")
    batch, stats = collector.collect(horizon=2, circuit_sampler=make_mergeable_sampler())

    # 终态环境已被重置并从新回合起点走完 2 步，采样全程无需空掩码。
    assert env.steps == 2
    assert stats == []
    assert batch.masks.reshape(2, -1).any(dim=-1).all()


# ---------------------------------------------------------------------------
# flatten
# ---------------------------------------------------------------------------
def test_flatten_shapes_and_order() -> None:
    """展平视图形状为 ``N*H``，且与 reshape 的“先环境后时间”顺序一致。"""
    policy = make_policy()
    envs = make_envs(num_envs=2)
    collector = RolloutCollector(policy, envs, device="cpu")
    batch, _stats = collector.collect(horizon=3, circuit_sampler=make_sampler())

    flat = batch.flatten()
    num_samples = 2 * 3
    assert flat.num_samples == num_samples
    assert flat.observations.shape == (num_samples, N_CHANNELS, N_QUBITS, N_MOMENTS)
    assert flat.actions.shape == (num_samples,)
    assert flat.log_probs.shape == (num_samples,)
    assert flat.values.shape == (num_samples,)
    assert flat.rewards.shape == (num_samples,)
    assert flat.dones.shape == (num_samples,)
    assert flat.masks.shape == (num_samples, N_RULES, N_QUBITS, N_MOMENTS)
    # 展平顺序：第 i 个环境、第 t 个时隙对应下标 i*horizon+t。
    assert torch.equal(flat.rewards, batch.rewards.reshape(-1))
    assert torch.equal(flat.observations[1], batch.observations[0, 1])


# ---------------------------------------------------------------------------
# 参数校验
# ---------------------------------------------------------------------------
def test_collector_argument_validation() -> None:
    """空环境、非法 horizon、非法采样器等输入显式报错。"""
    policy = make_policy()
    with pytest.raises(ValueError):
        RolloutCollector(policy, [])
    collector = RolloutCollector(policy, make_envs(num_envs=1), device="cpu")
    with pytest.raises(ValueError):
        collector.collect(horizon=0, circuit_sampler=make_sampler())
    with pytest.raises(TypeError):
        collector.collect(horizon=2, circuit_sampler=None)  # type: ignore[arg-type]
