"""``qco.ppo`` 的单元测试。

重点验证契约第 12 节的硬性正确性要求：

1. GAE 与**逐项手算**的参考值一致，且包含最后一步的 bootstrap 价值；
2. 初始（参数未更新时）策略 ratio 严格为 1；
3. 一次更新后参数确实发生变化；
4. ``update`` 返回的指标键齐全、数值有限；
5. 近似 KL 早停能提前终止更新而不报错。
"""

from __future__ import annotations

import math

import torch

from qco.agent import CircuitPolicyNet
from qco.gates import N_CHANNELS
from qco.ppo import PPO, PPOConfig
from qco.rollout import RolloutBatch
from qco.rules import RULES

N_QUBITS = 3
N_MOMENTS = 5
N_RULES = len(RULES)
ACTION_SIZE = N_RULES * N_QUBITS * N_MOMENTS


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def make_policy(hidden_channels: int = 8) -> CircuitPolicyNet:
    """构造一个小型策略网络。"""
    torch.manual_seed(0)
    return CircuitPolicyNet(
        in_channels=N_CHANNELS,
        n_rules=N_RULES,
        n_qubits=N_QUBITS,
        n_moments=N_MOMENTS,
        hidden_channels=hidden_channels,
    )


def make_masks(num_envs: int, horizon: int, seed: int = 3) -> torch.Tensor:
    """构造 ``[N, H, R, Q, M]`` 掩码，每个位置至少一个合法动作。"""
    generator = torch.Generator().manual_seed(seed)
    flat = torch.zeros(num_envs * horizon, ACTION_SIZE, dtype=torch.bool)
    for row in range(num_envs * horizon):
        flat[row, torch.randint(0, ACTION_SIZE, (4,), generator=generator)] = True
    return flat.reshape(num_envs, horizon, N_RULES, N_QUBITS, N_MOMENTS)


def make_batch(
    policy: CircuitPolicyNet,
    num_envs: int = 2,
    horizon: int = 6,
    seed: int = 7,
) -> RolloutBatch:
    """用给定策略构造一份形状合法的 rollout 批次（观测为随机张量）。"""
    generator = torch.Generator().manual_seed(seed)
    observations = torch.rand(
        num_envs, horizon, N_CHANNELS, N_QUBITS, N_MOMENTS, generator=generator
    )
    masks = make_masks(num_envs, horizon)
    with torch.no_grad():
        actions, log_probs, values, _entropies = policy.act(
            observations.reshape(num_envs * horizon, N_CHANNELS, N_QUBITS, N_MOMENTS),
            masks.reshape(num_envs * horizon, N_RULES, N_QUBITS, N_MOMENTS),
        )
    rewards = torch.randn(num_envs, horizon, generator=generator) * 0.1
    dones = torch.zeros(num_envs, horizon)
    next_values = torch.randn(num_envs, generator=generator) * 0.1
    return RolloutBatch(
        observations=observations,
        actions=actions.reshape(num_envs, horizon),
        log_probs=log_probs.reshape(num_envs, horizon),
        values=values.reshape(num_envs, horizon),
        rewards=rewards,
        dones=dones,
        masks=masks,
        next_values=next_values,
    )


# ---------------------------------------------------------------------------
# GAE：手算参考值
# ---------------------------------------------------------------------------
def test_gae_matches_hand_computed_values() -> None:
    """单环境 3 步的 GAE 与逐项手算结果一致。

    参数：``γ=0.9, λ=0.8``，``rewards=[1, 2, 3]``，``values=[0.5, 1.0, 1.5]``，
    ``dones=[0, 0, 1]``，``next_values=[0]``。手算过程：

    - t=2：not_done=0 ⇒ δ₂ = 3 + 0.9·0·0 − 1.5 = 1.5，A₂ = 1.5；
    - t=1：δ₁ = 2 + 0.9·1.5 − 1.0 = 2.35，A₁ = 2.35 + 0.9·0.8·1.5 = 3.43；
    - t=0：δ₀ = 1 + 0.9·1.0 − 0.5 = 1.4，A₀ = 1.4 + 0.72·3.43 = 3.8696；
    - returns = advantages + values = [4.3696, 4.43, 3.0]。
    """
    rewards = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    values = torch.tensor([[0.5, 1.0, 1.5]], dtype=torch.float64)
    dones = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    next_values = torch.tensor([0.0], dtype=torch.float64)

    advantages, returns = PPO.compute_gae(rewards, values, dones, next_values, 0.9, 0.8)

    expected_advantages = torch.tensor([[3.8696, 3.43, 1.5]], dtype=torch.float64)
    expected_returns = torch.tensor([[4.3696, 4.43, 3.0]], dtype=torch.float64)
    assert torch.allclose(advantages, expected_advantages, atol=1e-9, rtol=0.0)
    assert torch.allclose(returns, expected_returns, atol=1e-9, rtol=0.0)
    assert torch.allclose(returns, advantages + values, atol=1e-9, rtol=0.0)


def test_gae_matches_hand_computed_multi_env_with_dones() -> None:
    """两个环境（含 done 截断）的 GAE 与手算参考值一致，并与参考循环逐项相等。

    参数：``γ=0.5, λ=0.5``；
    - env0：rewards=[1,1,1]，values=[0,0,0]，dones=[0,0,0]，next=2
      ⇒ A=[1.375, 1.5, 2]，returns=[1.375, 1.5, 2]；
    - env1：rewards=[0,2,0]，values=[1,1,1]，dones=[0,1,0]，next=3
      ⇒ A=[-0.25, 1.0, 0.5]，returns=[0.75, 2.0, 1.5]。
    """
    rewards = torch.tensor([[1.0, 1.0, 1.0], [0.0, 2.0, 0.0]], dtype=torch.float64)
    values = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
    dones = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    next_values = torch.tensor([2.0, 3.0], dtype=torch.float64)
    gamma, gae_lambda = 0.5, 0.5

    advantages, returns = PPO.compute_gae(rewards, values, dones, next_values, gamma, gae_lambda)

    expected_advantages = torch.tensor(
        [[1.375, 1.5, 2.0], [-0.25, 1.0, 0.5]], dtype=torch.float64
    )
    expected_returns = torch.tensor(
        [[1.375, 1.5, 2.0], [0.75, 2.0, 1.5]], dtype=torch.float64
    )
    assert torch.allclose(advantages, expected_advantages, atol=1e-9, rtol=0.0)
    assert torch.allclose(returns, expected_returns, atol=1e-9, rtol=0.0)


def test_gae_matches_reference_loop() -> None:
    """随机数据上 GAE 与“显式逐环境递推”的参考实现逐项一致。"""
    torch.manual_seed(11)
    num_envs, horizon = 3, 5
    rewards = torch.randn(num_envs, horizon, dtype=torch.float64)
    values = torch.randn(num_envs, horizon, dtype=torch.float64)
    dones = (torch.rand(num_envs, horizon, dtype=torch.float64) < 0.3).to(torch.float64)
    dones[:, -1] = 0.0  # 最后一步不设 done，专门验证使用 next_values 的情况
    next_values = torch.randn(num_envs, dtype=torch.float64)
    gamma, gae_lambda = 0.97, 0.92

    advantages, returns = PPO.compute_gae(rewards, values, dones, next_values, gamma, gae_lambda)

    # 参考实现：显式逐环境、从最后一步开始反向循环。
    expected_advantages = torch.zeros_like(rewards)
    for env in range(num_envs):
        running = 0.0
        for step in reversed(range(horizon)):
            next_value = next_values[env] if step == horizon - 1 else values[env, step + 1]
            not_done = 1.0 - dones[env, step]
            delta = rewards[env, step] + gamma * next_value * not_done - values[env, step]
            running = delta + gamma * gae_lambda * not_done * running
            expected_advantages[env, step] = running

    assert torch.allclose(advantages, expected_advantages, atol=1e-9, rtol=0.0)
    assert torch.allclose(returns, expected_advantages + values, atol=1e-9, rtol=0.0)


def test_gae_last_step_uses_next_values() -> None:
    """最后一步的 bootstrap 价值必须进入 delta（漏掉最后一步会得到 0）。"""
    rewards = torch.zeros(1, 2, dtype=torch.float64)
    values = torch.zeros(1, 2, dtype=torch.float64)
    dones = torch.zeros(1, 2, dtype=torch.float64)
    next_values = torch.tensor([10.0], dtype=torch.float64)

    advantages, _returns = PPO.compute_gae(rewards, values, dones, next_values, 0.9, 0.95)
    expected_last = 0.9 * 10.0
    expected_first = 0.9 * 0.95 * expected_last
    assert math.isclose(float(advantages[0, 1]), expected_last, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(advantages[0, 0]), expected_first, rel_tol=0.0, abs_tol=1e-9)


def test_gae_accepts_one_dimensional_input() -> None:
    """一维输入（单环境）与二维输入 ``[1, H]`` 的结果一致。"""
    rewards_2d = torch.tensor([[0.5, -0.2, 1.0]], dtype=torch.float64)
    values_2d = torch.tensor([[0.1, 0.3, -0.4]], dtype=torch.float64)
    dones_2d = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    next_values = torch.tensor([0.7], dtype=torch.float64)

    adv_2d, ret_2d = PPO.compute_gae(rewards_2d, values_2d, dones_2d, next_values, 0.99, 0.95)
    adv_1d, ret_1d = PPO.compute_gae(
        rewards_2d[0], values_2d[0], dones_2d[0], next_values[0], 0.99, 0.95
    )
    assert adv_1d.shape == (3,) and ret_1d.shape == (3,)
    assert torch.allclose(adv_1d, adv_2d[0], atol=1e-12, rtol=0.0)
    assert torch.allclose(ret_1d, ret_2d[0], atol=1e-12, rtol=0.0)


def test_gae_rejects_inconsistent_shapes() -> None:
    """形状不一致时显式报错。"""
    rewards = torch.zeros(2, 3)
    values = torch.zeros(2, 3)
    dones = torch.zeros(2, 3)
    try:
        PPO.compute_gae(rewards, values, dones, torch.zeros(1), 0.99, 0.95)
    except ValueError:
        pass
    else:  # pragma: no cover - 失败路径
        raise AssertionError("next_values 数量不一致应抛 ValueError")
    try:
        PPO.compute_gae(rewards, values[:, :2], dones, torch.zeros(2), 0.99, 0.95)
    except ValueError:
        pass
    else:  # pragma: no cover - 失败路径
        raise AssertionError("values 形状不一致应抛 ValueError")


# ---------------------------------------------------------------------------
# ratio / 更新 / 指标
# ---------------------------------------------------------------------------
def test_initial_ratio_is_one() -> None:
    """参数未更新时，重新前向得到的 log_prob 与采样时完全相同 ⇒ ratio 恒为 1。"""
    policy = make_policy()
    ppo = PPO(policy, PPOConfig())
    batch = make_batch(policy)
    flat = batch.flatten()

    with torch.no_grad():
        log_probs, _values = ppo._log_probs_and_values(flat.observations, flat.masks)

    new_selected = log_probs.gather(1, flat.actions.unsqueeze(1)).squeeze(1)
    ratio = torch.exp(new_selected - flat.log_probs)
    assert torch.allclose(new_selected, flat.log_probs, atol=1e-6)
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-6)


def test_update_changes_parameters_and_returns_all_metrics() -> None:
    """一次更新后参数发生变化，指标键齐全且数值有限。"""
    policy = make_policy()
    config = PPOConfig(
        lr=1e-2,
        update_epochs=2,
        minibatch_size=8,
        target_kl=10.0,  # 不触发早停
        normalize_advantage=True,
    )
    ppo = PPO(policy, config)
    batch = make_batch(policy, num_envs=2, horizon=6)

    before = [parameter.detach().clone() for parameter in policy.parameters()]
    metrics = ppo.update(batch)
    after = [parameter.detach().clone() for parameter in policy.parameters()]

    assert set(metrics) == set(PPO.METRIC_KEYS)
    assert all(isinstance(value, float) for value in metrics.values())
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["learning_rate"] == config.lr
    assert metrics["clip_fraction"] >= 0.0 and metrics["clip_fraction"] <= 1.0
    assert metrics["entropy"] >= 0.0
    assert metrics["grad_norm"] > 0.0
    assert "explained_variance" in metrics

    changed = any(
        not torch.equal(old, new) for old, new in zip(before, after)
    )
    assert changed, "PPO.update 之后参数应发生变化"


def test_update_without_advantage_normalization() -> None:
    """关闭优势标准化时同样可以正常更新。"""
    policy = make_policy()
    ppo = PPO(
        policy,
        PPOConfig(lr=1e-2, update_epochs=1, minibatch_size=8, normalize_advantage=False),
    )
    batch = make_batch(policy)
    metrics = ppo.update(batch)
    assert set(metrics) == set(PPO.METRIC_KEYS)
    assert all(math.isfinite(value) for value in metrics.values())


def test_kl_early_stop_terminates_updates() -> None:
    """近似 KL 超过 target_kl 时提前终止剩余小批量更新，且不报错。"""
    policy = make_policy()
    config = PPOConfig(
        lr=0.5,
        update_epochs=20,
        minibatch_size=4,
        target_kl=1e-9,
    )
    ppo = PPO(policy, config)
    batch = make_batch(policy, num_envs=2, horizon=6)

    step_calls = {"count": 0}
    original_step = ppo.optimizer.step

    def counted_step(*args, **kwargs):
        step_calls["count"] += 1
        return original_step(*args, **kwargs)

    ppo.optimizer.step = counted_step  # type: ignore[method-assign]
    metrics = ppo.update(batch)
    ppo.optimizer.step = original_step  # type: ignore[method-assign]

    total_possible = config.update_epochs * math.ceil(
        batch.num_envs * batch.horizon / config.minibatch_size
    )
    # 第一个小批量 ratio=1（KL=0）不会停；更新一次之后 KL 立刻超过 1e-9。
    assert 1 <= step_calls["count"] <= 3
    assert step_calls["count"] < total_possible
    assert set(metrics) == set(PPO.METRIC_KEYS)
    assert all(math.isfinite(value) for value in metrics.values())


def test_ppo_config_validation() -> None:
    """非法超参数在构造时显式报错。"""
    for kwargs in (
        {"lr": 0.0},
        {"gamma": 1.5},
        {"gae_lambda": -0.1},
        {"update_epochs": 0},
        {"minibatch_size": 0},
        {"target_kl": 0.0},
        {"max_grad_norm": 0.0},
    ):
        try:
            PPOConfig(**kwargs)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError(f"非法配置 {kwargs} 应抛 ValueError")
