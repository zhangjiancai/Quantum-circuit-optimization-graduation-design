"""``qco.agent.CircuitPolicyNet`` 的单元测试。

覆盖契约要求的四类行为：
1. 前向输出形状 ``logits [B, R, Q, M]`` 与 ``value [B]``；
2. 掩码后非法动作的概率严格为 0，且概率在合法动作上归一化；
3. ``deterministic=True`` 时选择掩码后 logits 的 argmax（动作编码与 ``qco.actions`` 一致）；
4. ``act()`` 返回的对数概率/价值/熵可以反传梯度。
"""

from __future__ import annotations

import torch

from qco.actions import encode_action
from qco.agent import CircuitPolicyNet
from qco.gates import N_CHANNELS
from qco.masking import masked_log_softmax
from qco.rules import RULES

N_QUBITS = 3
N_MOMENTS = 5
N_RULES = len(RULES)
ACTION_SIZE = N_RULES * N_QUBITS * N_MOMENTS


def make_policy(hidden_channels: int = 8) -> CircuitPolicyNet:
    """构造一个小型策略网络（隐藏通道少，CPU 上足够快）。"""
    torch.manual_seed(0)
    return CircuitPolicyNet(
        in_channels=N_CHANNELS,
        n_rules=N_RULES,
        n_qubits=N_QUBITS,
        n_moments=N_MOMENTS,
        hidden_channels=hidden_channels,
    )


def make_observations(batch_size: int = 2) -> torch.Tensor:
    """随机观测张量 ``[B, C, Q, M]``（数值范围与真实观测相近）。"""
    generator = torch.Generator().manual_seed(1)
    return torch.rand(batch_size, N_CHANNELS, N_QUBITS, N_MOMENTS, generator=generator)


def make_mask(batch_size: int, valid_actions: tuple[int, ...] | None = None) -> torch.Tensor:
    """构造 ``[B, R, Q, M]`` 掩码；默认每行随机 3 个合法动作。"""
    mask = torch.zeros(batch_size, ACTION_SIZE, dtype=torch.bool)
    if valid_actions is None:
        generator = torch.Generator().manual_seed(2)
        for row in range(batch_size):
            indices = torch.randint(0, ACTION_SIZE, (3,), generator=generator)
            mask[row, indices] = True
    else:
        for row in range(batch_size):
            for action in valid_actions:
                mask[row, action] = True
    return mask.reshape(batch_size, N_RULES, N_QUBITS, N_MOMENTS)


def test_forward_shapes_and_no_batchnorm() -> None:
    """前向输出形状正确，网络结构里没有 BatchNorm。"""
    policy = make_policy()
    observations = make_observations(batch_size=4)
    logits, value = policy(observations)

    assert logits.shape == (4, N_RULES, N_QUBITS, N_MOMENTS)
    assert value.shape == (4,)
    assert logits.dtype == torch.float32
    assert policy.policy_logits(observations).shape == logits.shape

    batchnorm_types = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
    assert not any(isinstance(module, batchnorm_types) for module in policy.modules())
    assert any(isinstance(module, torch.nn.GroupNorm) for module in policy.modules())


def test_forward_rejects_wrong_shape() -> None:
    """观测形状不符合契约时应显式报错，而不是静默广播。"""
    policy = make_policy()
    try:
        policy.forward(torch.zeros(2, N_CHANNELS, N_QUBITS, N_MOMENTS + 1))
    except ValueError:
        pass
    else:  # pragma: no cover - 失败路径
        raise AssertionError("错误形状的观测应抛 ValueError")
    try:
        policy.forward(torch.zeros(N_CHANNELS, N_QUBITS, N_MOMENTS))
    except ValueError:
        pass
    else:  # pragma: no cover - 失败路径
        raise AssertionError("缺少批次维的观测应抛 ValueError")


def test_masked_illegal_actions_have_zero_probability() -> None:
    """掩码后非法动作概率严格为 0，合法动作概率之和为 1。"""
    policy = make_policy()
    observations = make_observations(batch_size=3)
    mask = make_mask(batch_size=3)
    flat_mask = mask.reshape(3, ACTION_SIZE)

    logits = policy.policy_logits(observations)
    log_probs = masked_log_softmax(logits.reshape(3, -1), flat_mask)
    probs = log_probs.exp()

    assert torch.all(probs[~flat_mask] == 0.0)
    assert torch.all(probs[flat_mask] > 0.0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-6)


def test_act_returns_valid_actions_and_shapes() -> None:
    """``act`` 输出形状/类型正确，且采样动作一定落在掩码内。"""
    policy = make_policy()
    observations = make_observations(batch_size=3)
    mask = make_mask(batch_size=3)

    actions, log_probs, values, entropies = policy.act(observations, mask)
    assert actions.shape == (3,) and actions.dtype == torch.int64
    assert log_probs.shape == (3,) and values.shape == (3,) and entropies.shape == (3,)
    assert torch.all(log_probs <= 1e-6)  # 对数概率非正
    assert torch.all(entropies >= 0.0)

    flat_mask = mask.reshape(3, ACTION_SIZE)
    for row in range(3):
        action = int(actions[row])
        # 采样动作必须合法
        assert bool(flat_mask[row, action])
        # int64 动作可以还原为 (rule, qubit, moment) 并重新编码
        rule_index, remainder = divmod(action, N_QUBITS * N_MOMENTS)
        qubit_index, moment_index = divmod(remainder, N_MOMENTS)
        assert encode_action(rule_index, qubit_index, moment_index, N_QUBITS, N_MOMENTS) == action

    # value 与熵只由观测决定，与采样随机性无关
    _actions, _log_probs, values_retaken, entropies_retaken = policy.act(
        observations, mask.reshape(3, N_RULES, N_QUBITS, N_MOMENTS)
    )
    assert torch.allclose(values, values_retaken, atol=1e-7)
    assert torch.allclose(entropies, entropies_retaken, atol=1e-7)


def test_deterministic_selection_is_argmax_of_masked_logits() -> None:
    """``deterministic=True`` 选择掩码后 logits 的最大值位置。"""
    policy = make_policy()
    observations = make_observations(batch_size=2)
    valid = (5, 17, 40)
    mask = make_mask(batch_size=2, valid_actions=valid)

    actions, log_probs, _values, _entropies = policy.act(
        observations, mask, deterministic=True
    )
    flat_logits = policy.policy_logits(observations).reshape(2, -1)
    expected = flat_logits.masked_fill(~mask.reshape(2, -1), -1e9).argmax(dim=-1)
    assert torch.equal(actions, expected)
    # 只保留一个合法动作时，其对数概率为 0、熵为 0
    single_mask = torch.zeros(2, ACTION_SIZE, dtype=torch.bool)
    single_mask[:, 9] = True
    actions_single, log_probs_single, _values_single, entropies_single = policy.act(
        observations, single_mask.reshape(2, N_RULES, N_QUBITS, N_MOMENTS), deterministic=True
    )
    assert torch.equal(actions_single, torch.full((2,), 9, dtype=torch.int64))
    assert torch.allclose(log_probs_single, torch.zeros(2), atol=1e-6)
    assert torch.allclose(entropies_single, torch.zeros(2), atol=1e-6)
    del log_probs


def test_act_log_probs_match_masked_log_softmax() -> None:
    """``act`` 返回的 log_prob 与对同一 logits 做一次 masked_log_softmax 的结果一致。"""
    policy = make_policy()
    observations = make_observations(batch_size=2)
    mask = make_mask(batch_size=2)

    actions, log_probs, _values, _entropies = policy.act(observations, mask)
    flat_logits = policy.policy_logits(observations).reshape(2, -1)
    expected = masked_log_softmax(flat_logits, mask.reshape(2, -1))
    expected_selected = expected.gather(1, actions.unsqueeze(1)).squeeze(1)
    assert torch.allclose(log_probs, expected_selected, atol=1e-6)


def test_act_is_differentiable() -> None:
    """策略/价值/熵都能反传梯度，参数梯度有限且非全零。"""
    policy = make_policy()
    observations = make_observations(batch_size=2)
    mask = make_mask(batch_size=2)

    actions, log_probs, values, entropies = policy.act(observations, mask)
    assert actions.requires_grad is False
    loss = -log_probs.mean() + values.pow(2).mean() - 0.01 * entropies.mean()
    loss.backward()

    grads = [parameter.grad for parameter in policy.parameters() if parameter.grad is not None]
    assert len(grads) == len(list(policy.parameters()))
    assert all(torch.isfinite(grad).all() for grad in grads)
    assert any(float(grad.abs().sum()) > 0.0 for grad in grads)


def test_single_valid_action_mask_broadcast() -> None:
    """``[R, Q, M]`` 单条掩码可以广播到整个批次。"""
    policy = make_policy()
    observations = make_observations(batch_size=3)
    mask = torch.zeros(N_RULES, N_QUBITS, N_MOMENTS, dtype=torch.bool)
    mask[2, 1, 3] = True  # 对应动作下标 (2*Q+1)*M+3
    expected_action = encode_action(2, 1, 3, N_QUBITS, N_MOMENTS)

    actions, log_probs, _values, entropies = policy.act(observations, mask, deterministic=True)
    assert torch.equal(actions, torch.full((3,), expected_action, dtype=torch.int64))
    assert torch.allclose(log_probs, torch.zeros(3), atol=1e-6)
    assert torch.allclose(entropies, torch.zeros(3), atol=1e-6)


def test_act_rejects_empty_mask() -> None:
    """整行掩码全为 False（没有可执行动作）时应给出清晰的 ValueError。"""
    policy = make_policy()
    observations = make_observations(batch_size=2)
    empty_mask = torch.zeros(2, N_RULES, N_QUBITS, N_MOMENTS, dtype=torch.bool)
    for deterministic in (False, True):
        try:
            policy.act(observations, empty_mask, deterministic=deterministic)
        except ValueError as exc:
            assert "掩码" in str(exc)
        else:  # pragma: no cover - 失败路径
            raise AssertionError("全 False 掩码应抛 ValueError")
