"""动作掩码：屏蔽当前电路上不可用的 ``(规则, 比特, 时隙)`` 动作。

为什么必须有掩码
----------------
动作空间由 ``规则 × 比特 × 时隙`` 组合而成，其中绝大多数组合在当前电路上
没有意义（该比特上没有门、规则不适用、时隙越界）。若直接在这些动作上采样，
智能体会浪费大量步数并学到“乱试”行为；掩码后策略只在可行动作上分配概率，
这也是原项目 ``ActionMask`` 想做但从未真正实现的部分。
"""

from __future__ import annotations

import numpy as np
import torch

from qco.circuit import Circuit
from qco.rules import RULES

#: 用于屏蔽非法动作的负无穷（float32 下仍可参与 log_softmax）。
NEG_INF = -1e9


def build_action_mask(circuit: Circuit, n_moments: int) -> np.ndarray:
    """构造形状 ``[len(RULES), n_qubits, n_moments]`` 的布尔动作掩码。

    ``mask[r, q, m]`` 为真当且仅当规则 ``r`` 在 ``(q, m)`` 处可用。
    """
    n_rules = len(RULES)
    n_qubits = circuit.n_qubits
    n_moments = int(n_moments)
    if n_moments <= 0:
        raise ValueError(f"n_moments 必须为正整数，收到 {n_moments}")
    mask = np.zeros((n_rules, n_qubits, n_moments), dtype=bool)
    depth = circuit.depth
    for rule in RULES:
        for qubit in range(n_qubits):
            for moment in range(min(depth, n_moments)):
                if rule.is_applicable(circuit, qubit, moment):
                    mask[rule.rule_index, qubit, moment] = True
    return mask


def flatten_action_mask(mask: np.ndarray) -> np.ndarray:
    """把 ``[R, Q, M]`` 掩码展平为 ``[R*Q*M]``（行优先，与动作编码一致）。"""
    return np.asarray(mask, dtype=bool).reshape(-1)


def unflatten_action_mask(
    flat: np.ndarray, n_rules: int, n_qubits: int, n_moments: int
) -> np.ndarray:
    """把一维掩码还原为 ``[R, Q, M]``。"""
    flat = np.asarray(flat, dtype=bool)
    expected = int(n_rules) * int(n_qubits) * int(n_moments)
    if flat.size != expected:
        raise ValueError(f"掩码大小应为 {expected}，收到 {flat.size}")
    return flat.reshape((int(n_rules), int(n_qubits), int(n_moments)))


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """带掩码的 ``log_softmax``：非法动作的 log 概率为 ``NEG_INF``。

    Parameters
    ----------
    logits
        形状 ``[..., R, Q, M]`` 或 ``[..., R*Q*M]`` 的原始打分。
    mask
        布尔掩码，形状可为 ``[R, Q, M]``（自动广播）或与 ``logits`` 相同。
    """
    if not torch.is_tensor(logits):  # pragma: no cover - 类型防御
        raise TypeError("logits 必须是 torch.Tensor")
    mask = _broadcast_mask_to_logits(mask, logits)
    masked_logits = logits.masked_fill(~mask, NEG_INF)
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    # 二次掩码：若整行都被屏蔽，log_softmax 会给出均匀分布，这里统一压到 NEG_INF。
    return log_probs.masked_fill(~mask, NEG_INF)


def _broadcast_mask_to_logits(mask: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """把动作掩码规整为与 ``logits`` 同形状的布尔张量。

    支持三种输入布局：

    - 与 ``logits`` 同形状（``[B, R, Q, M]`` 或 ``[B, R*Q*M]``）；
    - ``[R, Q, M]`` 且 ``logits`` 为 ``[B, R, Q, M]``（按批次广播）；
    - ``[R, Q, M]`` 且 ``logits`` 为 ``[B, R*Q*M]``（先展平为 ``[R*Q*M]`` 再按批次广播）。

    最后一种布局容易踩坑：``[R, Q, M]`` 的张量不能直接 ``expand_as`` 到展平后的
    ``[B, R*Q*M]``，必须先按与 :mod:`qco.actions` 一致的行优先顺序展平。
    """
    mask = mask.to(device=logits.device, dtype=torch.bool)
    if mask.shape == logits.shape:
        return mask
    if mask.dim() == 3 and logits.dim() >= 2 and logits.shape[-1] == mask.numel():
        mask = mask.reshape(-1)
    if mask.numel() == logits.numel():
        return mask.reshape(logits.shape)
    try:
        return mask.expand_as(logits)
    except RuntimeError as exc:  # pragma: no cover - 形状不兼容时给出明确报错
        raise ValueError(
            f"动作掩码形状 {tuple(mask.shape)} 无法广播到 logits 形状 {tuple(logits.shape)}"
        ) from exc


def masked_entropy(log_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """计算带掩码分布的熵，形状为 ``[...]``。

    非法动作的 log 概率为 ``NEG_INF``，直接计算会出现 ``0 * -inf = nan``，
    因此先用掩码把非法位置的概率与贡献都置零。
    """
    mask = _broadcast_mask_to_logits(mask, log_probs)
    probs = torch.where(mask, log_probs.exp(), torch.zeros_like(log_probs))
    terms = torch.where(mask, probs * log_probs, torch.zeros_like(log_probs))
    return -terms.sum(dim=-1)
