"""动作掩码与带掩码 softmax 的单元测试。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qco.circuit import Circuit, Op
from qco.generator import random_circuits
from qco.masking import (
    NEG_INF,
    build_action_mask,
    flatten_action_mask,
    masked_entropy,
    masked_log_softmax,
    unflatten_action_mask,
)
from qco.rules import RULES, applicable_actions


def make(n_qubits: int, layers: list[list[tuple[str, tuple[int, ...]]]]) -> Circuit:
    return Circuit(n_qubits, [[Op(gate, qs) for gate, qs in layer] for layer in layers])


def test_mask_shape_and_type() -> None:
    circuit = make(2, [[("H", (0,))], [("H", (0,))]])
    mask = build_action_mask(circuit, n_moments=5)
    assert mask.shape == (len(RULES), 2, 5)
    assert mask.dtype == bool


def test_mask_matches_rule_applicability() -> None:
    for circuit in random_circuits(10, 3, 6, seed=4):
        mask = build_action_mask(circuit, n_moments=7)
        for rule in RULES:
            for qubit in range(circuit.n_qubits):
                for moment in range(7):
                    expected = moment < circuit.depth and rule.is_applicable(
                        circuit, qubit, moment
                    )
                    assert mask[rule.rule_index, qubit, moment] == expected


def test_mask_matches_applicable_actions() -> None:
    circuit = make(3, [[("T", (0,))], [("T", (0,))], [("CNOT", (1, 2))]])
    mask = build_action_mask(circuit, n_moments=4)
    mask_actions = {
        (int(rule), int(qubit), int(moment))
        for rule, qubit, moment in zip(*np.nonzero(mask))
    }
    assert mask_actions == set(applicable_actions(circuit))


def test_mask_out_of_depth_is_false() -> None:
    circuit = make(2, [[("X", (0,))]])
    mask = build_action_mask(circuit, n_moments=6)
    assert not mask[:, :, 1:].any()
    # 只有一层且该比特上只有一个门：没有任何规则可用
    assert not mask[:, :, 0].any()


def test_flatten_unflatten_roundtrip() -> None:
    circuit = make(2, [[("H", (0,))], [("H", (0,))]])
    mask = build_action_mask(circuit, n_moments=3)
    flat = flatten_action_mask(mask)
    assert flat.shape == (len(RULES) * 2 * 3,)
    restored = unflatten_action_mask(flat, len(RULES), 2, 3)
    assert np.array_equal(restored, mask)
    with pytest.raises(ValueError):
        unflatten_action_mask(flat[:-1], len(RULES), 2, 3)


def test_masked_log_softmax_zeros_invalid_actions() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 3, 5)
    mask = torch.zeros(4, 3, 5, dtype=torch.bool)
    mask[0, 0, 0] = True
    mask[1, 2, 1] = True
    mask[3, 1, 4] = True
    log_probs = masked_log_softmax(logits, mask)
    expanded = mask.expand_as(log_probs)
    assert (log_probs[~expanded] <= NEG_INF).all()
    probs = log_probs.exp()
    assert probs[~expanded].abs().max() == 0.0
    # 有可用动作的行概率和必须为 1；整行不可用的行概率和必须为 0
    valid_rows = expanded.any(dim=-1)
    assert torch.allclose(probs.sum(dim=-1), valid_rows.float(), atol=1e-6)


def test_masked_log_softmax_accepts_flat_mask() -> None:
    logits = torch.randn(2, 3 * 2)
    flat_mask = torch.tensor([True, False, False, True, False, False])
    log_probs = masked_log_softmax(logits, flat_mask)
    assert (log_probs[:, 1] <= NEG_INF).all()
    assert torch.isfinite(log_probs[:, 0]).all()


def test_masked_log_softmax_accepts_3d_mask_for_flat_logits() -> None:
    """``[R, Q, M]`` 掩码应能广播到展平后的 ``[B, R*Q*M]`` logits（按行优先展平）。"""
    torch.manual_seed(0)
    n_rules, n_qubits, n_moments = 2, 2, 3
    logits = torch.randn(2, n_rules * n_qubits * n_moments)
    mask = torch.zeros(n_rules, n_qubits, n_moments, dtype=torch.bool)
    mask[0, 0, 0] = True
    mask[1, 1, 2] = True
    log_probs = masked_log_softmax(logits, mask)
    expected_allowed = torch.zeros(n_rules * n_qubits * n_moments, dtype=torch.bool)
    expected_allowed[0] = True
    expected_allowed[(1 * n_qubits + 1) * n_moments + 2] = True
    assert torch.isfinite(log_probs[:, expected_allowed]).all()
    assert (log_probs[:, ~expected_allowed] <= NEG_INF).all()
    assert torch.allclose(log_probs.exp().sum(dim=-1), torch.ones(2), atol=1e-6)
    entropy = masked_entropy(log_probs, mask)
    assert torch.isfinite(entropy).all()


def test_masked_log_softmax_rejects_incompatible_mask() -> None:
    logits = torch.randn(2, 6)
    with pytest.raises(ValueError):
        masked_log_softmax(logits, torch.ones(5, dtype=torch.bool))


def test_masked_log_softmax_rejects_impossible_mask() -> None:
    """全部动作被屏蔽时，所有 log 概率都应被压到 NEG_INF 以下（调用方保证不采样）。"""
    logits = torch.randn(1, 2, 2)
    mask = torch.zeros(2, 2, dtype=torch.bool)
    log_probs = masked_log_softmax(logits, mask)
    assert (log_probs <= NEG_INF).all()


def test_masked_entropy() -> None:
    # 两个可用动作、logits 相同 -> 概率 0.5/0.5 -> 熵 ln 2
    logits = torch.zeros(1, 1, 2)
    mask = torch.tensor([[[True, True]]])
    log_probs = masked_log_softmax(logits, mask)
    entropy = masked_entropy(log_probs, mask)
    assert torch.allclose(entropy, torch.tensor([np.log(2.0)], dtype=torch.float32), atol=1e-6)
    # 只有一个可用动作 -> 熵为 0
    mask_single = torch.tensor([[[True, False]]])
    log_probs_single = masked_log_softmax(logits, mask_single)
    entropy_single = masked_entropy(log_probs_single, mask_single)
    assert torch.allclose(entropy_single, torch.zeros(1), atol=1e-6)
    assert torch.isfinite(entropy_single).all()
