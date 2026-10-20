"""重写规则的单元测试。

核心不变量（每个可适用动作都必须满足）：

1. 应用后电路合法；
2. 代价严格下降（``num_ops + lambda_depth * depth``）；
3. 与变换前**模全局相位语义等价**。
"""

from __future__ import annotations

import pytest

from qco.circuit import Circuit, Op
from qco.generator import random_circuits
from qco.rules import (
    RULES,
    CancelTwoQubitRule,
    MergeSingleQubitRule,
    MoveEarlierRule,
    MoveLaterRule,
    RULES_BY_NAME,
    applicable_actions,
    rule_names,
)
from qco.simulator import circuits_equivalent

LAMBDA_DEPTH = 0.5


def make(n_qubits: int, layers: list[list[tuple[str, tuple[int, ...]]]]) -> Circuit:
    return Circuit(n_qubits, [[Op(gate, qs) for gate, qs in layer] for layer in layers])


def test_rule_order_and_names() -> None:
    """规则下标即动作空间中的规则编号，必须稳定。"""
    assert rule_names() == (
        "merge_single_qubit",
        "cancel_two_qubit",
        "move_earlier",
        "move_later",
    )
    for index, rule in enumerate(RULES):
        assert rule.rule_index == index
        assert RULES_BY_NAME[rule.name] is rule


# ----------------------------------------------------------------------
# merge_single_qubit
# ----------------------------------------------------------------------
def test_merge_t_t_becomes_s() -> None:
    circuit = make(1, [[("T", (0,))], [("T", (0,))]])
    rule = MergeSingleQubitRule()
    assert rule.is_applicable(circuit, 0, 0)
    merged = rule.apply(circuit, 0, 0)
    assert merged.depth == 1
    assert merged.num_ops == 1
    assert merged.layer(0)[0].gate == "S"
    assert merged.cost(LAMBDA_DEPTH) < circuit.cost(LAMBDA_DEPTH)
    assert circuits_equivalent(circuit, merged)


def test_merge_self_inverse_removes_both() -> None:
    circuit = make(2, [[("H", (0,))], [("H", (0,))]])
    merged = MergeSingleQubitRule().apply(circuit, 0, 0)
    assert merged.num_ops == 0
    assert merged.depth == 0
    assert circuits_equivalent(circuit, merged)


def test_merge_up_to_global_phase() -> None:
    """Y·Z = iX，模全局相位应合并为 X。"""
    circuit = make(1, [[("Y", (0,))], [("Z", (0,))]])
    merged = MergeSingleQubitRule().apply(circuit, 0, 0)
    assert merged.num_ops == 1
    assert merged.layer(0)[0].gate == "X"
    assert circuits_equivalent(circuit, merged)


def test_merge_works_across_idle_layers() -> None:
    circuit = make(2, [[("H", (0,))], [("Y", (1,))], [("H", (0,))]])
    rule = MergeSingleQubitRule()
    assert rule.is_applicable(circuit, 0, 0)
    merged = rule.apply(circuit, 0, 0)
    # H·H = I 被删除，只剩另一比特上的 Y
    assert merged.num_ops == 1
    assert merged.layer(0)[0].gate == "Y"
    assert circuits_equivalent(circuit, merged)


def test_merge_requires_adjacent_in_qubit_order() -> None:
    circuit = make(2, [[("H", (0,))], [("CNOT", (0, 1))], [("H", (0,))]])
    rule = MergeSingleQubitRule()
    assert not rule.is_applicable(circuit, 0, 0)
    assert not rule.is_applicable(circuit, 0, 2)


def test_merge_not_applicable_for_non_catalog_product() -> None:
    circuit = make(1, [[("H", (0,))], [("S", (0,))]])
    assert not MergeSingleQubitRule().is_applicable(circuit, 0, 0)


def test_merge_not_applicable_on_two_qubit_gate() -> None:
    circuit = make(2, [[("CNOT", (0, 1))], [("CNOT", (0, 1))]])
    assert not MergeSingleQubitRule().is_applicable(circuit, 0, 0)


# ----------------------------------------------------------------------
# cancel_two_qubit
# ----------------------------------------------------------------------
@pytest.mark.parametrize("gate", ["CNOT", "CZ", "SWAP"])
def test_cancel_two_qubit_identical(gate: str) -> None:
    circuit = make(2, [[(gate, (0, 1))], [(gate, (0, 1))]])
    rule = CancelTwoQubitRule()
    assert rule.is_applicable(circuit, 0, 0)
    assert rule.is_applicable(circuit, 1, 0)
    cancelled = rule.apply(circuit, 0, 0)
    assert cancelled.num_ops == 0
    assert circuits_equivalent(circuit, cancelled)


def test_cancel_two_qubit_rejects_different_gates() -> None:
    circuit = make(2, [[("CNOT", (0, 1))], [("CZ", (0, 1))]])
    assert not CancelTwoQubitRule().is_applicable(circuit, 0, 0)


def test_cancel_two_qubit_rejects_swapped_roles() -> None:
    circuit = make(2, [[("CNOT", (0, 1))], [("CNOT", (1, 0))]])
    assert not CancelTwoQubitRule().is_applicable(circuit, 0, 0)


def test_cancel_two_qubit_rejects_intervening_gate() -> None:
    circuit = make(2, [[("CNOT", (0, 1))], [("X", (1,))], [("CNOT", (0, 1))]])
    assert not CancelTwoQubitRule().is_applicable(circuit, 0, 0)


# ----------------------------------------------------------------------
# move_earlier / move_later
# ----------------------------------------------------------------------
def test_move_earlier_compresses_depth() -> None:
    circuit = make(2, [[("X", (0,))], [("Y", (1,))], [("Z", (0,))]])
    rule = MoveEarlierRule()
    assert rule.is_applicable(circuit, 0, 2)
    moved = rule.apply(circuit, 0, 2)
    assert moved.depth == 2
    assert moved.cost(LAMBDA_DEPTH) < circuit.cost(LAMBDA_DEPTH)
    assert circuits_equivalent(circuit, moved)


def test_move_later_compresses_depth() -> None:
    circuit = make(2, [[("Z", (0,))], [("Y", (1,))]])
    rule = MoveLaterRule()
    assert rule.is_applicable(circuit, 0, 0)
    moved = rule.apply(circuit, 0, 0)
    assert moved.depth == 1
    assert moved.num_ops == 2
    assert circuits_equivalent(circuit, moved)


def test_move_rejected_when_order_would_change() -> None:
    circuit = make(2, [[("X", (0,)), ("Y", (1,))], [("Z", (0,))]])
    assert not MoveEarlierRule().is_applicable(circuit, 0, 1)
    assert not MoveLaterRule().is_applicable(circuit, 0, 1)


def test_move_rejected_when_layer_not_singleton() -> None:
    """独占时隙才保证深度下降；共享时隙不允许移动（否则是空动作）。"""
    circuit = make(2, [[("Z", (0,)), ("Y", (1,))], [("X", (0,))]])
    assert not MoveLaterRule().is_applicable(circuit, 0, 0)
    assert not MoveEarlierRule().is_applicable(circuit, 0, 0)


def test_move_at_boundaries_not_applicable() -> None:
    circuit = make(2, [[("X", (0,))], [("Y", (1,))]])
    assert not MoveEarlierRule().is_applicable(circuit, 0, 0)
    assert not MoveLaterRule().is_applicable(circuit, 1, 1)


def test_move_two_qubit_gate() -> None:
    """单比特门可以绕过其他比特上的门移动，深度随之压缩。"""
    circuit = make(3, [[("CNOT", (0, 1))], [("X", (1,))], [("H", (2,))]])
    rule = MoveEarlierRule()
    assert rule.is_applicable(circuit, 2, 2)
    moved = rule.apply(circuit, 2, 2)
    assert moved.depth == 2
    assert circuits_equivalent(circuit, moved)
    # 二比特门同样可以移动（这里向后移动一个时隙）
    circuit2 = make(3, [[("CNOT", (0, 1))], [("X", (2,))]])
    rule_later = MoveLaterRule()
    assert rule_later.is_applicable(circuit2, 0, 0)
    moved2 = rule_later.apply(circuit2, 0, 0)
    assert moved2.depth == 1
    assert circuits_equivalent(circuit2, moved2)


# ----------------------------------------------------------------------
# 全局性质
# ----------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_all_applicable_actions_satisfy_invariants(seed: int) -> None:
    """随机电路上，每个可适用动作都必须降代价且保持语义等价。"""
    circuits = random_circuits(15, 4, 6, seed=seed)
    checked = 0
    for circuit in circuits:
        for rule in RULES:
            for qubit in range(circuit.n_qubits):
                for moment in range(circuit.depth):
                    if not rule.is_applicable(circuit, qubit, moment):
                        continue
                    transformed = rule.apply(circuit, qubit, moment)
                    checked += 1
                    assert transformed.n_qubits == circuit.n_qubits
                    assert transformed.cost(LAMBDA_DEPTH) < circuit.cost(LAMBDA_DEPTH)
                    assert transformed.depth <= circuit.depth
                    assert circuits_equivalent(circuit, transformed)
    assert checked > 0, "随机电路上应至少存在一些可用动作"


def test_applicable_actions_matches_per_rule_scan() -> None:
    circuits = random_circuits(10, 3, 5, seed=9)
    for circuit in circuits:
        listed = set(applicable_actions(circuit))
        manual = {
            (rule.rule_index, qubit, moment)
            for rule in RULES
            for qubit in range(circuit.n_qubits)
            for moment in range(circuit.depth)
            if rule.is_applicable(circuit, qubit, moment)
        }
        assert listed == manual


def test_apply_raises_when_not_applicable() -> None:
    circuit = make(2, [[("X", (0,))]])
    for rule in RULES:
        # 找一个该规则不可用的位置
        for qubit in range(2):
            for moment in range(circuit.depth):
                if not rule.is_applicable(circuit, qubit, moment):
                    with pytest.raises(ValueError):
                        rule.apply(circuit, qubit, moment)
                    return
