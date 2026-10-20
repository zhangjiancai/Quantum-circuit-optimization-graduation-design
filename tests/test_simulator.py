"""状态向量 / 幺正矩阵模拟器的单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from qco.circuit import Circuit, Op
from qco.simulator import (
    apply_gate_to_state,
    circuit_state_vector,
    circuit_unitary,
    circuits_equivalent,
    equivalent_up_to_global_phase,
    probabilities,
)


def make(n_qubits: int, layers: list[list[tuple[str, tuple[int, ...]]]]) -> Circuit:
    """测试辅助：用 ``(门名, 比特)`` 列表构造电路。"""
    return Circuit(n_qubits, [[Op(gate, qubits) for gate, qubits in layer] for layer in layers])


def test_state_vector_initial_state() -> None:
    assert np.allclose(circuit_state_vector(make(2, [])), [1, 0, 0, 0])


def test_single_qubit_gate_on_qubit_zero_is_most_significant() -> None:
    """比特 0 是最高位：X(q0) 作用在 |00> 上应得到 |10>，即下标 2。"""
    state = circuit_state_vector(make(2, [[("X", (0,))]]))
    assert np.allclose(state, [0, 0, 1, 0])
    state = circuit_state_vector(make(2, [[("X", (1,))]]))
    assert np.allclose(state, [0, 1, 0, 0])


def test_hadamard_and_probabilities() -> None:
    circuit = make(1, [[("H", (0,))]])
    state = circuit_state_vector(circuit)
    assert np.allclose(state, [1 / np.sqrt(2), 1 / np.sqrt(2)])
    assert np.allclose(probabilities(circuit), [0.5, 0.5])


def test_cnot_entanglement() -> None:
    """H(q0) + CNOT(q0,q1) 生成 Bell 态。"""
    circuit = make(2, [[("H", (0,))], [("CNOT", (0, 1))]])
    state = circuit_state_vector(circuit)
    expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
    assert np.allclose(state, expected)


def test_swap_gate() -> None:
    """X(q0) 后接 SWAP 应把激发移动到 q1。"""
    circuit = make(2, [[("X", (0,))], [("SWAP", (0, 1))]])
    state = circuit_state_vector(circuit)
    assert np.allclose(state, [0, 1, 0, 0])


def test_apply_gate_to_state_validation() -> None:
    with pytest.raises(ValueError):
        apply_gate_to_state(np.zeros(3, dtype=complex), "X", (0,), 2)
    with pytest.raises(ValueError):
        apply_gate_to_state(np.zeros(4, dtype=complex), "X", (0, 0), 2)
    with pytest.raises(ValueError):
        apply_gate_to_state(np.zeros(4, dtype=complex), "X", (5,), 2)


def test_circuit_unitary_is_unitary() -> None:
    rng = np.random.default_rng(0)
    from qco.generator import random_circuits

    for circuit in random_circuits(10, 3, 5, seed=3):
        unitary = circuit_unitary(circuit)
        dim = 2**circuit.n_qubits
        assert unitary.shape == (dim, dim)
        assert np.allclose(unitary @ unitary.conj().T, np.eye(dim), atol=1e-8)
        assert circuit.unitary.shape == unitary.shape
        assert np.allclose(circuit.unitary, unitary)
    del rng


def test_circuit_unitary_matches_state_vector() -> None:
    circuit = make(3, [[("H", (0,)), ("X", (2,))], [("CNOT", (0, 2))], [("T", (1,))]])
    unitary = circuit_unitary(circuit)
    zero_state = np.zeros(8, dtype=complex)
    zero_state[0] = 1.0
    assert np.allclose(unitary @ zero_state, circuit_state_vector(circuit))


def test_equivalent_up_to_global_phase() -> None:
    identity = np.eye(2)
    assert equivalent_up_to_global_phase(identity, 1j * identity)
    assert equivalent_up_to_global_phase(np.diag([1, 1]), -np.diag([1, 1]))
    assert not equivalent_up_to_global_phase(identity, np.diag([1, -1]))
    assert not equivalent_up_to_global_phase(np.eye(2), np.eye(3))
    assert not equivalent_up_to_global_phase(np.zeros((2, 2)), np.eye(2))


def test_circuits_equivalent_examples() -> None:
    # X 后接 X 等价于空电路
    assert circuits_equivalent(make(1, [[("X", (0,))], [("X", (0,))]]), make(1, []))
    # H 后接 H 等价于空电路
    assert circuits_equivalent(make(1, [[("H", (0,))], [("H", (0,))]]), make(1, []))
    # S 后接 S 等价于 Z
    assert circuits_equivalent(make(1, [[("S", (0,))], [("S", (0,))]]), make(1, [[("Z", (0,))]]))
    # 不同比特数不相等
    assert not circuits_equivalent(make(1, [[("X", (0,))]]), make(2, [[("X", (0,))]]))


def test_circuit_unitary_matches_cirq() -> None:
    """安装了 cirq 时，交叉验证自研模拟器与 cirq 的幺正矩阵一致。"""
    cirq = pytest.importorskip("cirq")
    from qco.generator import random_circuits

    for circuit in random_circuits(5, 3, 6, seed=11):
        expected = cirq.unitary(circuit.to_cirq())
        actual = circuit_unitary(circuit)
        assert equivalent_up_to_global_phase(actual, np.asarray(expected), atol=1e-8)
