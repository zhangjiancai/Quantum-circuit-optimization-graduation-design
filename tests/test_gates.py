"""门库与门矩阵的单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from qco.gates import (
    CONTROL_CHANNEL,
    GATE_CHANNEL_INDEX,
    GATE_NAMES,
    GATES,
    IDLE_CHANNEL,
    N_CHANNELS,
    SINGLE_QUBIT_GATE_NAMES,
    TARGET_CHANNEL,
    TWO_QUBIT_GATE_NAMES,
    gate_matrix,
    gate_matrix_on,
    gate_num_qubits,
    global_phase_equal,
    is_unitary,
    match_single_qubit_gate,
)


def test_gate_catalog_order_is_fixed() -> None:
    """门库顺序是状态张量通道与动作空间的约定，必须保持稳定。"""
    assert GATE_NAMES == (
        "X", "Y", "Z", "H", "S", "Sdag", "T", "Tdag", "CNOT", "CZ", "SWAP",
    )
    assert SINGLE_QUBIT_GATE_NAMES == ("X", "Y", "Z", "H", "S", "Sdag", "T", "Tdag")
    assert TWO_QUBIT_GATE_NAMES == ("CNOT", "CZ", "SWAP")
    assert GATE_CHANNEL_INDEX["X"] == 0 and GATE_CHANNEL_INDEX["SWAP"] == 10
    assert CONTROL_CHANNEL == 11 and TARGET_CHANNEL == 12 and IDLE_CHANNEL == 13
    assert N_CHANNELS == 14


@pytest.mark.parametrize("name", list(GATES))
def test_all_gates_are_unitary(name: str) -> None:
    """每个门矩阵都必须是幺正矩阵。"""
    matrix = gate_matrix(name)
    assert matrix.shape == (2 ** gate_num_qubits(name),) * 2
    assert is_unitary(matrix)


def test_known_matrices() -> None:
    """抽查几个门矩阵的具体数值与比特顺序。"""
    assert np.allclose(gate_matrix("X"), np.array([[0, 1], [1, 0]]))
    assert np.allclose(gate_matrix("H"), np.array([[1, 1], [1, -1]]) / np.sqrt(2))
    assert np.allclose(gate_matrix("S"), np.diag([1, 1j]))
    assert np.allclose(gate_matrix("Tdag"), np.diag([1, np.exp(-1j * np.pi / 4)]))
    # CNOT 基矢顺序 |00>,|01>,|10>,|11>，第一个比特为控制位
    expected_cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    )
    assert np.allclose(gate_matrix("CNOT"), expected_cnot)
    assert np.allclose(gate_matrix("CZ"), np.diag([1, 1, 1, -1]))
    assert np.allclose(
        gate_matrix("SWAP"),
        np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
    )


def test_unknown_gate_raises() -> None:
    with pytest.raises(KeyError):
        gate_matrix("NOT_A_GATE")
    with pytest.raises(KeyError):
        gate_num_qubits("NOT_A_GATE")


def test_global_phase_equal() -> None:
    x = gate_matrix("X")
    assert global_phase_equal(x, 1j * x)
    assert global_phase_equal(1j * x, -x)
    assert not global_phase_equal(x, gate_matrix("Y"))
    assert not global_phase_equal(x, 2 * x)  # 不是相位因子（模长不为 1）
    assert not global_phase_equal(x, np.eye(2))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("X", "X"),
        ("H", "H"),
        ("S", "S"),
        ("Sdag", "Sdag"),
        ("T", "T"),
        ("Tdag", "Tdag"),
    ],
)
def test_match_single_qubit_gate_identity(name: str, expected: str) -> None:
    assert match_single_qubit_gate(gate_matrix(name)) == expected


def test_match_single_qubit_gate_up_to_phase() -> None:
    """模全局相位的匹配：``1j·X`` 仍应识别为 X。"""
    x = gate_matrix("X")
    assert match_single_qubit_gate(1j * x) == "X"
    assert match_single_qubit_gate(np.exp(0.7j) * np.eye(2)) == "I"
    # Y·Z = iX，因此应合并为 X
    assert match_single_qubit_gate(gate_matrix("Y") @ gate_matrix("Z")) == "X"
    # T·T = S
    assert match_single_qubit_gate(gate_matrix("T") @ gate_matrix("T")) == "S"
    # S·S = Z
    assert match_single_qubit_gate(gate_matrix("S") @ gate_matrix("S")) == "Z"


def test_match_single_qubit_gate_rejects_non_catalog() -> None:
    """非门库内的幺正矩阵（如绕 X 轴旋转 0.3π）应返回 None。"""
    theta = 0.3 * np.pi
    rotation = np.cos(theta / 2) * np.eye(2) - 1j * np.sin(theta / 2) * gate_matrix("X")
    assert is_unitary(rotation)
    assert match_single_qubit_gate(rotation) is None
    # 非幺正矩阵必须返回 None
    assert match_single_qubit_gate(np.ones((2, 2), dtype=complex)) is None
    # 形状不对也返回 None
    assert match_single_qubit_gate(np.eye(3)) is None


def test_gate_matrix_on_single_qubit() -> None:
    """嵌入到多比特系统后，未涉及比特上应为单位阵。"""
    x0 = gate_matrix_on("X", (0,), 2)
    expected = np.kron(gate_matrix("X"), np.eye(2))
    assert np.allclose(x0, expected)

    z1 = gate_matrix_on("Z", (1,), 2)
    assert np.allclose(z1, np.kron(np.eye(2), gate_matrix("Z")))


def test_gate_matrix_on_two_qubits_ordering() -> None:
    """CNOT(control=1, target=0) 与 CNOT(control=0, target=1) 矩阵不同。"""
    cnot01 = gate_matrix_on("CNOT", (0, 1), 2)
    cnot10 = gate_matrix_on("CNOT", (1, 0), 2)
    assert np.allclose(cnot01, gate_matrix("CNOT"))
    expected = np.array(
        [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex
    )
    assert np.allclose(cnot10, expected)
    assert is_unitary(gate_matrix_on("SWAP", (1, 0), 3))


def test_gate_matrix_on_validation() -> None:
    with pytest.raises(ValueError):
        gate_matrix_on("X", (0, 1), 2)  # 单比特门给了两个比特
    with pytest.raises(ValueError):
        gate_matrix_on("CNOT", (0, 0), 2)  # 比特重复
    with pytest.raises(ValueError):
        gate_matrix_on("X", (2,), 2)  # 越界
