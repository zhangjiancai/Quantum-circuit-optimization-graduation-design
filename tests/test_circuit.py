"""不可变电路表示的单元测试。"""

from __future__ import annotations

import pytest

from qco.circuit import Circuit, Op
from qco.gates import GATE_NAMES


def test_op_validation() -> None:
    with pytest.raises(ValueError):
        Op("NOT_A_GATE", (0,))
    with pytest.raises(ValueError):
        Op("X", (0, 1))  # 单比特门给了两个比特
    with pytest.raises(ValueError):
        Op("CNOT", (0,))  # 二比特门只给了一个比特
    with pytest.raises(ValueError):
        Op("X", (-1,))
    with pytest.raises(ValueError):
        Op("CNOT", (1, 1))  # 比特重复


def test_op_helpers() -> None:
    op = Op("CNOT", (1, 2))
    assert op.num_qubits == 2
    assert "CNOT" in op.label()
    assert Op("X", (0,)).label() == "X(q0)"


def test_circuit_validation() -> None:
    with pytest.raises(ValueError):
        Circuit(0, [])
    with pytest.raises(ValueError):
        Circuit(2, [[Op("X", (0,)), Op("Y", (0,))]])  # 同层同比特冲突
    with pytest.raises(ValueError):
        Circuit(2, [[Op("X", (2,))]])  # 比特越界
    with pytest.raises(TypeError):
        Circuit(2, [[("X", (0,))]])  # 不是 Op 对象


def test_empty_layers_are_dropped() -> None:
    circuit = Circuit(2, [[], [Op("X", (0,))], [], [Op("Y", (1,))], []])
    assert circuit.depth == 2
    assert circuit.layers == ((Op("X", (0,)),), (Op("Y", (1,)),))


def test_metrics() -> None:
    circuit = Circuit(
        3,
        [
            [Op("H", (0,)), Op("X", (2,))],
            [Op("CNOT", (0, 1))],
            [Op("T", (2,))],
        ],
    )
    assert circuit.depth == 3
    assert circuit.num_ops == 4
    assert circuit.num_single_qubit_ops == 3
    assert circuit.num_two_qubit_ops == 1
    assert circuit.cost(lambda_depth=0.5) == pytest.approx(4 + 1.5)
    assert circuit.cost(lambda_depth=0.0) == pytest.approx(4.0)


def test_queries() -> None:
    circuit = Circuit(
        2,
        [
            [Op("H", (0,)), Op("X", (1,))],
            [Op("CNOT", (0, 1))],
            [Op("H", (0,))],
        ],
    )
    assert circuit.layer_qubits(0) == frozenset({0, 1})
    assert circuit.op_on(0, 0).gate == "H"
    assert circuit.op_on(1, 0).gate == "X"
    assert circuit.op_on(0, 5) is None
    assert circuit.op_on(0, -1) is None
    assert circuit.next_op_on(0, 0) == (1, Op("CNOT", (0, 1)))
    assert circuit.next_op_on(1, 1) is None
    assert circuit.prev_op_on(0, 2) == (1, Op("CNOT", (0, 1)))
    assert circuit.prev_op_on(0, 0) is None
    assert [op.gate for _, op in circuit.qubit_ops(0)] == ["H", "CNOT", "H"]
    assert [index for index, _ in circuit.iter_ops()] == [0, 0, 1, 2]


def test_serialization_roundtrip() -> None:
    circuit = Circuit(
        3,
        [
            [Op("H", (0,)), Op("CNOT", (1, 2))],
            [Op("T", (2,))],
        ],
    )
    data = circuit.to_dict()
    assert data["n_qubits"] == 3
    restored = Circuit.from_dict(data)
    assert restored == circuit
    assert restored.layers == circuit.layers


def test_with_layers_and_equality_hash() -> None:
    first = Circuit(2, [[Op("X", (0,))]])
    second = Circuit(2, [[Op("X", (0,))], []])
    assert first == second
    assert hash(first) == hash(second)
    third = first.with_layers([[Op("Y", (0,))]])
    assert third != first
    assert len({first, second, third}) == 2


def test_to_ascii_and_text() -> None:
    circuit = Circuit(2, [[Op("H", (0,))], [Op("CNOT", (0, 1))], [Op("X", (1,))]])
    ascii_text = circuit.to_ascii()
    assert "q0" in ascii_text and "q1" in ascii_text
    assert "H" in ascii_text
    assert isinstance(circuit.to_text(), str)


def test_unitary_property_matches_simulator() -> None:
    from qco.simulator import circuit_unitary

    circuit = Circuit(2, [[Op("H", (0,))], [Op("CNOT", (0, 1))]])
    assert circuit.unitary.shape == (4, 4)
    assert (circuit.unitary == circuit_unitary(circuit)).all()
    # 缓存：第二次访问是同一对象
    assert circuit.unitary is circuit.unitary


def test_cirq_roundtrip() -> None:
    cirq = pytest.importorskip("cirq")
    circuit = Circuit(
        3,
        [
            [Op("H", (0,)), Op("X", (2,))],
            [Op("CNOT", (0, 1))],
            [Op("S", (1,))],
        ],
    )
    cirq_circuit = circuit.to_cirq()
    assert isinstance(cirq_circuit, cirq.Circuit)
    restored = Circuit.from_cirq(cirq_circuit)
    assert restored == circuit
    assert restored.equivalent(circuit)


def test_circuit_unknown_cirq_gate() -> None:
    cirq = pytest.importorskip("cirq")
    qubit = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.rx(0.3)(qubit))
    with pytest.raises(ValueError):
        Circuit.from_cirq(circuit)


def test_gate_names_are_covered() -> None:
    """门库顺序变化会破坏观测/动作约定，这里做一次显式守卫。"""
    assert len(GATE_NAMES) == 11
