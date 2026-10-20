"""随机电路生成与数据集存取的单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from qco.generator import load_circuits, random_circuit, random_circuits, save_circuits


def test_random_circuit_shape_and_validity() -> None:
    rng = np.random.default_rng(0)
    circuit = random_circuit(5, 8, rng)
    assert circuit.n_qubits == 5
    assert circuit.depth == 8
    # 每层至少一个门；构造器已经保证同一比特不会被重复占用
    for layer in circuit.layers:
        assert len(layer) >= 1


def test_random_circuit_reproducible() -> None:
    first = random_circuit(4, 6, np.random.default_rng(42))
    second = random_circuit(4, 6, np.random.default_rng(42))
    third = random_circuit(4, 6, np.random.default_rng(43))
    assert first == second
    assert first != third


def test_random_circuit_p_two_zero_gives_single_qubit_only() -> None:
    circuit = random_circuit(4, 5, np.random.default_rng(1), p_two=0.0)
    assert circuit.num_two_qubit_ops == 0
    assert circuit.num_single_qubit_ops == circuit.num_ops


def test_random_circuit_p_two_one_has_two_qubit_gates() -> None:
    circuit = random_circuit(5, 6, np.random.default_rng(2), p_two=1.0)
    assert circuit.num_two_qubit_ops > 0


def test_random_circuit_argument_validation() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        random_circuit(1, 3, rng)
    with pytest.raises(ValueError):
        random_circuit(3, 0, rng)
    with pytest.raises(ValueError):
        random_circuit(3, 3, rng, p_two=1.5)


def test_random_circuits_reproducible() -> None:
    first = random_circuits(5, 4, 5, seed=7)
    second = random_circuits(5, 4, 5, seed=7)
    third = random_circuits(5, 4, 5, seed=8)
    assert first == second
    assert first != third
    assert len(first) == 5


def test_dataset_roundtrip(tmp_path) -> None:
    circuits = random_circuits(3, 4, 6, seed=5)
    path = tmp_path / "dataset.json"
    save_circuits(circuits, path)
    assert path.exists()
    restored = load_circuits(path)
    assert restored == circuits


def test_save_empty_dataset_raises(tmp_path) -> None:
    with pytest.raises(ValueError):
        save_circuits([], tmp_path / "empty.json")
