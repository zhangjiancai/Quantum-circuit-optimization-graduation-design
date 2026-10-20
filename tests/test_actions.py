"""动作编码的单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from qco.actions import (
    action_space_size,
    decode_action,
    decode_actions,
    encode_action,
    encode_actions,
)

N_RULES = 4
N_QUBITS = 3
N_MOMENTS = 5


def test_action_space_size() -> None:
    assert action_space_size(N_RULES, N_QUBITS, N_MOMENTS) == 60
    with pytest.raises(ValueError):
        action_space_size(0, 1, 1)
    with pytest.raises(ValueError):
        action_space_size(1, -1, 1)


def test_roundtrip_all_actions() -> None:
    seen = set()
    for rule in range(N_RULES):
        for qubit in range(N_QUBITS):
            for moment in range(N_MOMENTS):
                index = encode_action(rule, qubit, moment, N_QUBITS, N_MOMENTS)
                assert 0 <= index < action_space_size(N_RULES, N_QUBITS, N_MOMENTS)
                assert index not in seen
                seen.add(index)
                assert decode_action(index, N_QUBITS, N_MOMENTS) == (rule, qubit, moment)
    assert len(seen) == N_RULES * N_QUBITS * N_MOMENTS


def test_layout_matches_tensor_flattening() -> None:
    """动作下标必须等于 ``[R, Q, M]`` 张量按行优先展平后的下标。"""
    index = encode_action(2, 1, 4, N_QUBITS, N_MOMENTS)
    manual = np.ravel_multi_index((2, 1, 4), (N_RULES, N_QUBITS, N_MOMENTS))
    assert index == manual


def test_encode_validation() -> None:
    with pytest.raises(ValueError):
        encode_action(0, N_QUBITS, 0, N_QUBITS, N_MOMENTS)
    with pytest.raises(ValueError):
        encode_action(0, 0, N_MOMENTS, N_QUBITS, N_MOMENTS)
    with pytest.raises(ValueError):
        encode_action(-1, 0, 0, N_QUBITS, N_MOMENTS)
    with pytest.raises(ValueError):
        decode_action(-1, N_QUBITS, N_MOMENTS)


def test_vectorized_roundtrip() -> None:
    indices = np.arange(N_RULES * N_QUBITS * N_MOMENTS, dtype=np.int64)
    rules, qubits, moments = decode_actions(indices, N_QUBITS, N_MOMENTS)
    restored = encode_actions(rules, qubits, moments, N_QUBITS, N_MOMENTS)
    assert np.array_equal(restored, indices)
    assert np.array_equal(rules[:N_MOMENTS], np.zeros(N_MOMENTS, dtype=indices.dtype))


def test_vectorized_shape_validation() -> None:
    with pytest.raises(ValueError):
        encode_actions(np.zeros(2), np.zeros(3), np.zeros(2), N_QUBITS, N_MOMENTS)
