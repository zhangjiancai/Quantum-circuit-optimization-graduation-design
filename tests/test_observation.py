"""观测编码的单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from qco.circuit import Circuit, Op
from qco.gates import (
    CONTROL_CHANNEL,
    GATE_CHANNEL_INDEX,
    IDLE_CHANNEL,
    N_CHANNELS,
    TARGET_CHANNEL,
)
from qco.observation import encode_circuit, observation_shape


def test_shape_and_dtype() -> None:
    circuit = Circuit(3, [[Op("H", (0,))], [Op("CNOT", (0, 2))]])
    observation = encode_circuit(circuit, n_moments=5)
    assert observation.shape == (N_CHANNELS, 3, 5)
    assert observation.dtype == np.float32
    assert observation_shape(3, 5) == (N_CHANNELS, 3, 5)


def test_single_qubit_gate_channels() -> None:
    circuit = Circuit(2, [[Op("H", (0,))]], )
    observation = encode_circuit(circuit, n_moments=3)
    assert observation[GATE_CHANNEL_INDEX["H"], 0, 0] == 1.0
    assert observation[GATE_CHANNEL_INDEX["X"], 0, 0] == 0.0
    assert observation[IDLE_CHANNEL, 0, 0] == 0.0
    # 同一层另一个比特空闲
    assert observation[IDLE_CHANNEL, 1, 0] == 1.0
    # 超出实际深度的时隙全部空闲
    assert observation[IDLE_CHANNEL, 0, 1] == 1.0
    assert observation[IDLE_CHANNEL, 1, 2] == 1.0
    assert observation[:, 0, 1:].sum() == 2.0  # 两个时隙各一个 idle 通道


def test_two_qubit_gate_control_target_channels() -> None:
    circuit = Circuit(2, [[Op("CNOT", (0, 1))]])
    observation = encode_circuit(circuit, n_moments=2)
    channel = GATE_CHANNEL_INDEX["CNOT"]
    assert observation[channel, 0, 0] == 1.0
    assert observation[channel, 1, 0] == 1.0
    assert observation[CONTROL_CHANNEL, 0, 0] == 1.0
    assert observation[CONTROL_CHANNEL, 1, 0] == 0.0
    assert observation[TARGET_CHANNEL, 1, 0] == 1.0
    assert observation[TARGET_CHANNEL, 0, 0] == 0.0
    assert observation[IDLE_CHANNEL, 0, 0] == 0.0
    assert observation[IDLE_CHANNEL, 1, 0] == 0.0


def test_each_position_has_exactly_one_occupancy_marker() -> None:
    """每个 (比特, 时隙) 的门通道之和为 0 或 1，且与 idle 通道互补。"""
    circuit = Circuit(
        3,
        [
            [Op("H", (0,)), Op("T", (2,))],
            [Op("CZ", (0, 2))],
            [Op("SWAP", (1, 2))],
        ],
    )
    observation = encode_circuit(circuit, n_moments=3)
    gate_sum = observation[: len(GATE_CHANNEL_INDEX), :, :].sum(axis=0)
    assert np.all((gate_sum == 0.0) | (gate_sum == 1.0))
    assert np.allclose(gate_sum + observation[IDLE_CHANNEL], 1.0)


def test_depth_exceeds_window_raises() -> None:
    circuit = Circuit(2, [[Op("H", (0,))], [Op("X", (1,))]])
    with pytest.raises(ValueError):
        encode_circuit(circuit, n_moments=1)


def test_invalid_window_raises() -> None:
    with pytest.raises(ValueError):
        encode_circuit(Circuit(2, []), n_moments=0)


def test_empty_circuit_is_all_idle() -> None:
    observation = encode_circuit(Circuit(2, []), n_moments=4)
    assert np.all(observation[IDLE_CHANNEL] == 1.0)
    assert np.all(observation[:IDLE_CHANNEL] == 0.0)
