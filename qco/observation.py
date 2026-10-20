"""电路 -> 观测张量的编码。

状态表示：``[N_CHANNELS, n_qubits, n_moments]`` 的 float32 张量，通道含义见
:mod:`qco.gates`：
- 通道 0..10：该比特在该时隙参与了哪种门的编码（one-hot，二比特门在两个比特上同时置 1）；
- 通道 11（``CONTROL_CHANNEL``）：二比特门的控制比特标记；
- 通道 12（``TARGET_CHANNEL``）：二比特门的目标比特标记；
- 通道 13（``IDLE_CHANNEL``）：该比特在该时隙空闲。

之所以显式编码 ``idle``：空时隙是“把门前移/后移”这类调度动作的关键信息，
全靠零向量无法区分“空闲”与“不存在”。超出电路实际深度的时隙全部标记为空闲，
因此无论电路被优化到多浅，观测张量形状都保持不变。
"""

from __future__ import annotations

import numpy as np

from qco.circuit import Circuit
from qco.gates import (
    CONTROL_CHANNEL,
    GATE_CHANNEL_INDEX,
    IDLE_CHANNEL,
    N_CHANNELS,
    TARGET_CHANNEL,
)


def encode_circuit(circuit: Circuit, n_moments: int) -> np.ndarray:
    """把电路编码为形状 ``[N_CHANNELS, n_qubits, n_moments]`` 的 float32 观测。

    Parameters
    ----------
    circuit
        待编码电路。
    n_moments
        观测时间窗口宽度；必须不小于 ``circuit.depth``。

    Raises
    ------
    ValueError
        当 ``circuit.depth > n_moments`` 时（说明配置窗口过小，应扩大 ``n_moments``）。
    """
    n_moments = int(n_moments)
    if n_moments <= 0:
        raise ValueError(f"n_moments 必须为正整数，收到 {n_moments}")
    if circuit.depth > n_moments:
        raise ValueError(
            f"电路深度 {circuit.depth} 超过观测窗口 n_moments={n_moments}，"
            "请增大 EnvConfig.n_moments 或减小初始电路深度"
        )

    observation = np.zeros(
        (N_CHANNELS, circuit.n_qubits, n_moments), dtype=np.float32
    )
    # 默认全部空闲，再逐个时隙覆盖。
    observation[IDLE_CHANNEL, :, :] = 1.0
    for moment, layer in enumerate(circuit.layers):
        occupied: set[int] = set()
        for op in layer:
            gate_channel = GATE_CHANNEL_INDEX[op.gate]
            for position, qubit in enumerate(op.qubits):
                observation[gate_channel, qubit, moment] = 1.0
                occupied.add(qubit)
                if op.num_qubits == 2:
                    if position == 0:
                        observation[CONTROL_CHANNEL, qubit, moment] = 1.0
                    else:
                        observation[TARGET_CHANNEL, qubit, moment] = 1.0
        for qubit in occupied:
            observation[IDLE_CHANNEL, qubit, moment] = 0.0
    return observation


def observation_shape(n_qubits: int, n_moments: int) -> tuple[int, int, int]:
    """返回观测张量形状 ``(N_CHANNELS, n_qubits, n_moments)``。"""
    return (N_CHANNELS, int(n_qubits), int(n_moments))
