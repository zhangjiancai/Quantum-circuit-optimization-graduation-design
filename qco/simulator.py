"""基于 numpy 的状态向量模拟器。

用于两件事：

1. 计算电路的幺正矩阵，从而验证重写规则**模全局相位保持语义不变**；
2. 计算末态概率分布，便于可视化与教学演示。

约定
----
- 比特 0 是最高位（big-endian），基矢顺序 ``|0...0>, |0...1>, ...``；
- 张量形状统一为 ``(2,) * n_qubits``，最后可附加任意 batch 轴。

本模块不依赖 cirq；与 cirq 的一致性由测试交叉验证。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from qco.gates import gate_matrix

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查
    from qco.circuit import Circuit


def _apply_operator_to_axes(
    tensor: np.ndarray,
    matrix: np.ndarray,
    axes: Sequence[int],
    n_qubits: int,
) -> np.ndarray:
    """把 ``matrix`` 作用到张量的指定轴上（其余轴视为 batch 或恒等）。

    Parameters
    ----------
    tensor
        形状为 ``(2,) * n_qubits + batch_shape`` 的张量。
    matrix
        作用在 ``axes`` 上的 ``2**len(axes) × 2**len(axes)`` 矩阵。
    axes
        涉及的比特轴，顺序与 ``matrix`` 的行列顺序一致。
    n_qubits
        量子比特数（即前 ``n_qubits`` 个轴）。
    """
    axes = tuple(int(a) for a in axes)
    k = len(axes)
    extra_axes = list(range(n_qubits, tensor.ndim))
    permutation = list(axes) + [i for i in range(n_qubits) if i not in axes] + extra_axes
    transposed = np.transpose(tensor, permutation)

    leading_shape = (2,) * k
    rest_shape = transposed.shape[k:]
    flat = transposed.reshape((2**k,) + rest_shape)
    applied = np.tensordot(matrix, flat, axes=([1], [0]))
    restored = applied.reshape(leading_shape + rest_shape)
    return np.transpose(restored, np.argsort(permutation))


def apply_gate_to_state(
    state: np.ndarray,
    gate: str,
    qubits: Sequence[int],
    n_qubits: int,
) -> np.ndarray:
    """把一个门作用到状态向量上，返回新状态（不修改入参）。"""
    state = np.asarray(state, dtype=np.complex128)
    expected = 2**n_qubits
    if state.ndim != 1 or state.shape[0] != expected:
        raise ValueError(f"状态向量形状应为 ({expected},)，收到 {state.shape}")
    qs = tuple(int(q) for q in qubits)
    if len(set(qs)) != len(qs):
        raise ValueError(f"比特不能重复：{qs}")
    if any(q < 0 or q >= n_qubits for q in qs):
        raise ValueError(f"比特下标越界：{qs}（n_qubits={n_qubits}）")

    tensor = state.reshape((2,) * n_qubits)
    matrix = gate_matrix(gate)
    applied = _apply_operator_to_axes(tensor, matrix, qs, n_qubits)
    return applied.reshape(-1)


def circuit_state_vector(circuit: "Circuit") -> np.ndarray:
    """从 ``|0...0>`` 出发计算电路末态。"""
    n_qubits = circuit.n_qubits
    state = np.zeros(2**n_qubits, dtype=np.complex128)
    state[0] = 1.0
    for layer in circuit.layers:
        for op in layer:
            state = apply_gate_to_state(state, op.gate, op.qubits, n_qubits)
    return state


def circuit_unitary(circuit: "Circuit") -> np.ndarray:
    """计算电路的 ``2**n × 2**n`` 幺正矩阵（big-endian 基矢顺序）。"""
    n_qubits = circuit.n_qubits
    dim = 2**n_qubits
    # 每一列是一个基矢经电路演化后的结果；对整个单位阵同时施加门即可。
    tensor = np.eye(dim, dtype=np.complex128).reshape((2,) * n_qubits + (dim,))
    for layer in circuit.layers:
        for op in layer:
            tensor = _apply_operator_to_axes(tensor, gate_matrix(op.gate), op.qubits, n_qubits)
    return tensor.reshape(dim, dim)


def equivalent_up_to_global_phase(
    a: np.ndarray, b: np.ndarray, atol: float = 1e-8
) -> bool:
    """判断两个矩阵是否只相差一个全局相位。"""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape or a.size == 0:
        return False
    index = int(np.argmax(np.abs(b)))
    pivot = b.flat[index]
    if abs(pivot) < atol:
        return False
    phase = a.flat[index] / pivot
    if abs(abs(phase) - 1.0) > 1e-6:
        return False
    return bool(np.allclose(a, phase * b, atol=atol, rtol=0.0))


def circuits_equivalent(a: "Circuit", b: "Circuit", atol: float = 1e-8) -> bool:
    """判断两个电路是否实现相同的幺正演化（模全局相位）。

    若两电路比特数不同，直接返回 ``False``。
    """
    if a.n_qubits != b.n_qubits:
        return False
    return equivalent_up_to_global_phase(circuit_unitary(a), circuit_unitary(b), atol=atol)


def probabilities(circuit: "Circuit") -> np.ndarray:
    """返回计算基下的末态概率分布。"""
    state = circuit_state_vector(circuit)
    return np.abs(state) ** 2
