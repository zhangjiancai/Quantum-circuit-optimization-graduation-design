"""量子门库：门矩阵、门库顺序与“模全局相位”的单比特门匹配。

本模块是所有模块共享的门字典。``GATE_NAMES`` 的顺序决定观测张量的通道顺序，
也决定重写规则与网络输入通道数，**禁止随意调整**。

约定
----
- 基矢顺序为 ``|0...0>, |0...1>, ..., |1...1>``，即比特 0 是最高位（big-endian）；
- 单比特门矩阵为 2×2，二比特门矩阵为 4×4，比特顺序与 :class:`qco.circuit.Op`
  的 ``qubits`` 一致（``CNOT`` 为 ``(control, target)``）；
- 全局相位不可观测，因此“门等价”与“电路等价”都按模全局相位判定；
  ``"I"`` 是特殊名字，仅用于表示合并结果为单位阵，不是可放置的门。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

_SQRT1_2 = 1.0 / np.sqrt(2.0)


@dataclass(frozen=True, eq=False)
class GateSpec:
    """单个门的静态描述。"""

    name: str
    num_qubits: int
    matrix: np.ndarray
    label: str
    cirq_factory: Callable[[], Any] | None = None


# ---------------------------------------------------------------------------
# 门矩阵定义
# ---------------------------------------------------------------------------

_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_Z = np.diag(np.array([1.0, -1.0], dtype=np.complex128))
_H = _SQRT1_2 * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
_S = np.diag(np.array([1.0, 1.0j], dtype=np.complex128))
_SDAG = _S.conj().T.copy()
_T = np.diag(np.array([1.0, np.exp(1.0j * np.pi / 4.0)], dtype=np.complex128))
_TDAG = _T.conj().T.copy()
_CNOT = np.array(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 1.0],
     [0.0, 0.0, 1.0, 0.0]],
    dtype=np.complex128,
)
_CZ = np.diag(np.array([1.0, 1.0, 1.0, -1.0], dtype=np.complex128))
_SWAP = np.array(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=np.complex128,
)

IDENTITY_2 = np.eye(2, dtype=np.complex128)


def _make_cirq_gate(name: str) -> Any:
    """延迟构造 cirq 门对象；cirq 为可选依赖。"""
    try:
        import cirq
    except ImportError as exc:  # pragma: no cover - 依赖缺失时给出明确提示
        raise RuntimeError("需要安装 cirq 才能导出 cirq 电路：pip install cirq") from exc

    table = {
        "X": cirq.X,
        "Y": cirq.Y,
        "Z": cirq.Z,
        "H": cirq.H,
        "S": cirq.S,
        "Sdag": cirq.S**-1,
        "T": cirq.T,
        "Tdag": cirq.T**-1,
        "CNOT": cirq.CNOT,
        "CZ": cirq.CZ,
        "SWAP": cirq.SWAP,
    }
    return table[name]


def _spec(name: str, num_qubits: int, matrix: np.ndarray, label: str) -> GateSpec:
    return GateSpec(
        name=name,
        num_qubits=num_qubits,
        matrix=matrix,
        label=label,
        cirq_factory=lambda n=name: _make_cirq_gate(n),
    )


#: 门名到门描述的映射，顺序即张量通道顺序。
GATES: dict[str, GateSpec] = {
    "X": _spec("X", 1, _X, "X"),
    "Y": _spec("Y", 1, _Y, "Y"),
    "Z": _spec("Z", 1, _Z, "Z"),
    "H": _spec("H", 1, _H, "H"),
    "S": _spec("S", 1, _S, "S"),
    "Sdag": _spec("Sdag", 1, _SDAG, "S†"),
    "T": _spec("T", 1, _T, "T"),
    "Tdag": _spec("Tdag", 1, _TDAG, "T†"),
    "CNOT": _spec("CNOT", 2, _CNOT, "CNOT"),
    "CZ": _spec("CZ", 2, _CZ, "CZ"),
    "SWAP": _spec("SWAP", 2, _SWAP, "SWAP"),
}

GATE_NAMES: tuple[str, ...] = tuple(GATES)
SINGLE_QUBIT_GATE_NAMES: tuple[str, ...] = tuple(
    name for name, spec in GATES.items() if spec.num_qubits == 1
)
TWO_QUBIT_GATE_NAMES: tuple[str, ...] = tuple(
    name for name, spec in GATES.items() if spec.num_qubits == 2
)
SELF_INVERSE_TWO_QUBIT: tuple[str, ...] = ("CNOT", "CZ", "SWAP")

#: 门名 -> 观测张量通道下标。
GATE_CHANNEL_INDEX: dict[str, int] = {name: i for i, name in enumerate(GATE_NAMES)}
CONTROL_CHANNEL: int = len(GATE_NAMES)
TARGET_CHANNEL: int = len(GATE_NAMES) + 1
IDLE_CHANNEL: int = len(GATE_NAMES) + 2
N_CHANNELS: int = len(GATE_NAMES) + 3


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def gate_matrix(gate: str) -> np.ndarray:
    """返回门的幺正矩阵（复数副本，调用方不要原地修改）。"""
    try:
        return GATES[gate].matrix.copy()
    except KeyError as exc:
        raise KeyError(f"未知门名：{gate!r}；可用门：{GATE_NAMES}") from exc


def gate_num_qubits(gate: str) -> int:
    """返回门作用的比特数。"""
    try:
        return GATES[gate].num_qubits
    except KeyError as exc:
        raise KeyError(f"未知门名：{gate!r}；可用门：{GATE_NAMES}") from exc


def is_unitary(matrix: np.ndarray, atol: float = 1e-8) -> bool:
    """判断方阵是否为幺正矩阵。"""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    dim = matrix.shape[0]
    product = matrix @ matrix.conj().T
    return bool(np.allclose(product, np.eye(dim), atol=atol, rtol=0.0))


def global_phase_equal(a: np.ndarray, b: np.ndarray, atol: float = 1e-8) -> bool:
    """判断两个矩阵是否只相差一个全局相位（``a = e^{iφ} b``）。"""
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


def match_single_qubit_gate(matrix: np.ndarray, atol: float = 1e-8) -> str | None:
    """把 2×2 幺正矩阵在模全局相位意义下匹配到单比特门库。

    Returns
    -------
    str | None
        命中时返回门名（``"X"`` 等）；若等价于单位阵（即两个门可整体删除）
        返回 ``"I"``；否则返回 ``None``。
    """
    matrix = np.asarray(matrix, dtype=np.complex128)
    if matrix.shape != (2, 2):
        return None
    if not is_unitary(matrix, atol=atol):
        return None
    if global_phase_equal(matrix, IDENTITY_2, atol=atol):
        return "I"
    for name in SINGLE_QUBIT_GATE_NAMES:
        if global_phase_equal(matrix, GATES[name].matrix, atol=atol):
            return name
    return None


def gate_matrix_on(gate: str, qubits: Sequence[int], n_qubits: int) -> np.ndarray:
    """把门矩阵嵌入到 ``n_qubits`` 个比特的希尔伯特空间。

    返回 ``2**n_qubits × 2**n_qubits`` 的矩阵，未涉及比特上为单位阵。
    比特 0 为最高位；``qubits`` 的顺序与门矩阵的行列顺序一致。
    """
    # 延迟导入，避免 gates <-> simulator 的循环依赖。
    from qco.simulator import apply_gate_to_state

    qs = tuple(int(q) for q in qubits)
    spec = GATES[gate]
    if len(qs) != spec.num_qubits:
        raise ValueError(f"门 {gate} 需要 {spec.num_qubits} 个比特，收到 {qs}")
    if len(set(qs)) != len(qs):
        raise ValueError(f"比特不能重复：{qs}")
    if any(q < 0 or q >= n_qubits for q in qs):
        raise ValueError(f"比特下标越界：{qs}（n_qubits={n_qubits}）")

    dim = 2**n_qubits
    result = np.empty((dim, dim), dtype=np.complex128)
    for column in range(dim):
        basis = np.zeros(dim, dtype=np.complex128)
        basis[column] = 1.0
        result[:, column] = apply_gate_to_state(basis, gate, qs, n_qubits)
    return result
