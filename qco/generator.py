"""随机量子电路生成与数据集存取。

生成策略
--------
逐层生成：把比特随机排序后依次尝试放置门；若当前比特未被占用，
以概率 ``p_two`` 尝试与相邻比特 ``(q, q+1)`` 放一个二比特门，否则放随机单比特门。
每层至少放一个门，因此 ``depth`` 就是真实层数。

固定随机种子即可复现实验数据；训练集与评估集使用不同种子，避免数据泄漏。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from qco.circuit import Circuit, Op
from qco.gates import SINGLE_QUBIT_GATE_NAMES, TWO_QUBIT_GATE_NAMES


def random_circuit(
    n_qubits: int,
    depth: int,
    rng: np.random.Generator,
    p_two: float = 0.15,
) -> Circuit:
    """生成一个随机量子电路。

    Parameters
    ----------
    n_qubits
        比特数。
    depth
        电路层数（深度）。
    rng
        ``numpy.random.Generator`` 实例。
    p_two
        在每个未被占用的比特上尝试放置二比特门的概率。

    Returns
    -------
    Circuit
        深度恰为 ``depth``、每层至少有一个门的随机电路。
    """
    if n_qubits <= 1:
        raise ValueError("n_qubits 至少为 2，才能生成含二比特门的随机电路")
    if depth <= 0:
        raise ValueError(f"depth 必须为正整数，收到 {depth}")
    if not (0.0 <= p_two <= 1.0):
        raise ValueError(f"p_two 必须在 [0, 1] 内，收到 {p_two}")

    single = np.asarray(SINGLE_QUBIT_GATE_NAMES)
    double = np.asarray(TWO_QUBIT_GATE_NAMES)
    layers: list[list[Op]] = []
    for _ in range(depth):
        layer: list[Op] = []
        used: set[int] = set()
        order = rng.permutation(n_qubits)
        for qubit in order:
            qubit = int(qubit)
            if qubit in used:
                continue
            can_pair = qubit + 1 < n_qubits and (qubit + 1) not in used
            if can_pair and rng.random() < p_two:
                gate = str(double[int(rng.integers(len(double)))])
                layer.append(Op(gate, (qubit, qubit + 1)))
                used.update((qubit, qubit + 1))
            else:
                gate = str(single[int(rng.integers(len(single)))])
                layer.append(Op(gate, (qubit,)))
                used.add(qubit)
        if not layer:  # 理论上不会发生（n_qubits >= 2），保留兜底
            gate = str(single[int(rng.integers(len(single)))])
            layer.append(Op(gate, (int(rng.integers(n_qubits)),)))
        layers.append(layer)
    return Circuit(n_qubits, layers)


def random_circuits(
    count: int,
    n_qubits: int,
    depth: int,
    seed: int = 0,
    p_two: float = 0.15,
) -> list[Circuit]:
    """批量生成随机电路（同一 seed 结果可复现）。"""
    if count <= 0:
        raise ValueError(f"count 必须为正整数，收到 {count}")
    rng = np.random.default_rng(int(seed))
    return [random_circuit(n_qubits, depth, rng, p_two=p_two) for _ in range(int(count))]


def save_circuits(circuits: Sequence[Circuit], path: str | Path) -> None:
    """把电路列表保存为 JSON 数据集。"""
    circuits = list(circuits)
    if not circuits:
        raise ValueError("电路列表不能为空")
    n_qubits = circuits[0].n_qubits
    payload = {
        "meta": {
            "count": len(circuits),
            "n_qubits": n_qubits,
            "depth": max(circuit.depth for circuit in circuits),
        },
        "circuits": [circuit.to_dict() for circuit in circuits],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_circuits(path: str | Path) -> list[Circuit]:
    """从 JSON 数据集读取电路列表。"""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [Circuit.from_dict(item) for item in payload["circuits"]]
