"""不可变的量子电路表示。

设计要点
--------
- 电路由若干**时隙（layer / moment）**组成，每个时隙内不同比特上最多一个门，
  同一时隙内任意两个门作用于不相交的比特集合，因此它们对易；
- :class:`Op` 与 :class:`Circuit` 都不可变，重写规则一律返回新电路。这避免了原项目
  “规则函数原地改电路、状态与电路不一致”的一类 bug，也让电路可以作为字典键、
  便于做搜索与单元测试；
- 空时隙不会保留：构造时自动丢弃，因此 ``depth == len(layers)`` 始终成立。

与 cirq 的互操作是**可选**的：安装 cirq 后可导出为 ``cirq.Circuit`` 画图或交叉验证，
核心逻辑与模拟器不依赖 cirq。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Iterable, Iterator, Sequence

import numpy as np

from qco.gates import GATES, gate_num_qubits

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查
    AnyCirq = object


@dataclass(frozen=True)
class Op:
    """一个量子门操作。

    Attributes
    ----------
    gate
        门名，必须是 :data:`qco.gates.GATES` 中的键。
    qubits
        作用比特；``CNOT`` 的顺序是 ``(control, target)``。
    """

    gate: str
    qubits: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.gate not in GATES:
            raise ValueError(f"未知门名：{self.gate!r}；可用门：{tuple(GATES)}")
        expected = gate_num_qubits(self.gate)
        if len(self.qubits) != expected:
            raise ValueError(
                f"门 {self.gate} 需要 {expected} 个比特，收到 {self.qubits}"
            )
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError(f"比特不能重复：{self.qubits}")
        if any(q < 0 for q in self.qubits):
            raise ValueError(f"比特下标不能为负：{self.qubits}")

    @property
    def num_qubits(self) -> int:
        """作用比特数。"""
        return len(self.qubits)

    def label(self) -> str:
        """人类可读标签，例如 ``X(q0)``、``CNOT(q1→q2)``。"""
        if self.num_qubits == 1:
            return f"{self.gate}(q{self.qubits[0]})"
        return f"{self.gate}(" + ",".join(f"q{q}" for q in self.qubits) + ")"


def _normalize_layers(
    n_qubits: int, layers: Iterable[Iterable[Op]]
) -> tuple[tuple[Op, ...], ...]:
    """校验并冻结层结构：检查越界、同层比特冲突，并丢弃空层。"""
    if n_qubits <= 0:
        raise ValueError(f"n_qubits 必须为正整数，收到 {n_qubits}")
    frozen: list[tuple[Op, ...]] = []
    for index, layer in enumerate(layers):
        ops: list[Op] = []
        used: set[int] = set()
        for op in layer:
            if not isinstance(op, Op):
                raise TypeError(f"第 {index} 层包含非 Op 对象：{op!r}")
            for qubit in op.qubits:
                if qubit >= n_qubits:
                    raise ValueError(
                        f"第 {index} 层的比特 {qubit} 越界（n_qubits={n_qubits}）"
                    )
                if qubit in used:
                    raise ValueError(
                        f"第 {index} 层比特 {qubit} 被多个门同时占用：{ops + [op]}"
                    )
                used.add(qubit)
            ops.append(op)
        if ops:
            frozen.append(tuple(ops))
    return tuple(frozen)


class Circuit:
    """不可变的量子电路。

    Parameters
    ----------
    n_qubits
        量子比特数。
    layers
        可迭代的层，每层是若干 :class:`Op`；空层会被自动丢弃。
    """

    __slots__ = ("n_qubits", "layers", "__dict__")

    def __init__(self, n_qubits: int, layers: Iterable[Iterable[Op]] = ()) -> None:
        self.n_qubits = int(n_qubits)
        self.layers = _normalize_layers(self.n_qubits, layers)

    # ------------------------------------------------------------------
    # 基本属性
    # ------------------------------------------------------------------
    @property
    def depth(self) -> int:
        """电路深度（时隙数）。"""
        return len(self.layers)

    @property
    def num_ops(self) -> int:
        """门的总数。"""
        return sum(len(layer) for layer in self.layers)

    @property
    def num_single_qubit_ops(self) -> int:
        """单比特门数量。"""
        return sum(1 for layer in self.layers for op in layer if op.num_qubits == 1)

    @property
    def num_two_qubit_ops(self) -> int:
        """二比特门数量。"""
        return sum(1 for layer in self.layers for op in layer if op.num_qubits == 2)

    def cost(self, lambda_depth: float = 0.5) -> float:
        """优化目标代价：``门数量 + lambda_depth × 深度``。"""
        return float(self.num_ops) + float(lambda_depth) * float(self.depth)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def layer(self, index: int) -> tuple[Op, ...]:
        """返回第 ``index`` 层的门元组。"""
        return self.layers[index]

    def layer_qubits(self, index: int) -> frozenset[int]:
        """返回第 ``index`` 层占用的比特集合。"""
        return frozenset(q for op in self.layers[index] for q in op.qubits)

    def op_on(self, qubit: int, layer: int) -> Op | None:
        """返回第 ``layer`` 层中作用于 ``qubit`` 的门；没有则返回 ``None``。"""
        if not (0 <= layer < self.depth):
            return None
        for op in self.layers[layer]:
            if qubit in op.qubits:
                return op
        return None

    def next_op_on(self, qubit: int, layer: int) -> tuple[int, Op] | None:
        """返回 ``layer`` 之后（严格大于）该比特上的下一个门及其层号。"""
        for index in range(layer + 1, self.depth):
            op = self.op_on(qubit, index)
            if op is not None:
                return index, op
        return None

    def prev_op_on(self, qubit: int, layer: int) -> tuple[int, Op] | None:
        """返回 ``layer`` 之前（严格小于）该比特上的上一个门及其层号。"""
        for index in range(layer - 1, -1, -1):
            op = self.op_on(qubit, index)
            if op is not None:
                return index, op
        return None

    def qubit_ops(self, qubit: int) -> list[tuple[int, Op]]:
        """返回某比特按时间顺序排列的 ``(层号, 门)`` 列表。"""
        result: list[tuple[int, Op]] = []
        for index, layer in enumerate(self.layers):
            for op in layer:
                if qubit in op.qubits:
                    result.append((index, op))
        return result

    def iter_ops(self) -> Iterator[tuple[int, Op]]:
        """按时间顺序迭代 ``(层号, 门)``。"""
        for index, layer in enumerate(self.layers):
            for op in layer:
                yield index, op

    # ------------------------------------------------------------------
    # 变换
    # ------------------------------------------------------------------
    def with_layers(self, layers: Iterable[Iterable[Op]]) -> "Circuit":
        """用新的层结构构造电路（会重新校验并丢弃空层）。"""
        return Circuit(self.n_qubits, layers)

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """转换为可 JSON 序列化的字典。"""
        return {
            "n_qubits": self.n_qubits,
            "layers": [
                [[op.gate, list(op.qubits)] for op in layer] for layer in self.layers
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Circuit":
        """从 :meth:`to_dict` 的结果恢复电路。"""
        n_qubits = int(data["n_qubits"])
        layers = [
            [Op(str(gate), tuple(int(q) for q in qubits)) for gate, qubits in layer]
            for layer in data["layers"]
        ]
        return cls(n_qubits, layers)

    # ------------------------------------------------------------------
    # cirq 互操作
    # ------------------------------------------------------------------
    def to_cirq(self):
        """导出为 ``cirq.Circuit``；cirq 未安装时抛 ``RuntimeError``。"""
        try:
            import cirq
        except ImportError as exc:  # pragma: no cover - 可选依赖
            raise RuntimeError("需要安装 cirq：pip install cirq") from exc

        qubits = cirq.LineQubit.range(self.n_qubits)
        moments = []
        for layer in self.layers:
            ops = []
            for op in layer:
                gate = GATES[op.gate].cirq_factory()
                ops.append(gate.on(*(qubits[q] for q in op.qubits)))
            moments.append(cirq.Moment(ops))
        return cirq.Circuit.from_moments(*moments) if moments else cirq.Circuit()

    @classmethod
    def from_cirq(cls, circuit) -> "Circuit":
        """从 ``cirq.Circuit`` 导入；比特按 ``sorted(circuit.all_qubits())`` 编号。"""
        qubits = sorted(circuit.all_qubits())
        index = {qubit: i for i, qubit in enumerate(qubits)}
        name_by_gate = {}
        for name, spec in GATES.items():
            try:
                key = str(spec.cirq_factory())
            except RuntimeError:  # pragma: no cover - 未安装 cirq
                raise
            name_by_gate[key] = name

        layers = []
        for moment in circuit.moments:
            ops = []
            for operation in moment.operations:
                key = str(operation.gate)
                gate_name = name_by_gate.get(key)
                if gate_name is None:
                    raise ValueError(f"无法识别的 cirq 门：{operation.gate!r}")
                op_qubits = tuple(index[q] for q in operation.qubits)
                ops.append(Op(gate_name, op_qubits))
            if ops:
                layers.append(ops)
        return cls(len(qubits), layers)

    # ------------------------------------------------------------------
    # 语义
    # ------------------------------------------------------------------
    @cached_property
    def unitary(self) -> np.ndarray:
        """电路的幺正矩阵；延迟计算并缓存。"""
        from qco.simulator import circuit_unitary

        return circuit_unitary(self)

    def equivalent(self, other: "Circuit", atol: float = 1e-8) -> bool:
        """判断两个电路是否模全局相位等价。"""
        from qco.simulator import circuits_equivalent

        return circuits_equivalent(self, other, atol=atol)

    # ------------------------------------------------------------------
    # 展示
    # ------------------------------------------------------------------
    def to_text(self) -> str:
        """返回可读文本电路图（优先 cirq，缺失时退化为门列表）。"""
        try:
            return str(self.to_cirq())
        except RuntimeError:  # pragma: no cover - 未安装 cirq
            return self.to_ascii()

    def to_ascii(self) -> str:
        """不依赖第三方库的简单 ASCII 电路图。"""
        depth = self.depth
        if depth == 0:
            return "<empty circuit>"
        columns: list[list[str]] = []
        for index in range(depth):
            entries = ["" for _ in range(self.n_qubits)]
            for op in self.layers[index]:
                if op.num_qubits == 1:
                    entries[op.qubits[0]] = op.gate
                else:
                    head, tail = op.qubits
                    marker = "●" if op.gate == "CNOT" else op.gate[0]
                    entries[head] = marker
                    entries[tail] = "X" if op.gate == "CNOT" else marker
            width = max(len(entry) for entry in entries) if any(entries) else 1
            columns.append([entry.center(width) for entry in entries])
        lines = []
        for qubit in range(self.n_qubits):
            row = "-".join(column[qubit] for column in columns)
            lines.append(f"q{qubit}: {row}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 协议
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.depth

    def __iter__(self) -> Iterator[tuple[Op, ...]]:
        return iter(self.layers)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Circuit):
            return NotImplemented
        return self.n_qubits == other.n_qubits and self.layers == other.layers

    def __hash__(self) -> int:
        return hash((self.n_qubits, self.layers))

    def __repr__(self) -> str:
        return f"Circuit(n_qubits={self.n_qubits}, depth={self.depth}, ops={self.num_ops})"

    def __str__(self) -> str:
        return self.to_text()
