"""电路重写规则集合。

每条规则都是**语义保持**的等价变换：变换前后电路实现的幺正矩阵只相差一个全局相位。
规则在两个层次上被约束：

1. 适用性 ``is_applicable``：只回答“这个动作在这个位置是否有定义”；
2. 不变量 ``apply`` 之后 ``cost`` 严格下降（门数减少或深度减少）。因此每走一步
   都必然在优化目标上前进，动作掩码只需过滤“不可用”的动作即可，
   不存在“看似合法但毫无收益”的空动作。

动作锚点是 ``(qubit, moment)``：规则从这个位置向外寻找可配对的邻居。
这一设计直接对应原项目的 ``(rule_index, qubit_moment_index)`` 动作表示，
但补上了原项目缺失的适用性判断、掩码与语义验证。

规则顺序 ``RULES`` 就是动作空间中的规则下标，**禁止调换**。
"""

from __future__ import annotations

import abc

from qco.circuit import Circuit, Op
from qco.gates import (
    SELF_INVERSE_TWO_QUBIT,
    gate_matrix,
    match_single_qubit_gate,
)


class RewriteRule(abc.ABC):
    """重写规则的抽象基类。"""

    name: str = ""
    description: str = ""
    rule_index: int = -1

    @abc.abstractmethod
    def is_applicable(self, circuit: Circuit, qubit: int, moment: int) -> bool:
        """判断在 ``(qubit, moment)`` 锚点处该规则是否可用。"""

    @abc.abstractmethod
    def apply(self, circuit: Circuit, qubit: int, moment: int) -> Circuit:
        """应用规则并返回新电路；调用前必须保证 ``is_applicable`` 为真。"""

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"<{type(self).__name__} {self.name!r}>"


class MergeSingleQubitRule(RewriteRule):
    """合并同一比特上相邻的两个单比特门。

    找到锚点之后该比特上的下一个门；两者都是单比特门时，计算乘积
    ``U = U_next @ U_prev``，若它（模全局相位）等于门库中的某个门则合并，
    等于单位阵则两个门一起删除。例如 ``T·T → S``、``H·H → 删除``、``Z·Y → X``。
    """

    name = "merge_single_qubit"
    description = "合并同一比特上相邻的两个单比特门"
    rule_index = 0

    def _merged_gate(self, circuit: Circuit, qubit: int, moment: int) -> tuple[int, Op, str] | None:
        if moment < 0 or moment >= circuit.depth:
            return None
        first = circuit.op_on(qubit, moment)
        if first is None or first.num_qubits != 1:
            return None
        nxt = circuit.next_op_on(qubit, moment)
        if nxt is None:
            return None
        second_moment, second = nxt
        if second.num_qubits != 1:
            return None
        product = gate_matrix(second.gate) @ gate_matrix(first.gate)
        merged = match_single_qubit_gate(product)
        if merged is None:
            return None
        return second_moment, second, merged

    def is_applicable(self, circuit: Circuit, qubit: int, moment: int) -> bool:
        return self._merged_gate(circuit, qubit, moment) is not None

    def apply(self, circuit: Circuit, qubit: int, moment: int) -> Circuit:
        result = self._merged_gate(circuit, qubit, moment)
        if result is None:
            raise ValueError("MergeSingleQubitRule 在此位置不可用")
        second_moment, second, merged = result
        first = circuit.op_on(qubit, moment)
        assert first is not None  # _merged_gate 已保证

        layers = [list(layer) for layer in circuit.layers]
        layers[moment].remove(first)
        layers[second_moment].remove(second)
        if merged != "I":
            layers[moment].append(Op(merged, (qubit,)))
        return Circuit(circuit.n_qubits, layers)


class CancelTwoQubitRule(RewriteRule):
    """抵消同一有序比特对上相邻的两个相同自逆二比特门。

    ``CNOT``、``CZ``、``SWAP`` 都是自逆门，连续作用两次等于单位阵；
    “相邻”指两个门之间没有任何门触及这两个比特中的任何一个。
    """

    name = "cancel_two_qubit"
    description = "抵消相邻的相同自逆二比特门（CNOT/CZ/SWAP）"
    rule_index = 1

    def _pair(self, circuit: Circuit, qubit: int, moment: int) -> tuple[Op, int, Op] | None:
        if moment < 0 or moment >= circuit.depth:
            return None
        first = circuit.op_on(qubit, moment)
        if first is None or first.num_qubits != 2 or first.gate not in SELF_INVERSE_TWO_QUBIT:
            return None
        nexts = [circuit.next_op_on(q, moment) for q in first.qubits]
        if any(item is None for item in nexts):
            return None
        (m_a, op_a), (m_b, op_b) = nexts  # type: ignore[misc]
        if m_a != m_b or op_a != first or op_b != first:
            return None
        return first, m_a, op_a

    def is_applicable(self, circuit: Circuit, qubit: int, moment: int) -> bool:
        return self._pair(circuit, qubit, moment) is not None

    def apply(self, circuit: Circuit, qubit: int, moment: int) -> Circuit:
        result = self._pair(circuit, qubit, moment)
        if result is None:
            raise ValueError("CancelTwoQubitRule 在此位置不可用")
        first, second_moment, _ = result
        layers = [list(layer) for layer in circuit.layers]
        layers[moment].remove(first)
        layers[second_moment].remove(first)
        return Circuit(circuit.n_qubits, layers)


class MoveEarlierRule(RewriteRule):
    """把一个独占时隙的门前移到最早允许的位置（深度 -1）。

    门只能在“其所有比特都空闲”的时隙之间移动，因此各比特上的门顺序不变，
    电路语义不变；原时隙变空后被丢弃，深度减一。
    """

    name = "move_earlier"
    description = "把独占时隙的门前移到最早空闲位置以压缩深度"
    rule_index = 2

    def _target(self, circuit: Circuit, qubit: int, moment: int) -> tuple[Op, int] | None:
        if moment < 0 or moment >= circuit.depth:
            return None
        op = circuit.op_on(qubit, moment)
        if op is None or len(circuit.layers[moment]) != 1:
            return None
        position = moment - 1
        while position >= 0 and not any(
            q in circuit.layer_qubits(position) for q in op.qubits
        ):
            position -= 1
        target = position + 1
        if target >= moment:
            return None
        return op, target

    def is_applicable(self, circuit: Circuit, qubit: int, moment: int) -> bool:
        return self._target(circuit, qubit, moment) is not None

    def apply(self, circuit: Circuit, qubit: int, moment: int) -> Circuit:
        result = self._target(circuit, qubit, moment)
        if result is None:
            raise ValueError("MoveEarlierRule 在此位置不可用")
        op, target = result
        layers = [list(layer) for layer in circuit.layers]
        layers[moment].remove(op)
        layers[target].append(op)
        return Circuit(circuit.n_qubits, layers)


class MoveLaterRule(RewriteRule):
    """把一个独占时隙的门后移到最晚允许的位置（深度 -1）。

    与 :class:`MoveEarlierRule` 对称；后移同样只跨越其所有比特均空闲的时隙。
    """

    name = "move_later"
    description = "把独占时隙的门后移到最晚空闲位置以压缩深度"
    rule_index = 3

    def _target(self, circuit: Circuit, qubit: int, moment: int) -> tuple[Op, int] | None:
        if moment < 0 or moment >= circuit.depth:
            return None
        op = circuit.op_on(qubit, moment)
        if op is None or len(circuit.layers[moment]) != 1:
            return None
        position = moment + 1
        while position < circuit.depth and not any(
            q in circuit.layer_qubits(position) for q in op.qubits
        ):
            position += 1
        target = position - 1
        if target <= moment:
            return None
        return op, target

    def is_applicable(self, circuit: Circuit, qubit: int, moment: int) -> bool:
        return self._target(circuit, qubit, moment) is not None

    def apply(self, circuit: Circuit, qubit: int, moment: int) -> Circuit:
        result = self._target(circuit, qubit, moment)
        if result is None:
            raise ValueError("MoveLaterRule 在此位置不可用")
        op, target = result
        layers = [list(layer) for layer in circuit.layers]
        layers[moment].remove(op)
        layers[target].append(op)
        return Circuit(circuit.n_qubits, layers)


#: 规则顺序即动作空间中的规则下标，禁止调换。
RULES: tuple[RewriteRule, ...] = (
    MergeSingleQubitRule(),
    CancelTwoQubitRule(),
    MoveEarlierRule(),
    MoveLaterRule(),
)

RULES_BY_NAME: dict[str, RewriteRule] = {rule.name: rule for rule in RULES}


def rule_names() -> tuple[str, ...]:
    """返回规则名元组（按动作空间顺序）。"""
    return tuple(rule.name for rule in RULES)


def applicable_actions(circuit: Circuit) -> list[tuple[int, int, int]]:
    """列出当前电路上所有可用的 ``(rule_index, qubit, moment)`` 动作。"""
    actions: list[tuple[int, int, int]] = []
    for rule in RULES:
        for qubit in range(circuit.n_qubits):
            for moment in range(circuit.depth):
                if rule.is_applicable(circuit, qubit, moment):
                    actions.append((rule.rule_index, qubit, moment))
    return actions
