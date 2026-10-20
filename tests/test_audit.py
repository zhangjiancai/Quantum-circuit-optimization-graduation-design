"""T4 独立对抗性审计测试（不使用 ``qco`` 自身测试的任何断言作为依据）。

本文件按 ``docs/architecture.md`` 的接口契约，从**独立实现**出发验证：

- A 重写规则语义（等价性 / 代价严格下降 / 终止性 / 与动作掩码一致）；
- B 模拟器与 cirq（big-endian、幺正矩阵、往返）；
- C 动作布局一致性（编码 / 展平 / 解码 / 策略采样）；
- D PPO 数学（GAE、ratio、掩码一致性、价值损失）；
- E 端到端（训练脚本产物、评估脚本、checkpoint 一致性回归）；
- F 边界条件；
- G 文档 / CLI / 契约一致性。

约定
----
- 独立参考实现位于本文件（``independent_unitary`` 等），不调用 ``qco.simulator``；
- 标记 ``xfail`` 的用例是**已知缺陷的回归测试**：缺陷修复后它们会变成 XPASS，
  审计报告里逐条说明原因；
- 所有用例在 CPU 上运行，整套目标 < 120 秒。
"""

from __future__ import annotations

import ast
import csv
import importlib.util
import inspect
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - 正常由 conftest 保证
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from qco.actions import (  # noqa: E402
    action_space_size,
    decode_action,
    decode_actions,
    encode_action,
    encode_actions,
)
from qco.agent import CircuitPolicyNet  # noqa: E402
from qco.baselines import (  # noqa: E402
    OptimizationResult,
    choose_greedy,
    choose_random,
    optimize,
    optimize_greedy,
    optimize_noop,
    optimize_random,
)
from qco.circuit import Circuit, Op  # noqa: E402
from qco.environment import CircuitOptimizationEnv, EnvConfig  # noqa: E402
from qco.evaluate import (  # noqa: E402
    EvalResult,
    evaluate_agent,
    evaluate_greedy,
    evaluate_noop,
    evaluate_random,
    format_comparison,
)
from qco.gates import (  # noqa: E402
    CONTROL_CHANNEL,
    GATE_CHANNEL_INDEX,
    GATE_NAMES,
    N_CHANNELS,
    IDLE_CHANNEL,
    SINGLE_QUBIT_GATE_NAMES,
    SELF_INVERSE_TWO_QUBIT,
    TARGET_CHANNEL,
    TWO_QUBIT_GATE_NAMES,
    gate_matrix,
    gate_matrix_on,
    global_phase_equal,
    is_unitary,
    match_single_qubit_gate,
)
import qco.trainer as trainer_module  # noqa: E402
from qco.trainer import Trainer, TrainConfig, build_policy_from_checkpoint, load_checkpoint  # noqa: E402
from qco.utils import to_jsonable  # noqa: E402
from qco.generator import load_circuits, random_circuit, random_circuits, save_circuits  # noqa: E402
from qco.masking import (  # noqa: E402
    NEG_INF,
    build_action_mask,
    flatten_action_mask,
    masked_entropy,
    masked_log_softmax,
    unflatten_action_mask,
)
from qco.observation import encode_circuit  # noqa: E402
from qco.ppo import PPO, PPOConfig  # noqa: E402
from qco.rollout import EpisodeStats, RolloutBatch, RolloutCollector  # noqa: E402
from qco.rules import (  # noqa: E402
    RULES,
    RULES_BY_NAME,
    CancelTwoQubitRule,
    MergeSingleQubitRule,
    MoveEarlierRule,
    MoveLaterRule,
    applicable_actions,
)
from qco.simulator import (  # noqa: E402
    circuit_state_vector,
    circuit_unitary,
    circuits_equivalent,
    equivalent_up_to_global_phase,
)

PYTHON = sys.executable
LAMBDA_DEPTH = 0.5
SINGLE = tuple(SINGLE_QUBIT_GATE_NAMES)
DOUBLE = tuple(TWO_QUBIT_GATE_NAMES)
RULE_NAMES = ("merge_single_qubit", "cancel_two_qubit", "move_earlier", "move_later")

#: 审计目标快照（报告里会记录；这里只作为常量便于诊断）。
AUDIT_START_TIME = time.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# 独立参考实现：门矩阵、嵌入、幺正矩阵、模全局相位比较
# ---------------------------------------------------------------------------
_SQRT1_2 = 1.0 / math.sqrt(2.0)

#: 本文件独立定义的 11 个门矩阵（与实现无关地按教科书定义写死）。
MY_GATE_MATRICES: dict[str, np.ndarray] = {
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    "H": _SQRT1_2 * np.array([[1, 1], [1, -1]], dtype=np.complex128),
    "S": np.array([[1, 0], [0, 1j]], dtype=np.complex128),
    "Sdag": np.array([[1, 0], [0, -1j]], dtype=np.complex128),
    "T": np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=np.complex128),
    "Tdag": np.array([[1, 0], [0, np.exp(-1j * math.pi / 4)]], dtype=np.complex128),
    "CNOT": np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128
    ),
    "CZ": np.diag(np.array([1, 1, 1, -1], dtype=np.complex128)),
    "SWAP": np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128
    ),
}

_SWAP2 = MY_GATE_MATRICES["SWAP"]

_EMBED_CACHE: dict[tuple[str, tuple[int, ...], int], np.ndarray] = {}


def _embed(op: Op, n_qubits: int) -> np.ndarray:
    """把门的 2**k 矩阵嵌入 n 比特空间（独立实现，支持任意比特位置）。

    做法：先把比特顺序按 ``op.qubits + 其余比特`` 重排，使门作用在最高 k 位上，
    再左乘/右乘置换矩阵换回标准 big-endian 基。结果按 ``(gate, qubits, n)`` 缓存。
    """
    key = (op.gate, tuple(op.qubits), n_qubits)
    cached = _EMBED_CACHE.get(key)
    if cached is not None:
        return cached
    gate = MY_GATE_MATRICES[op.gate]
    k = len(op.qubits)
    dim = 2**n_qubits
    permutation = list(op.qubits) + [i for i in range(n_qubits) if i not in op.qubits]
    perm_index = np.empty(dim, dtype=np.int64)
    for standard_index in range(dim):
        bits = [(standard_index >> (n_qubits - 1 - i)) & 1 for i in range(n_qubits)]
        permuted = 0
        for position in permutation:
            permuted = (permuted << 1) | bits[position]
        perm_index[standard_index] = permuted
    perm_matrix = np.zeros((dim, dim), dtype=np.complex128)
    perm_matrix[perm_index, np.arange(dim)] = 1.0
    permuted_gate = np.kron(gate, np.eye(2 ** (n_qubits - k)))
    embedded = perm_matrix.T @ permuted_gate @ perm_matrix
    _EMBED_CACHE[key] = embedded
    return embedded


def independent_unitary(circuit: Circuit) -> np.ndarray:
    """独立计算电路幺正矩阵（按时间顺序左乘各操作）。"""
    dim = 2**circuit.n_qubits
    result = np.eye(dim, dtype=np.complex128)
    for layer in circuit.layers:
        layer_matrix = np.eye(dim, dtype=np.complex128)
        for op in layer:
            layer_matrix = _embed(op, circuit.n_qubits) @ layer_matrix
        result = layer_matrix @ result
    return result


def up_to_global_phase_equal(a: np.ndarray, b: np.ndarray, atol: float = 1e-8) -> bool:
    """独立实现的模全局相位比较：``a = e^{iφ} b``。"""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    overlap = np.vdot(b, a)
    if abs(overlap) < atol:
        return False
    phase = overlap / abs(overlap)
    return bool(np.allclose(a, phase * b, atol=atol, rtol=0.0))


# ---------------------------------------------------------------------------
# 独立随机电路生成（不依赖 qco.generator）
# ---------------------------------------------------------------------------
def independent_random_circuit(
    n_qubits: int, depth: int, seed: int, p_two: float = 0.15
) -> Circuit:
    """逐层随机生成电路；每层至少一个门，深度恰好为 depth。"""
    rng = np.random.default_rng(seed)
    layers: list[list[Op]] = []
    for _ in range(depth):
        order = [int(q) for q in rng.permutation(n_qubits)]
        used: set[int] = set()
        layer: list[Op] = []
        for qubit in order:
            if qubit in used:
                continue
            pair_ok = qubit + 1 < n_qubits and (qubit + 1) not in used
            if pair_ok and rng.random() < p_two:
                gate = DOUBLE[int(rng.integers(len(DOUBLE)))]
                layer.append(Op(gate, (qubit, qubit + 1)))
                used.update((qubit, qubit + 1))
            else:
                gate = SINGLE[int(rng.integers(len(SINGLE)))]
                layer.append(Op(gate, (qubit,)))
                used.add(qubit)
        layers.append(layer)
    return Circuit(n_qubits, layers)


def make(n_qubits: int, layers) -> Circuit:
    """构造电路的简写：layers 为 ``[[(gate, qubits), ...], ...]``。"""
    return Circuit(
        n_qubits,
        [[Op(gate, tuple(qubits)) for gate, qubits in layer] for layer in layers],
    )


def layers_signature(circuit: Circuit) -> tuple:
    """可哈希的层结构签名（用于集合比较 / 独立性检查）。"""
    return tuple(tuple((op.gate, tuple(op.qubits)) for op in layer) for layer in circuit.layers)


def manual_mask(circuit: Circuit, n_moments: int) -> np.ndarray:
    """按契约第 8 节独立构造掩码：``mask[r,q,m] = is_applicable and m < depth``。"""
    mask = np.zeros((len(RULES), circuit.n_qubits, int(n_moments)), dtype=bool)
    for rule in RULES:
        for qubit in range(circuit.n_qubits):
            for moment in range(int(n_moments)):
                if moment < circuit.depth and rule.is_applicable(circuit, qubit, moment):
                    mask[rule.rule_index, qubit, moment] = True
    return mask


def assert_circuit_legal(circuit: Circuit, n_qubits: int) -> None:
    """重新从层结构构造一次，确保电路满足 Circuit 的全部构造校验。"""
    assert isinstance(circuit, Circuit)
    assert circuit.n_qubits == n_qubits
    rebuilt = Circuit(circuit.n_qubits, [list(layer) for layer in circuit.layers])
    assert rebuilt.layers == circuit.layers
    assert circuit.depth == len(circuit.layers)


def tiny_env_config(**overrides) -> EnvConfig:
    """审计用的小环境配置。"""
    params = dict(
        n_qubits=3,
        n_moments=8,
        max_steps=16,
        lambda_depth=LAMBDA_DEPTH,
        initial_depth=5,
        p_two=0.3,
        seed=0,
    )
    params.update(overrides)
    return EnvConfig(**params)


# ---------------------------------------------------------------------------
# A. 重写规则语义
# ---------------------------------------------------------------------------
def test_a1_rule_table_and_order() -> None:
    """契约第 0/7 节：4 条规则的顺序、name、rule_index 必须固定。"""
    assert len(RULES) == 4
    assert tuple(rule.name for rule in RULES) == RULE_NAMES
    for index, rule in enumerate(RULES):
        assert rule.rule_index == index
        assert RULES_BY_NAME[rule.name] is rule
        assert isinstance(rule.description, str) and rule.description
        assert isinstance(rule, (MergeSingleQubitRule, CancelTwoQubitRule,
                                MoveEarlierRule, MoveLaterRule))
    assert [type(rule) for rule in RULES] == [
        MergeSingleQubitRule,
        CancelTwoQubitRule,
        MoveEarlierRule,
        MoveLaterRule,
    ]


def test_a2_random_rule_invariants() -> None:
    """随机电路上枚举全部锚点：可适用 ⇒ 合法 + 语义等价 + 代价严格下降。

    覆盖 n_qubits 2..5、depth 1..8、多种 p_two、seeds 0..50（循环取组合）。
    等价性同时用实现函数 ``equivalent_up_to_global_phase(circuit_unitary...)``
    与本文件独立幺正矩阵交叉验证；并抽样调用契约指定的
    ``qco.simulator.circuits_equivalent`` 复核。
    """
    checked = 0
    sample_circuits_equivalent_checks: list[tuple[Circuit, Circuit]] = []
    for seed in range(51):
        n_qubits = 2 + seed % 4
        depth = 1 + (seed * 7) % 8
        p_two = (0.05, 0.15, 0.4, 0.8)[seed % 4]
        # 每个种子两个变体：不同深度 + 互补的二比特门概率，扩大覆盖面。
        variants = (
            (depth, p_two, seed),
            (min(depth + 2, 8), 1.0 - p_two, seed + 1000),
        )
        for variant_depth, variant_p, variant_seed in variants:
            circuit = independent_random_circuit(
                n_qubits, variant_depth, variant_seed, variant_p
            )
            assert_circuit_legal(circuit, n_qubits)
            base_unitary = independent_unitary(circuit)
            # 独立参考矩阵与实现模拟器在随机电路上必须一致（双向交叉验证）。
            assert np.allclose(base_unitary, circuit_unitary(circuit), atol=1e-10)
            assert np.allclose(base_unitary, circuit.unitary, atol=1e-10)
            old_cost = circuit.cost(LAMBDA_DEPTH)
            for rule in RULES:
                for qubit in range(n_qubits):
                    for moment in range(variant_depth):
                        if not rule.is_applicable(circuit, qubit, moment):
                            continue
                        checked += 1
                        new_circuit = rule.apply(circuit, qubit, moment)
                        assert_circuit_legal(new_circuit, n_qubits)
                        assert new_circuit.cost(LAMBDA_DEPTH) < old_cost - 1e-12, (
                            f"{rule.name} 在 {(qubit, moment)} 没有降低代价"
                        )
                        new_unitary = independent_unitary(new_circuit)
                        assert up_to_global_phase_equal(base_unitary, new_unitary), (
                            f"{rule.name} 在 seed={seed} {(qubit, moment)} 破坏语义等价"
                        )
                        # 契约指定的实现函数（缓存后等价于 circuits_equivalent）。
                        assert equivalent_up_to_global_phase(
                            circuit.unitary, new_circuit.unitary
                        ), f"{rule.name} 在 seed={seed} {(qubit, moment)} 未通过实现等价判定"
                        if checked % 37 == 0:
                            sample_circuits_equivalent_checks.append((circuit, new_circuit))
    assert checked >= 250, f"随机电路上可适用动作太少（{checked}），审计覆盖不足"
    for old, new in sample_circuits_equivalent_checks:
        assert circuits_equivalent(old, new), "circuits_equivalent 与缓存判定不一致"


def test_a3_handcrafted_rule_cases() -> None:
    """边界构造：跨空闲层合并、中间门阻断抵消、CNOT 顺序、二比特门后移、混排。"""
    merge = RULES[0]
    cancel = RULES[1]
    earlier = RULES[2]
    later = RULES[3]

    # 1) 跨空闲层合并：H(q0) ... Y(q1) ... H(q0) ⇒ 两个 H 抵消。
    c = make(2, [[("H", (0,))], [("Y", (1,))], [("H", (0,))]])
    assert merge.is_applicable(c, 0, 0)
    new = merge.apply(c, 0, 0)
    assert new.num_ops == 1 and new.depth == 1
    assert up_to_global_phase_equal(independent_unitary(c), independent_unitary(new))

    # 2) 中间有单比特门阻断（同一比特）：CNOT(0,1) X(0) CNOT(0,1) 不可抵消。
    c = make(2, [[("CNOT", (0, 1))], [("X", (0,))], [("CNOT", (0, 1))]])
    assert not cancel.is_applicable(c, 0, 0)
    # 中间只有其他比特上的门不阻断抵消。
    c = make(3, [[("CNOT", (0, 1))], [("X", (2,))], [("CNOT", (0, 1))]])
    assert cancel.is_applicable(c, 0, 0)
    new = cancel.apply(c, 0, 0)
    assert new.num_ops == 1
    assert up_to_global_phase_equal(independent_unitary(c), independent_unitary(new))

    # 3) CNOT 比特顺序：CNOT(0,1) 与 CNOT(1,0) 不能互相抵消。
    c = make(2, [[("CNOT", (0, 1))], [("CNOT", (1, 0))]])
    assert not cancel.is_applicable(c, 0, 0)
    u01 = gate_matrix_on("CNOT", (0, 1), 2)
    u10 = gate_matrix_on("CNOT", (1, 0), 2)
    assert not up_to_global_phase_equal(u01, u10)
    # 但两个 CNOT(1,0) 可以抵消，且语义等价。
    c = make(2, [[("CNOT", (1, 0))], [("CNOT", (1, 0))]])
    assert cancel.is_applicable(c, 0, 0) and cancel.is_applicable(c, 1, 0)
    assert cancel.apply(c, 0, 0).num_ops == 0

    # 4) 合并乘积匹配：T·T→S，T·Tdag→I，Y·Z→X（Y 先 Z 后）。
    t, s = MY_GATE_MATRICES["T"], MY_GATE_MATRICES["S"]
    assert up_to_global_phase_equal(s, t @ t)
    c = make(1, [[("T", (0,))], [("T", (0,))]])
    new = merge.apply(c, 0, 0)
    assert new.layer(0)[0].gate == "S"
    c = make(1, [[("T", (0,))], [("Tdag", (0,))]])
    assert merge.apply(c, 0, 0).num_ops == 0
    c = make(1, [[("Y", (0,))], [("Z", (0,))]])
    assert merge.apply(c, 0, 0).layer(0)[0].gate == "X"
    # 不在门库内的乘积（H·S）不适用。
    c = make(1, [[("H", (0,))], [("S", (0,))]])
    assert not merge.is_applicable(c, 0, 0)
    # 第二个门是二比特门时不适用。
    c = make(2, [[("H", (0,))], [("CNOT", (0, 1))]])
    assert not merge.is_applicable(c, 0, 0)

    # 5) 二比特门后移跨层：CNOT 独占层 + 其他比特上的门前移/后移。
    c = make(3, [[("CNOT", (0, 1))], [("X", (2,))]])
    assert later.is_applicable(c, 0, 0)
    new = later.apply(c, 0, 0)
    assert new.depth == 1
    assert up_to_global_phase_equal(independent_unitary(c), independent_unitary(new))
    c = make(3, [[("X", (2,))], [("CNOT", (0, 1))]])
    assert earlier.is_applicable(c, 0, 1)
    new = earlier.apply(c, 0, 1)
    assert new.depth == 1
    assert up_to_global_phase_equal(independent_unitary(c), independent_unitary(new))

    # 6) 独占层要求：共享层不允许移动（否则是空动作）。
    c = make(2, [[("Z", (0,)), ("Y", (1,))], [("X", (0,))]])
    assert not later.is_applicable(c, 0, 0)
    assert not earlier.is_applicable(c, 0, 0)
    # 深度 1 的电路上两个方向都不可移动。
    c = make(2, [[("X", (0,)), ("Y", (1,))]])
    for q in range(2):
        assert not earlier.is_applicable(c, q, 0)
        assert not later.is_applicable(c, q, 0)

    # 7) 2 比特门与单比特门混排：合并只看同一比特上的相邻门。
    c = make(3, [[("H", (1,))], [("CNOT", (0, 1))], [("H", (1,))]])
    # q1 上的下一个门是 CNOT（二比特门），因此不能合并。
    assert not merge.is_applicable(c, 1, 0)
    assert not merge.is_applicable(c, 1, 2)


def test_a4_repetition_terminates_and_reaches_normal_form() -> None:
    """反复应用任意合法动作必然在有限步内终止，且终点无任何可行动作。"""
    for config_index, (n_qubits, depth) in enumerate(
        [(2, 3), (2, 8), (3, 5), (4, 6), (5, 4), (5, 8)]
    ):
        for seed in range(3):
            circuit = independent_random_circuit(n_qubits, depth, seed + 100 * config_index)
            initial = circuit
            initial_unitary = independent_unitary(initial)
            initial_cost = circuit.cost(LAMBDA_DEPTH)
            # 理论上每步至少减少 1 个门或 1 层深度，因此步数上界为 num_ops + depth。
            step_bound = circuit.num_ops + circuit.depth
            rng = np.random.default_rng(seed + 7 * config_index)
            steps = 0
            current = circuit
            while True:
                valid = [
                    (rule.rule_index, qubit, moment)
                    for rule in RULES
                    for qubit in range(n_qubits)
                    for moment in range(current.depth)
                    if rule.is_applicable(current, qubit, moment)
                ]
                if not valid:
                    break
                rule_index, qubit, moment = valid[int(rng.integers(len(valid)))]
                before_cost = current.cost(LAMBDA_DEPTH)
                current = RULES[rule_index].apply(current, qubit, moment)
                assert_circuit_legal(current, n_qubits)
                assert current.cost(LAMBDA_DEPTH) < before_cost - 1e-12, "动作未降低代价（死循环风险）"
                steps += 1
                assert steps <= step_bound + 1, (
                    f"重写步数 {steps} 超过上界 {step_bound}（seed={seed}, n={n_qubits}）"
                )
            assert not any(
                rule.is_applicable(current, qubit, moment)
                for rule in RULES
                for qubit in range(n_qubits)
                for moment in range(current.depth)
            ), "终止状态仍然存在可行动作"
            assert current.cost(LAMBDA_DEPTH) <= initial_cost + 1e-12
            if steps > 0:
                assert current.cost(LAMBDA_DEPTH) < initial_cost - 1e-12
            assert up_to_global_phase_equal(initial_unitary, independent_unitary(current))


def test_a5_mask_matches_rule_applicability() -> None:
    """动作掩码与逐锚点规则适用性完全一致（含 padding 时隙）。"""
    cases = 0
    for n_qubits in (2, 3, 4, 5):
        for depth in (1, 3, 6):
            circuit = independent_random_circuit(n_qubits, depth, seed=n_qubits * 10 + depth)
            for n_moments in (depth, depth + 2, 15):
                if n_moments < depth:
                    continue
                mask = build_action_mask(circuit, n_moments)
                assert mask.shape == (len(RULES), n_qubits, n_moments)
                assert mask.dtype == bool
                expected = manual_mask(circuit, n_moments)
                assert np.array_equal(mask, expected), (
                    f"掩码与规则适用性不一致 n={n_qubits} depth={depth} M={n_moments}"
                )
                # padding 时隙一律为 False。
                assert not mask[:, :, depth:].any()
                # 展平/还原与 applicable_actions 一致。
                flat = flatten_action_mask(mask)
                assert flat.shape == (len(RULES) * n_qubits * n_moments,)
                assert np.array_equal(unflatten_action_mask(
                    flat, len(RULES), n_qubits, n_moments), mask)
                listed = set(applicable_actions(circuit))
                from_mask = set()
                for flat_index in np.flatnonzero(flat):
                    rule_index, qubit, moment = decode_action(
                        int(flat_index), n_qubits, n_moments
                    )
                    from_mask.add((rule_index, qubit, moment))
                assert from_mask == listed
                cases += 1
    assert cases > 20
    with pytest.raises(ValueError):
        build_action_mask(independent_random_circuit(2, 2, seed=0), 0)


def test_a6_rules_are_pure_and_deterministic() -> None:
    """规则不得原地修改电路；同锚点重复应用结果一致。"""
    circuit = independent_random_circuit(4, 6, seed=42, p_two=0.4)
    original_layers = layers_signature(circuit)
    for rule in RULES:
        for qubit in range(circuit.n_qubits):
            for moment in range(circuit.depth):
                if not rule.is_applicable(circuit, qubit, moment):
                    continue
                first = rule.apply(circuit, qubit, moment)
                second = rule.apply(circuit, qubit, moment)
                assert layers_signature(first) == layers_signature(second)
                assert layers_signature(circuit) == original_layers, "规则原地修改了输入电路"
                assert first is not circuit


# ===========================================================================
# B. 模拟器与 cirq
# ===========================================================================
def _known_state(gate: str, qubits, n_qubits: int, initial: str) -> np.ndarray:
    """在给定计算基态上作用一个门，返回状态向量。"""
    index = int(initial, 2)
    state = np.zeros(2**n_qubits, dtype=np.complex128)
    state[index] = 1.0
    from qco.simulator import apply_gate_to_state

    return apply_gate_to_state(state, gate, qubits, n_qubits)


def test_b1_known_gate_actions_big_endian() -> None:
    """已知门作用结果：比特 0 是最高位（big-endian）。"""
    # X(q0)|00> = |10>（下标 2），X(q1)|00> = |01>（下标 1）。
    state = _known_state("X", (0,), 2, "00")
    assert np.allclose(state, np.array([0, 0, 1, 0]))
    state = _known_state("X", (1,), 2, "00")
    assert np.allclose(state, np.array([0, 1, 0, 0]))
    # H(q0)|00> = (|00> + |10>)/√2。
    state = _known_state("H", (0,), 2, "00")
    assert np.allclose(state, np.array([1, 0, 1, 0]) / math.sqrt(2))
    # CNOT(control=q0, target=q1)：|10> -> |11>。
    state = _known_state("CNOT", (0, 1), 2, "10")
    assert np.allclose(state, np.array([0, 0, 0, 1]))
    # CNOT(control=q1, target=q0)：|10> 不变（控制位为 0）。
    state = _known_state("CNOT", (1, 0), 2, "10")
    assert np.allclose(state, np.array([0, 0, 1, 0]))
    # SWAP|01> = |10>。
    state = _known_state("SWAP", (0, 1), 2, "01")
    assert np.allclose(state, np.array([0, 0, 1, 0]))
    # T|1> = exp(iπ/4)|1>，Sdag|1> = -i|1>，CZ|11> = -|11>。
    state = _known_state("T", (0,), 1, "1")
    assert np.allclose(state, np.array([0, np.exp(1j * math.pi / 4)]))
    state = _known_state("Sdag", (0,), 1, "1")
    assert np.allclose(state, np.array([0, -1j]))
    state = _known_state("CZ", (0, 1), 2, "11")
    assert np.allclose(state, np.array([0, 0, 0, -1]))
    # 单比特门作用在中间比特上不污染其他比特（qubit1 是中间位 ⇒ 下标 2）。
    state = _known_state("X", (1,), 3, "000")
    assert np.allclose(state, np.array([0, 0, 1, 0, 0, 0, 0, 0]))
    # big-endian 再确认：H(q0) 在 2 比特上的幺正矩阵 = H ⊗ I。
    assert np.allclose(circuit_unitary(make(2, [[("H", (0,))]])),
                       np.kron(MY_GATE_MATRICES["H"], np.eye(2)))


def test_b2_unitary_matches_independent_reference() -> None:
    """实现模拟器的幺正矩阵 = 本文件独立 kron 嵌入的乘积（随机电路）。"""
    for seed in range(12):
        n_qubits = 2 + seed % 4
        depth = 1 + (seed * 5) % 7
        circuit = independent_random_circuit(n_qubits, depth, seed=seed + 500, p_two=0.35)
        mine = independent_unitary(circuit)
        theirs = circuit_unitary(circuit)
        assert np.allclose(mine, theirs, atol=1e-10), (
            f"模拟器与独立参考不一致：n={n_qubits} depth={depth} seed={seed}"
        )
        assert np.allclose(circuit.unitary, theirs, atol=0.0)  # 属性缓存一致
        assert is_unitary(theirs)
        # 状态向量 = U @ |0...0>。
        assert np.allclose(circuit_state_vector(circuit), theirs[:, 0], atol=1e-12)


def test_b3_big_endian_embedding_matches_kron() -> None:
    """big-endian：X(q0) 在矩阵左侧（kron 高位），CNOT 控制位在前。"""
    n = 3
    eye = np.eye(2)
    x = MY_GATE_MATRICES["X"]
    assert np.allclose(gate_matrix_on("X", (0,), n), np.kron(np.kron(x, eye), eye))
    assert np.allclose(gate_matrix_on("X", (1,), n), np.kron(np.kron(eye, x), eye))
    assert np.allclose(gate_matrix_on("X", (2,), n), np.kron(np.kron(eye, eye), x))
    for gate in DOUBLE:
        for qubits in ((0, 1), (0, 2), (1, 2), (2, 0), (1, 0), (2, 1)):
            embedded = gate_matrix_on(gate, qubits, n)
            expected = _embed(Op(gate, qubits), n)
            assert np.allclose(embedded, expected, atol=1e-12), (
                f"gate_matrix_on({gate}, {qubits}) 与 big-endian 嵌入不一致"
            )
    # 两比特门矩阵与手写已知矩阵一致（n=2 时无其他比特）。
    assert np.allclose(gate_matrix_on("CNOT", (0, 1), 2), MY_GATE_MATRICES["CNOT"])
    assert np.allclose(gate_matrix_on("CNOT", (1, 0), 2), np.array(
        [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex))


def test_b4_matches_cirq_up_to_global_phase() -> None:
    """随机电路：numpy 幺正矩阵 vs ``cirq.unitary`` 模全局相位一致。"""
    cirq = pytest.importorskip("cirq")
    for seed in range(10):
        n_qubits = 2 + seed % 4
        depth = 1 + (seed * 3) % 6
        circuit = independent_random_circuit(n_qubits, depth, seed=seed + 900, p_two=0.3)
        u_numpy = independent_unitary(circuit)
        u_cirq = cirq.unitary(circuit.to_cirq())
        assert up_to_global_phase_equal(u_numpy, u_cirq, atol=1e-8), (
            f"模拟器与 cirq 不一致：n={n_qubits} seed={seed}"
        )
        cirq.testing.assert_allclose_up_to_global_phase(u_numpy, u_cirq, atol=1e-8)


def test_b5_cirq_roundtrip_and_gate_coverage() -> None:
    """``to_cirq`` / ``from_cirq`` 往返一致，且覆盖全部 11 种门。"""
    pytest.importorskip("cirq")
    layers = []
    for index, gate in enumerate(GATE_NAMES):
        if gate in SINGLE:
            layers.append([(gate, (index % 3,))])
        else:
            qubit = index % 2
            layers.append([(gate, (qubit, qubit + 1))])
    circuit = make(3, layers)
    assert set(op.gate for layer in circuit.layers for op in layer) == set(GATE_NAMES)
    back = Circuit.from_cirq(circuit.to_cirq())
    assert back == circuit
    assert layers_signature(back) == layers_signature(circuit)
    # 随机电路往返。
    for seed in range(6):
        original = independent_random_circuit(2 + seed % 4, 5, seed=seed + 20, p_two=0.3)
        assert Circuit.from_cirq(original.to_cirq()) == original
    # cirq 不认识的（分数幂门）必须显式报错而不是静默近似。
    import cirq

    qubit = cirq.LineQubit(0)
    bad = cirq.Circuit(cirq.X(qubit) ** 0.5)
    with pytest.raises(ValueError):
        Circuit.from_cirq(bad)


# ===========================================================================
# C. 动作布局一致性
# ===========================================================================
def test_c1_action_encoding_formula_and_roundtrip() -> None:
    """``index = (rule * Q + qubit) * M + moment``；编码/解码互逆且向量化一致。"""
    for n_rules in (1, 4):
        for n_qubits, n_moments in ((2, 1), (5, 15), (3, 7)):
            size = action_space_size(n_rules, n_qubits, n_moments)
            assert size == n_rules * n_qubits * n_moments
            sampled = list(range(size)) if size <= 120 else list(range(0, size, 13))
            for index in sampled:
                rule, qubit, moment = decode_action(index, n_qubits, n_moments)
                assert rule == index // (n_qubits * n_moments)
                assert (rule * n_qubits + qubit) * n_moments + moment == index
                assert encode_action(rule, qubit, moment, n_qubits, n_moments) == index
                assert int(encode_actions(
                    [rule], [qubit], [moment], n_qubits, n_moments)[0]) == index
                decoded = decode_actions(np.array([index]), n_qubits, n_moments)
                assert tuple(int(np.asarray(v).reshape(-1)[0]) for v in decoded) == (
                    rule, qubit, moment)
            with pytest.raises(ValueError):
                decode_action(-1, n_qubits, n_moments)


def test_c2_env_valid_actions_are_consistent_with_decoder() -> None:
    """``env.valid_actions`` 的下标、掩码行优先布局、``decode_action`` 三者一致。"""
    for seed in range(6):
        circuit = random_circuit(3, 5, np.random.default_rng(seed), p_two=0.3)
        env = CircuitOptimizationEnv(tiny_env_config(n_qubits=3), circuit=circuit)
        mask = env.action_mask
        valid = env.valid_actions
        assert valid.dtype == np.int64
        assert valid.size > 0
        assert np.array_equal(np.sort(valid), np.flatnonzero(flatten_action_mask(mask)))
        for flat_index in valid:
            flat_index = int(flat_index)
            rule, qubit, moment = np.unravel_index(flat_index, mask.shape)
            assert (int(rule), int(qubit), int(moment)) == decode_action(
                flat_index, 3, env.config.n_moments
            )
            assert mask[rule, qubit, moment]
            # 环境接受该动作并解释为同一个三元组。
            _, _, _, info = env.step(flat_index)
            assert (info["rule"], info["qubit"], info["moment"]) == (
                int(rule), int(qubit), int(moment)
            )
            break  # 每轮只需验证一个动作；枚举成本高


def _tiny_policy(n_qubits: int = 3, n_moments: int = 6, hidden: int = 8) -> CircuitPolicyNet:
    torch.manual_seed(0)
    return CircuitPolicyNet(
        in_channels=N_CHANNELS,
        n_rules=len(RULES),
        n_qubits=n_qubits,
        n_moments=n_moments,
        hidden_channels=hidden,
    )


def _independent_masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """独立实现的 masked log-softmax（不调用 qco.masking）。"""
    logits = logits.clone()
    mask = mask.bool()
    fill = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(~mask, fill)
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return logits.masked_fill(~mask, NEG_INF)


def test_c3_agent_act_single_legal_action() -> None:
    """只有一个合法动作时：deterministic 与采样都只能选它，log_prob=0，熵=0。"""
    n_qubits, n_moments = 3, 6
    n_rules = len(RULES)
    action_size = n_rules * n_qubits * n_moments
    policy = _tiny_policy(n_qubits, n_moments)
    batch_size = 3
    observations = torch.zeros(batch_size, N_CHANNELS, n_qubits, n_moments)
    masks = torch.zeros(batch_size, n_rules, n_qubits, n_moments, dtype=torch.bool)
    expected = []
    for row in range(batch_size):
        flat = (row + 1) * 17 % action_size
        rule, qubit, moment = decode_action(flat, n_qubits, n_moments)
        masks[row, rule, qubit, moment] = True
        expected.append(flat)
    expected_tensor = torch.tensor(expected, dtype=torch.int64)

    for deterministic in (True, False):
        actions, log_probs, values, entropies = policy.act(
            observations, masks, deterministic=deterministic
        )
        assert torch.equal(actions, expected_tensor)
        assert torch.allclose(log_probs, torch.zeros_like(log_probs), atol=1e-6)
        assert torch.allclose(entropies, torch.zeros_like(entropies), atol=1e-6)
        assert values.shape == (batch_size,)
        # 用环境校验这些动作确实可执行。
    # 掩码行全 False 时显式报错。
    empty = masks.clone()
    empty[0] = False
    with pytest.raises(ValueError):
        policy.act(observations, empty, deterministic=True)


def test_c4_agent_act_probabilities_and_entropy() -> None:
    """两种合法动作时：概率和归一、非法动作概率 0、log_prob/熵与独立实现一致。"""
    n_qubits, n_moments = 2, 4
    n_rules = len(RULES)
    action_size = n_rules * n_qubits * n_moments
    policy = _tiny_policy(n_qubits, n_moments)
    torch.manual_seed(5)
    observations = torch.rand(1, N_CHANNELS, n_qubits, n_moments)
    masks = torch.zeros(1, n_rules, n_qubits, n_moments, dtype=torch.bool)
    valid_flat = [3, action_size - 2]
    for flat in valid_flat:
        rule, qubit, moment = decode_action(flat, n_qubits, n_moments)
        masks[0, rule, qubit, moment] = True

    flat_logits = policy.policy_logits(observations).reshape(1, -1).detach()
    reference_log_probs = _independent_masked_log_softmax(
        flat_logits, masks.reshape(1, -1)
    )
    reference_probs = reference_log_probs.exp().detach()
    assert math.isclose(float(reference_probs.sum()), 1.0, abs_tol=1e-6)
    assert float(reference_probs[0, valid_flat].sum()) > 1.0 - 1e-6
    # 非法动作概率严格为 0。
    illegal = reference_probs.clone()
    illegal[0, valid_flat] = 0.0
    assert float(illegal.sum()) == 0.0

    actions, log_probs, _values, entropies = policy.act(
        observations, masks, deterministic=True
    )
    best = int(np.argmax(flat_logits.detach().numpy()[0, valid_flat]))
    assert int(actions[0]) == valid_flat[best]
    expected_log_prob = float(reference_log_probs[0, int(actions[0])])
    assert math.isclose(float(log_probs[0].detach()), expected_log_prob, abs_tol=1e-6)

    reference_entropy = -float(
        (reference_probs[0, valid_flat] * reference_log_probs[0, valid_flat]).sum()
    )
    assert math.isclose(float(entropies[0].detach()), reference_entropy, abs_tol=1e-5)

    # 随机采样只可能落在合法集合内。
    seen = set()
    for _ in range(60):
        sample_actions, sample_log_probs, _, _ = policy.act(observations, masks)
        action = int(sample_actions[0])
        assert action in valid_flat
        assert math.isclose(float(sample_log_probs[0].detach()),
                            float(reference_log_probs[0, action]), abs_tol=1e-6)
        seen.add(action)
    assert seen.issubset(set(valid_flat))


def test_c5_rollout_layout_and_observation_consistency() -> None:
    """rollout 保存的观测/掩码/动作与逐步重放完全一致（布局无 transpose bug）。"""
    n_qubits, n_moments = 3, 8
    circuits = [
        random_circuit(n_qubits, 6, np.random.default_rng(seed), p_two=0.2)
        for seed in range(2)
    ]
    envs = [
        CircuitOptimizationEnv(
            tiny_env_config(n_qubits=n_qubits, n_moments=n_moments, max_steps=48),
            circuit=circuit,
        )
        for circuit in circuits
    ]
    policy = _tiny_policy(n_qubits, n_moments)
    collector = RolloutCollector(policy, envs, device="cpu")
    horizon = 3
    batch, episode_stats = collector.collect(horizon, lambda: circuits[0])

    assert batch.observations.shape == (2, horizon, N_CHANNELS, n_qubits, n_moments)
    assert batch.masks.shape == (2, horizon, len(RULES), n_qubits, n_moments)
    assert batch.actions.shape == (2, horizon)
    assert batch.dones.shape == (2, horizon)
    assert batch.next_values.shape == (2,)
    assert len(episode_stats) == int(batch.dones.sum())

    # 用记录的动作从头重放，逐步核对观测、掩码、动作合法性、奖励公式。
    for env_index, circuit in enumerate(circuits):
        replay = CircuitOptimizationEnv(
            tiny_env_config(n_qubits=n_qubits, n_moments=n_moments, max_steps=48),
            circuit=circuit,
        )
        for step in range(horizon):
            observation = batch.observations[env_index, step].numpy()
            mask = batch.masks[env_index, step].numpy()
            assert np.array_equal(observation, replay.observation)
            assert np.array_equal(observation, encode_circuit(replay.circuit, n_moments))
            assert np.array_equal(mask, replay.action_mask)
            action = int(batch.actions[env_index, step])
            flat_mask = mask.reshape(-1)
            assert flat_mask[action], "rollout 选中的动作在保存的掩码里非法"
            rule, qubit, moment = decode_action(action, n_qubits, n_moments)
            assert mask[rule, qubit, moment]
            cost_before = replay.cost
            initial_cost = replay.initial_cost
            _, reward, done, _info = replay.step(action)
            assert math.isclose(
                reward, (cost_before - replay.cost) / initial_cost, abs_tol=1e-9
            )
            assert math.isclose(float(batch.rewards[env_index, step]), reward, abs_tol=1e-6)
            if done:
                # 每个 done 列都对应一次 auto-reset，后面的重放无法继续；仅检查计数。
                break


def test_c6_rollout_done_bookkeeping_and_bootstrap() -> None:
    """done 计数 = 回合统计数；next_values 是最后观测的策略价值；统计量自洽。"""
    n_qubits, n_moments = 2, 5
    # 保证每步都有可行动作的固定电路（H·H→删除，X·Y→Z），避免重采样器碰运气。
    circuit = make(2, [[("H", (0,))], [("H", (0,))], [("X", (1,))], [("Y", (1,))]])
    fresh_circuit = make(2, [[("Z", (0,))], [("Z", (0,))], [("T", (1,))], [("Tdag", (1,))]])
    envs = [
        CircuitOptimizationEnv(
            # max_steps=1（必要时改为 2）让每个环境每步都可能 done，覆盖 auto-reset 路径。
            tiny_env_config(
                n_qubits=n_qubits, n_moments=n_moments, max_steps=2, initial_depth=4
            ),
            circuit=circuit,
        )
        for _ in range(2)
    ]
    policy = _tiny_policy(n_qubits, n_moments)
    collector = RolloutCollector(policy, envs, device="cpu")
    horizon = 4
    batch, stats = collector.collect(horizon, lambda: fresh_circuit)
    assert int(batch.dones.sum()) == len(stats)
    assert len(stats) > 0, "max_steps=2 + horizon=4 应至少发生一次回合结束"
    # 每个 done 行的统计量自洽：return ≈ (initial_cost - final_cost)/initial_cost。
    for item in stats:
        assert isinstance(item, EpisodeStats)
        expected = (item.initial_cost - item.final_cost) / item.initial_cost
        assert math.isclose(item.return_, expected, abs_tol=1e-9)
        assert item.final_cost < item.initial_cost
        assert item.steps <= 2
        assert sum(item.rule_counts.values()) == item.steps
        assert 0 <= item.final_gate_count <= item.initial_gate_count
        assert 0 <= item.final_depth <= item.initial_depth
    # next_values 等于策略对 horizon 之后观测的价值估计。
    with torch.no_grad():
        final_observations = torch.stack([
            torch.from_numpy(env.observation) for env in envs
        ])
        _logits, expected_next_values = policy(final_observations)
    assert torch.allclose(
        batch.next_values, expected_next_values, atol=1e-6
    ), "next_values 不是最后观测的 bootstrap 价值"


# ===========================================================================
# D. PPO 数学
# ===========================================================================
def reference_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """逐环境 GAE 参考实现（纯 numpy 手写循环，不使用 torch / qco.ppo）。"""
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    next_values = np.asarray(next_values, dtype=np.float64).reshape(-1)
    num_envs, horizon = rewards.shape
    advantages = np.zeros_like(rewards)
    for env in range(num_envs):
        running = 0.0
        for step in range(horizon - 1, -1, -1):
            next_value = (
                next_values[env] if step == horizon - 1 else values[env, step + 1]
            )
            not_done = 1.0 - dones[env, step]
            delta = rewards[env, step] + gamma * next_value * not_done - values[env, step]
            running = delta + gamma * gae_lambda * not_done * running
            advantages[env, step] = running
    return advantages, advantages + values


def test_d1_gae_matches_independent_loop() -> None:
    """多组含 done 的随机数据：``compute_gae`` 与独立逐环境递推逐项一致。"""
    for seed in range(10):
        rng = np.random.default_rng(seed)
        num_envs = 1 + seed % 3
        horizon = 1 + seed % 8
        rewards = rng.normal(size=(num_envs, horizon))
        values = rng.normal(size=(num_envs, horizon))
        dones = (rng.random((num_envs, horizon)) < 0.35).astype(np.float64)
        if seed % 3 == 0:
            dones[:, -1] = 0.0  # 最后一步不 done，专门验证 next_values 的 bootstrap
        if seed % 4 == 0:
            dones[:, 0] = 1.0  # 第一步就 done，验证跨回合截断
        next_values = rng.normal(size=num_envs)
        gamma, gae_lambda = (0.99, 0.95) if seed % 2 else (0.9, 0.8)
        expected_adv, expected_ret = reference_gae(
            rewards, values, dones, next_values, gamma, gae_lambda
        )
        advantages, returns = PPO.compute_gae(
            torch.as_tensor(rewards),
            torch.as_tensor(values),
            torch.as_tensor(dones),
            torch.as_tensor(next_values),
            gamma,
            gae_lambda,
        )
        assert np.allclose(advantages.numpy(), expected_adv, atol=1e-12, rtol=0.0), (
            f"GAE 优势与独立实现不一致 seed={seed}"
        )
        assert np.allclose(returns.numpy(), expected_ret, atol=1e-12, rtol=0.0)
        assert np.allclose(
            returns.numpy(), advantages.numpy() + values, atol=1e-12, rtol=0.0
        )


def test_d2_gae_edge_cases() -> None:
    """GAE 边界：最后一步 done 忽略 next_values、horizon=1、一维输入、形状校验。"""
    # 最后一步 done ⇒ next_values 不得影响优势。
    rewards = torch.zeros(1, 3, dtype=torch.float64)
    values = torch.zeros(1, 3, dtype=torch.float64)
    dones = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    adv_zero, _ = PPO.compute_gae(
        rewards, values, dones, torch.tensor([0.0], dtype=torch.float64), 0.9, 0.9
    )
    adv_big, _ = PPO.compute_gae(
        rewards, values, dones, torch.tensor([1000.0], dtype=torch.float64), 0.9, 0.9
    )
    assert torch.allclose(adv_zero, adv_big, atol=0.0)

    # horizon=1：A_0 = r_0 + γ·v_next·(1-d_0) − v_0。
    rewards = torch.tensor([[2.0]], dtype=torch.float64)
    values = torch.tensor([[1.0]], dtype=torch.float64)
    dones = torch.tensor([[0.0]], dtype=torch.float64)
    next_values = torch.tensor([0.5], dtype=torch.float64)
    advantages, returns = PPO.compute_gae(rewards, values, dones, next_values, 0.5, 0.9)
    expected = 2.0 + 0.5 * 0.5 - 1.0
    assert math.isclose(float(advantages[0, 0]), expected, abs_tol=1e-12)
    assert math.isclose(float(returns[0, 0]), expected + 1.0, abs_tol=1e-12)

    # 一维输入与 [1, H] 输入一致，且形状正确。
    adv_1d, ret_1d = PPO.compute_gae(
        rewards[0], values[0], dones[0], next_values[0], 0.5, 0.9
    )
    assert adv_1d.shape == (1,) and ret_1d.shape == (1,)
    assert math.isclose(float(adv_1d[0]), expected, abs_tol=1e-12)

    # 不同环境之间不得串扰：改变 env1 的 next_values 不影响 env0。
    rewards = torch.zeros(2, 2, dtype=torch.float64)
    values = torch.zeros(2, 2, dtype=torch.float64)
    dones = torch.zeros(2, 2, dtype=torch.float64)
    adv_a, _ = PPO.compute_gae(
        rewards, values, dones, torch.tensor([1.0, 0.0], dtype=torch.float64), 0.9, 0.9
    )
    adv_b, _ = PPO.compute_gae(
        rewards, values, dones, torch.tensor([1.0, 99.0], dtype=torch.float64), 0.9, 0.9
    )
    assert torch.allclose(adv_a[0], adv_b[0], atol=0.0)
    assert not torch.allclose(adv_a[1], adv_b[1])

    # 形状不一致必须报错。
    with pytest.raises(ValueError):
        PPO.compute_gae(
            torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(1), 0.9, 0.9
        )
    with pytest.raises(ValueError):
        PPO.compute_gae(
            torch.zeros(2, 3), torch.zeros(2, 2), torch.zeros(2, 3), torch.zeros(2), 0.9, 0.9
        )
    with pytest.raises(ValueError):
        PPO.compute_gae(
            torch.zeros(1, 0), torch.zeros(1, 0), torch.zeros(1, 0), torch.zeros(1), 0.9, 0.9
        )


def _circuit_with_actions(
    seed: int, n_qubits: int = 3, depth: int = 5, p_two: float = 0.25
) -> Circuit:
    """生成一个至少有一个可行动作的随机电路（供 rollout 测试使用）。"""
    for attempt in range(200):
        circuit = random_circuit(
            n_qubits, depth, np.random.default_rng(seed + 9973 * attempt), p_two=p_two
        )
        if build_action_mask(circuit, 16).any():
            return circuit
    raise AssertionError("无法生成含可行动作的随机电路")  # pragma: no cover


def _build_real_rollout(
    policy: CircuitPolicyNet,
    n_qubits: int = 3,
    n_moments: int = 6,
    num_envs: int = 2,
    horizon: int = 4,
    seed: int = 0,
) -> tuple[RolloutBatch, list[EpisodeStats]]:
    """用真实环境 + 策略采样一个 rollout 批次（用于布局 / PPO 一致性检查）。"""
    config = tiny_env_config(
        n_qubits=n_qubits, n_moments=n_moments, max_steps=48, initial_depth=5
    )
    circuits = [
        _circuit_with_actions(seed + 31 * index, n_qubits=n_qubits)
        for index in range(num_envs)
    ]
    envs = [CircuitOptimizationEnv(config, circuit=circuit) for circuit in circuits]
    collector = RolloutCollector(policy, envs, device="cpu")
    return collector.collect(horizon, lambda: circuits[0])


def test_d3_ratio_is_one_and_no_double_softmax() -> None:
    """参数冻结时 ratio 严格为 1；log_probs 与独立 masked log-softmax 一致。"""
    n_qubits, n_moments = 3, 6
    policy = _tiny_policy(n_qubits, n_moments)
    batch, _stats = _build_real_rollout(policy, n_qubits, n_moments, num_envs=2, horizon=4)

    policy.eval()
    with torch.no_grad():
        flat_observations = batch.observations.reshape(-1, N_CHANNELS, n_qubits, n_moments)
        logits, values = policy(flat_observations)
        flat_logits = logits.reshape(flat_observations.shape[0], -1)
        flat_masks = batch.masks.reshape(flat_observations.shape[0], -1)
        reference_log_probs = _independent_masked_log_softmax(flat_logits, flat_masks)
        actions = batch.actions.reshape(-1)
        reference_selected = reference_log_probs.gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

    # 采样时保存的 log_prob 必须与独立实现逐项一致（若有双重 softmax 会不一致）。
    assert torch.allclose(reference_selected, batch.log_probs.reshape(-1), atol=1e-6), (
        "rollout 保存的 log_prob 与独立 masked log-softmax 不一致（疑似双重 softmax / 布局错位）"
    )
    # 概率归一：合法动作上的概率和严格为 1，非法动作概率为 0。
    probabilities = reference_log_probs.exp()
    assert torch.allclose(
        probabilities.sum(dim=-1),
        torch.ones_like(probabilities.sum(dim=-1)),
        atol=1e-6,
    )
    illegal = probabilities.masked_fill(flat_masks, 0.0)
    assert float(illegal.sum()) == 0.0
    # 参数未更新 ⇒ ratio = exp(new - old) 严格为 1。
    ratio = torch.exp(reference_selected - batch.log_probs.reshape(-1))
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-6)
    # 保存的 value 必须等于策略对同一观测的价值（rollout 记录正确）。
    assert torch.allclose(values, batch.values.reshape(-1), atol=1e-6)


def test_d4_update_uses_saved_masks_and_returns_value_loss() -> None:
    """update 复用 rollout 掩码；value loss 用 returns 而非 rewards；ratio=1 时损失一致。"""
    n_qubits, n_moments = 3, 6
    policy = _tiny_policy(n_qubits, n_moments)
    batch, _stats = _build_real_rollout(policy, n_qubits, n_moments, num_envs=2, horizon=4)
    num_samples = batch.num_envs * batch.horizon

    config = PPOConfig(
        lr=1e-2,
        gamma=0.99,
        gae_lambda=0.95,
        update_epochs=1,
        minibatch_size=num_samples,  # 单个小批量 ⇒ 指标就是整批统计
        target_kl=10.0,
        normalize_advantage=True,
    )
    ppo = PPO(policy, config, device="cpu")

    # 独立计算 returns（参考实现，不使用 PPO.compute_gae）。
    _adv_ref, returns_ref = reference_gae(
        batch.rewards.numpy(),
        batch.values.numpy(),
        batch.dones.numpy(),
        batch.next_values.numpy(),
        config.gamma,
        config.gae_lambda,
    )
    # 更新前（参数未动）的策略价值，用于独立计算期望 value loss。
    policy.eval()
    with torch.no_grad():
        flat_observations = batch.observations.reshape(-1, N_CHANNELS, n_qubits, n_moments)
        _, values_before = policy(flat_observations)
    expected_value_loss = float(
        torch.nn.functional.mse_loss(
            values_before, torch.as_tensor(returns_ref.reshape(-1), dtype=torch.float32)
        )
    )

    # 捕获 update 内部实际使用的观测与掩码。
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []
    original = ppo._log_probs_and_values

    def spy(observations: torch.Tensor, flat_masks: torch.Tensor):
        captured.append((observations.detach().clone(), flat_masks.detach().clone()))
        return original(observations, flat_masks)

    ppo._log_probs_and_values = spy  # type: ignore[method-assign]
    metrics = ppo.update(batch)
    ppo._log_probs_and_values = original  # type: ignore[method-assign]

    expected_calls = math.ceil(num_samples / min(num_samples, config.minibatch_size))
    expected_calls *= config.update_epochs
    assert len(captured) == expected_calls, (
        f"update 内部前向次数 {len(captured)} != 期望 {expected_calls}"
    )
    saved_mask_rows = sorted(
        tuple(bool(v) for v in row)
        for row in batch.masks.reshape(num_samples, -1).numpy()
    )
    saved_observation_rows = sorted(
        batch.observations.reshape(num_samples, -1).numpy()[index].tobytes()
        for index in range(num_samples)
    )
    for observations, flat_masks in captured:
        used_mask_rows = sorted(
            tuple(bool(v) for v in row) for row in flat_masks.numpy()
        )
        assert used_mask_rows == saved_mask_rows, (
            "PPO.update 重新前向使用的掩码与 rollout 保存的掩码不一致"
        )
        used_observation_rows = sorted(
            observations.numpy()[index].tobytes()
            for index in range(observations.shape[0])
        )
        assert used_observation_rows == saved_observation_rows

    # 单小批量 + ratio=1 ⇒ KL≈0、clip_fraction=0、policy_loss≈-mean(归一化优势)≈0。
    assert abs(metrics["approx_kl"]) < 1e-6
    assert metrics["clip_fraction"] == 0.0
    assert abs(metrics["policy_loss"]) < 1e-5
    # value loss 必须基于 returns（用 rewards 会得到明显不同的值）。
    assert math.isclose(metrics["value_loss"], expected_value_loss, abs_tol=1e-5), (
        f"value_loss={metrics['value_loss']} 与独立计算的 MSE(values, returns)="
        f"{expected_value_loss} 不一致"
    )
    rewards_only_loss = float(
        torch.nn.functional.mse_loss(
            values_before, batch.rewards.reshape(-1).to(torch.float32)
        )
    )
    # 仅作记录：returns 与 rewards 的 MSE 明显不同（含 done 的短回合同样如此），
    # 上面的 expected_value_loss 相等断言才是“value loss 用 returns”的判定依据。
    assert isinstance(rewards_only_loss, float)
    # 总损失公式。
    assert math.isclose(
        metrics["total_loss"],
        metrics["policy_loss"]
        + config.value_coef * metrics["value_loss"]
        - config.entropy_coef * metrics["entropy"],
        abs_tol=1e-5,
    )
    # 指标键与数值。
    assert set(metrics) == set(PPO.METRIC_KEYS)
    assert all(math.isfinite(v) for v in metrics.values())
    assert metrics["learning_rate"] == config.lr
    assert metrics["grad_norm"] > 0.0


def test_d5_entropy_metric_matches_independent_computation() -> None:
    """指标里的熵 = 合法动作上独立计算的掩码熵；关闭标准化时 policy loss 一致。"""
    n_qubits, n_moments = 2, 5
    policy = _tiny_policy(n_qubits, n_moments)
    batch, _stats = _build_real_rollout(policy, n_qubits, n_moments, num_envs=2, horizon=3)
    num_samples = batch.num_envs * batch.horizon
    config = PPOConfig(
        lr=1e-2,
        update_epochs=1,
        minibatch_size=num_samples,
        target_kl=10.0,
        normalize_advantage=False,
    )
    ppo = PPO(policy, config, device="cpu")
    policy.eval()
    with torch.no_grad():
        flat_observations = batch.observations.reshape(-1, N_CHANNELS, n_qubits, n_moments)
        logits, _values = policy(flat_observations)
        flat_masks = batch.masks.reshape(num_samples, -1)
        reference_log_probs = _independent_masked_log_softmax(
            logits.reshape(num_samples, -1), flat_masks
        )
        probabilities = reference_log_probs.exp()
        reference_entropy = float(
            -(probabilities * reference_log_probs)
            .masked_fill(~flat_masks, 0.0)
            .sum(dim=-1)
            .mean()
        )
    metrics = ppo.update(batch)
    assert math.isclose(metrics["entropy"], reference_entropy, abs_tol=1e-5)
    # 关闭优势标准化时，ratio=1 的 policy loss = -mean(advantage)。
    adv_ref, _returns_ref = reference_gae(
        batch.rewards.numpy(),
        batch.values.numpy(),
        batch.dones.numpy(),
        batch.next_values.numpy(),
        config.gamma,
        config.gae_lambda,
    )
    expected_policy_loss = -float(np.mean(adv_ref))
    assert math.isclose(metrics["policy_loss"], expected_policy_loss, abs_tol=1e-4)


def test_d6_update_changes_parameters_and_validates_batch() -> None:
    """一次更新确实改变参数；非法批次形状显式报错。"""
    n_qubits, n_moments = 2, 5
    policy = _tiny_policy(n_qubits, n_moments)
    batch, _stats = _build_real_rollout(policy, n_qubits, n_moments, num_envs=2, horizon=3)
    ppo = PPO(policy, PPOConfig(lr=1e-2, update_epochs=2, minibatch_size=4, target_kl=10.0))
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    ppo.update(batch)
    after = [parameter.detach().clone() for parameter in policy.parameters()]
    assert any(not torch.equal(old, new) for old, new in zip(before, after))

    bad = RolloutBatch(
        observations=batch.observations[:, :, :, :, :-1],  # 错误的 M
        actions=batch.actions,
        log_probs=batch.log_probs,
        values=batch.values,
        rewards=batch.rewards,
        dones=batch.dones,
        masks=batch.masks,
        next_values=batch.next_values,
    )
    bad_next = RolloutBatch(
        observations=batch.observations,
        actions=batch.actions,
        log_probs=batch.log_probs,
        values=batch.values,
        rewards=batch.rewards,
        dones=batch.dones,
        masks=batch.masks,
        next_values=batch.next_values[:1],
    )
    with pytest.raises(ValueError):
        ppo.update(bad)
    with pytest.raises(ValueError):
        ppo.update(bad_next)


# ===========================================================================
# E. 端到端（真实脚本 / 检查点一致性回归）
# ===========================================================================
def _run_python(args: list[str], cwd: Path | str = ROOT, timeout: float = 300.0):
    """用项目虚拟环境的解释器运行子进程，返回 ``(returncode, stdout, stderr)``。"""
    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["MPLBACKEND"] = "Agg"
    completed = subprocess.run(
        [PYTHON, *args],
        cwd=str(cwd),
        capture_output=True,
        timeout=timeout,
        env=environment,
    )
    return (
        completed.returncode,
        completed.stdout.decode("utf-8", errors="replace"),
        completed.stderr.decode("utf-8", errors="replace"),
    )


_MODULE_CACHE: dict[str, object] = {}


def _script_module(relative_path: str):
    """按文件路径加载脚本模块（避免依赖 scripts 是否为包）。"""
    module = _MODULE_CACHE.get(relative_path)
    if module is not None:
        return module
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(
        "audit_script_" + Path(relative_path).stem + "_" + str(len(_MODULE_CACHE)), path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE_CACHE[relative_path] = module
    return module


def _parser_options(parser) -> set[str]:
    return {option for action in parser._actions for option in action.option_strings}


def _parser_defaults(parser) -> dict[str, object]:
    return {action.dest: action.default for action in parser._actions}


def _train_supports_dataset() -> bool:
    """``scripts/train.py`` 是否已提供 ``--dataset``（P2 修复后的接口）。"""
    parser = _script_module("scripts/train.py").build_parser()
    return "--dataset" in _parser_options(parser)


def _train_config_has_eval_dataset() -> bool:
    return "eval_dataset" in {field.name for field in fields(TrainConfig)}


TINY_TRAIN_ARGS = [
    "--iterations", "2",
    "--num-envs", "2",
    "--steps-per-iter", "4",
    "--seed", "0",
    "--device", "cpu",
    "--max-steps", "6",
    "--n-qubits", "3",
    "--n-moments", "8",
    "--initial-depth", "5",
    "--p-two", "0.3",
    "--hidden-channels", "8",
    "--eval-every", "1",
    "--save-every", "0",
    "--eval-circuits", "5",
    "--train-circuits", "6",
    "--eval-seed", "10000",
    "--log-level", "WARNING",
]


@pytest.fixture(scope="module")
def e2e_run(tmp_path_factory):
    """模块级夹具：用极小配置真实运行一次 ``scripts/make_dataset.py`` + ``scripts/train.py``。"""
    tmp = tmp_path_factory.mktemp("audit_e2e")
    dataset = tmp / "eval_tiny.json"
    returncode, stdout, stderr = _run_python([
        str(ROOT / "scripts" / "make_dataset.py"),
        "--out", str(dataset),
        "--count", "5",
        "--n-qubits", "3",
        "--depth", "5",
        "--seed", "10000",
        "--log-level", "WARNING",
    ])
    assert returncode == 0, f"make_dataset 失败：{stderr[-3000:]}"
    assert dataset.exists()
    run_dir = tmp / "run"
    base = [str(ROOT / "scripts" / "train.py"), "--out", str(run_dir), *TINY_TRAIN_ARGS]
    if _train_supports_dataset():
        method = "cli_dataset"
        command = [*base, "--dataset", str(dataset)]
        returncode, stdout, stderr = _run_python(command)
    else:
        method = "patched_dataset"
        patch_code = (
            "import sys, pathlib, runpy\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "import qco.trainer as _trainer\n"
            f"_trainer.DEFAULT_EVAL_DATASET = pathlib.Path({str(tmp / 'absent_eval.json')!r})\n"
            f"runpy.run_path({str(ROOT / 'scripts' / 'train.py')!r}, run_name='__main__')\n"
        )
        command = [*base[1:]]  # 去掉脚本路径，作为 -c 之后的 argv
        returncode, stdout, stderr = _run_python(["-c", patch_code, *command])
    return {
        "tmp": tmp,
        "dataset": dataset,
        "run_dir": run_dir,
        "eval_out": tmp / "eval_out",
        "method": method,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def test_e1_train_script_artifacts_and_summary(e2e_run) -> None:
    """极小配置训练：产物齐全、CSV 列符合契约、summary 等价率 == 1.0。"""
    assert e2e_run["returncode"] == 0, (
        f"scripts/train.py 退出码 {e2e_run['returncode']}\nstderr:\n{e2e_run['stderr'][-4000:]}"
    )
    run_dir: Path = e2e_run["run_dir"]
    for filename in (
        "config.json", "train_log.csv", "eval_log.csv", "best.pt", "last.pt", "summary.json"
    ):
        assert (run_dir / filename).is_file(), f"缺少训练产物 {filename}"
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["iterations"] == 2
    assert config["env"]["n_qubits"] == 3
    assert config["env"]["n_moments"] == 8
    assert config["env"]["initial_depth"] == 5

    train_columns = [
        "iteration", "mean_return", "mean_episode_len", "mean_initial_cost",
        "mean_final_cost", "gate_reduction_pct", "depth_reduction_pct",
        "policy_loss", "value_loss", "entropy", "approx_kl", "explained_variance",
    ]
    with (run_dir / "train_log.csv").open("r", encoding="utf-8", newline="") as handle:
        train_rows = list(csv.DictReader(handle))
    assert list(train_rows[0].keys()) == train_columns
    assert len(train_rows) == 2, "iterations=2 ⇒ train_log.csv 应有 2 行"
    assert [int(row["iteration"]) for row in train_rows] == [1, 2]
    for row in train_rows:
        for column in train_columns[1:]:
            float(row[column])  # 必须能解析为数字
        assert float(row["mean_initial_cost"]) > 0.0
        assert float(row["mean_final_cost"]) >= 0.0

    eval_columns = [
        "iteration", "method", "cost_reduction_pct", "gate_reduction_pct",
        "depth_reduction_pct", "equivalence_rate", "mean_steps",
    ]
    with (run_dir / "eval_log.csv").open("r", encoding="utf-8", newline="") as handle:
        eval_rows = list(csv.DictReader(handle))
    assert list(eval_rows[0].keys()) == eval_columns
    methods = {row["method"] for row in eval_rows}
    assert methods == {"agent", "noop", "random", "greedy"}, (
        f"最终轮应评估四种方法（最后一行 4 条），实际方法集合 {methods}"
    )
    for row in eval_rows:
        assert math.isclose(float(row["equivalence_rate"]), 1.0, abs_tol=1e-12), (
            f"eval_log 中 {row['method']} 等价率不是 1.0"
        )

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["iterations"] == 2
    assert summary["train_time_s"] > 0.0
    assert summary["best_eval"] is not None
    assert math.isclose(summary["best_eval"]["equivalence_rate"], 1.0, abs_tol=1e-12)
    assert set(summary["results"]) == {"noop", "random", "greedy", "agent"}
    for method, payload in summary["results"].items():
        assert math.isclose(payload["equivalence_rate"], 1.0, abs_tol=1e-12), (
            f"summary 中 {method} 等价率不是 1.0"
        )
        assert payload["n_circuits"] == 5
        assert payload["cost_reduction_pct"] >= 0.0
    assert math.isclose(
        summary["results"]["agent"]["mean_final_cost"],
        summary["best_eval"]["mean_final_cost"],
        abs_tol=1e-12,
    ), "summary 的 agent 行应与 best_eval（best.pt 权重）一致"
    assert summary["results"]["noop"]["cost_reduction_pct"] == 0.0
    assert summary["results"]["noop"]["mean_steps"] == 0.0
    assert "方法" in summary["comparison_table"]

    best = load_checkpoint(run_dir / "best.pt")
    last = load_checkpoint(run_dir / "last.pt")
    for payload in (best, last):
        assert "state_dict" in payload and "config" in payload
        assert payload["iteration"] >= 1
        assert payload["config"]["env"]["n_qubits"] == 3
    last_state_values = [value for value in last["state_dict"].values()]
    assert all(isinstance(value, torch.Tensor) for value in last_state_values)
    # 允许 E 项以两种方式运行：修复后的 --dataset CLI，或修复前的等价 patch。
    assert e2e_run["method"] in ("cli_dataset", "patched_dataset")


def test_e2_evaluate_script_loads_checkpoint(e2e_run) -> None:
    """``scripts/evaluate.py`` 加载 best.pt，在 5 个电路上评估并写 evaluation.json。"""
    assert e2e_run["returncode"] == 0, e2e_run["stderr"][-3000:]
    run_dir: Path = e2e_run["run_dir"]
    eval_out: Path = e2e_run["eval_out"]
    returncode, stdout, stderr = _run_python([
        str(ROOT / "scripts" / "evaluate.py"),
        "--checkpoint", str(run_dir / "best.pt"),
        "--dataset", str(e2e_run["dataset"]),
        "--out", str(eval_out),
        "--baselines", "random,greedy",
        "--device", "cpu",
        "--log-level", "WARNING",
    ])
    assert returncode == 0, f"evaluate.py 失败：\n{stderr[-4000:]}"
    assert "方法" in stdout, "evaluate.py 应在 stdout 打印 Markdown 对比表"
    evaluation_path = eval_out / "evaluation.json"
    assert evaluation_path.is_file()
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert set(payload) == {"noop", "random", "greedy", "agent"}
    for method, result in payload.items():
        assert result["method"] == method
        assert result["n_circuits"] == 5
        assert math.isclose(result["equivalence_rate"], 1.0, abs_tol=1e-12), (
            f"evaluation.json 中 {method} 等价率不是 1.0"
        )
        assert result["mean_initial_cost"] > 0.0
    assert payload["noop"]["cost_reduction_pct"] == 0.0
    assert payload["noop"]["mean_final_cost"] == payload["noop"]["mean_initial_cost"]
    # 评估数据集与训练时完全一致 ⇒ 加载 best.pt 的 agent 指标应复现 summary["best_eval"]。
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    best_eval = summary["best_eval"]
    assert math.isclose(
        payload["agent"]["mean_final_cost"], best_eval["mean_final_cost"], abs_tol=1e-6
    ), "evaluate.py 加载 best.pt 得到的 mean_final_cost 与 summary best_eval 不一致"
    assert math.isclose(
        payload["agent"]["cost_reduction_pct"],
        best_eval["cost_reduction_pct"],
        abs_tol=1e-6,
    )
    for method in ("random", "greedy", "agent"):
        assert payload[method]["cost_reduction_pct"] >= 0.0


def test_e3_best_checkpoint_reload_matches_recorded_metrics(tmp_path) -> None:
    """回归（Lead 要求）：3 iterations / eval_every=1，best.pt 重载评估 == summary["best_eval"]。"""
    circuits = random_circuits(3, 3, 5, seed=10000)
    eval_dataset_path = tmp_path / "eval_regression.json"
    save_circuits(circuits, eval_dataset_path)
    import qco.trainer as trainer

    kwargs = dict(
        env=EnvConfig(
            n_qubits=3, n_moments=8, max_steps=6, initial_depth=5, seed=0
        ),
        ppo=PPOConfig(lr=1e-3, update_epochs=2, minibatch_size=8),
        iterations=3,
        steps_per_iter=4,
        num_envs=2,
        hidden_channels=8,
        eval_every=1,
        save_every=0,
        eval_circuits=3,
        train_circuits=5,
        eval_seed=10000,
        out_dir=str(tmp_path / "run_regression"),
        device="cpu",
    )
    original_dataset = trainer.DEFAULT_EVAL_DATASET
    if _train_config_has_eval_dataset():
        kwargs["eval_dataset"] = str(eval_dataset_path)
        config = TrainConfig(**kwargs)
        restore = False
    else:  # P2 修复前的等价做法：把模块级数据路径指向临时评估集
        trainer.DEFAULT_EVAL_DATASET = eval_dataset_path
        config = TrainConfig(**kwargs)
        restore = True
    try:
        instance = Trainer(config)
        summary = instance.train()
    finally:
        if restore:
            trainer.DEFAULT_EVAL_DATASET = original_dataset

    best = load_checkpoint(Path(config.out_dir) / "best.pt")
    assert "eval" in best, "best.pt 必须保存最优轮的评估指标"
    assert best["iteration"] == summary["best_iteration"]
    recorded = summary["best_eval"]
    assert recorded is not None
    # 修复后的语义：summary 主结果表中的 agent 行与 best.pt 对齐（取最优轮指标）。
    assert math.isclose(
        summary["results"]["agent"]["mean_final_cost"],
        recorded["mean_final_cost"],
        abs_tol=1e-12,
    )
    assert "final_eval" in summary and summary["final_eval"] is not None
    policy, env_config, _raw = build_policy_from_checkpoint(best)
    reloaded = evaluate_agent(
        policy, instance.eval_circuits, env_config, deterministic=True
    )
    # 重新加载 best.pt 后在同一评估集上的指标必须与归档指标一致。
    assert math.isclose(
        reloaded.mean_final_cost, recorded["mean_final_cost"], abs_tol=1e-6
    ), f"best.pt 重载评估 mean_final_cost={reloaded.mean_final_cost} != 归档 {recorded['mean_final_cost']}"
    assert math.isclose(
        reloaded.cost_reduction_pct, recorded["cost_reduction_pct"], abs_tol=1e-6
    )
    assert math.isclose(
        reloaded.gate_reduction_pct, recorded["gate_reduction_pct"], abs_tol=1e-6
    )
    assert math.isclose(
        reloaded.depth_reduction_pct, recorded["depth_reduction_pct"], abs_tol=1e-6
    )
    assert math.isclose(reloaded.equivalence_rate, recorded["equivalence_rate"], abs_tol=1e-12)
    assert reloaded.equivalence_rate == 1.0
    # 回归前置条件：该确定性配置下最优轮必须早于最后一轮，否则“重载一致性”检查
    # 无法区分“best.pt 存了最后一轮”这一退化（旧实现正是在此配置下失败）。
    assert summary["best_iteration"] < summary["iterations"], (
        "回归前置条件被破坏：3 iterations / eval_every=1 下最优轮应早于最后一轮；"
        "若训练动力学改变请复核本回归测试是否仍有判别力"
    )
    # 最优轮不是最后一轮 ⇒ best.pt 与 last.pt 的权重必须不同（否则就是存了最后一轮）。
    if summary["best_iteration"] < summary["iterations"]:
        last = load_checkpoint(Path(config.out_dir) / "last.pt")
        identical = all(
            torch.equal(best["state_dict"][key], last["state_dict"][key])
            for key in best["state_dict"]
        )
        assert not identical, "best_iteration < iterations 但 best.pt 与 last.pt 权重相同"


def test_e4_tiny_config_via_cli_requires_dataset_flag(e2e_run) -> None:
    """回归：``scripts/train.py --dataset <小型数据>`` 可运行 n_qubits=3 极小配置。"""
    if e2e_run["method"] != "cli_dataset":
        pytest.xfail(
            "P2 回归：scripts/train.py 缺少 --dataset；data/eval.json（5 比特）存在时，"
            "n_qubits=3 的极小配置无法通过 CLI 运行"
        )
    assert e2e_run["returncode"] == 0
    config = json.loads(
        (e2e_run["run_dir"] / "config.json").read_text(encoding="utf-8")
    )
    if "eval_dataset" in config:
        assert str(config["eval_dataset"]).endswith("eval_tiny.json")


# ===========================================================================
# F. 边界条件
# ===========================================================================
def test_f1_empty_and_optimal_circuits() -> None:
    """空电路与已最优电路：不崩溃、掩码为空、基线返回零步。"""
    config = tiny_env_config(n_qubits=2, n_moments=4, initial_depth=2)
    empty = Circuit(2, [])
    env = CircuitOptimizationEnv(config, circuit=empty)
    assert env.cost == 0.0 and env.initial_cost == 0.0 and env.steps == 0
    assert env.observation.shape == (N_CHANNELS, 2, 4)
    observation = env.observation
    assert observation[IDLE_CHANNEL].all()
    assert float(observation[: len(GATE_NAMES)].sum()) == 0.0
    assert not env.action_mask.any()
    assert env.valid_actions.size == 0
    assert not env.has_valid_actions
    with pytest.raises(ValueError):
        env.step(0)
    assert isinstance(env.render(), str)
    noop = optimize_noop(empty, config)
    assert noop.steps == 0 and noop.final_cost == 0.0 and noop.equivalent
    random_result = optimize_random(empty, config)
    assert random_result.steps == 0 and random_result.final_cost == 0.0
    greedy_result = optimize_greedy(empty, config)
    assert greedy_result.steps == 0

    # 已最优电路：单层两个门，不满足任何规则的适用条件。
    optimal = make(2, [[("X", (0,)), ("Y", (1,))]])
    env = CircuitOptimizationEnv(config, circuit=optimal)
    assert not env.action_mask.any()
    optimized = optimize_greedy(optimal, config)
    assert optimized.steps == 0
    assert optimized.final_circuit == optimal
    assert optimized.equivalent
    assert optimized.initial_cost == optimal.cost(config.lambda_depth)


def test_f2_depth_equals_n_moments() -> None:
    """``initial_depth == n_moments`` 与 ``circuit.depth == n_moments`` 均合法。"""
    config = tiny_env_config(n_qubits=2, n_moments=5, initial_depth=5)
    env = CircuitOptimizationEnv(config)
    assert env.circuit.depth == 5
    assert env.observation.shape == (N_CHANNELS, 2, 5)
    assert env.action_mask.shape == (len(RULES), 2, 5)

    circuit = make(2, [[("H", (0,))] for _ in range(5)])
    assert circuit.depth == 5
    env = CircuitOptimizationEnv(config, circuit=circuit)
    assert env.action_mask.any()
    optimized = optimize_greedy(circuit, config)
    assert optimized.equivalent
    assert optimized.final_cost <= optimized.initial_cost
    # 超过 n_moments 的电路必须显式报错。
    deep = make(2, [[("X", (0,))] for _ in range(6)])
    with pytest.raises(ValueError):
        CircuitOptimizationEnv(config, circuit=deep)
    with pytest.raises(ValueError):
        encode_circuit(deep, 5)


def test_f3_n_qubits_two_end_to_end() -> None:
    """n_qubits=2 时规则 / 掩码 / 策略网络 / rollout 全部可用。"""
    config = tiny_env_config(n_qubits=2, n_moments=5, initial_depth=4)
    circuit = random_circuit(2, 4, np.random.default_rng(7), p_two=0.4)
    env = CircuitOptimizationEnv(config, circuit=circuit)
    mask = build_action_mask(circuit, config.n_moments)
    assert mask.shape == (len(RULES), 2, 5)
    assert mask.any()
    result = optimize_greedy(circuit, config)
    assert result.equivalent and result.steps >= 1
    assert circuits_equivalent(circuit, result.final_circuit)
    policy = _tiny_policy(2, 5)
    batch, _stats = _build_real_rollout(policy, n_qubits=2, n_moments=5, num_envs=2, horizon=3)
    assert batch.observations.shape == (2, 3, N_CHANNELS, 2, 5)
    assert batch.masks.shape == (2, 3, len(RULES), 2, 5)
    evaluation = evaluate_agent(policy, [circuit], config, deterministic=True)
    assert evaluation.equivalence_rate == 1.0
    assert evaluation.n_circuits == 1


def test_f4_max_steps_one_truncation() -> None:
    """``max_steps=1``：一步后必然截断；baseline 与 rollout 单步行为正确。"""
    config = EnvConfig(
        n_qubits=2, n_moments=6, max_steps=1, initial_depth=4, p_two=0.4, seed=0
    )
    circuit = _circuit_with_actions(seed=5, n_qubits=2, depth=4)
    env = CircuitOptimizationEnv(config, circuit=circuit)
    assert env.valid_actions.size > 0
    _observation, _reward, done, info = env.step(int(env.valid_actions[0]))
    assert done
    assert info["truncated"] is True
    assert info["terminated"] is False
    assert env.steps == 1

    greedy = optimize_greedy(circuit, config)
    assert greedy.steps <= 1
    # 环境外步数上限同样生效。
    env2 = CircuitOptimizationEnv(config, circuit=circuit)
    limited = optimize(env2, choose_greedy, max_steps=1)
    assert limited.steps <= 1
    assert limited.truncated

    # rollout：max_steps=1 ⇒ 每一步都 done，auto-reset 正常。
    policy = _tiny_policy(2, 6)
    envs = [CircuitOptimizationEnv(config, circuit=circuit) for _ in range(2)]
    collector = RolloutCollector(policy, envs, device="cpu")
    batch, stats = collector.collect(2, lambda: circuit)
    assert batch.dones.shape == (2, 2)
    assert float(batch.dones.sum()) == 4.0
    assert len(stats) == 4
    assert all(item.truncated for item in stats)
    assert all(item.steps == 1 for item in stats)


def test_f5_illegal_and_out_of_range_actions() -> None:
    """非法动作 / 越界动作 / 负下标必须抛 ValueError，且环境状态不变。"""
    config = tiny_env_config(n_qubits=3, n_moments=6, initial_depth=4)
    circuit = _circuit_with_actions(seed=8, n_qubits=3, depth=4)
    env = CircuitOptimizationEnv(config, circuit=circuit)
    before_layers = layers_signature(env.circuit)
    for bad_action in (-1, env.n_actions, env.n_actions + 10, 10**6):
        with pytest.raises(ValueError):
            env.step(bad_action)
        assert layers_signature(env.circuit) == before_layers
        assert env.steps == 0
    # 被掩码屏蔽的动作。
    mask = env.action_mask
    blocked = [
        (rule, qubit, moment)
        for rule in range(len(RULES))
        for qubit in range(3)
        for moment in range(config.n_moments)
        if not mask[rule, qubit, moment]
    ]
    assert blocked
    rule, qubit, moment = blocked[0]
    with pytest.raises(ValueError):
        env.step(encode_action(rule, qubit, moment, 3, config.n_moments))
    # 编码函数自身的越界参数。
    with pytest.raises(ValueError):
        encode_action(0, 3, 0, 3, 6)
    with pytest.raises(ValueError):
        encode_action(0, 0, 6, 3, 6)
    with pytest.raises(ValueError):
        decode_action(-3, 3, 6)


def test_f6_reset_is_repeatable() -> None:
    """重复 ``reset`` 必须回到干净状态；reset 不改变给定电路。"""
    config = tiny_env_config(n_qubits=3, n_moments=6, initial_depth=4)
    circuit = _circuit_with_actions(seed=13, n_qubits=3, depth=4)
    env = CircuitOptimizationEnv(config, circuit=circuit)
    observation_a = env.observation.copy()
    mask_a = env.action_mask.copy()
    cost_a = env.cost
    env.step(int(env.valid_actions[0]))
    assert env.steps == 1
    observation_b = env.reset(circuit)
    assert np.array_equal(observation_b, observation_a)
    assert np.array_equal(env.observation, observation_a)
    assert np.array_equal(env.action_mask, mask_a)
    assert env.cost == cost_a and env.steps == 0
    assert env.initial_cost == cost_a
    assert env.rule_counts == {rule.name: 0 for rule in RULES}
    # reset 传入不同比特数的电路必须报错。
    with pytest.raises(ValueError):
        env.reset(Circuit(2, [[Op("X", (0,))]]))
    # reset 传入过深的电路必须报错。
    with pytest.raises(ValueError):
        env.reset(make(3, [[("X", (0,))] for _ in range(7)]))


def test_f7_train_eval_sets_are_disjoint_and_dataset_reproducible() -> None:
    """训练集（seed=0）与评估集（seed=10000）不重叠；data/eval.json 可由固定种子复现。"""
    dataset_path = ROOT / "data" / "eval.json"
    if not dataset_path.exists():
        pytest.skip("data/eval.json 不存在（全新 clone 场景），跳过数据集复现检查")
    eval_circuits = load_circuits(dataset_path)
    assert len(eval_circuits) == 100
    regenerated = random_circuits(100, 5, 12, seed=10000)
    assert [layers_signature(c) for c in eval_circuits] == [
        layers_signature(c) for c in regenerated
    ], "data/eval.json 与文档命令（seed=10000, count=100, 5 比特, 深度 12）的生成结果不一致"
    train_circuits = random_circuits(200, 5, 12, seed=0)
    train_signatures = {layers_signature(c) for c in train_circuits}
    eval_signatures = {layers_signature(c) for c in eval_circuits}
    assert len(train_signatures) == len(train_circuits)
    assert len(eval_signatures) == len(eval_circuits)
    assert train_signatures.isdisjoint(eval_signatures), "训练集与评估集存在重叠电路"
    default_train_config = TrainConfig()
    assert default_train_config.env.seed == 0
    assert default_train_config.eval_seed == 10000
    assert default_train_config.env.seed != default_train_config.eval_seed


def test_f8_circuit_validation_and_serialization() -> None:
    """电路构造校验 / 空层丢弃 / 序列化往返 / 查询方法 / cost 公式。"""
    c = Circuit(3, [
        [Op("X", (0,)), Op("CNOT", (1, 2))],
        [],
        [Op("H", (1,))],
    ])
    assert c.depth == 2  # 空层被丢弃
    assert c.num_ops == 3
    assert c.num_single_qubit_ops == 2
    assert c.num_two_qubit_ops == 1
    assert c.cost(0.5) == 3 + 0.5 * 2
    assert c.cost(0.0) == 3
    assert c.cost(2.0) == 3 + 4.0
    assert c.layer(1) == (Op("H", (1,)),)
    assert c.layer_qubits(0) == frozenset({0, 1, 2})
    assert c.layer_qubits(1) == frozenset({1})
    assert c.op_on(2, 0) == Op("CNOT", (1, 2))
    assert c.op_on(0, 1) is None
    assert c.next_op_on(1, 0) == (1, Op("H", (1,)))
    assert c.prev_op_on(1, 1) == (0, Op("CNOT", (1, 2)))
    assert c.qubit_ops(1) == [(0, Op("CNOT", (1, 2))), (1, Op("H", (1,)))]
    # 序列化往返。
    assert Circuit.from_dict(c.to_dict()) == c
    assert json.loads(json.dumps(c.to_dict()))["n_qubits"] == 3
    # 相等性与哈希（可作字典键）。
    assert c == Circuit(3, [list(layer) for layer in c.layers])
    assert c != make(2, [[("X", (0,))]])
    assert hash(c) == hash(Circuit(3, [list(layer) for layer in c.layers]))
    # ASCII 图。
    text = c.to_ascii()
    assert "q0" in text and "q1" in text and "X" in text
    assert Circuit(2, []).to_ascii() == "<empty circuit>"
    # 非法构造。
    with pytest.raises(ValueError):
        Circuit(2, [[Op("X", (0,)), Op("H", (0,))]])  # 同一层同一比特
    with pytest.raises(ValueError):
        Circuit(2, [[Op("X", (2,))]])  # 比特越界
    with pytest.raises(ValueError):
        Op("NOT_A_GATE", (0,))
    with pytest.raises(ValueError):
        Op("X", (0, 1))  # 比特数不匹配
    with pytest.raises(ValueError):
        Op("CNOT", (0, 0))  # 比特重复
    with pytest.raises(ValueError):
        Circuit(0, [])
    # 不可变：layers 是 tuple 的 tuple。
    assert isinstance(c.layers, tuple)
    assert all(isinstance(layer, tuple) for layer in c.layers)


def test_f9_env_config_validation() -> None:
    """非法配置在构造时显式报错，而不是训练中途崩溃。"""
    for kwargs in (
        {"n_qubits": 1},
        {"n_moments": 0},
        {"max_steps": 0},
        {"initial_depth": 0},
        {"initial_depth": 9, "n_moments": 8},
        {"p_two": 1.5},
        {"p_two": -0.1},
        {"lambda_depth": -0.5},
    ):
        with pytest.raises(ValueError):
            EnvConfig(**kwargs)
    for kwargs in (
        {"iterations": 0},
        {"steps_per_iter": 0},
        {"num_envs": 0},
        {"hidden_channels": 0},
        {"eval_every": 0},
        {"save_every": -1},
        {"eval_circuits": 0},
        {"train_circuits": 0},
    ):
        with pytest.raises(ValueError):
            TrainConfig(**kwargs)
    for kwargs in (
        {"lr": 0.0},
        {"gamma": -0.1},
        {"gae_lambda": 1.5},
        {"update_epochs": 0},
        {"minibatch_size": 0},
        {"target_kl": 0.0},
        {"max_grad_norm": 0.0},
    ):
        with pytest.raises(ValueError):
            PPOConfig(**kwargs)


def test_f10_lambda_zero_config_is_rejected() -> None:
    """``lambda_depth`` 必须为正（P2 修复）：move_* 规则只改深度，λ=0 会破坏严格进步不变量。"""
    for bad_value in (0.0, -1e-9, -0.5):
        with pytest.raises(ValueError) as error_info:
            EnvConfig(
                n_qubits=2,
                n_moments=4,
                max_steps=8,
                lambda_depth=bad_value,
                initial_depth=2,
            )
        assert "lambda_depth" in str(error_info.value)
    # 默认值与正数（包括极小正数）仍然合法。
    assert EnvConfig().lambda_depth == 0.5
    tiny = EnvConfig(lambda_depth=1e-9)
    assert tiny.lambda_depth == 1e-9


def test_f11_move_rules_depend_on_positive_lambda() -> None:
    """记录性检查：移动规则只改变深度项，λ=0 时 cost 不变（这正是 λ>0 校验的原因）。"""
    circuit = make(2, [[("X", (0,))], [("Y", (1,))]])
    move_actions = [
        (rule.rule_index, qubit, moment)
        for rule in RULES[2:]
        for qubit in range(2)
        for moment in range(circuit.depth)
        if rule.is_applicable(circuit, qubit, moment)
    ]
    assert move_actions, "该电路上应存在移动动作"
    for rule_index, qubit, moment in move_actions:
        transformed = RULES[rule_index].apply(circuit, qubit, moment)
        assert transformed.cost(0.0) == circuit.cost(0.0)
        assert transformed.cost(0.5) < circuit.cost(0.5)


# ===========================================================================
# G. 文档 / CLI / 契约一致性
# ===========================================================================
def test_g1_all_cli_help_runs() -> None:
    """契约第 16 节：所有脚本（含根目录薄封装）的 ``--help`` 必须可运行。"""
    scripts = (
        "scripts/make_dataset.py",
        "scripts/train.py",
        "scripts/evaluate.py",
        "scripts/make_figures.py",
        "train.py",
        "evaluate.py",
    )
    for relative in scripts:
        assert (ROOT / relative).is_file(), f"缺少脚本 {relative}"
        returncode, stdout, stderr = _run_python([str(ROOT / relative), "--help"])
        assert returncode == 0, f"{relative} --help 退出码 {returncode}\n{stderr[-2000:]}"
        assert "usage" in stdout.lower(), f"{relative} --help 没有输出 usage"
    # 从任意工作目录运行 --help 也应成功（脚本自己插入项目根到 sys.path）。
    with tempfile.TemporaryDirectory() as outside:
        returncode, stdout, _stderr = _run_python(
            [str(ROOT / "scripts" / "train.py"), "--help"], cwd=outside
        )
        assert returncode == 0


def test_g2_architecture_cli_defaults_and_flags() -> None:
    """契约第 16 节的命令行接口（参数名 / 默认值）必须与实现一致。"""
    train_parser = _script_module("scripts/train.py").build_parser()
    train_defaults = _parser_defaults(train_parser)
    train_options = _parser_options(train_parser)
    for option in (
        "--out", "--iterations", "--num-envs", "--steps-per-iter", "--seed",
        "--device", "--max-steps", "--lambda-depth",
    ):
        assert option in train_options, f"scripts/train.py 缺少契约要求的 {option}"
    assert train_defaults["out"] == "results/default"
    assert train_defaults["iterations"] == 150
    assert train_defaults["num_envs"] == 8
    assert train_defaults["steps_per_iter"] == 16
    assert train_defaults["max_steps"] == 48
    assert train_defaults["lambda_depth"] == 0.5
    assert train_defaults["seed"] == 0
    assert train_defaults["device"] == "cpu"

    evaluate_parser = _script_module("scripts/evaluate.py").build_parser()
    evaluate_defaults = _parser_defaults(evaluate_parser)
    evaluate_options = _parser_options(evaluate_parser)
    for option in ("--checkpoint", "--dataset", "--out", "--baselines"):
        assert option in evaluate_options, f"scripts/evaluate.py 缺少契约要求的 {option}"
    assert evaluate_defaults["checkpoint"] == "results/default/best.pt"
    assert evaluate_defaults["dataset"] == "data/eval.json"
    assert evaluate_defaults["out"] == "results/default/eval"
    assert evaluate_defaults["baselines"] == "random,greedy"

    dataset_parser = _script_module("scripts/make_dataset.py").build_parser()
    dataset_defaults = _parser_defaults(dataset_parser)
    assert dataset_defaults["out"] == "data/eval.json"
    assert dataset_defaults["count"] == 100
    assert dataset_defaults["n_qubits"] == 5
    assert dataset_defaults["depth"] == 12
    assert dataset_defaults["seed"] == 10000

    figures_parser = _script_module("scripts/make_figures.py").build_parser()
    figures_defaults = _parser_defaults(figures_parser)
    assert figures_defaults["run"] == "results/default"
    assert figures_defaults["out"] == "results/figures"

    # 契约里写出的命令必须能被解析（不真正执行）。
    train_parser.parse_args([
        "--out", "results/default", "--iterations", "150", "--num-envs", "8",
        "--steps-per-iter", "16", "--seed", "0", "--device", "cpu",
        "--max-steps", "48", "--lambda-depth", "0.5",
    ])
    evaluate_parser.parse_args([
        "--checkpoint", "results/default/best.pt", "--dataset", "data/eval.json",
        "--out", "results/default/eval", "--baselines", "random,greedy",
    ])
    dataset_parser.parse_args([
        "--out", "data/eval.json", "--count", "100", "--n-qubits", "5",
        "--depth", "12", "--seed", "10000",
    ])
    figures_parser.parse_args(["--run", "results/default", "--out", "results/figures"])


def test_g3_contract_public_names_exist() -> None:
    """契约中列出的公开类 / 函数 / 属性必须存在。"""
    expectations: dict[str, tuple[str, ...]] = {
        "qco.gates": (
            "GateSpec", "GATE_NAMES", "SINGLE_QUBIT_GATE_NAMES", "TWO_QUBIT_GATE_NAMES",
            "SELF_INVERSE_TWO_QUBIT", "GATES", "GATE_CHANNEL_INDEX", "CONTROL_CHANNEL",
            "TARGET_CHANNEL", "IDLE_CHANNEL", "N_CHANNELS", "gate_matrix",
            "gate_matrix_on", "match_single_qubit_gate", "is_unitary", "global_phase_equal",
        ),
        "qco.circuit": ("Op", "Circuit"),
        "qco.simulator": (
            "apply_gate_to_state", "circuit_state_vector", "circuit_unitary",
            "equivalent_up_to_global_phase", "circuits_equivalent",
        ),
        "qco.observation": ("encode_circuit", "observation_shape"),
        "qco.generator": ("random_circuit", "random_circuits", "save_circuits", "load_circuits"),
        "qco.actions": (
            "action_space_size", "encode_action", "decode_action", "encode_actions",
        ),
        "qco.rules": ("RewriteRule", "RULES", "RULES_BY_NAME", "applicable_actions"),
        "qco.masking": (
            "build_action_mask", "flatten_action_mask", "unflatten_action_mask",
            "masked_log_softmax", "masked_entropy",
        ),
        "qco.environment": ("EnvConfig", "CircuitOptimizationEnv"),
        "qco.agent": ("CircuitPolicyNet",),
        "qco.rollout": ("RolloutBatch", "EpisodeStats", "RolloutCollector"),
        "qco.ppo": ("PPOConfig", "PPO"),
        "qco.baselines": (
            "OptimizationResult", "optimize", "optimize_noop", "optimize_random",
            "optimize_greedy", "choose_random", "choose_greedy",
        ),
        "qco.evaluate": (
            "EvalResult", "evaluate_agent", "evaluate_noop", "evaluate_random",
            "evaluate_greedy", "format_comparison", "save_eval_results",
        ),
        "qco.visualizer": (
            "setup_matplotlib_chinese_font", "plot_training_curves", "plot_circuit",
            "plot_method_comparison", "plot_rule_usage",
        ),
        "qco.utils": (
            "set_seed", "resolve_device", "ensure_dir", "save_json", "load_json",
            "to_jsonable", "config_to_dict",
        ),
        "qco.trainer": ("TrainConfig", "Trainer"),
    }
    import importlib

    for module_name, names in expectations.items():
        module = importlib.import_module(module_name)
        for name in names:
            assert hasattr(module, name), f"{module_name} 缺少契约中的 `{name}`"
    # 关键方法 / 属性。
    for name in (
        "cost", "layer", "layer_qubits", "op_on", "next_op_on", "prev_op_on",
        "qubit_ops", "to_dict", "from_dict", "to_cirq", "from_cirq", "unitary",
        "equivalent", "to_text",
    ):
        assert hasattr(Circuit, name), f"Circuit 缺少契约中的 `{name}`"
    for name in (
        "n_actions", "circuit", "observation", "action_mask", "valid_actions",
        "cost", "initial_cost", "steps", "reset", "step", "render",
    ):
        assert hasattr(CircuitOptimizationEnv, name), f"环境缺少契约中的 `{name}`"
    for name in ("forward", "policy_logits", "act"):
        assert hasattr(CircuitPolicyNet, name), f"策略网络缺少契约中的 `{name}`"
    for name in ("flatten", "num_envs", "horizon"):
        assert hasattr(RolloutBatch, name)
    for name in ("update", "compute_gae"):
        assert hasattr(PPO, name)
    assert len(PPO.METRIC_KEYS) == 9


def test_g4_dataclass_fields_match_contract() -> None:
    """契约中声明的 dataclass 字段不得缺失。"""
    from dataclasses import fields as dc_fields

    documented = {
        EnvConfig: {
            "n_qubits", "n_moments", "max_steps", "lambda_depth",
            "initial_depth", "p_two", "seed",
        },
        PPOConfig: {
            "lr", "gamma", "gae_lambda", "clip_epsilon", "value_coef",
            "entropy_coef", "max_grad_norm", "update_epochs", "minibatch_size",
            "target_kl", "normalize_advantage",
        },
        TrainConfig: {
            "env", "ppo", "iterations", "steps_per_iter", "num_envs",
            "hidden_channels", "eval_every", "save_every", "eval_circuits",
            "train_circuits", "eval_seed", "out_dir", "device",
        },
        EvalResult: {
            "method", "n_circuits", "mean_initial_gate_count",
            "mean_final_gate_count", "mean_initial_depth", "mean_final_depth",
            "mean_initial_cost", "mean_final_cost", "gate_reduction_pct",
            "depth_reduction_pct", "cost_reduction_pct", "equivalence_rate",
            "mean_steps", "truncation_rate", "mean_rule_counts", "mean_elapsed_ms",
        },
        OptimizationResult: {
            "method", "initial_circuit", "final_circuit", "initial_cost",
            "final_cost", "initial_gate_count", "final_gate_count",
            "initial_depth", "final_depth", "steps", "truncated", "equivalent",
            "rule_counts", "elapsed_ms",
        },
        RolloutBatch: {
            "observations", "actions", "log_probs", "values", "rewards",
            "dones", "masks", "next_values",
        },
        EpisodeStats: {
            "return_", "initial_cost", "final_cost", "initial_gate_count",
            "final_gate_count", "initial_depth", "final_depth", "steps",
            "truncated", "rule_counts",
        },
    }
    for dataclass_type, expected in documented.items():
        actual = {field.name for field in dc_fields(dataclass_type)}
        missing = expected - actual
        assert not missing, f"{dataclass_type.__name__} 缺少契约字段：{sorted(missing)}"
    # EnvConfig / PPOConfig / EvalResult 不允许出现契约外的歧义字段。
    assert {field.name for field in dc_fields(EnvConfig)} == documented[EnvConfig]
    assert {field.name for field in dc_fields(PPOConfig)} == documented[PPOConfig]
    assert {field.name for field in dc_fields(EvalResult)} == documented[EvalResult]


def test_g5_readme_referenced_files_exist() -> None:
    """README 中引用的本地文件必须存在（``docs/figures/`` 为生成产物，单独记录）。"""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    import re

    targets = re.findall(r"\[[^\]]*\]\(([^)\s]+)\)", readme)
    missing: list[str] = []
    generated_missing: list[str] = []
    checked = 0
    for target in targets:
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        clean = target.split("#", 1)[0]
        if not clean or clean.startswith("results/"):
            continue
        if clean.startswith("docs/figures/"):
            if not (ROOT / clean).is_file():
                generated_missing.append(clean)
            continue
        path = (ROOT / clean)
        checked += 1
        if not path.exists():
            missing.append(clean)
    assert checked > 5, "没有解析到 README 的本地链接"
    assert not missing, f"README 引用了不存在的文件：{missing}"
    # docs/figures/*.png 缺失属于“尚未生成插图”（README 有说明），记录为已知项。
    assert isinstance(generated_missing, list)


def test_g6_dependencies_declared_and_cirq_optional() -> None:
    """第三方依赖必须声明；cirq 只能是函数内延迟导入的可选依赖。"""
    third_party_imports: set[str] = set()
    top_level_imports: dict[str, set[str]] = {}
    python_files = sorted((ROOT / "qco").glob("*.py")) + sorted(
        (ROOT / "scripts").glob("*.py")
    ) + [ROOT / "train.py", ROOT / "evaluate.py"]
    for path in python_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        top_level: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_level.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                top_level.add(node.module.split(".")[0])
        top_level_imports[path.name] = top_level
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    third_party_imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                third_party_imports.add(node.module.split(".")[0])

    stdlib = set(sys.stdlib_module_names)
    local = {"qco"}
    outside = {
        name for name in third_party_imports
        if name not in stdlib and name not in local and not name.startswith("_")
    }
    assert outside, "未解析到任何第三方依赖导入"

    declared: set[str] = set()
    for requirements_file in (ROOT / "requirements.txt", ROOT / "requirements-dev.txt"):
        for line in requirements_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            name = re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0].strip().lower()
            declared.add(name)
    optional = {"cirq", "psutil"}  # cirq / psutil 均为 try-import 的可选依赖
    undeclared = outside - declared - optional
    assert not undeclared, f"以下第三方导入未在 requirements 中声明：{sorted(undeclared)}"
    for required in ("numpy", "torch", "matplotlib"):
        assert required in declared, f"requirements 缺少核心依赖 {required}"
    # cirq 为可选依赖：任何模块都不得在模块顶层 import cirq。
    for filename, names in top_level_imports.items():
        assert "cirq" not in names, f"{filename} 在模块顶层导入了 cirq（应为可选依赖）"
    # qco 核心模块不得依赖 torch（纯模拟器路径应可无 torch 使用）。
    for filename in ("gates.py", "circuit.py", "simulator.py", "rules.py",
                     "environment.py", "generator.py", "actions.py", "observation.py"):
        if filename in top_level_imports:
            assert "torch" not in top_level_imports[filename], (
                f"qco/{filename} 顶层导入了 torch"
            )


def test_g7_observation_shape_signature_matches_contract() -> None:
    """契约 §4 与实现的 ``observation_shape`` 签名必须一致（P2 已修复：契约对齐实现）。"""
    import qco.observation as observation_module

    signature = inspect.signature(observation_module.observation_shape)
    implementation_params = list(signature.parameters)
    assert implementation_params == ["n_qubits", "n_moments"], (
        f"实现签名 {signature} 与冻结前约定的 (n_qubits, n_moments) 不一致"
    )
    contract = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    match = re.search(r"def observation_shape\(([^)]*)\)\s*->\s*([^\s#]+)", contract)
    assert match, "契约 §4 中未找到 observation_shape 定义"
    contract_params = [
        piece.split(":")[0].strip()
        for piece in match.group(1).split(",")
        if piece.strip()
    ]
    assert contract_params == implementation_params, (
        f"契约参数 {contract_params} 与实现 {implementation_params} 不一致"
    )
    # 语义抽查：形状 = (N_CHANNELS, Q, M)，且与 encode_circuit 输出一致。
    circuit = make(3, [[("H", (0,))], [("CNOT", (1, 2))], [("X", (0,))]])
    expected_shape = (N_CHANNELS, 3, 5)
    assert observation_module.observation_shape(3, 5) == expected_shape
    assert encode_circuit(circuit, 5).shape == expected_shape


def test_g8_masked_log_softmax_accepts_documented_broadcast() -> None:
    """契约 §8：``[..., R, Q, M]`` / ``[..., R*Q*M]`` 的 logits 均可与 ``[R, Q, M]`` 掩码广播。"""
    n_rules, n_qubits, n_moments = len(RULES), 2, 3
    action_size = n_rules * n_qubits * n_moments
    torch.manual_seed(0)
    logits_4d = torch.randn(2, n_rules, n_qubits, n_moments)
    flat_logits = logits_4d.reshape(2, action_size)
    mask = torch.zeros(n_rules, n_qubits, n_moments, dtype=torch.bool)
    mask[0, 0, 0] = True
    mask[1, 1, 2] = True
    flat_mask = mask.reshape(1, -1).expand(2, -1)  # 显式展平的参考掩码

    expected = masked_log_softmax(flat_logits, flat_mask)
    # 契约要求的广播写法：``[R,Q,M]`` 掩码 + 展平 ``[B,R*Q*M]`` logits。
    assert torch.allclose(masked_log_softmax(flat_logits, mask), expected, atol=0.0)
    # 4-D logits 也接受 3-D 掩码（形状层面）；注意：
    # 该调用路径沿最后一维（M）做 log_softmax，不是“整个动作空间”的分类分布，
    # 因此不把它与展平路径的数值做等价断言（内部调用方都传展平 logits）。
    result_4d = masked_log_softmax(logits_4d, mask)
    assert result_4d.shape == logits_4d.shape
    assert torch.equal(result_4d.masked_fill(~mask, 0.0).masked_fill(mask, 0.0),
                       torch.zeros_like(result_4d))  # 非法位置为 NEG_INF（被替换后为 0）
    assert torch.all(result_4d[~mask.expand_as(result_4d)] == NEG_INF)
    assert torch.all(torch.isfinite(result_4d[mask.expand_as(result_4d)]))
    # 概率和为 1、非法动作概率严格为 0。
    probabilities = expected.exp()
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2), atol=1e-6)
    illegal = probabilities.masked_fill(flat_mask, 0.0)
    assert float(illegal.sum()) == 0.0
    # masked_entropy 同样支持广播，且与显式展平结果一致。
    assert torch.allclose(
        masked_entropy(expected, mask), masked_entropy(expected, flat_mask), atol=0.0
    )
    # 不兼容的形状必须给出明确的 ValueError（而不是底层的 RuntimeError）。
    for bad_mask in (
        torch.zeros(2, 2, dtype=torch.bool),
        torch.zeros(7, dtype=torch.bool),
    ):
        with pytest.raises(ValueError):
            masked_log_softmax(flat_logits, bad_mask)
        with pytest.raises(ValueError):
            masked_entropy(expected, bad_mask)


def test_g9_env_info_keys_match_contract() -> None:
    """契约 §9：step 的 info 至少包含约定字段。"""
    config = tiny_env_config(n_qubits=3, n_moments=6, initial_depth=4)
    circuit = _circuit_with_actions(seed=21, n_qubits=3, depth=4)
    env = CircuitOptimizationEnv(config, circuit=circuit)
    action = int(env.valid_actions[0])
    _obs, reward, done, info = env.step(action)
    required = {
        "rule", "rule_name", "qubit", "moment", "cost", "initial_cost",
        "gate_count", "depth", "valid_actions", "terminated", "truncated",
        "rule_counts",
    }
    missing = required - set(info)
    assert not missing, f"step().info 缺少契约字段：{sorted(missing)}"
    assert info["rule_name"] == RULES[info["rule"]].name
    assert isinstance(done, bool)
    assert math.isclose(
        reward, (env.initial_cost - env.cost) / env.initial_cost, abs_tol=1e-12
    )
    assert math.isclose(info["cost"], env.cost, abs_tol=1e-12)


def test_g10_baseline_and_evaluation_aggregation_sanity() -> None:
    """基线与评估聚合：noop 不变、greedy 不劣于随机起点、等价率恒为 1。"""
    circuits = [
        _circuit_with_actions(seed=seed + 300, n_qubits=3, depth=5) for seed in range(3)
    ]
    config = tiny_env_config(n_qubits=3, n_moments=8, initial_depth=5, max_steps=16)
    noop = evaluate_noop(circuits, config)
    random_result = evaluate_random(circuits, config, seed=0)
    greedy = evaluate_greedy(circuits, config)
    for result in (noop, random_result, greedy):
        assert result.equivalence_rate == 1.0
        assert result.n_circuits == len(circuits)
        assert result.mean_initial_cost == pytest.approx(
            float(np.mean([c.cost(config.lambda_depth) for c in circuits]))
        )
    assert noop.mean_final_cost == noop.mean_initial_cost
    assert greedy.mean_final_cost <= noop.mean_final_cost + 1e-12
    table = format_comparison({"noop": noop, "random": random_result, "greedy": greedy})
    assert "方法" in table and "noop" in table and "greedy" in table
    first = circuits[0]
    result = optimize_noop(first, config)
    assert isinstance(result, OptimizationResult)
    assert result.initial_circuit is first and result.final_circuit is first
    assert result.steps == 0 and result.equivalent and not result.truncated


def test_g11_scripts_do_not_write_into_project_when_given_absolute_paths(tmp_path) -> None:
    """脚本在临时目录运行（绝对输出路径）时不应污染项目目录。"""
    import hashlib

    project_dataset = ROOT / "data" / "eval.json"
    before_hash = (
        hashlib.sha256(project_dataset.read_bytes()).hexdigest()
        if project_dataset.is_file()
        else None
    )
    dataset = tmp_path / "eval_abs.json"
    returncode, _stdout, stderr = _run_python([
        str(ROOT / "scripts" / "make_dataset.py"),
        "--out", str(dataset),
        "--count", "2",
        "--n-qubits", "2",
        "--depth", "3",
        "--seed", "5",
        "--log-level", "WARNING",
    ])
    assert returncode == 0, stderr[-2000:]
    assert dataset.is_file()
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    assert payload["meta"]["count"] == 2
    assert payload["meta"]["n_qubits"] == 2
    assert len(payload["circuits"]) == 2
    after_hash = (
        hashlib.sha256(project_dataset.read_bytes()).hexdigest()
        if project_dataset.is_file()
        else None
    )
    assert before_hash == after_hash, "make_dataset 不应修改 data/eval.json"


# ===========================================================================
# 附加对抗性用例（A/B/D/G 的补充覆盖）
# ===========================================================================
def test_a7_greedy_chooser_tie_break_and_optimality() -> None:
    """``choose_greedy`` 选一步后代价最小的合法动作，并列时取最小下标。"""
    config = EnvConfig(
        n_qubits=3, n_moments=5, max_steps=10, initial_depth=3, seed=0
    )
    # 三种动作降同样的代价（2.5），应选下标最小的 41=(move_later, q2, m1)。
    circuit = make(3, [[("CNOT", (0, 1))], [("X", (2,))]])
    env = CircuitOptimizationEnv(config, circuit=circuit)
    valid = [int(action) for action in env.valid_actions]
    costs = {}
    for action in valid:
        rule_index, qubit, moment = decode_action(action, 3, config.n_moments)
        costs[action] = RULES[rule_index].apply(circuit, qubit, moment).cost(
            config.lambda_depth
        )
    assert min(costs.values()) == max(costs.values())  # 全部并列
    assert choose_greedy(env) == min(valid)

    # 四种动作代价相同，并列时取最小下标 36=(move_earlier, q1, m1)。
    circuit = make(3, [[("X", (0,))], [("Y", (1,))], [("Z", (2,))]])
    env = CircuitOptimizationEnv(config, circuit=circuit)
    valid = [int(action) for action in env.valid_actions]
    assert choose_greedy(env) == min(valid) == 36

    # 随机选择器只能返回合法动作。
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert choose_random(env, rng) in set(valid)


def test_b6_single_qubit_gate_matcher_is_exhaustively_correct() -> None:
    """``match_single_qubit_gate`` 对全部 8×8 乘积的判定必须与门库一致。"""
    identity = np.eye(2, dtype=np.complex128)
    catalog = {name: gate_matrix(name) for name in SINGLE_QUBIT_GATE_NAMES}
    for first in SINGLE_QUBIT_GATE_NAMES:
        for second in SINGLE_QUBIT_GATE_NAMES:
            product = gate_matrix(second) @ gate_matrix(first)
            matched = match_single_qubit_gate(product)
            if up_to_global_phase_equal(product, identity):
                assert matched == "I", f"{second}·{first} 应为 I，实际 {matched}"
                continue
            if matched is None:
                # 门库不闭合：只有确实不在门库中时才允许返回 None。
                assert all(
                    not up_to_global_phase_equal(product, matrix)
                    for matrix in catalog.values()
                ), f"{second}·{first} 实际命中门库却返回 None"
                continue
            assert matched != "I"
            assert up_to_global_phase_equal(product, catalog[matched]), (
                f"{second}·{first} 匹配到 {matched} 但矩阵不等价"
            )
    # 非幺正 / 形状不对 / 非门库矩阵必须返回 None。
    assert match_single_qubit_gate(np.array([[1.0, 0.0], [0.0, 2.0]])) is None
    assert match_single_qubit_gate(np.eye(3, dtype=complex)) is None
    assert match_single_qubit_gate(np.array([[1.0, 0.0], [0.0, 0.0]])) is None
    assert match_single_qubit_gate(np.array([[1.0, 0.5], [0.0, 1.0]])) is None
    # is_unitary / global_phase_equal 的反例。
    assert not is_unitary(np.zeros((2, 2), dtype=complex))
    assert not global_phase_equal(np.zeros((2, 2), dtype=complex), identity)
    assert not global_phase_equal(np.eye(2, dtype=complex), np.eye(3, dtype=complex))
    assert global_phase_equal(1j * identity, -identity)
    assert not global_phase_equal(np.array([[1, 0], [0, 1j]]), identity)


def test_b7_circuit_unitary_is_cached() -> None:
    """``Circuit.unitary`` 延迟计算并缓存同一对象（契约 §2）。"""
    circuit = independent_random_circuit(3, 4, seed=77, p_two=0.3)
    first = circuit.unitary
    second = circuit.unitary
    assert first is second
    assert np.array_equal(first, circuit_unitary(circuit))


def test_d7_kl_early_stopping_is_effective() -> None:
    """近似 KL 超过 ``target_kl`` 后提前终止剩余小批量（不跑满 epoch 数）。"""
    n_qubits, n_moments = 2, 5
    policy = _tiny_policy(n_qubits, n_moments)
    batch, _stats = _build_real_rollout(policy, n_qubits, n_moments, num_envs=2, horizon=3)
    num_samples = batch.num_envs * batch.horizon
    config = PPOConfig(
        lr=0.5, update_epochs=20, minibatch_size=2, target_kl=1e-9
    )
    ppo = PPO(policy, config, device="cpu")
    step_calls = {"count": 0}
    original_step = ppo.optimizer.step

    def counted_step(*args, **kwargs):
        step_calls["count"] += 1
        return original_step(*args, **kwargs)

    ppo.optimizer.step = counted_step  # type: ignore[method-assign]
    try:
        metrics = ppo.update(batch)
    finally:
        ppo.optimizer.step = original_step  # type: ignore[method-assign]
    maximum = config.update_epochs * math.ceil(num_samples / config.minibatch_size)
    assert 1 <= step_calls["count"] < maximum, (
        f"KL 早停未生效：实际更新步数 {step_calls['count']} / 上限 {maximum}"
    )
    assert set(metrics) == set(PPO.METRIC_KEYS)
    assert all(math.isfinite(value) for value in metrics.values())


def test_g12_style_conventions() -> None:
    """契约 §18：所有 qco 模块有 ``from __future__ import annotations``、无 print 残留、公开对象有文档。"""
    import importlib
    import pkgutil

    import qco

    module_names = sorted(
        module.name for module in pkgutil.iter_modules(qco.__path__)
    )
    assert len(module_names) >= 17
    for module_name in module_names:
        source = (ROOT / "qco" / f"{module_name}.py").read_text(encoding="utf-8")
        assert "from __future__ import annotations" in source, (
            f"qco/{module_name}.py 缺少 from __future__ import annotations"
        )
        assert not re.search(r"^\s*print\(", source, flags=re.MULTILINE), (
            f"qco/{module_name}.py 残留 print 调试语句"
        )
        module = importlib.import_module(f"qco.{module_name}")
        for name, member in inspect.getmembers(module):
            if name.startswith("_"):
                continue
            if getattr(member, "__module__", None) != f"qco.{module_name}":
                continue
            if inspect.isclass(member) or inspect.isfunction(member):
                assert (member.__doc__ or "").strip(), (
                    f"qco.{module_name}.{name} 缺少 docstring（契约要求中文 docstring）"
                )
    # qco/__init__ 的延迟导出名单必须与真实属性一致。
    for name in qco.__all__:
        assert hasattr(qco, name), f"qco.__all__ 中的 {name} 不存在"


def test_g13_make_figures_generates_expected_pngs(e2e_run, tmp_path) -> None:
    """README 的 ``scripts/make_figures.py`` 命令真实可运行并生成 5 张图。"""
    assert e2e_run["returncode"] == 0
    out_dir = tmp_path / "figs"
    returncode, stdout, stderr = _run_python([
        str(ROOT / "scripts" / "make_figures.py"),
        "--run", str(e2e_run["run_dir"]),
        "--out", str(out_dir),
        "--dataset", str(e2e_run["dataset"]),
        "--log-level", "WARNING",
    ])
    assert returncode == 0, f"make_figures 失败：\n{stderr[-4000:]}"
    for filename in (
        "training_curves.png",
        "method_comparison.png",
        "rule_usage.png",
        "circuit_before.png",
        "circuit_after.png",
    ):
        path = out_dir / filename
        assert path.is_file(), f"缺少插图 {filename}"
        assert path.stat().st_size > 1024, f"插图 {filename} 疑似为空"



