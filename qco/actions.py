"""动作编码工具。

动作空间是 ``(规则, 量子比特, 时隙)`` 的笛卡尔积，按固定布局展平为一维下标：

    index = (rule * n_qubits + qubit) * n_moments + moment

该布局必须与策略网络输出 ``[B, n_rules, n_qubits, n_moments]`` 的展平顺序、
动作掩码 ``[n_rules, n_qubits, n_moments]`` 的展平顺序完全一致；
任何一端改动都会使策略学到的动作错位，因此集中在本模块实现。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def action_space_size(n_rules: int, n_qubits: int, n_moments: int) -> int:
    """返回展平后的动作空间大小。"""
    if n_rules <= 0 or n_qubits <= 0 or n_moments <= 0:
        raise ValueError("n_rules / n_qubits / n_moments 必须为正整数")
    return int(n_rules) * int(n_qubits) * int(n_moments)


def encode_action(rule: int, qubit: int, moment: int, n_qubits: int, n_moments: int) -> int:
    """把 ``(rule, qubit, moment)`` 编码为一维动作下标。"""
    if not (0 <= rule < 10**9):
        raise ValueError(f"非法的规则下标：{rule}")
    if not (0 <= qubit < n_qubits):
        raise ValueError(f"非法的量子比特下标：{qubit}（n_qubits={n_qubits}）")
    if not (0 <= moment < n_moments):
        raise ValueError(f"非法的时隙下标：{moment}（n_moments={n_moments}）")
    return (int(rule) * int(n_qubits) + int(qubit)) * int(n_moments) + int(moment)


def decode_action(index: int, n_qubits: int, n_moments: int) -> tuple[int, int, int]:
    """把一维动作下标解码为 ``(rule, qubit, moment)``。"""
    index = int(index)
    if index < 0:
        raise ValueError(f"非法的动作下标：{index}")
    rule, remainder = divmod(index, int(n_qubits) * int(n_moments))
    qubit, moment = divmod(remainder, int(n_moments))
    return rule, qubit, moment


def encode_actions(
    rules: Sequence[int] | np.ndarray,
    qubits: Sequence[int] | np.ndarray,
    moments: Sequence[int] | np.ndarray,
    n_qubits: int,
    n_moments: int,
) -> np.ndarray:
    """向量化版本的动作编码，返回 int64 数组。"""
    rules_arr = np.asarray(rules, dtype=np.int64)
    qubits_arr = np.asarray(qubits, dtype=np.int64)
    moments_arr = np.asarray(moments, dtype=np.int64)
    if not (rules_arr.shape == qubits_arr.shape == moments_arr.shape):
        raise ValueError("rules / qubits / moments 形状必须一致")
    return (rules_arr * int(n_qubits) + qubits_arr) * int(n_moments) + moments_arr


def decode_actions(index: np.ndarray, n_qubits: int, n_moments: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """向量化版本的动作解码。"""
    index = np.asarray(index, dtype=np.int64)
    if np.any(index < 0):
        raise ValueError("动作下标不能为负数")
    rule, remainder = np.divmod(index, int(n_qubits) * int(n_moments))
    qubit, moment = np.divmod(remainder, int(n_moments))
    return rule, qubit, moment
