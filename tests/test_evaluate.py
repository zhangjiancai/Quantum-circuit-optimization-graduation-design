"""基线与评估模块的单元测试。

覆盖 ``qco.baselines``（noop / random / greedy 与通用优化循环）以及
``qco.evaluate``（指标汇总、Markdown 表格、JSON 保存、智能体评估）。
所有断言都只使用小规模随机电路，CPU 上可快速完成。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qco.actions import decode_action
from qco.baselines import (
    OptimizationResult,
    choose_greedy,
    choose_random,
    optimize,
    optimize_greedy,
    optimize_noop,
    optimize_random,
)
from qco.circuit import Circuit, Op
from qco.environment import CircuitOptimizationEnv, EnvConfig
from qco.evaluate import (
    EvalResult,
    evaluate_greedy,
    evaluate_noop,
    evaluate_random,
    format_comparison,
    save_eval_results,
)
from qco.generator import random_circuits
from qco.rules import RULES
from qco.simulator import circuits_equivalent


@pytest.fixture()
def env_config() -> EnvConfig:
    """小规模环境配置：3 比特、时隙窗口 8、最多 8 步。"""
    return EnvConfig(
        n_qubits=3,
        n_moments=8,
        max_steps=8,
        lambda_depth=0.5,
        initial_depth=5,
        p_two=0.3,
        seed=1,
    )


@pytest.fixture()
def circuits() -> list[Circuit]:
    """固定种子的评估电路集合。"""
    return random_circuits(3, 3, 5, seed=7, p_two=0.3)


# ---------------------------------------------------------------------------
# 基线：动作选择与通用优化循环
# ---------------------------------------------------------------------------


def test_choose_greedy_picks_minimum_one_step_cost() -> None:
    """贪心选择必须等价于“一步后代价最小、并列取最小动作下标”。"""
    circuit = Circuit(
        2,
        [
            [Op("X", (0,)), Op("X", (1,))],
            [Op("X", (0,)), Op("Z", (1,))],
        ],
    )
    config = EnvConfig(
        n_qubits=2,
        n_moments=8,
        max_steps=8,
        initial_depth=5,
        p_two=0.0,
        seed=1,
    )
    env = CircuitOptimizationEnv(config, circuit=circuit)
    action = choose_greedy(env)
    assert int(action) in set(int(item) for item in env.valid_actions)

    candidates: list[tuple[float, int]] = []
    for raw in env.valid_actions:
        rule_index, qubit, moment = decode_action(
            int(raw), config.n_qubits, config.n_moments
        )
        candidate = RULES[rule_index].apply(env.circuit, qubit, moment)
        candidates.append((candidate.cost(config.lambda_depth), int(raw)))
    best_cost = min(cost for cost, _ in candidates)
    expected = min(raw for cost, raw in candidates if abs(cost - best_cost) < 1e-12)
    assert int(action) == expected


def test_choose_random_stays_within_valid_actions(env_config: EnvConfig) -> None:
    """随机选择只返回合法动作，且同一随机种子结果可复现。"""
    circuit = random_circuits(1, 3, 5, seed=3)[0]
    env = CircuitOptimizationEnv(env_config, circuit=circuit)
    valid = set(int(item) for item in env.valid_actions)
    first = choose_random(env, np.random.default_rng(123))
    second = choose_random(env, np.random.default_rng(123))
    assert first in valid
    assert first == second


def test_optimize_merges_and_keeps_semantics() -> None:
    """两个相邻的 X 门应被合并删除：1 步后电路为空，语义保持不变。"""
    circuit = Circuit(2, [[Op("X", (0,))], [Op("X", (0,))]])
    config = EnvConfig(
        n_qubits=2,
        n_moments=8,
        max_steps=8,
        initial_depth=5,
        p_two=0.0,
        seed=1,
    )
    env = CircuitOptimizationEnv(config, circuit=circuit)
    result = optimize(env, choose_greedy)
    assert result.steps == 1
    assert result.final_gate_count == 0
    assert result.final_depth == 0
    assert result.final_cost == pytest.approx(0.0)
    assert result.rule_counts["merge_single_qubit"] == 1
    assert result.equivalent is True
    assert result.truncated is False
    assert circuits_equivalent(result.initial_circuit, result.final_circuit)


def test_optimize_records_truncation(env_config: EnvConfig) -> None:
    """达到步数上限时应标记 ``truncated=True``。"""
    short_config = EnvConfig(
        n_qubits=3,
        n_moments=8,
        max_steps=1,
        initial_depth=5,
        p_two=0.3,
        seed=2,
    )
    circuit = random_circuits(1, 3, 5, seed=11, p_two=0.3)[0]
    result = optimize_random(circuit, short_config, seed=0)
    assert result.steps == 1
    assert result.truncated is True


def test_optimize_noop_returns_initial_circuit(env_config: EnvConfig, circuits: list[Circuit]) -> None:
    """noop 结果：最终电路与初始电路完全一致、零步、语义等价。"""
    result = optimize_noop(circuits[0], env_config)
    assert isinstance(result, OptimizationResult)
    assert result.method == "noop"
    assert result.steps == 0
    assert result.truncated is False
    assert result.equivalent is True
    assert result.final_circuit == result.initial_circuit
    assert result.initial_cost == pytest.approx(result.final_cost)
    assert result.rule_counts == {rule.name: 0 for rule in RULES}


def test_baselines_always_improve_cost(env_config: EnvConfig, circuits: list[Circuit]) -> None:
    """random / greedy 每一步都严格降低代价，因此最终代价不高于初始代价。"""
    for circuit in circuits:
        for result in (
            optimize_random(circuit, env_config, seed=2),
            optimize_greedy(circuit, env_config),
        ):
            assert result.equivalent is True
            assert result.final_cost <= result.initial_cost + 1e-9
            assert result.final_gate_count <= result.initial_gate_count
            assert result.steps >= 0
            assert result.elapsed_ms >= 0.0
            assert set(result.rule_counts) == {rule.name for rule in RULES}


# ---------------------------------------------------------------------------
# 评估指标
# ---------------------------------------------------------------------------


def test_evaluate_noop_metrics(env_config: EnvConfig, circuits: list[Circuit]) -> None:
    """noop 的压缩率应为 0、等价率为 1。"""
    result = evaluate_noop(circuits, env_config)
    assert isinstance(result, EvalResult)
    assert result.method == "noop"
    assert result.n_circuits == len(circuits)
    assert result.equivalence_rate == pytest.approx(1.0)
    assert result.gate_reduction_pct == pytest.approx(0.0)
    assert result.depth_reduction_pct == pytest.approx(0.0)
    assert result.cost_reduction_pct == pytest.approx(0.0)
    assert result.mean_steps == pytest.approx(0.0)
    assert result.truncation_rate == pytest.approx(0.0)
    assert result.mean_final_cost == pytest.approx(result.mean_initial_cost)


def test_evaluate_random_and_greedy_metrics(
    env_config: EnvConfig, circuits: list[Circuit]
) -> None:
    """随机与贪心基线都应降低代价，且等价率恒为 1.0。"""
    noop = evaluate_noop(circuits, env_config)
    random_result = evaluate_random(circuits, env_config, seed=0)
    greedy_result = evaluate_greedy(circuits, env_config)

    for result in (random_result, greedy_result):
        assert result.n_circuits == len(circuits)
        assert result.equivalence_rate == pytest.approx(1.0)
        assert result.cost_reduction_pct > 0.0
        assert result.mean_final_cost < noop.mean_final_cost
        assert result.mean_steps > 0.0
        assert 0.0 <= result.truncation_rate <= 1.0
        assert set(result.mean_rule_counts) == {rule.name for rule in RULES}

    # 契约要求放宽“随机不优于贪心”的断言：贪心的单步最优不等于全局最优，
    # 这里只要求两者处于同一量级（差距不超过 10 个百分点）。
    assert greedy_result.cost_reduction_pct >= random_result.cost_reduction_pct - 10.0


def test_evaluate_respects_lambda_depth(circuits: list[Circuit]) -> None:
    """``EnvConfig.lambda_depth`` 必须进入代价函数：noop 代价 = 门数 + λ×深度。"""
    config = EnvConfig(
        n_qubits=3,
        n_moments=8,
        max_steps=8,
        lambda_depth=2.0,
        initial_depth=5,
        p_two=0.3,
        seed=1,
    )
    result = evaluate_noop(circuits, config)
    expected_cost = result.mean_initial_gate_count + 2.0 * result.mean_initial_depth
    assert result.mean_initial_cost == pytest.approx(expected_cost)
    assert result.mean_final_cost == pytest.approx(result.mean_initial_cost)


def test_evaluate_agent_equivalence(
    env_config: EnvConfig, circuits: list[Circuit]
) -> None:
    """策略网络评估：等价率必须为 1.0，指标字段合理。"""
    pytest.importorskip("qco.agent")
    from qco.agent import CircuitPolicyNet
    from qco.evaluate import evaluate_agent

    policy = CircuitPolicyNet(
        n_rules=len(RULES),
        n_qubits=env_config.n_qubits,
        n_moments=env_config.n_moments,
        hidden_channels=8,
    )
    result = evaluate_agent(policy, circuits, env_config, device="cpu", deterministic=True)
    assert result.method == "agent"
    assert result.n_circuits == len(circuits)
    assert result.equivalence_rate == pytest.approx(1.0)
    assert result.mean_final_cost <= result.mean_initial_cost
    assert 0.0 <= result.truncation_rate <= 1.0


def test_format_comparison_is_markdown_table(
    env_config: EnvConfig, circuits: list[Circuit]
) -> None:
    """对比表应包含表头、分隔行与所有方法名。"""
    results = {
        "noop": evaluate_noop(circuits, env_config),
        "greedy": evaluate_greedy(circuits, env_config),
    }
    table = format_comparison(results)
    lines = table.splitlines()
    assert len(lines) == 2 + len(results)
    assert lines[0].startswith("|")
    assert "等价率" in lines[0]
    assert set(lines[1].replace("|", " ").split()) == {"---"}
    assert "noop" in table
    assert "greedy" in table


def test_save_eval_results_roundtrip(
    env_config: EnvConfig, circuits: list[Circuit], tmp_path: Path
) -> None:
    """评估结果应可保存为 JSON 并完整读回。"""
    results = {
        "noop": evaluate_noop(circuits, env_config),
        "random": evaluate_random(circuits, env_config, seed=1),
    }
    path = tmp_path / "nested" / "evaluation.json"
    save_eval_results(results, path)
    assert path.exists()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {"noop", "random"}
    greedy_payload = payload["random"]
    assert greedy_payload["method"] == "random"
    assert greedy_payload["n_circuits"] == len(circuits)
    assert greedy_payload["equivalence_rate"] == pytest.approx(1.0)
    assert set(greedy_payload["mean_rule_counts"]) == {rule.name for rule in RULES}
