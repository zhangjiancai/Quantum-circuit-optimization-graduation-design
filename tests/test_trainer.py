"""训练器测试：极小配置端到端跑通、产物文件齐全、检查点可恢复、训练可复现。

覆盖点
------
- 2 iterations 的极小配置能产出全部约定产物（模块级 fixture 只训练一次，其余测试复用）；
- ``best.pt`` 保存的是**评估指标最好那一轮**的权重，重新加载后评估结果与
  ``summary.json`` 记录的 ``best_eval`` 一致（回归测试：曾出现 best 权重被最后一轮覆盖）；
- ``eval_dataset`` 可配置：显式数据集优先、缺失报错、规模不匹配报错、缺省时现场生成；
- 相同种子与配置的训练可复现。
"""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("qco.agent")
pytest.importorskip("qco.ppo")
pytest.importorskip("qco.rollout")

from qco.agent import CircuitPolicyNet  # noqa: E402
from qco.environment import EnvConfig  # noqa: E402
from qco.evaluate import evaluate_agent  # noqa: E402
from qco.gates import N_CHANNELS  # noqa: E402
from qco.generator import random_circuits, save_circuits  # noqa: E402
from qco.ppo import PPOConfig  # noqa: E402
from qco.rules import RULES  # noqa: E402
from qco.trainer import (  # noqa: E402
    DEFAULT_EVAL_DATASET,
    EVAL_LOG_COLUMNS,
    PROJECT_ROOT,
    TRAIN_LOG_COLUMNS,
    TrainConfig,
    Trainer,
    build_policy_from_checkpoint,
    load_checkpoint,
)

#: 验收要求必须存在的产物文件。
REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "config.json",
    "train_log.csv",
    "eval_log.csv",
    "best.pt",
    "last.pt",
    "summary.json",
)

#: 测试用评估集电路数（刻意与 TrainConfig.eval_circuits 不同，以验证数据集优先级）。
DATASET_CIRCUITS = 3


def make_dataset_file(
    path: Path,
    *,
    n_qubits: int = 3,
    depth: int = 5,
    count: int = DATASET_CIRCUITS,
    seed: int = 123,
) -> Path:
    """生成一个固定种子的评估数据集文件并返回其路径。"""
    circuits = random_circuits(count, n_qubits, depth, seed=seed, p_two=0.2)
    save_circuits(circuits, path)
    return path


def make_tiny_config(
    out_dir: Path,
    *,
    iterations: int = 2,
    eval_every: int = 2,
    eval_dataset: str | None = None,
    eval_circuits: int = 2,
) -> TrainConfig:
    """构造 CPU 上几秒内即可跑完的极小训练配置。"""
    return TrainConfig(
        env=EnvConfig(
            n_qubits=3,
            n_moments=8,
            max_steps=6,
            lambda_depth=0.5,
            initial_depth=5,
            p_two=0.2,
            seed=3,
        ),
        ppo=PPOConfig(lr=1e-3, update_epochs=1, minibatch_size=8),
        iterations=iterations,
        steps_per_iter=4,
        num_envs=2,
        hidden_channels=8,
        eval_every=eval_every,
        save_every=1,
        eval_circuits=eval_circuits,
        train_circuits=4,
        eval_seed=123,
        out_dir=str(out_dir),
        device="cpu",
        eval_dataset=eval_dataset,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    """读取 CSV 为字典列表。"""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory: pytest.TempPathFactory):
    """训练一次极小配置，返回 ``(config, summary, run_dir)``。

    评估集使用临时目录里显式生成的数据集文件，避免测试读取项目根目录下的
    ``data/eval.json``（正式实验数据，规模大且会拖慢测试）。
    """
    base = tmp_path_factory.mktemp("trainer_run")
    dataset_path = make_dataset_file(base / "eval.json")
    config = make_tiny_config(
        base / "run", eval_every=1, eval_dataset=str(dataset_path)
    )
    trainer = Trainer(config)
    summary = trainer.train()
    return config, summary, base / "run"


def test_training_creates_all_artifacts(trained_run) -> None:
    """2 iterations 的极小训练应产出全部约定文件，且日志内容合法。"""
    config, summary, run_dir = trained_run

    for name in REQUIRED_ARTIFACTS:
        assert (run_dir / name).exists(), f"缺少产物文件 {name}"

    # config.json 完整记录配置（含显式评估数据集路径）
    config_payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config_payload["env"]["n_qubits"] == config.env.n_qubits
    assert config_payload["iterations"] == config.iterations
    assert config_payload["hidden_channels"] == config.hidden_channels
    assert config_payload["eval_dataset"] == config.eval_dataset

    # train_log.csv：每个 iteration 一行，列名与数值合法
    train_rows = _read_csv(run_dir / "train_log.csv")
    assert len(train_rows) == config.iterations
    assert tuple(train_rows[0].keys()) == TRAIN_LOG_COLUMNS
    for row in train_rows:
        assert int(row["iteration"]) >= 1
        for column in TRAIN_LOG_COLUMNS[1:]:
            value = float(row[column])
            assert np.isfinite(value), f"列 {column} 不是有限数值：{row[column]}"

    # eval_log.csv：至少包含 agent 行，最后一轮评估包含三种基线
    eval_rows = _read_csv(run_dir / "eval_log.csv")
    assert eval_rows, "eval_log.csv 不应为空"
    assert tuple(eval_rows[0].keys()) == EVAL_LOG_COLUMNS
    methods = {row["method"] for row in eval_rows}
    assert {"noop", "random", "greedy", "agent"} <= methods
    for row in eval_rows:
        assert float(row["equivalence_rate"]) == pytest.approx(1.0)

    # summary.json：四种方法的最终评估；n_circuits 证明用的是显式数据集（而非 eval_circuits 现场生成）
    summary_payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert set(summary_payload["results"]) == {"noop", "random", "greedy", "agent"}
    for method, payload in summary_payload["results"].items():
        assert payload["method"] == method
        assert payload["equivalence_rate"] == pytest.approx(1.0)
        assert 0.0 <= payload["cost_reduction_pct"] <= 100.0 + 1e-6
        assert payload["n_circuits"] == DATASET_CIRCUITS
    # 主结果表的 agent 行必须与 best.pt 对应的最优轮指标一致；
    # 最后一轮指标单独记录在 final_eval（供曲线分析），不参与主结果表。
    assert summary_payload["results"]["agent"] == summary_payload["best_eval"]
    assert summary_payload["final_eval"]["method"] == "agent"
    assert summary_payload["final_eval"]["n_circuits"] == DATASET_CIRCUITS
    assert 1 <= summary_payload["best_iteration"] <= config.iterations
    assert summary_payload["train_time_s"] > 0.0
    assert "| noop |" in summary_payload["comparison_table"]
    assert summary["results"] == summary_payload["results"]
    assert summary["final_eval"] == summary_payload["final_eval"]
    assert summary["out_dir"] == str(run_dir)


def test_best_checkpoint_weights_match_reported_best(tmp_path: Path) -> None:
    """``best.pt`` 必须是“最优评估那一轮”的权重，而不是最后一轮权重。

    回归测试：修复前 ``best.pt`` 在训练结束时才写盘，保存的是最后一轮权重却
    标注 ``best_eval``；这里用 3 iterations + 每轮评估，把 best.pt 重新加载后
    在评估集上复算，指标必须与 ``summary.json`` 的 ``best_eval`` 一致。
    """
    import qco.trainer as trainer_module

    monkeypatch_dataset = tmp_path / "unused_default.json"
    original_dataset = trainer_module.DEFAULT_EVAL_DATASET
    trainer_module.DEFAULT_EVAL_DATASET = monkeypatch_dataset
    try:
        dataset_path = make_dataset_file(tmp_path / "eval.json")
        config = make_tiny_config(
            tmp_path / "run", iterations=3, eval_every=1, eval_dataset=str(dataset_path)
        )
        trainer = Trainer(config)
        summary = trainer.train()
    finally:
        trainer_module.DEFAULT_EVAL_DATASET = original_dataset

    best_checkpoint = load_checkpoint(tmp_path / "run" / "best.pt")
    best_policy, env_config, _ = build_policy_from_checkpoint(best_checkpoint, device="cpu")
    recomputed = evaluate_agent(
        best_policy, trainer.eval_circuits, env_config, device="cpu", deterministic=True
    )

    reported = summary["best_eval"]
    assert reported is not None
    assert best_checkpoint["iteration"] == summary["best_iteration"]
    # checkpoint 内元数据、重载后复算、summary 三者一致（评估是确定性的，误差仅来自浮点）。
    assert recomputed.mean_final_cost == pytest.approx(reported["mean_final_cost"], abs=1e-6)
    assert best_checkpoint["eval"]["mean_final_cost"] == pytest.approx(
        reported["mean_final_cost"], abs=1e-6
    )
    # 主结果表中的 agent 行也必须是“最优轮”指标，才能和 best.pt 的评估数字对齐。
    assert summary["results"]["agent"] == reported

    # final_eval 记录最后一轮指标：用 last.pt 重新评估应当复现它。
    final_eval = summary["final_eval"]
    last_checkpoint = load_checkpoint(tmp_path / "run" / "last.pt")
    last_policy, _, _ = build_policy_from_checkpoint(last_checkpoint, device="cpu")
    recomputed_last = evaluate_agent(
        last_policy, trainer.eval_circuits, env_config, device="cpu", deterministic=True
    )
    assert final_eval is not None
    assert final_eval["method"] == "agent"
    assert recomputed_last.mean_final_cost == pytest.approx(
        final_eval["mean_final_cost"], abs=1e-6
    )

    # 历史 bug 的判据：最优轮次不是最后一轮时，best.pt 权重必须与 last.pt 不同；
    # 若相等说明 best.pt 又退化成“最后一轮权重”。
    if summary["best_iteration"] < config.iterations:
        best_state = best_checkpoint["state_dict"]
        last_state = last_checkpoint["state_dict"]
        assert set(best_state) == set(last_state)
        assert any(
            not torch.allclose(best_state[key], last_state[key], atol=0.0)
            for key in best_state
        ), "best.pt 权重与 last.pt 完全相同：best.pt 保存了错误轮次的权重"


def test_checkpoints_are_recoverable(trained_run) -> None:
    """best.pt / last.pt / 周期性检查点都能重建策略网络并正常前向。"""
    config, _summary, run_dir = trained_run

    assert (run_dir / "checkpoint_0001.pt").exists()
    last_checkpoint = load_checkpoint(run_dir / "last.pt")
    assert last_checkpoint["iteration"] == config.iterations
    best_checkpoint = load_checkpoint(run_dir / "best.pt")
    assert 1 <= best_checkpoint["iteration"] <= config.iterations
    assert "eval" in best_checkpoint

    for name in ("best.pt", "last.pt"):
        checkpoint = load_checkpoint(run_dir / name)
        policy, env_config, raw_config = build_policy_from_checkpoint(checkpoint, device="cpu")
        assert isinstance(policy, CircuitPolicyNet)
        assert env_config.n_qubits == config.env.n_qubits
        assert env_config.n_moments == config.env.n_moments
        assert raw_config["hidden_channels"] == config.hidden_channels

        observations = torch.zeros(
            (2, N_CHANNELS, config.env.n_qubits, config.env.n_moments), dtype=torch.float32
        )
        with torch.no_grad():
            logits, values = policy(observations)
        assert tuple(logits.shape) == (
            2,
            len(RULES),
            config.env.n_qubits,
            config.env.n_moments,
        )
        assert tuple(values.shape) == (2,)

    # --device auto 也必须在检查点加载路径上生效（无 GPU 时自动回退 CPU）。
    policy_auto, env_auto, _ = build_policy_from_checkpoint(best_checkpoint, device="auto")
    assert env_auto.n_qubits == config.env.n_qubits
    assert next(policy_auto.parameters()).device.type in ("cpu", "cuda")


def test_training_is_deterministic(tmp_path: Path) -> None:
    """相同种子与配置的两次训练应得到一致的训练曲线与最终参数。"""
    dataset_path = make_dataset_file(tmp_path / "eval.json")
    runs = []
    for sub_dir in ("a", "b"):
        config = make_tiny_config(
            tmp_path / sub_dir, iterations=1, eval_every=1, eval_dataset=str(dataset_path)
        )
        trainer = Trainer(config)
        trainer.train()
        runs.append((tmp_path / sub_dir, trainer))

    first_rows = _read_csv(runs[0][0] / "train_log.csv")
    second_rows = _read_csv(runs[1][0] / "train_log.csv")
    assert len(first_rows) == len(second_rows) == 1
    for column in TRAIN_LOG_COLUMNS:
        assert first_rows[0][column] == second_rows[0][column], column

    first_state = runs[0][1].policy.state_dict()
    second_state = runs[1][1].policy.state_dict()
    assert set(first_state) == set(second_state)
    for key in first_state:
        assert torch.allclose(first_state[key], second_state[key], atol=1e-6), key


def test_train_config_validation() -> None:
    """非法训练配置应在构造时立即报错。"""
    with pytest.raises(ValueError):
        TrainConfig(iterations=0)
    with pytest.raises(ValueError):
        TrainConfig(eval_every=0)
    with pytest.raises(ValueError):
        TrainConfig(steps_per_iter=0)
    with pytest.raises(ValueError):
        TrainConfig(num_envs=0)


def test_default_eval_dataset_is_project_root_relative() -> None:
    """评估集默认路径相对项目根解析，不受当前工作目录影响。"""
    assert PROJECT_ROOT == Path(__file__).resolve().parents[1]
    assert DEFAULT_EVAL_DATASET == PROJECT_ROOT / "data" / "eval.json"
    assert DEFAULT_EVAL_DATASET.is_absolute()


def test_explicit_eval_dataset_is_loaded(tmp_path: Path) -> None:
    """显式 ``eval_dataset`` 优先于现场生成（即使 ``eval_circuits`` 更小）。"""
    dataset_path = make_dataset_file(tmp_path / "dataset.json", count=5)
    config = dataclasses.replace(
        make_tiny_config(tmp_path / "run", eval_circuits=2, eval_dataset=str(dataset_path)),
        eval_seed=999999,
    )
    trainer = Trainer(config)
    assert len(trainer.eval_circuits) == 5
    assert all(circuit.n_qubits == 3 for circuit in trainer.eval_circuits)


def test_explicit_missing_eval_dataset_raises(tmp_path: Path) -> None:
    """显式给出但不存在的数据集路径应报错（而不是静默现场生成）。"""
    config = make_tiny_config(
        tmp_path / "run", eval_dataset=str(tmp_path / "does_not_exist.json")
    )
    with pytest.raises(FileNotFoundError):
        Trainer(config)


def test_eval_dataset_scale_mismatch_raises(tmp_path: Path) -> None:
    """数据集规模（比特数）与训练配置不一致时应报错并提示重新生成。"""
    dataset_path = make_dataset_file(tmp_path / "eval7.json", n_qubits=4)
    config = make_tiny_config(tmp_path / "run", eval_dataset=str(dataset_path))
    with pytest.raises(ValueError, match="比特数"):
        Trainer(config)


def test_generates_eval_set_when_no_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """未配置数据集且默认文件不存在时，按 ``eval_seed`` 现场生成评估集。"""
    import qco.trainer as trainer_module

    monkeypatch.setattr(
        trainer_module, "DEFAULT_EVAL_DATASET", tmp_path / "missing_default.json"
    )
    config = make_tiny_config(tmp_path / "run", eval_circuits=4, eval_dataset=None)
    trainer = Trainer(config)
    assert len(trainer.eval_circuits) == 4
    expected = random_circuits(
        4, config.env.n_qubits, config.env.initial_depth, seed=config.eval_seed, p_two=config.env.p_two
    )
    assert trainer.eval_circuits == expected


def test_trainer_honors_env_and_network_options(tmp_path: Path) -> None:
    """``TrainConfig`` 中的 ``lambda_depth`` 与网络尺寸必须传给环境与策略网络。"""
    dataset_path = make_dataset_file(
        tmp_path / "eval4.json", n_qubits=4, depth=6, count=2
    )
    config = dataclasses.replace(
        make_tiny_config(tmp_path / "run", eval_dataset=str(dataset_path)),
        env=EnvConfig(
            n_qubits=4,
            n_moments=9,
            max_steps=7,
            lambda_depth=1.5,
            initial_depth=6,
            p_two=0.2,
            seed=3,
        ),
        hidden_channels=12,
        num_envs=3,
    )
    trainer = Trainer(config)
    assert len(trainer.envs) == 3
    for env in trainer.envs:
        assert env.config.lambda_depth == pytest.approx(1.5)
        assert env.config.n_qubits == 4
        assert env.config.max_steps == 7
    assert trainer.policy.n_qubits == 4
    assert trainer.policy.n_moments == 9
    assert trainer.policy.hidden_channels == 12
    assert trainer.collector.n_qubits == 4
    # 代价函数确实使用 λ=1.5：noop 代价 = 门数 + 1.5 × 深度。
    first_circuit = trainer.eval_circuits[0]
    assert first_circuit.cost(1.5) == pytest.approx(
        first_circuit.num_ops + 1.5 * first_circuit.depth
    )
