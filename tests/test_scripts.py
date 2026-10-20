"""脚本入口测试：``--help`` 可用、数据集生成、评估流水线与插图生成。

所有脚本都以子进程方式从项目根目录运行，验证真实用户的使用路径
（包括 ``python scripts/xxx.py`` 与根目录薄封装 ``python train.py``）。
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
trainer_module = pytest.importorskip("qco.trainer")
pytest.importorskip("qco.agent")

from qco.agent import CircuitPolicyNet  # noqa: E402
from qco.environment import EnvConfig  # noqa: E402
from qco.evaluate import evaluate_greedy, evaluate_noop  # noqa: E402
from qco.generator import random_circuits, save_circuits  # noqa: E402
from qco.rules import RULES  # noqa: E402
from qco.utils import config_to_dict, to_jsonable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

#: 需要验证 ``--help`` 的入口脚本（含根目录薄封装）。
ENTRY_SCRIPTS: tuple[str, ...] = (
    "scripts/make_dataset.py",
    "scripts/train.py",
    "scripts/evaluate.py",
    "scripts/make_figures.py",
    "train.py",
    "evaluate.py",
)


def run_script(
    args: list[str],
    timeout: float = 180.0,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """以子进程方式运行脚本；``cwd`` 默认项目根目录。"""
    return subprocess.run(
        [PYTHON, *args],
        cwd=str(cwd if cwd is not None else ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


@pytest.mark.parametrize("script", ENTRY_SCRIPTS)
def test_script_help_runs(script: str) -> None:
    """所有脚本都必须支持 ``--help`` 并以退出码 0 结束。"""
    result = run_script([script, "--help"])
    assert result.returncode == 0, f"{script} --help 失败：\n{result.stderr}"
    assert "usage" in result.stdout.lower()


def test_make_dataset_from_outside_project_root(tmp_path: Path) -> None:
    """脚本在项目根目录之外调用时仍能 import qco 并解析路径参数。"""
    dataset_path = tmp_path / "data" / "eval.json"
    result = run_script(
        [
            str(ROOT / "scripts" / "make_dataset.py"),
            "--out",
            str(dataset_path),
            "--count",
            "2",
            "--n-qubits",
            "3",
            "--depth",
            "4",
            "--seed",
            "5",
        ],
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert dataset_path.exists()

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    assert payload["meta"]["count"] == 2
    assert payload["meta"]["n_qubits"] == 3
    assert len(payload["circuits"]) == 2


def _write_checkpoint(path: Path, n_qubits: int, n_moments: int, hidden_channels: int) -> None:
    """构造一个随机初始化策略网络的检查点文件（不依赖训练）。"""
    env_config = EnvConfig(
        n_qubits=n_qubits,
        n_moments=n_moments,
        max_steps=6,
        initial_depth=4,
        p_two=0.2,
        seed=0,
    )
    policy = CircuitPolicyNet(
        n_rules=len(RULES),
        n_qubits=n_qubits,
        n_moments=n_moments,
        hidden_channels=hidden_channels,
    )
    torch.save(
        {
            "iteration": 1,
            "config": {
                "env": config_to_dict(env_config),
                "hidden_channels": hidden_channels,
            },
            "state_dict": policy.state_dict(),
        },
        path,
    )


def test_evaluate_script_end_to_end(tmp_path: Path) -> None:
    """``evaluate.py`` 应加载检查点、评估四种方法并写出 evaluation.json。"""
    n_qubits, n_moments = 3, 8
    checkpoint_path = tmp_path / "model.pt"
    _write_checkpoint(checkpoint_path, n_qubits, n_moments, hidden_channels=8)
    dataset_path = tmp_path / "eval.json"
    save_circuits(random_circuits(2, n_qubits, 4, seed=5, p_two=0.2), dataset_path)
    out_dir = tmp_path / "eval_out"
    result = run_script(
        [
            str(ROOT / "scripts" / "evaluate.py"),
            "--checkpoint",
            str(checkpoint_path),
            "--dataset",
            str(dataset_path),
            "--out",
            str(out_dir),
            "--baselines",
            "random,greedy",
            "--device",
            "auto",
            "--seed",
            "7",
        ],
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "|" in result.stdout  # Markdown 对比表输出到 stdout

    evaluation_path = out_dir / "evaluation.json"
    assert evaluation_path.exists()
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert set(payload) == {"noop", "random", "greedy", "agent"}
    for method, metrics in payload.items():
        assert metrics["method"] == method
        assert metrics["n_circuits"] == 2
        assert metrics["equivalence_rate"] == pytest.approx(1.0)


def test_make_figures_generates_pngs(tmp_path: Path) -> None:
    """``make_figures.py`` 应依据训练产物生成论文插图 PNG。"""
    n_qubits, n_moments = 3, 8
    dataset_path = tmp_path / "eval.json"
    circuits = random_circuits(2, n_qubits, 4, seed=5, p_two=0.2)
    save_circuits(circuits, dataset_path)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # 合成训练日志（只需列名与数值合法，用于测试绘图流程）
    with (run_dir / "train_log.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(trainer_module.TRAIN_LOG_COLUMNS)
        for iteration in (1, 2):
            writer.writerow([iteration, 0.1 * iteration, 5.0, 20.0, 18.0, 10.0, 5.0, 0.1, 0.5, 1.0, 0.01, 0.2])
    with (run_dir / "eval_log.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(trainer_module.EVAL_LOG_COLUMNS)
        for iteration in (1, 2):
            writer.writerow([iteration, "agent", 10.0, 8.0, 4.0, 1.0, 12.0])

    env_config = EnvConfig(
        n_qubits=n_qubits,
        n_moments=n_moments,
        max_steps=6,
        initial_depth=4,
        p_two=0.2,
        seed=0,
    )
    results = {
        "noop": to_jsonable(evaluate_noop(circuits, env_config)),
        "greedy": to_jsonable(evaluate_greedy(circuits, env_config)),
    }
    (run_dir / "summary.json").write_text(
        json.dumps({"results": results}, ensure_ascii=False), encoding="utf-8"
    )

    out_dir = tmp_path / "figures"
    result = run_script(
        [
            "scripts/make_figures.py",
            "--run",
            str(run_dir),
            "--out",
            str(out_dir),
            "--dataset",
            str(dataset_path),
        ]
    )
    assert result.returncode == 0, result.stderr
    expected = (
        "training_curves.png",
        "method_comparison.png",
        "rule_usage.png",
        "circuit_before.png",
        "circuit_after.png",
    )
    for name in expected:
        path = out_dir / name
        assert path.exists(), f"缺少插图 {name}"
        assert path.stat().st_size > 0

    # 每张图都能被 matplotlib 重新打开，确认不是损坏文件
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.image as matplotlib_image

    for name in expected:
        image = matplotlib_image.imread(str(out_dir / name))
        assert image.ndim in (2, 3)


def test_train_script_parameters_are_wired() -> None:
    """``train.py`` 的命令行参数必须真正进入训练配置（含 --lambda-depth / --device auto）。"""
    sys.path.insert(0, str(ROOT))
    from scripts.train import build_config, build_parser

    args = build_parser().parse_args(
        [
            "--out",
            "results/_wiring_check",
            "--iterations",
            "3",
            "--num-envs",
            "2",
            "--steps-per-iter",
            "5",
            "--seed",
            "7",
            "--device",
            "auto",
            "--max-steps",
            "20",
            "--lambda-depth",
            "1.5",
            "--n-qubits",
            "4",
            "--n-moments",
            "9",
            "--initial-depth",
            "6",
            "--hidden-channels",
            "12",
            "--eval-every",
            "2",
            "--lr",
            "1e-3",
            "--dataset",
            "data/eval7.json",
        ]
    )
    config = build_config(args)
    assert config.env.lambda_depth == pytest.approx(1.5)
    assert config.env.n_qubits == 4
    assert config.env.n_moments == 9
    assert config.env.max_steps == 20
    assert config.device == "auto"
    assert config.iterations == 3
    assert config.steps_per_iter == 5
    assert config.num_envs == 2
    assert config.hidden_channels == 12
    assert config.eval_every == 2
    assert config.ppo.lr == pytest.approx(1e-3)
    # 相对路径参数统一解析到项目根目录，保证从任意 cwd 调用结果一致。
    assert Path(config.out_dir).is_absolute()
    assert Path(config.out_dir) == ROOT / "results" / "_wiring_check"
    assert config.eval_dataset is not None
    assert Path(config.eval_dataset) == ROOT / "data" / "eval7.json"

    # 不传 --dataset 时保持 None，交由 Trainer 使用默认 data/eval.json 或现场生成。
    default_config = build_config(build_parser().parse_args([]))
    assert default_config.eval_dataset is None


def test_train_script_with_custom_dataset(tmp_path: Path) -> None:
    """``train.py --dataset`` 指向不同规模（3 比特）的数据集时应能跑通。"""
    dataset_path = tmp_path / "eval3.json"
    save_circuits(random_circuits(2, 3, 4, seed=5, p_two=0.2), dataset_path)
    out_dir = tmp_path / "run3"
    result = run_script(
        [
            str(ROOT / "scripts" / "train.py"),
            "--out",
            str(out_dir),
            "--dataset",
            str(dataset_path),
            "--n-qubits",
            "3",
            "--n-moments",
            "8",
            "--initial-depth",
            "4",
            "--max-steps",
            "6",
            "--lambda-depth",
            "1.5",
            "--iterations",
            "1",
            "--num-envs",
            "1",
            "--steps-per-iter",
            "2",
            "--eval-every",
            "1",
            "--save-every",
            "0",
            "--eval-circuits",
            "2",
            "--train-circuits",
            "2",
            "--hidden-channels",
            "8",
            "--device",
            "auto",
            "--seed",
            "1",
        ],
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr

    config_payload = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    assert config_payload["env"]["n_qubits"] == 3
    assert config_payload["env"]["lambda_depth"] == pytest.approx(1.5)
    assert config_payload["eval_dataset"] == str(dataset_path)

    summary_payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert set(summary_payload["results"]) == {"noop", "random", "greedy", "agent"}
    for payload in summary_payload["results"].values():
        assert payload["n_circuits"] == 2  # 用的是 --dataset 指定的 2 个电路
        assert payload["equivalence_rate"] == pytest.approx(1.0)
    # 主结果表 agent 行 = 最优轮指标；最后一轮指标单独记录在 final_eval。
    assert summary_payload["results"]["agent"] == summary_payload["best_eval"]
    assert summary_payload["final_eval"]["method"] == "agent"
