"""基于强化学习的量子电路优化（Quantum Circuit Optimization, QCO）。

重写版毕业设计代码包。模块划分：

- :mod:`qco.gates`       门库与门矩阵
- :mod:`qco.circuit`      不可变电路表示
- :mod:`qco.simulator`    numpy 状态向量 / 幺正矩阵模拟器
- :mod:`qco.observation`  电路 -> 张量观测
- :mod:`qco.generator`    随机电路与数据集
- :mod:`qco.actions`      动作编码
- :mod:`qco.rules`        等价重写规则
- :mod:`qco.masking`      动作掩码
- :mod:`qco.environment`  RL 环境
- :mod:`qco.agent`        CNN Actor-Critic 策略网络
- :mod:`qco.ppo`          PPO 算法
- :mod:`qco.rollout`      轨迹采样
- :mod:`qco.trainer`      训练器
- :mod:`qco.baselines`    随机 / 贪心基线
- :mod:`qco.evaluate`     评估与对比
- :mod:`qco.visualizer`   训练曲线与电路图
- :mod:`qco.config`       配置集中入口与旧版常量对照

轻量模块（门库、电路、模拟器、规则、环境）在 ``import qco`` 时直接可用；
依赖 torch 的模块通过模块级 ``__getattr__`` 延迟导入，避免只做电路模拟时
被迫加载 PyTorch。
"""

from __future__ import annotations

from qco.actions import (
    action_space_size,
    decode_action,
    decode_actions,
    encode_action,
    encode_actions,
)
from qco.circuit import Circuit, Op
from qco.environment import CircuitOptimizationEnv, EnvConfig
from qco.gates import GATES, GATE_NAMES, N_CHANNELS
from qco.generator import (
    load_circuits,
    random_circuit,
    random_circuits,
    save_circuits,
)
from qco.masking import build_action_mask
from qco.observation import encode_circuit
from qco.rules import RULES, RewriteRule, applicable_actions
from qco.simulator import circuits_equivalent, circuit_unitary
from qco.utils import set_seed

__version__ = "1.0.0"

_LAZY_EXPORTS: dict[str, str] = {
    # 名称 -> 所在模块
    "CircuitPolicyNet": "qco.agent",
    "PPO": "qco.ppo",
    "PPOConfig": "qco.ppo",
    "RolloutBatch": "qco.rollout",
    "RolloutCollector": "qco.rollout",
    "EpisodeStats": "qco.rollout",
    "Trainer": "qco.trainer",
    "TrainConfig": "qco.trainer",
    "OptimizationResult": "qco.baselines",
    "optimize_greedy": "qco.baselines",
    "optimize_random": "qco.baselines",
    "EvalResult": "qco.evaluate",
    "evaluate_agent": "qco.evaluate",
    "evaluate_greedy": "qco.evaluate",
    "evaluate_random": "qco.evaluate",
}

__all__ = [
    "__version__",
    "Circuit",
    "Op",
    "GATES",
    "GATE_NAMES",
    "N_CHANNELS",
    "CircuitOptimizationEnv",
    "EnvConfig",
    "RULES",
    "RewriteRule",
    "applicable_actions",
    "action_space_size",
    "encode_action",
    "decode_action",
    "encode_actions",
    "decode_actions",
    "build_action_mask",
    "encode_circuit",
    "random_circuit",
    "random_circuits",
    "save_circuits",
    "load_circuits",
    "circuit_unitary",
    "circuits_equivalent",
    "set_seed",
    *_LAZY_EXPORTS,
]


def __getattr__(name: str):
    """模块级延迟导入（PEP 562）：只在真正访问时加载 torch 相关模块。"""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value  # 缓存，后续访问不再走 __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
