"""配置集中入口与旧版配置对照。

重写版把配置拆成三个不可变 dataclass：

- :class:`qco.environment.EnvConfig` —— 环境（电路规模、观测窗口、代价函数权重、随机种子）；
- :class:`qco.ppo.PPOConfig` —— PPO 超参数；
- :class:`qco.trainer.TrainConfig` —— 训练流程（迭代数、并行环境、评估频率、输出目录）。

这里统一再导出，并给出与原仓库 ``config.py`` 常量的对照表，方便对照论文与答辩：
原项目把电路规模、规则数、训练轮数都写成全局常量，且 ``N_RULES=6`` 与实际的
规则实现（``rules.py`` 里的 ``RULES`` 只有 6 个但其中两条在“加门”）并不一致；
重写版的规则数由 ``len(RULES)`` 自动决定（当前为 4），不再手写常量。
"""

from __future__ import annotations

from qco.environment import EnvConfig
from qco.ppo import PPOConfig
from qco.trainer import TrainConfig

__all__ = [
    "EnvConfig",
    "PPOConfig",
    "TrainConfig",
    "DEFAULT_ENV_CONFIG",
    "DEFAULT_PPO_CONFIG",
    "DEFAULT_TRAIN_CONFIG",
    "LEGACY_CONFIG_MAPPING",
]

DEFAULT_ENV_CONFIG = EnvConfig()
DEFAULT_PPO_CONFIG = PPOConfig()
DEFAULT_TRAIN_CONFIG = TrainConfig()

#: 原仓库 ``config.py`` 常量 -> 重写版对应位置。
LEGACY_CONFIG_MAPPING: dict[str, str] = {
    "N_QUBITS": "EnvConfig.n_qubits（默认 5）",
    "N_MOMENTS": "EnvConfig.n_moments（默认 15）",
    "N_GATE_CLASSES": "len(qco.gates.GATE_NAMES)（默认 11，作为观测通道数）",
    "N_RULES": "len(qco.rules.RULES)（默认 4，由规则列表自动决定）",
    "N_STEPS": "EnvConfig.max_steps（默认 48）",
    "EPOCHS": "TrainConfig.iterations（默认 150）",
    "STEPS_PER_EPOCH": "TrainConfig.steps_per_iter（默认 16）",
    "LEARNING_RATE": "PPOConfig.lr（默认 3e-4）",
    "gamma": "PPOConfig.gamma（默认 0.99）",
    "clip_epsilon": "PPOConfig.clip_epsilon（默认 0.2）",
}
