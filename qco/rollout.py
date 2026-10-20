"""并行轨迹采样（Rollout）。

一个 rollout 批次由 ``num_envs`` 个并行环境、每个环境连续执行 ``horizon`` 步组成，
采样结果按 ``[num_envs, horizon, ...]`` 存放，供 PPO 逐环境计算 GAE 后再展平。

两个容易写错、但直接决定训练正确性的细节在这里集中处理：

1. **auto-reset**：某个环境本步 ``done`` 时，先记录该回合的统计量（
   :class:`EpisodeStats`），再立刻用 ``circuit_sampler()`` 采一个新电路
   :meth:`~qco.environment.CircuitOptimizationEnv.reset`。这样并行采样不需要
   为“回合结束”中断，也不会把终止步之后的奖励错误地算进下一回合；
   GAE 用 ``dones`` 把 bootstrap 截断，因此 auto-reset 不影响优势估计。
2. **bootstrap 价值**：``next_values[env]`` 取“horizon 步之后观测”的回报估计。
   对发生过 auto-reset 的环境，这个观测就是新回合的初始观测，但因为该步
   ``done=1``，GAE 不会用它来 bootstrap，所以取新回合初始观测是安全的。
3. **终态环境的兜底**：环境构造时的随机初始电路可能恰好没有任何可用动作
   （此时 :meth:`~qco.environment.CircuitOptimizationEnv.step` 无法执行）。
   :meth:`RolloutCollector.collect` 开始采样前会检查每个环境，终态环境同样用
   ``circuit_sampler()`` 重置，保证交给策略网络的掩码至少有一个合法动作。

采样结果统一存放在 **CPU** 上（即使策略在 GPU 上），便于日志记录与后续
``PPO.update`` 按需搬移到训练设备。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
import torch

from qco.circuit import Circuit
from qco.environment import CircuitOptimizationEnv

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查，避免运行时循环导入
    from qco.agent import CircuitPolicyNet

__all__ = ["RolloutBatch", "FlatRolloutBatch", "EpisodeStats", "RolloutCollector"]

#: auto-reset 时连续采样新电路的最大尝试次数。
#: 随机初始电路几乎总有可用的重写动作；若采样器连续给出“无动作可做”的电路，
#: 说明配置或采样器有问题，此时应显式报错而不是静默死循环。
_MAX_RESAMPLE_ATTEMPTS = 32


@dataclass
class FlatRolloutBatch:
    """把 ``[num_envs, horizon]`` 展平成 ``[num_envs * horizon]`` 后的样本视图。

    主要用于 PPO 的小批量更新；张量通过 ``reshape`` 得到，可能与原批次共享存储。
    """

    observations: torch.Tensor  # [N*H, C, Q, M]
    actions: torch.Tensor  # [N*H] int64
    log_probs: torch.Tensor  # [N*H] float32
    values: torch.Tensor  # [N*H] float32
    rewards: torch.Tensor  # [N*H] float32
    dones: torch.Tensor  # [N*H] float32
    masks: torch.Tensor  # [N*H, R, Q, M] bool

    @property
    def num_samples(self) -> int:
        """展平后的样本数 ``num_envs * horizon``。"""
        return int(self.actions.shape[0])


@dataclass
class RolloutBatch:
    """一个并行 rollout 批次的张量容器。

    Attributes
    ----------
    observations
        ``[num_envs, horizon, C, Q, M]``，每一步**执行动作前**的观测。
    actions
        ``[num_envs, horizon]``，int64 动作下标。
    log_probs
        ``[num_envs, horizon]``，采样时旧策略给出所选动作的 log 概率（PPO 的 ratio 基准）。
    values
        ``[num_envs, horizon]``，旧策略对每一步观测的价值估计。
    rewards
        ``[num_envs, horizon]``，环境返回的塑形奖励。
    dones
        ``[num_envs, horizon]``，float32；1.0 表示该步回合终止/截断（用于截断 GAE）。
    masks
        ``[num_envs, horizon, R, Q, M]``，执行动作时的动作掩码（PPO 重新前向必须用同一组掩码）。
    next_values
        ``[num_envs]``，horizon 步之后观测的 bootstrap 价值。
    """

    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    masks: torch.Tensor
    next_values: torch.Tensor

    @property
    def num_envs(self) -> int:
        """并行环境数。"""
        return int(self.actions.shape[0])

    @property
    def horizon(self) -> int:
        """每个环境连续采样的步数。"""
        return int(self.actions.shape[1])

    def flatten(self) -> FlatRolloutBatch:
        """返回展平后的样本视图（``num_envs * horizon`` 个样本）。

        展平顺序是“先环境、后时间”的行优先顺序；``PPO.compute_gae`` 需要的
        ``[num_envs, horizon]`` 结构由本类保留，展平只用于策略/价值网络的批量更新。
        """
        num_envs = self.num_envs
        horizon = self.horizon
        num_samples = num_envs * horizon
        return FlatRolloutBatch(
            observations=self.observations.reshape(num_samples, *self.observations.shape[2:]),
            actions=self.actions.reshape(num_samples),
            log_probs=self.log_probs.reshape(num_samples),
            values=self.values.reshape(num_samples),
            rewards=self.rewards.reshape(num_samples),
            dones=self.dones.reshape(num_samples),
            masks=self.masks.reshape(num_samples, *self.masks.shape[2:]),
        )


@dataclass
class EpisodeStats:
    """一个完整回合（终止或截断）的统计量。"""

    return_: float
    initial_cost: float
    final_cost: float
    initial_gate_count: int
    final_gate_count: int
    initial_depth: int
    final_depth: int
    steps: int
    truncated: bool
    rule_counts: dict[str, int]


class RolloutCollector:
    """用一组并行环境采样轨迹。

    Parameters
    ----------
    policy
        策略网络；采样时临时切换到 ``eval()`` 模式并在结束后恢复原模式。
    envs
        并行环境序列；初始化时它们应处于某个回合的开头（构造函数已 reset），
        但本类也允许从回合中途开始采样（统计量从回合起点累计）。
    device
        策略所在的设备；观测/掩码会搬到该设备上做前向。

    Notes
    -----
    采样结果（:class:`RolloutBatch`）统一在 CPU 上，设备搬运留给
    :meth:`qco.ppo.PPO.update`。
    """

    def __init__(
        self,
        policy: "CircuitPolicyNet",
        envs: Sequence[CircuitOptimizationEnv],
        device: torch.device | str = "cpu",
    ) -> None:
        self.policy = policy
        self.envs: list[CircuitOptimizationEnv] = list(envs)
        if not self.envs:
            raise ValueError("RolloutCollector 至少需要一个环境")
        self.device = torch.device(device)

        # 所有环境必须共享同样的张量形状，否则无法堆叠成批次。
        shapes = {(env.config.n_qubits, env.config.n_moments) for env in self.envs}
        if len(shapes) != 1:
            raise ValueError(f"所有环境必须共享同样的 (n_qubits, n_moments)，收到 {sorted(shapes)}")
        self.n_qubits, self.n_moments = next(iter(shapes))
        mask_shape = tuple(self.envs[0].action_mask.shape)
        for env in self.envs[1:]:
            if tuple(env.action_mask.shape) != mask_shape:
                raise ValueError(
                    f"所有环境的动作掩码形状必须一致：{mask_shape} vs {tuple(env.action_mask.shape)}"
                )
        self.mask_shape = mask_shape  # (R, Q, M)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------
    def collect(
        self,
        horizon: int,
        circuit_sampler: Callable[[], Circuit],
    ) -> tuple[RolloutBatch, list[EpisodeStats]]:
        """并行采样 ``horizon`` 步，返回轨迹批次与期间结束的回合统计。

        Parameters
        ----------
        horizon
            每个环境连续执行的步数。
        circuit_sampler
            回合结束（``done``）后用于生成新电路的采样器，例如
            ``lambda: random_circuit(n_qubits, depth, rng)``。

        Returns
        -------
        tuple[RolloutBatch, list[EpisodeStats]]
            轨迹批次（CPU 张量），以及本次采样期间结束的回合统计列表
            （按结束顺序追加；没有回合结束时为空列表）。
        """
        horizon = int(horizon)
        if horizon <= 0:
            raise ValueError(f"horizon 必须为正整数，收到 {horizon}")
        if not callable(circuit_sampler):
            raise TypeError("circuit_sampler 必须是可调用对象 Callable[[], Circuit]")

        num_envs = len(self.envs)
        obs_buf = np.zeros(
            (num_envs, horizon, *self.envs[0].observation.shape), dtype=np.float32
        )
        mask_buf = np.zeros((num_envs, horizon, *self.mask_shape), dtype=bool)
        action_buf = np.zeros((num_envs, horizon), dtype=np.int64)
        log_prob_buf = np.zeros((num_envs, horizon), dtype=np.float32)
        value_buf = np.zeros((num_envs, horizon), dtype=np.float32)
        reward_buf = np.zeros((num_envs, horizon), dtype=np.float32)
        done_buf = np.zeros((num_envs, horizon), dtype=np.float32)
        episode_stats: list[EpisodeStats] = []

        policy = self.policy
        was_training = policy.training
        policy.eval()
        try:
            # 环境可能在上一回合已经终止（没有可执行动作）；先统一重置到可用状态。
            self._ensure_active_envs(circuit_sampler)
            for step in range(horizon):
                observations_np = np.stack([env.observation for env in self.envs])
                masks_np = np.stack([env.action_mask for env in self.envs])
                observations_t = torch.as_tensor(observations_np, device=self.device)
                masks_t = torch.as_tensor(masks_np, device=self.device)
                with torch.no_grad():
                    actions, log_probs, values, _entropies = policy.act(observations_t, masks_t)
                actions_np = actions.detach().cpu().numpy()
                log_probs_np = log_probs.detach().cpu().numpy()
                values_np = values.detach().cpu().numpy()

                obs_buf[:, step] = observations_np
                mask_buf[:, step] = masks_np
                for index, env in enumerate(self.envs):
                    action = int(actions_np[index])
                    _obs, reward, done, info = env.step(action)
                    action_buf[index, step] = action
                    log_prob_buf[index, step] = float(log_probs_np[index])
                    value_buf[index, step] = float(values_np[index])
                    reward_buf[index, step] = float(reward)
                    done_buf[index, step] = 1.0 if done else 0.0
                    if done:
                        episode_stats.append(self._episode_stats(env, info))
                        self._reset_env(env, circuit_sampler)

            # horizon 步之后（含 auto-reset 之后）的观测用于 bootstrap 价值。
            final_observations_np = np.stack([env.observation for env in self.envs])
            with torch.no_grad():
                _logits, next_values_t = policy(
                    torch.as_tensor(final_observations_np, device=self.device)
                )
            next_values = next_values_t.detach().to(device="cpu", dtype=torch.float32)
        finally:
            policy.train(was_training)

        batch = RolloutBatch(
            observations=torch.from_numpy(obs_buf),
            actions=torch.from_numpy(action_buf),
            log_probs=torch.from_numpy(log_prob_buf),
            values=torch.from_numpy(value_buf),
            rewards=torch.from_numpy(reward_buf),
            dones=torch.from_numpy(done_buf),
            masks=torch.from_numpy(mask_buf),
            next_values=next_values,
        )
        return batch, episode_stats

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    def _ensure_active_envs(self, circuit_sampler: Callable[[], Circuit]) -> None:
        """保证采样开始时每个环境都有可执行动作；终态环境用采样器重置。"""
        for env in self.envs:
            if not env.has_valid_actions:
                self._reset_env(env, circuit_sampler)

    def _episode_stats(
        self, env: CircuitOptimizationEnv, info: dict
    ) -> EpisodeStats:
        """在 ``reset`` 之前把当前回合的统计量抓取出来（reset 会清空这些状态）。"""
        initial_circuit = env.initial_circuit
        return EpisodeStats(
            return_=float(info.get("episode_return", info.get("return", 0.0))),
            initial_cost=float(env.initial_cost),
            final_cost=float(env.cost),
            initial_gate_count=int(initial_circuit.num_ops),
            final_gate_count=int(env.circuit.num_ops),
            initial_depth=int(initial_circuit.depth),
            final_depth=int(env.circuit.depth),
            steps=int(env.steps),
            truncated=bool(info.get("truncated", False)),
            rule_counts=dict(info.get("rule_counts", env.rule_counts)),
        )

    def _reset_env(
        self, env: CircuitOptimizationEnv, circuit_sampler: Callable[[], Circuit]
    ) -> None:
        """用采样电路 auto-reset 环境；重采样直到新回合至少有一个可用动作。"""
        for _attempt in range(_MAX_RESAMPLE_ATTEMPTS):
            env.reset(circuit_sampler())
            if env.has_valid_actions:
                return
        raise RuntimeError(
            "circuit_sampler 连续采样到的电路都没有可用的重写动作，无法开始新回合；"
            "请检查电路生成参数（深度/比特数/门概率）"
        )
