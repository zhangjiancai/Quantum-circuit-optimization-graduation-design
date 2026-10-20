"""PPO（Proximal Policy Optimization）算法实现。

与原项目的区别（正确性要点）
----------------------------
原项目的 PPO 不可用，原因包括：GAE 少算最后一步、对网络输出先 softmax 再 softmax
（双重 softmax）、更新阶段重新前向时使用了与采样时不同的掩码、ratio 与裁剪公式
混用等。本实现严格按下面的约定执行：

- **逐环境 GAE**：对 ``[num_envs, horizon]`` 的每个环境分别反向递推，
  ``delta_t = r_t + γ v_{t+1} (1-d_t) - v_t``，其中 ``t`` 为最后一步时
  ``v_{t+1} = next_values[env]``；``A_t = delta_t + γλ(1-d_t)A_{t+1}``。
  ``returns = advantages + values``。
- **只做一次 softmax**：网络输出 logits，策略分布由
  :func:`qco.masking.masked_log_softmax` 统一归一化；ratio 定义为
  ``exp(new_log_prob - old_log_prob)``，初始（参数未更新时）严格为 1。
- **同一组掩码**：更新阶段重新前向时使用采样时保存的动作掩码，保证新旧策略
  在同一个合法动作子集上比较。
- 策略损失用裁剪替代目标，价值损失用 ``returns``（detach）上的 MSE，
  熵正则按合法动作计算，梯度裁剪后做一步 Adam，并用近似 KL 做早停。

指标字段与 ``docs/architecture.md`` 第 12 节一致：
``policy_loss, value_loss, entropy, approx_kl, clip_fraction, total_loss,
explained_variance, learning_rate, grad_norm``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from qco.masking import masked_entropy, masked_log_softmax

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查
    from qco.agent import CircuitPolicyNet
    from qco.rollout import RolloutBatch

__all__ = ["PPOConfig", "PPO"]


@dataclass(frozen=True)
class PPOConfig:
    """PPO 超参数。

    Attributes
    ----------
    lr
        Adam 学习率。
    gamma
        回报折扣因子 ``γ``。
    gae_lambda
        GAE 的平滑参数 ``λ``。
    clip_epsilon
        策略比率裁剪范围 ``[1-ε, 1+ε]``。
    value_coef
        价值损失权重。
    entropy_coef
        熵正则权重。
    max_grad_norm
        梯度范数裁剪阈值。
    update_epochs
        每批数据重复更新（epoch）的次数。
    minibatch_size
        小批量大小；样本数不足时自动缩小为该批样本数。
    target_kl
        近似 KL 早停阈值：某个小批量更新后 KL 超过该值即停止本轮剩余更新。
    normalize_advantage
        是否对优势做标准化（零均值、单位标准差）。
    """

    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 64
    target_kl: float = 0.02
    normalize_advantage: bool = True

    def __post_init__(self) -> None:
        if self.lr <= 0.0:
            raise ValueError("lr 必须为正数")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError("gamma 必须在 [0, 1] 内")
        if not (0.0 <= self.gae_lambda <= 1.0):
            raise ValueError("gae_lambda 必须在 [0, 1] 内")
        if self.clip_epsilon < 0.0:
            raise ValueError("clip_epsilon 不能为负")
        if self.value_coef < 0.0 or self.entropy_coef < 0.0:
            raise ValueError("value_coef / entropy_coef 不能为负")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm 必须为正数")
        if self.update_epochs < 1:
            raise ValueError("update_epochs 至少为 1")
        if self.minibatch_size < 1:
            raise ValueError("minibatch_size 至少为 1")
        if self.target_kl <= 0.0:
            raise ValueError("target_kl 必须为正数")


class PPO:
    """PPO 训练器：从 :class:`qco.rollout.RolloutBatch` 计算优势并更新策略。

    Parameters
    ----------
    policy
        Actor-Critic 策略网络（:class:`qco.agent.CircuitPolicyNet`）。
    config
        PPO 超参数。
    device
        训练设备；策略与数据都会被搬移到该设备。

    Attributes
    ----------
    optimizer
        Adam 优化器（测试/训练器可访问，用于记录学习率或做诊断）。
    METRIC_KEYS
        :meth:`update` 返回的指标键，顺序固定。
    """

    #: ``update`` 返回的指标键（与契约一致，用于测试与日志列名）。
    METRIC_KEYS: tuple[str, ...] = (
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "total_loss",
        "explained_variance",
        "learning_rate",
        "grad_norm",
    )

    def __init__(
        self,
        policy: "CircuitPolicyNet",
        config: PPOConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.config = config
        self.policy = policy.to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.lr)

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------
    @staticmethod
    def compute_gae(
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_values: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """逐环境计算 GAE 优势与回报，返回 ``(advantages, returns)``。

        Parameters
        ----------
        rewards
            形状 ``[num_envs, horizon]``（或一维 ``[horizon]``）的奖励。
        values
            与 ``rewards`` 同形状的旧价值估计。
        dones
            与 ``rewards`` 同形状；``1.0`` 表示该步回合已结束（终止或截断）。
        next_values
            形状 ``[num_envs]``；每一步序列之后观测的 bootstrap 价值。
        gamma
            折扣因子。
        gae_lambda
            GAE 平滑参数。

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``advantages`` 与 ``returns = advantages + values``，形状与输入一致。

        Notes
        -----
        反向递推从**最后一步**开始：``t`` 为序列最后一步时用 ``next_values[env]``
        作为 ``v_{t+1}``，因此最后一步的优势不会漏掉 bootstrap 项。``dones`` 在
        每一步同时截断 ``delta`` 中来自下一状态的项与优势的跨步递推，保证
        GAE 不跨越回合边界。
        """
        squeeze = rewards.dim() == 1
        if squeeze:
            rewards_2d = rewards.unsqueeze(0)
            values_2d = values.unsqueeze(0)
            dones_2d = dones.unsqueeze(0)
        else:
            rewards_2d, values_2d, dones_2d = rewards, values, dones
        if rewards_2d.dim() != 2:
            raise ValueError(f"rewards 必须是 [num_envs, horizon] 或 [horizon]，收到 {rewards.shape}")
        if values_2d.shape != rewards_2d.shape or dones_2d.shape != rewards_2d.shape:
            raise ValueError("rewards / values / dones 形状必须一致")
        num_envs, horizon = rewards_2d.shape
        if next_values.dim() == 0:
            next_values = next_values.reshape(1)
        next_values = next_values.reshape(-1)
        if next_values.shape[0] != num_envs:
            raise ValueError(
                f"next_values 应有 {num_envs} 个元素，收到 {next_values.shape[0]}"
            )
        if horizon == 0:
            raise ValueError("horizon 必须大于 0")

        advantages = torch.zeros_like(rewards_2d)
        gae_running = torch.zeros(
            num_envs, dtype=rewards_2d.dtype, device=rewards_2d.device
        )
        with torch.no_grad():
            for step in reversed(range(horizon)):
                if step == horizon - 1:
                    next_value = next_values.to(
                        device=rewards_2d.device, dtype=rewards_2d.dtype
                    )
                else:
                    next_value = values_2d[:, step + 1]
                not_done = 1.0 - dones_2d[:, step]
                delta = (
                    rewards_2d[:, step]
                    + gamma * next_value * not_done
                    - values_2d[:, step]
                )
                gae_running = delta + gamma * gae_lambda * not_done * gae_running
                advantages[:, step] = gae_running
            returns = advantages + values_2d
        if squeeze:
            return advantages.squeeze(0), returns.squeeze(0)
        return advantages, returns

    # ------------------------------------------------------------------
    # 策略前向（更新时复用）
    # ------------------------------------------------------------------
    def _log_probs_and_values(
        self,
        observations: torch.Tensor,
        flat_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """重新前向得到 ``(log_probs [B, R*Q*M], values [B])``。

        只调用一次 :func:`masked_log_softmax`，与采样阶段使用完全相同的归一化，
        因此参数未更新时 ``exp(new_log_prob - old_log_prob) == 1``。
        """
        logits, values = self.policy(observations)
        flat_logits = logits.reshape(logits.shape[0], -1)
        masks = flat_masks.to(device=flat_logits.device, dtype=torch.bool)
        if masks.dim() == 4:  # [B, R, Q, M] -> [B, R*Q*M]
            masks = masks.reshape(masks.shape[0], -1)
        if masks.dim() != 2:
            raise ValueError(f"flat_masks 应为 [B, R*Q*M] 或 [B, R, Q, M]，收到 {tuple(masks.shape)}")
        log_probs = masked_log_softmax(flat_logits, masks)
        return log_probs, values

    # ------------------------------------------------------------------
    # 更新
    # ------------------------------------------------------------------
    def update(self, batch: "RolloutBatch") -> dict[str, float]:
        """用一批 rollout 数据做多轮小批量 PPO 更新，返回训练指标。

        Parameters
        ----------
        batch
            :class:`qco.rollout.RolloutBatch`，形状
            ``observations [N, H, C, Q, M]``、``actions/log_probs/values/rewards/dones
            [N, H]``、``masks [N, H, R, Q, M]``、``next_values [N]``。

        Returns
        -------
        dict[str, float]
            指标字典，键见 :attr:`METRIC_KEYS`。所有指标均为整批更新的加权平均
            （``explained_variance`` / ``learning_rate`` 除外，前者在更新前基于
            rollout 价值计算，后者是当前学习率）。
        """
        cfg = self.config
        device = self.device

        observations = batch.observations.to(device=device, dtype=torch.float32)
        actions = batch.actions.to(device=device, dtype=torch.long)
        old_log_probs = batch.log_probs.to(device=device, dtype=torch.float32)
        values = batch.values.to(device=device, dtype=torch.float32)
        rewards = batch.rewards.to(device=device, dtype=torch.float32)
        dones = batch.dones.to(device=device, dtype=torch.float32)
        masks = batch.masks.to(device=device, dtype=torch.bool)
        next_values = batch.next_values.to(device=device, dtype=torch.float32)

        if observations.dim() != 5:
            raise ValueError(f"observations 应为 [N, H, C, Q, M]，收到 {tuple(observations.shape)}")
        num_envs, horizon = rewards.shape
        if actions.shape != rewards.shape or values.shape != rewards.shape:
            raise ValueError("actions / values / rewards 形状必须一致")
        if old_log_probs.shape != rewards.shape or dones.shape != rewards.shape:
            raise ValueError("log_probs / dones / rewards 形状必须一致")
        if masks.shape[:2] != (num_envs, horizon):
            raise ValueError(f"masks 前两维应为 {(num_envs, horizon)}，收到 {tuple(masks.shape[:2])}")
        if next_values.shape != (num_envs,):
            raise ValueError(f"next_values 应为 [{num_envs}]，收到 {tuple(next_values.shape)}")
        num_samples = num_envs * horizon

        # 1) 逐环境 GAE；returns = advantages + values。
        advantages, returns = self.compute_gae(
            rewards, values, dones, next_values, cfg.gamma, cfg.gae_lambda
        )
        returns = returns.detach()
        values_flat = values.reshape(-1)
        returns_flat = returns.reshape(-1)

        # 2) 更新前基于 rollout 价值计算解释方差（训练监控指标）。
        returns_var = returns_flat.var(unbiased=False)
        if float(returns_var) > 1e-12:
            residual_var = (returns_flat - values_flat).var(unbiased=False)
            explained_variance = float(1.0 - residual_var / returns_var)
        else:  # 回报几乎是常数时解释方差没有意义
            explained_variance = 0.0

        # 3) 展平成样本，按需做优势标准化。
        flat_observations = observations.reshape(num_samples, *observations.shape[2:])
        flat_actions = actions.reshape(-1)
        flat_old_log_probs = old_log_probs.reshape(-1)
        flat_masks = masks.reshape(num_samples, -1)
        flat_advantages = advantages.reshape(-1)
        if cfg.normalize_advantage and num_samples > 1:
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (
                flat_advantages.std(unbiased=False) + 1e-8
            )

        # 4) 多轮小批量更新。
        minibatch_size = min(int(cfg.minibatch_size), num_samples)
        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "total_loss": 0.0,
            "grad_norm": 0.0,
        }
        processed = 0
        early_stopped = False
        self.policy.train()
        for _epoch in range(cfg.update_epochs):
            permutation = torch.randperm(num_samples, device=device)
            for start in range(0, num_samples, minibatch_size):
                index = permutation[start : start + minibatch_size]
                batch_count = int(index.numel())
                mb_observations = flat_observations[index]
                mb_actions = flat_actions[index]
                mb_old_log_probs = flat_old_log_probs[index]
                mb_advantages = flat_advantages[index]
                mb_returns = returns_flat[index]
                mb_masks = flat_masks[index]

                # 同一组掩码上重新前向：策略分布只做一次 masked_log_softmax。
                log_probs, new_values = self._log_probs_and_values(
                    mb_observations, mb_masks
                )
                new_log_probs = log_probs.gather(1, mb_actions.unsqueeze(1)).squeeze(1)
                # ratio = exp(new - old)，参数未更新时严格为 1。
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                unclipped = ratio * mb_advantages
                clipped = (
                    torch.clamp(
                        ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon
                    )
                    * mb_advantages
                )
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(new_values, mb_returns)
                entropy = masked_entropy(log_probs, mb_masks).mean()
                total_loss = (
                    policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                )

                with torch.no_grad():
                    approx_kl = float((mb_old_log_probs - new_log_probs).mean())
                    clip_fraction = float(
                        ((ratio - 1.0).abs() > cfg.clip_epsilon).to(torch.float32).mean()
                    )

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), cfg.max_grad_norm
                    )
                )
                self.optimizer.step()

                totals["policy_loss"] += float(policy_loss.detach()) * batch_count
                totals["value_loss"] += float(value_loss.detach()) * batch_count
                totals["entropy"] += float(entropy.detach()) * batch_count
                totals["total_loss"] += float(total_loss.detach()) * batch_count
                totals["approx_kl"] += approx_kl * batch_count
                totals["clip_fraction"] += clip_fraction * batch_count
                totals["grad_norm"] += grad_norm * batch_count
                processed += batch_count

                # 近似 KL 早停：KL 已经过大说明本批数据被“过度更新”，停止剩余小批量。
                if approx_kl > cfg.target_kl:
                    early_stopped = True
                    break
            if early_stopped:
                break

        metrics = {
            name: (value_sum / processed if processed > 0 else 0.0)
            for name, value_sum in totals.items()
        }
        metrics["explained_variance"] = explained_variance
        metrics["learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
        return metrics
