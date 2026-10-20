"""CNN Actor-Critic 策略网络。

网络结构
--------
观测张量 ``[B, N_CHANNELS, n_qubits, n_moments]`` 被当作一张
``n_qubits × n_moments`` 的“图像”：卷积核在比特方向与时隙方向上同时滑动，
因此能直接感知“相邻比特上的门”“相邻时隙上的门”这类局部结构（合并、消去、
移位规则恰好都只依赖这种局部关系）。

主干是 3 层 ``3×3 卷积 + GroupNorm + ELU``，之后分两个头：

- **策略头**：``1×1`` 卷积把隐藏特征直接映射为 ``[B, R, Q, M]`` 的**原始打分
  （logits）**，输出通道数等于规则数 ``R``；
- **价值头**：``1×1`` 卷积 → ELU → Flatten → 全连接 → 标量状态价值 ``[B]``。

归一化层使用 ``GroupNorm`` 而**不是** ``BatchNorm``：PPO 的更新使用小批量、且
单步 rollout 的数据相关性很强，BatchNorm 的批统计量会引入噪声，还会让
“训练模式”和“推理模式”的行为不一致；GroupNorm 与批大小无关，更适合本任务。

数值约定
--------
网络只输出 logits。动作分布的唯一一次归一化发生在 :meth:`CircuitPolicyNet.act`
里的 :func:`qco.masking.masked_log_softmax`；**绝不**先 softmax 再取 log 或再次
归一化（原项目的“双重 softmax”正是动作概率错乱的根因）。``log_probs.exp()``
只是把已经归一化好的对数概率还原成概率供 ``torch.multinomial`` 采样，没有再除以
任何归一化常数。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from qco.gates import N_CHANNELS
from qco.masking import NEG_INF, masked_entropy, masked_log_softmax
from qco.rules import RULES

__all__ = ["CircuitPolicyNet"]


def _make_group_norm(channels: int) -> nn.GroupNorm:
    """返回分组数能整除 ``channels`` 的 ``GroupNorm``。

    ``GroupNorm`` 要求通道数能被分组数整除；这里优先用 8 组（常规配置
    ``hidden_channels=64`` 时即 8 组），退化到 4/2/1 组以兼容小网络的测试配置。
    """
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    raise ValueError(f"无法为 {channels} 个通道选择 GroupNorm 分组数")  # pragma: no cover


class CircuitPolicyNet(nn.Module):
    """用于量子电路重写优化的 CNN Actor-Critic 网络。

    Parameters
    ----------
    in_channels
        观测通道数，默认 ``N_CHANNELS``（11 个门通道 + 控制/目标/空闲 3 个通道）。
    n_rules
        重写规则数，也是策略输出通道数 ``R``。
    n_qubits
        量子比特数 ``Q``，必须与环境的观测一致。
    n_moments
        时隙窗口宽度 ``M``，必须与环境的观测一致。
    hidden_channels
        共享卷积分支的隐藏通道数 ``H``。

    Attributes
    ----------
    trunk
        共享卷积主干（3×3 卷积 + GroupNorm + ELU，无 BatchNorm）。
    policy_head
        1×1 卷积策略头，输出形状 ``[B, R, Q, M]``。
    value_head
        价值头，输出形状 ``[B, 1]``（``forward`` 返回时压成 ``[B]``）。
    """

    def __init__(
        self,
        in_channels: int = N_CHANNELS,
        n_rules: int = len(RULES),
        n_qubits: int = 5,
        n_moments: int = 15,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        in_channels = int(in_channels)
        n_rules = int(n_rules)
        n_qubits = int(n_qubits)
        n_moments = int(n_moments)
        hidden_channels = int(hidden_channels)
        if min(in_channels, n_rules, n_qubits, n_moments, hidden_channels) <= 0:
            raise ValueError("in_channels / n_rules / n_qubits / n_moments / hidden_channels 必须为正整数")

        self.in_channels = in_channels
        self.n_rules = n_rules
        self.n_qubits = n_qubits
        self.n_moments = n_moments
        self.hidden_channels = hidden_channels

        # 共享卷积主干：感受野随层数扩大，能覆盖相邻比特/相邻时隙的局部模式。
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            _make_group_norm(hidden_channels),
            nn.ELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            _make_group_norm(hidden_channels),
            nn.ELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            _make_group_norm(hidden_channels),
            nn.ELU(),
        )
        # 策略头：逐位置的 1×1 卷积，把每个 (规则, 比特, 时隙) 的三元组打成 logit。
        self.policy_head = nn.Conv2d(hidden_channels, n_rules, kernel_size=1)
        # 价值头：先降维到 32 通道，再摊平为向量，最后回归一个标量状态价值。
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, kernel_size=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(32 * n_qubits * n_moments, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )
        self._init_parameters()

    # ------------------------------------------------------------------
    # 初始化与输入校验
    # ------------------------------------------------------------------
    def _init_parameters(self) -> None:
        """正交初始化：卷积/全连接用默认增益，策略头用小增益让初始策略接近均匀。"""
        for module in self.trunk:
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=2.0**0.5)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        for module in self.value_head:
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=2.0**0.5)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

    def _validate_observations(self, observations: torch.Tensor) -> None:
        """校验观测张量的形状是否为 ``[B, C, Q, M]`` 且与网络配置一致。"""
        if not torch.is_tensor(observations):
            raise TypeError("observations 必须是 torch.Tensor")
        if observations.dim() != 4:
            raise ValueError(
                f"observations 形状必须是 [B, C, Q, M]，收到 {tuple(observations.shape)}"
            )
        expected = (self.in_channels, self.n_qubits, self.n_moments)
        if tuple(observations.shape[1:]) != expected:
            raise ValueError(
                f"observations 后三维应为 {expected}，收到 {tuple(observations.shape[1:])}"
            )

    def _flatten_masks(self, masks: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        """把动作掩码规整为 ``[B, R*Q*M]`` 的布尔张量。

        支持三种常见布局：``[R, Q, M]``（自动广播到整个批次）、
        ``[B, R, Q, M]`` 与已展平的 ``[B, R*Q*M]``。展平顺序必须与
        :mod:`qco.actions` 的动作编码一致（``(rule*Q + q)*M + m``），
        这里直接按行优先 ``reshape``，与该布局天然一致。
        """
        if not torch.is_tensor(masks):
            masks = torch.as_tensor(masks)
        mask = masks.to(device=observations.device, dtype=torch.bool)
        batch_size = int(observations.shape[0])
        action_size = self.n_rules * self.n_qubits * self.n_moments
        mask_shape = (self.n_rules, self.n_qubits, self.n_moments)
        if mask.dim() == 3:
            if tuple(mask.shape) != mask_shape:
                raise ValueError(f"masks 形状应为 {mask_shape}，收到 {tuple(mask.shape)}")
            mask = mask.unsqueeze(0).expand(batch_size, *mask_shape)
        if mask.dim() == 4:
            if tuple(mask.shape[1:]) != mask_shape:
                raise ValueError(
                    f"masks 后三维应为 {mask_shape}，收到 {tuple(mask.shape[1:])}"
                )
            return mask.reshape(batch_size, action_size)
        if mask.dim() == 2:
            if tuple(mask.shape) != (batch_size, action_size):
                raise ValueError(
                    f"展平后的 masks 形状应为 {(batch_size, action_size)}，收到 {tuple(mask.shape)}"
                )
            return mask
        raise ValueError(
            f"masks 形状必须是 [R, Q, M]、[B, R, Q, M] 或 [B, R*Q*M]，收到 {tuple(mask.shape)}"
        )

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """一次前向同时给出动作 logits 与状态价值。

        Parameters
        ----------
        observations
            形状 ``[B, C, Q, M]`` 的 float32 观测。

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``logits`` 形状 ``[B, R, Q, M]``（**未 softmax** 的原始打分），
            ``value`` 形状 ``[B]``。
        """
        self._validate_observations(observations)
        features = self.trunk(observations)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value

    def policy_logits(self, observations: torch.Tensor) -> torch.Tensor:
        """只计算动作 logits（不计算价值），返回形状 ``[B, R, Q, M]``。"""
        self._validate_observations(observations)
        return self.policy_head(self.trunk(observations))

    # ------------------------------------------------------------------
    # 采样
    # ------------------------------------------------------------------
    def act(
        self,
        observations: torch.Tensor,
        masks: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """按掩码后的策略采样动作，并返回该动作的对数概率、价值与策略熵。

        Parameters
        ----------
        observations
            形状 ``[B, C, Q, M]`` 的观测。
        masks
            可用动作掩码，``[B, R, Q, M]``、``[R, Q, M]`` 或 ``[B, R*Q*M]`` 均可。
        deterministic
            为真时取掩码后 logits 的 ``argmax``（评估用）；否则按类别分布采样。

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            ``(action [B] int64, log_prob [B], value [B], entropy [B])``，
            其中 ``entropy`` 是所有**合法**动作上的策略熵。

        Notes
        -----
        ``logits → masked_log_softmax`` 只做一次；非法动作的 log 概率为
        ``NEG_INF``，其概率 ``exp(NEG_INF)`` 在 float32 下严格等于 0，
        因此采样永远不会选中被掩码的动作。

        Raises
        ------
        ValueError
            批次中某一行（某个环境）的动作掩码全为 ``False``，即没有可执行动作。
            环境在这种状态下本来就是回合终止条件，调用方应先重置环境。
        """
        self._validate_observations(observations)
        batch_size = int(observations.shape[0])
        flat_masks = self._flatten_masks(masks, observations)
        empty_rows = ~flat_masks.any(dim=-1)
        if bool(empty_rows.any()):
            rows = empty_rows.nonzero(as_tuple=False).reshape(-1).tolist()
            raise ValueError(
                f"批次中第 {rows} 行的动作掩码全为 False，没有可采样动作；"
                "环境应在没有可用动作时结束回合并重置"
            )

        features = self.trunk(observations)
        logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        flat_logits = logits.reshape(batch_size, -1)
        # 唯一一次归一化：网络输出 logits，这里只做一次 masked log_softmax。
        log_probs = masked_log_softmax(flat_logits, flat_masks)

        if deterministic:
            masked_logits = flat_logits.masked_fill(~flat_masks, NEG_INF)
            actions = masked_logits.argmax(dim=-1)
        else:
            # log_probs 已经是归一化对数概率，exp 即真实概率；不再做任何再归一化。
            actions = torch.multinomial(log_probs.exp(), num_samples=1).squeeze(-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        entropies = masked_entropy(log_probs, flat_masks)
        return actions.to(torch.int64), action_log_probs, values, entropies

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return (
            f"CircuitPolicyNet(in_channels={self.in_channels}, n_rules={self.n_rules}, "
            f"n_qubits={self.n_qubits}, n_moments={self.n_moments}, "
            f"hidden_channels={self.hidden_channels})"
        )
