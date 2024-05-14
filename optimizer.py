# optimizer.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# 定义PPO类，实现近端策略优化算法

class PPO:
    """
    近端策略优化（Proximal Policy Optimization，PPO）算法。

    一种强化学习算法，通过信任区域方法进行策略更新，平衡探索与利用。
    """

    def __init__(self, agent, lr=3e-4, gamma=0.99, clip_epsilon=0.2):
        """
        初始化PPO算法。

        参数:
        - agent: 强化学习智能体模型，通常为神经网络
        - lr: 优化器的学习率
        - gamma: 未来奖励的折现因子
        - clip_epsilon: PPO中用于限制策略更新的裁剪参数
        """
        self.agent = agent  # 设置智能体模型
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=lr)  # 使用Adam优化器调整模型参数
        self.gamma = gamma  # 折现因子，用于计算未来奖励的当前价值
        self.clip_epsilon = clip_epsilon  # 策略更新时的裁剪范围界限

    def update(self, states, actions, rewards, dones, old_log_probs, values):
        """
        根据收集的经验数据更新PPO智能体。

        步骤包括计算广义优势估计（GAE）、新旧策略的比率、策略损失、价值损失以及加入熵正则化项的总损失，
        并执行反向传播和参数更新。
        """
        # 计算广义优势估计（GAE）和回报
        returns, advantages = self.compute_gae(rewards, dones, values)

        # 从当前策略中获取新的动作分布和状态价值
        new_policy, new_values = self.agent(states)

        # 计算新策略下动作的对数概率
        new_log_probs = self.compute_log_probs(new_policy, actions)

        # 计算重要性采样比值（ratio）
        ratio = torch.exp(new_log_probs - old_log_probs)

        # 假设advantages是(batch_size, )或(batch_size, sequence_length)的形状
        mean_advantages = advantages.mean(dim=0 if advantages.dim() == 2 else None)  # 正确处理一维或二维张量
        std_advantages = advantages.std(dim=0 if advantages.dim() == 2 else None, unbiased=False)  # 同上，unbiased=False忽略样本数减1
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-8)

        # 计算损失函数的两部分：策略损失（通过裁剪）和价值函数损失（可选的裁剪形式）
        # 假设surr1和surr2的计算方式
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 价值函数损失计算，考虑了裁剪以减少更新波动
        # 假设returns的原始形状是[batch_size, sequence_length]
        returns = returns.sum(dim=0)  # 累加sequence_length维度
        #returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(0)  # 如果returns是Python标量，转换为张量并增加一维
        returns = returns.clone().detach().unsqueeze(0) #这段代码首先使用.clone()创建returns张量的一个副本，这样就断开了与原始计算图的连接。然后，使用.detach()确保新创建的张量不会跟踪任何历史梯度。最后，unsqueeze(0)用于在张量的最前面添加一个维度。这样，即使returns是一个标量，也能得到一个形状为(1, sequence_length)的张量。
        
        # 假设new_values的原始形状是[batch_size, sequence_length]
        new_values_aggregated = new_values.sum(dim=0)  # 累加sequence_length维度
        value_loss = F.mse_loss(new_values_aggregated, returns)
        # 修正 `value_loss` 计算
        #value_loss = F.mse_loss(new_values.squeeze(), returns)
        # 计算策略熵，鼓励探索
        new_policy_normalized = F.softmax(new_policy, dim=-1)
        # 现在使用归一化后的概率分布来计算熵
        entropy = Categorical(probs=new_policy_normalized).entropy().mean()

        # 总损失，结合策略损失、价值损失和熵项
        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        # 执行优化步骤：梯度清零、反向传播、参数更新
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # 返回训练统计信息，供监控和调试
        return {
            '策略损失': policy_loss.item(),
            '价值损失': value_loss.item(),
            '熵': entropy.item(),
            '总损失': total_loss.item(),
            '平均比率': ratio.mean().item()
        }

    def compute_gae(self, rewards, dones, values):
        """
        计算每个时间步的广义优势估计（Generalized Advantage Estimation, GAE）。
        """
        advantages = torch.zeros_like(rewards)
        gae_lambda = 0.95  # GAE中的衰减因子λ
        last_gae_lam = 0

        # 修改循环，使其不包括最后一个元素，避免索引越界
        for t in reversed(range(len(rewards) - 1)):  # 减1以避免索引越界
            next_non_terminal = 1.0 - dones[t+1]
            delta = rewards[t] + self.gamma * values[t+1] * next_non_terminal - values[t]
            advantages[t] = delta + self.gamma * gae_lambda * next_non_terminal * last_gae_lam
            last_gae_lam = advantages[t]

        # 对于序列的第一个元素（t=0），我们需要特殊处理，这里直接设置为累计的优势
        # 注意，因为我们已经处理到了t=1（原序列的倒数第二个元素），所以不需要额外的循环迭代
        #advantages[0] = advantages[0]  # 这一行实际上不需要操作，因为第一个元素的advantage已经在循环中计算好

        returns = advantages + values
        return returns, advantages

    def compute_log_probs(self,policy, actions):
        """
            计算在给定动作分布下选择的动作的对数概率。
    
        参数：
            policy (torch.Tensor): 策略输出，形状为 [batch_size, n_rules, n_qubits * n_moments]
            actions (torch.Tensor): 动作索引，形状为 [batch_size, 2]，其中包括规则索引和量子位时刻索引。
        """
        # 提取规则索引和量子位时刻索引
        rule_indices = actions[:, 0]
        qubit_moment_indices = actions[:, 1]

        # 批次大小
        batch_size = policy.shape[0]

        # 收集对应的概率
        log_probs = torch.zeros(batch_size)
        for i in range(batch_size):
            # 获取单个样本的规则对应的概率分布
            rule_prob = policy[i, rule_indices[i]]

            # 计算未规范化概率
            unnormalized_probs = torch.exp(rule_prob)

            # 规范化概率以满足概率分布的约束
            probs = unnormalized_probs / unnormalized_probs.sum(dim=0, keepdim=True)

            # 处理可能出现的NaN或无穷大值
            probs[torch.isnan(probs)] = 0.0
            probs[probs == float('inf')] = 0.0

            # 创建Categorical分布对象并计算对数概率
            dist = Categorical(probs=probs)
            log_prob = dist.log_prob(qubit_moment_indices[i])
            log_probs[i] = log_prob

        return log_probs