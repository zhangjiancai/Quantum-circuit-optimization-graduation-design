# evaluate.py
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
        returns = returns.sum(dim=1, keepdim=True)  # 累加sequence_length维度，添加keepdim保持二维结构
        # 假设new_values的原始形状是[batch_size, sequence_length]
        new_values_aggregated = new_values.sum(dim=1, keepdim=True)  # 累加sequence_length维度，保持二维结构
        value_loss = F.mse_loss(new_values_aggregated, returns)

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

    def compute_log_probs(self, policy_dist, actions):
        """
        计算在给定动作分布下选择的动作的对数概率。

        确保在传递给Categorical分布之前进行规范化。
        """
        # 计算未规范化概率
        unnormalized_probs = torch.exp(policy_dist)

        # 规范化概率以满足概率分布的约束
        probs = unnormalized_probs / unnormalized_probs.sum(dim=-1, keepdim=True)

        # 处理可能出现的NaN或无穷大值
        # 将它们设为0以避免数值问题
        probs[torch.isnan(probs)] = 0.0
        probs[probs == float('inf')] = 0.0

        # 创建Categorical分布对象并计算对数概率
        dist = Categorical(probs=probs)
        log_probs = dist.log_prob(actions)  # 计算对数概率

        return log_probs
       
class DummyAgent(nn.Module):
    """用于测试的简易代理模型"""
    def __init__(self):
        super(DummyAgent, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2_policy = nn.Linear(20, 3)
        self.fc2_value = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        policy = F.softmax(self.fc2_policy(x), dim=-1)
        value = self.fc2_value(x)
        return policy, value

def test_ppo():
    """测试 PPO 类"""
    # 创建测试数据
    agent = DummyAgent()
    ppo = PPO(agent)
    states = torch.randn(5, 10)
    actions = torch.tensor([0, 1, 2, 1, 0], dtype=torch.int64)
    rewards = torch.tensor([1.0, 0.5, -0.5, 1.0, 2.0], dtype=torch.float32)
    dones = torch.tensor([0, 0, 0, 1, 1], dtype=torch.float32)
    old_log_probs = torch.tensor([-0.1, -0.2, -0.3, -0.4, -0.5], dtype=torch.float32)
    values = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float32)

    # 更新代理模型
    ppo.update(states, actions, rewards, dones, old_log_probs, values)

    # 打印一些测试数据
    print("PPO Update Test Complete")
    print(f"States: {states}")
    print(f"Actions: {actions}")
    print(f"Rewards: {rewards}")
    print(f"Dones: {dones}")
    print(f"Old Log Probs: {old_log_probs}")
    print(f"Values: {values}")

#if __name__ == '__main__':
#    test_ppo()
