import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from config import LEARNING_RATE, gamma, clip_epsilon

class PPO:
    """
    近端策略优化（Proximal Policy Optimization，PPO）算法。

    一种强化学习算法，通过信任区域方法进行策略更新，平衡探索与利用。
    """

    def __init__(self, agent, lr=LEARNING_RATE, gamma=gamma, clip_epsilon=clip_epsilon):
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
        #states = states.squeeze(0)
        print("Adjusted states shape:", states.shape)
        new_policy, new_values = self.agent(states)

        # 计算新策略下动作的对数概率
        new_log_probs = self.compute_log_probs(new_policy, actions)

        # 计算重要性采样比值（ratio）
        ratio = torch.exp(new_log_probs - old_log_probs)

        mean_advantages = advantages.mean(dim=0 if advantages.dim() == 2 else None)  # 正确处理一维或二维张量
        std_advantages = advantages.std(dim=0 if advantages.dim() == 2 else None, unbiased=False)  # 同上，unbiased=False忽略样本数减1
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-8)

        # 计算损失函数的两部分：策略损失（通过裁剪）和价值函数损失（可选的裁剪形式）
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 价值函数损失计算，考虑了裁剪以减少更新波动
        returns = returns.clone().detach()
        value_loss = F.mse_loss(new_values, returns)

        # 计算策略熵，鼓励探索
        new_policy_normalized = F.softmax(new_policy, dim=-1)
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

        for t in reversed(range(len(rewards) - 1)):
            next_non_terminal = ~dones[t + 1]
            delta = rewards[t] + self.gamma * values[t + 1] * next_non_terminal - values[t]
            advantages[t] = delta + self.gamma * gae_lambda * next_non_terminal * last_gae_lam
            last_gae_lam = advantages[t]

        returns = advantages + values
        return returns, advantages

    def compute_log_probs(self, policy, actions):
        """
        计算在给定动作分布下选择的动作的对数概率。

        参数：
            policy (torch.Tensor): 策略输出，[probability, num_qubits, num_transform_rules, num_timesteps]
            actions (torch.Tensor): 动作索引，(rule_index, qubit_index, timestep_index)。
        """
        rule_indices = actions[:, 0]
        qubit_indices = actions[:, 1]
        timestep_indices = actions[:, 2]

        log_probs = []
        for i in range(policy.size(0)):
            # 从策略张量中获取指定规则、量子位和时间步的概率分布
            probs = policy[i, qubit_indices[i], rule_indices[i], timestep_indices[i]]
            
            # 创建Categorical分布对象并计算对数概率
            dist = Categorical(probs=probs)
            log_prob = dist.log_prob(torch.tensor([timestep_indices[i]]).to(probs.device))
            log_probs.append(log_prob)

        return torch.stack(log_probs)
