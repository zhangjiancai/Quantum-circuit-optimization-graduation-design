# evaluate.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PPO:
    """Proximal Policy Optimization (PPO) Algorithm."""
    def __init__(self, agent, lr=3e-4, gamma=0.99, clip_epsilon=0.2):
        """
        Initialize the PPO algorithm.

        :param agent: The reinforcement learning agent model
        :param lr: Learning rate
        :param gamma: Discount factor
        :param clip_epsilon: Clipping parameter for PPO
        """
        self.agent = agent
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

    def update(self, states, actions, rewards, dones, old_log_probs, values):
        """Update PPO agent."""
        returns, advantages = self.compute_advantages(rewards, dones, values)
        policy, values = self.agent(states)
        new_log_probs = self.compute_log_probs(policy, actions)

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, returns)

        loss = policy_loss + 0.5 * value_loss - 0.01 * policy.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def compute_advantages(self, rewards, dones, values):
        """Compute advantages using rewards and values."""
        returns = []
        advantage = 0
        for i in reversed(range(len(rewards))):
            mask = 1 - dones[i]
            advantage = rewards[i] + self.gamma * values[i + 1] * mask - values[i]
            returns.append(advantage)
        returns = torch.tensor(list(reversed(returns)), dtype=torch.float32)
        advantages = returns - values[:-1]
        return returns, advantages

    def compute_log_probs(self, policy, actions):
        """Compute log probabilities of the selected actions."""
        dist = Categorical(policy)
        return dist.log_prob(actions)

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
