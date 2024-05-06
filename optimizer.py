import torch
import torch.nn.functional as F
from torch.distributions import Categorical

class PPO:
    """Proximal Policy Optimization (PPO) Algorithm."""
    def __init__(self, agent, lr=3e-4, gamma=0.99, clip_epsilon=0.2):
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
