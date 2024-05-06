# collect_episode_data.py

import torch
from torch.distributions import Categorical
from agent import CircuitOptimizerAgent
from environment import QuantumCircuitEnvironment, ActionMask
from rules import RULES
from config import N_QUBITS, N_MOMENTS, N_GATE_CLASSES, N_RULES

# 加载 RL 智能体
agent = CircuitOptimizerAgent(N_QUBITS, N_MOMENTS, N_GATE_CLASSES, N_RULES)

# 创建模拟器环境
env = QuantumCircuitEnvironment(N_QUBITS, N_MOMENTS, RULES, N_GATE_CLASSES)
action_mask = ActionMask(N_RULES, N_QUBITS, N_MOMENTS)

def collect_episode_data(state, steps):
    """收集一整集的训练数据"""
    states, actions, rewards, dones, old_log_probs, values = [], [], [], [], [], []

    for _ in range(steps):
        state = state.squeeze(0)  # 移除多余的第一个维度
        with torch.no_grad():
            policy, value = agent(state.unsqueeze(0))  # 重新添加批次维度
            policy = policy.view(N_RULES, N_QUBITS * N_MOMENTS)
            masked_policy = policy * action_mask.mask(env.simulator.circuit, N_GATE_CLASSES)
            masked_policy = masked_policy.view(-1)

            # 防止所有动作被屏蔽的情况
            if masked_policy.sum() == 0:
                masked_policy = torch.ones_like(masked_policy)

            action_dist = Categorical(masked_policy)
            action = action_dist.sample()
            # 打印动作索引的取值范围
            print("Action index range:", 0, "-", len(env.rules) - 1)
            old_log_prob = action_dist.log_prob(action)
            print("Selected action index:", action.item())
            # 打印模型输出的策略
            print("Policy probabilities:", masked_policy)

        next_state, reward, done = env.apply_rule(action.item())

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        old_log_probs.append(old_log_prob)
        values.append(value)

        state = next_state
        if done:
            break

    states = torch.stack(states)
    actions = torch.tensor(actions, dtype=torch.int64)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)
    old_log_probs = torch.stack(old_log_probs)
    values = torch.stack(values)

    return states, actions, rewards, dones, old_log_probs, values

# 添加简单测试功能
if __name__ == "__main__":
    # 初始状态
    initial_state = env.reset()
    steps = 10

    # 收集训练数据
    states, actions, rewards, dones, old_log_probs, values = collect_episode_data(initial_state, steps)

    # 打印测试结果
    print("Collect Episode Data Test Complete")
    print(f"States Shape: {states.shape}")
    print(f"Actions Shape: {actions.shape}")
    print(f"Rewards: {rewards}")
    print(f"Dones: {dones}")
    print(f"Old Log Probs: {old_log_probs}")
    print(f"Values: {values}")
