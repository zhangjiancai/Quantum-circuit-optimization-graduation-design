# evaluate.py

import torch
from torch.distributions import Categorical
from agent import CircuitOptimizerAgent
from environment import QuantumCircuitEnvironment, ActionMask
from rules import RULES

# 配置
N_QUBITS = 12
N_MOMENTS = 50
N_GATE_CLASSES = 4
N_RULES = len(RULES)

# 加载已训练的智能体
agent = CircuitOptimizerAgent(N_QUBITS, N_MOMENTS, N_GATE_CLASSES, N_RULES)
agent.load_state_dict(torch.load('rl_agent.pth'))
agent.eval()

# 创建模拟器环境
env = QuantumCircuitEnvironment(N_QUBITS, N_MOMENTS, RULES)
action_mask = ActionMask(N_RULES, N_QUBITS, N_MOMENTS)

# 评估循环
state = env.reset()
done = False
total_reward = 0

while not done:
    with torch.no_grad():
        policy, _ = agent(state.unsqueeze(0))
        policy = policy.view(N_RULES, N_QUBITS * N_MOMENTS)
        masked_policy = policy * action_mask.mask(env.simulator.circuit)
        #action_dist = Categorical(masked_policy)
        action_dist = Categorical(masked_policy.flatten())
        action = action_dist.sample()

    state, reward, done = env.apply_rule(action.item())
    total_reward += reward
    print(f"Action: {action.item()}, Reward: {reward:.3f}")

print(f"Total Reward: {total_reward:.3f}")
print("Evaluation Complete.")
