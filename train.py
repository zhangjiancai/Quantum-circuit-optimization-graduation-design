# train.py

import torch
from torch.distributions import Categorical
from agent import CircuitOptimizerAgent
from environment import QuantumCircuitEnvironment, ActionMask
from optimizer import PPO
from rules import RULES
from collect_episode_data import collect_episode_data
from config import N_QUBITS, N_MOMENTS, N_GATE_CLASSES, N_RULES, EPOCHS, STEPS_PER_EPOCH, LEARNING_RATE

# 加载 RL 智能体
agent = CircuitOptimizerAgent(N_QUBITS, N_MOMENTS, N_GATE_CLASSES, N_RULES)
ppo = PPO(agent, lr=LEARNING_RATE)

# 创建模拟器环境
env = QuantumCircuitEnvironment(N_QUBITS, N_MOMENTS, RULES, N_GATE_CLASSES)
action_mask = ActionMask(N_RULES, N_QUBITS, N_MOMENTS)

# 训练循环
for epoch in range(EPOCHS):
    state = env.reset()
    states, actions, rewards, dones, old_log_probs, values = collect_episode_data(state, STEPS_PER_EPOCH)
    ppo.update(states, actions, rewards, dones, old_log_probs, values)
    print(f"Epoch {epoch + 1}/{EPOCHS} complete.")

# 保存训练好的模型
torch.save(agent.state_dict(), 'rl_agent.pth')
print("Training Complete. Model saved to 'rl_agent.pth'")
