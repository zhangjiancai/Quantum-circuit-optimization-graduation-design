# train.py

import torch
from torch.distributions import Categorical
from agent import CircuitOptimizerAgent
from environment import QuantumCircuitEnvironment, ActionMask
from optimizer import PPO
from rules import RULES
from collect_episode_data import collect_episode_data
from config import N_QUBITS, N_MOMENTS, N_GATE_CLASSES, N_RULES, EPOCHS, STEPS_PER_EPOCH, LEARNING_RATE, N_STEPS
from tqdm import trange

# 加载 RL 智能体
agent = CircuitOptimizerAgent(N_QUBITS, N_MOMENTS, N_GATE_CLASSES, N_RULES)
ppo = PPO(agent, lr=LEARNING_RATE)

# 创建模拟器环境
env = QuantumCircuitEnvironment(N_QUBITS, N_MOMENTS, RULES, N_GATE_CLASSES)
action_mask = ActionMask(N_RULES, N_QUBITS, N_MOMENTS)

# 训练循环
for epoch in trange(EPOCHS, desc="Epoch"):
    try:
        # 在每个epoch开始时重置环境
        states, actions, rewards, dones, old_log_probs, values = collect_episode_data(agent, env, action_mask, max_steps=STEPS_PER_EPOCH)
        
        # 将数据按步骤划分并批量更新模型
        for step in range(STEPS_PER_EPOCH):
            batch_states = states[step::STEPS_PER_EPOCH]
            batch_actions = actions[step::STEPS_PER_EPOCH]
            batch_rewards = rewards[step::STEPS_PER_EPOCH]
            batch_dones = dones[step::STEPS_PER_EPOCH]
            batch_old_log_probs = old_log_probs[step::STEPS_PER_EPOCH]
            batch_values = values[step::STEPS_PER_EPOCH]

            ppo.update(batch_states, batch_actions, batch_rewards, batch_dones, batch_old_log_probs, batch_values)

    except Exception as e:
        print(f"An error occurred during training: {e}")

    print(f"Epoch {epoch + 1}/{EPOCHS} complete.")

# 保存训练好的模型
torch.save(agent.state_dict(), 'rl_agent.pth')
print("Training Complete. Model saved to 'rl_agent.pth'")