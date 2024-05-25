import torch
from agent import CircuitOptimizerAgent
from environment import QuantumCircuitEnvironment, ActionMask
from optimizer import PPO
from rules import RULES
from collect_episode_data import collect_episode_data
from config import N_QUBITS, N_MOMENTS, N_GATE_CLASSES, N_RULES, EPOCHS, LEARNING_RATE, gamma, clip_epsilon
from tqdm import trange

# 初始化RL智能体和环境
agent = CircuitOptimizerAgent(N_QUBITS, N_GATE_CLASSES, N_RULES, N_MOMENTS)
ppo = PPO(agent, lr=LEARNING_RATE, gamma=gamma, clip_epsilon=clip_epsilon)
env = QuantumCircuitEnvironment(N_QUBITS, N_MOMENTS, RULES, N_GATE_CLASSES)
action_mask = ActionMask(N_RULES, N_QUBITS, N_MOMENTS)

# 训练循环
for epoch in trange(EPOCHS, desc="Training Epochs"):
        # 收集一个epoch的数据
        states, actions, rewards, dones, old_log_probs, values = collect_episode_data(agent, env, action_mask,N_GATE_CLASSES)

        # 使用PPO策略更新智能体，不分批次，直接更新整个数据集
        ppo.update(states, actions, rewards, dones, old_log_probs, values)


# 保存模型
torch.save(agent.state_dict(), 'rl_agent.pth')
print("Model saved to 'rl_agent.pth'")
