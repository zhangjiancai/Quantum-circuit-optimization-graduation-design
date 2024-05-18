import torch
from torch.distributions import Categorical
from agent import CircuitOptimizerAgent
from environment import QuantumCircuitEnvironment, ActionMask
from rules import RULES
from collect_episode_data import collect_episode_data  # 确保已经导入
import os
from config import N_QUBITS, N_MOMENTS, N_RULES, N_GATE_CLASSES, N_STEPS

# 初始化智能体和环境
agent = CircuitOptimizerAgent(N_QUBITS, N_MOMENTS, N_GATE_CLASSES, N_RULES)
env = QuantumCircuitEnvironment(N_QUBITS, N_MOMENTS, RULES, N_GATE_CLASSES)
action_mask = ActionMask(N_RULES, N_QUBITS, N_MOMENTS)

# 检查和加载模型
model_path = 'rl_agent.pth'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型文件 {model_path} 未找到，无法加载模型。")

try:
    agent.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    agent.eval()
except RuntimeError as e:
    print(f"加载模型时遇到错误: {e}")
    exit(1)

# 使用collect_episode_data进行评估
states, actions, rewards, dones, log_probs, values = collect_episode_data(agent, env, action_mask, max_steps=N_STEPS)

# 计算总奖励
total_reward = sum(rewards)
print(f"Total reward accumulated during the evaluation: {total_reward:.3f}")
print("Evaluation complete.")
