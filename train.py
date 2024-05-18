# train.py

import torch
import traceback
from torch.distributions import Categorical
from agent import CircuitOptimizerAgent
from environment import QuantumCircuitEnvironment, ActionMask
from optimizer import PPO
from rules import RULES
from collect_episode_data import collect_episode_data
from config import N_QUBITS, N_MOMENTS, N_GATE_CLASSES, N_RULES, EPOCHS, STEPS_PER_EPOCH, LEARNING_RATE, N_STEPS,gamma,clip_epsilon
from tqdm import trange

# 加载 RL 智能体
agent = CircuitOptimizerAgent(N_QUBITS, N_MOMENTS, N_GATE_CLASSES, N_RULES)
ppo = PPO(agent, lr=LEARNING_RATE,gamma=gamma,clip_epsilon=clip_epsilon)


# 创建模拟器环境
env = QuantumCircuitEnvironment(N_QUBITS, N_MOMENTS, RULES, N_GATE_CLASSES)
action_mask = ActionMask(N_RULES, N_QUBITS, N_MOMENTS)
def print_model_parameters(model):
    
    print("Model Parameters:")
    print(f"N_QUBITS: {N_QUBITS}, N_MOMENTS: {N_MOMENTS}, N_GATE_CLASSES: {N_GATE_CLASSES}, N_RULES: {N_RULES}")
    print("policy_linear output size:", agent.policy_linear.out_features)

def check_tensor_format(tensor, expected_shape, name):
    if not tensor.size() == expected_shape:
        raise ValueError(f"{name} has an unexpected shape. Expected {expected_shape}, got {tensor.size()}.")
    
# 训练循环
for epoch in trange(EPOCHS, desc="Epoch"):
    try:
        # 在每个epoch开始时重置环境
        states, actions, rewards, dones, old_log_probs, values = collect_episode_data(agent, env, action_mask, max_steps=STEPS_PER_EPOCH)
        check_tensor_format(actions, (STEPS_PER_EPOCH,2), "Actions")
        # 将数据按步骤划分并批量更新模型
        for step in range(STEPS_PER_EPOCH):
            batch_states = states[step::STEPS_PER_EPOCH]
            batch_actions = actions[step::STEPS_PER_EPOCH]
            batch_rewards = rewards[step::STEPS_PER_EPOCH]
            batch_dones = dones[step::STEPS_PER_EPOCH]
            batch_old_log_probs = old_log_probs[step::STEPS_PER_EPOCH]
            batch_values = values[step::STEPS_PER_EPOCH]

            ppo.update(batch_states, batch_actions, batch_rewards, batch_dones, batch_old_log_probs, batch_values)
            #print_model_parameters(agent)

    except Exception as e:
        # 获取详细的错误信息和堆栈跟踪
        error_message = f"""
        An error occurred during training:
        Error Type: {type(e).__name__}
        Description: {e}
    
        Traceback:
        {traceback.format_exc()}
        """
        print(error_message)

    print(f"Epoch {epoch + 1}/{EPOCHS} complete.")

# 保存训练好的模型
torch.save(agent.state_dict(), 'rl_agent.pth')
print("Training Complete. Model saved to 'rl_agent.pth'")