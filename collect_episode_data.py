# collect_episode_data.py

import torch
from torch.distributions import Categorical
import logging
from agent import CircuitOptimizerAgent
from environment import QuantumCircuitEnvironment, ActionMask
from rules import RULES
from config import N_QUBITS, N_MOMENTS, N_GATE_CLASSES, N_RULES, N_STEPS

# 设置日志记录
logging.basicConfig(level=logging.INFO)

# 加载 RL 智能体
agent = CircuitOptimizerAgent(N_QUBITS, N_MOMENTS, N_GATE_CLASSES, N_RULES)

# 创建模拟器环境
env = QuantumCircuitEnvironment(N_QUBITS, N_MOMENTS, RULES, N_GATE_CLASSES)
action_mask = ActionMask(N_RULES, N_QUBITS, N_MOMENTS)

def select_action(state, masked_policy, action_mask, env):
    """选择动作的逻辑，包括应用动作掩码和处理全零情况"""
    if masked_policy.sum() == 0:  # 如果所有动作被禁止，采取随机未禁止动作
        allowed_indices = torch.nonzero(action_mask.mask(env.simulator.circuit, N_GATE_CLASSES)).flatten()
        action = torch.randint(low=0, high=len(allowed_indices), size=(1,))
        action = allowed_indices[action]
    else:
        action_dist = Categorical(masked_policy.view(-1))
        action = action_dist.sample()
    return action

def collect_episode_data(agent, env, action_mask, max_steps=N_STEPS):
    """收集一整集的训练数据，进行了效率和健壮性的优化"""
    # 初始化数据容器
    states = torch.empty((max_steps, 1) + env.simulator.get_state().shape[1:], dtype=torch.float32)
    actions = torch.empty(max_steps, dtype=torch.int64)
    rewards = torch.empty(max_steps, dtype=torch.float32)
    dones = torch.empty(max_steps, dtype=torch.bool)
    old_log_probs = torch.empty(max_steps, dtype=torch.float32)
    values = torch.empty(max_steps, dtype=torch.float32)
    
    step_count = 0
    state = env.reset().squeeze(0)

    masked_policy = None  # 用于优化重复计算

    while step_count < max_steps:
        with torch.no_grad():
            policy, value = agent(state.unsqueeze(0))
            policy = policy.view(N_RULES, N_QUBITS * N_MOMENTS)
            
            if masked_policy is None or masked_policy.shape != policy.shape:  # 动作掩码不变时避免重复计算
                masked_policy = policy * action_mask.mask(env.simulator.circuit, N_GATE_CLASSES)

            action = select_action(state, masked_policy, action_mask, env)            
            old_log_prob = Categorical(masked_policy.view(-1)).log_prob(action)

        try:
            next_state, reward, done = env.apply_rule(action.item())
        except Exception as e:
            logging.error(f"Error applying rule: {e}")
            break  # 发生异常时终止循环

        state = next_state

        # 填充数据
        states[step_count] = state.unsqueeze(0)
        actions[step_count] = action
        rewards[step_count] = reward
        dones[step_count] = done
        old_log_probs[step_count] = old_log_prob
        values[step_count] = value
        
        if done:
            break
        step_count += 1

    # 截断未使用的部分（如果有必要）
    if step_count < max_steps:
        logging.warning(f"Episode ended before max_steps: {N_STEPS}")
    states = states[:step_count]
    actions = actions[:step_count]
    rewards = rewards[:step_count]
    dones = dones[:step_count]
    old_log_probs = old_log_probs[:step_count]
    values = values[:step_count]

    return states, actions, rewards, dones, old_log_probs, values
'''
if __name__ == "__main__":
    logging.info("Collect Episode Data Test Start")
    
    # 初始化环境和重置状态
    env = QuantumCircuitEnvironment(N_QUBITS, N_MOMENTS, RULES, N_GATE_CLASSES)
    action_mask = ActionMask(N_RULES, N_QUBITS, N_MOMENTS)
    initial_state = env.reset()
    
    # 收集训练数据，修正参数传递
    states, actions, rewards, dones, old_log_probs, values = collect_episode_data(agent, env, action_mask, max_steps=N_STEPS)

    # 打印测试结果
    logging.info("Collect Episode Data Test Complete")
    logging.info(f"States Shape: {states.shape}")
    logging.info(f"Actions Shape: {actions.shape}")
    logging.info(f"Rewards: {rewards}")
    logging.info(f"Dones: {dones}")
    logging.info(f"Old Log Probs: {old_log_probs}")
    logging.info(f"Values: {values}")
'''