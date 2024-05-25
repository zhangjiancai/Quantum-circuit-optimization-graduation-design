# collect_episode_data.py
import torch
from torch.distributions import Categorical
from config import N_QUBITS, N_MOMENTS, N_GATE_CLASSES, N_RULES, N_STEPS


class NoAvailableActionError(Exception):
    """自定义异常，表示没有可用的动作。"""
    pass


def _choose_random_action(allowed_indices):
    """
    从允许的动作中随机选择一个。

    参数：
        allowed_indices (Tensor): 允许的动作索引集合。

    返回：
        action_index (int): 随机选择的动作索引。
    """
    if len(allowed_indices) == 0:
        raise NoAvailableActionError("没有可用的动作。")
    random_index = torch.randint(low=0, high=len(allowed_indices), size=(1,))
    return allowed_indices[random_index].item()



def select_action(state, masked_policy, action_mask, env, n_gate_classes):
    """
    选择动作的逻辑，包括处理被屏蔽的动作。

    参数：
        state (Tensor): 当前状态。
        masked_policy (Tensor): 屏蔽后的策略分布。
        action_mask (ActionMask): ActionMask 实例。
        env (QuantumCircuitEnvironment): 环境实例。
        n_gate_classes (int): 门类数。

    返回：
        action (tuple): 选择的动作索引 (rule_index, qubit_index, timestep_index)。
        old_log_prob (Tensor): 选择动作的对数概率。
    """
    try:
        num_qubits = env.simulator.n_qubits
        num_timesteps = env.simulator.n_moments
        num_transform_rules = len(env.rules)

        # 将 masked_policy 调整为四维形状 (num_qubits, num_transform_rules, num_timesteps)
        masked_policy = masked_policy.view(num_qubits, num_transform_rules, num_timesteps)

        # 检查是否所有动作都被屏蔽
        if masked_policy.sum() == 0:
            # 如果所有动作都被屏蔽，则随机选择一个允许的动作
            allowed_indices = torch.nonzero(action_mask.mask(env.simulator.circuit, n_gate_classes)).flatten()
            action_index = _choose_random_action(allowed_indices)
            old_log_prob = torch.tensor(0.0)
        else:
            # 从非屏蔽的动作中选择一个动作
            qubit_index = torch.randint(0, num_qubits, (1,)).item()
            timestep_index = torch.randint(0, num_timesteps, (1,)).item()

            rule_probs = masked_policy[qubit_index, :, timestep_index]
            action_dist = Categorical(rule_probs)
            rule_index = action_dist.sample()
            old_log_prob = action_dist.log_prob(rule_index)
        action = torch.tensor([rule_index, qubit_index, timestep_index], dtype=torch.int64)

        return action, old_log_prob

    except Exception as e:
        print(f"An error occurred during action selection: {e}")
        raise

def collect_episode_data(agent, env, action_mask, N_GATE_CLASSES):
    states, actions, rewards, dones, old_log_probs, values = [], [], [], [], [], []

    state = env.reset()
    done = False
    
    while not done:
        # Fetch policy and value estimates for the current state
        policy, value = agent(state)
        
        # Apply the action mask to the fetched policy
        masked_policy = policy * action_mask.mask(env.simulator.circuit, N_GATE_CLASSES)
        
        # Select an action based on the masked policy and store log probability of the action
        action, old_log_prob = select_action(state, masked_policy, action_mask, env, N_GATE_CLASSES)
        
        # Apply the selected action to the environment to get new state and reward
        next_state, reward, done = env.apply_rule(action)

        # Store data (ensure all data are tensors before appending)
        states.append(state.clone())
        actions.append(action.clone())
        rewards.append(torch.tensor([reward], dtype=torch.float32))  # Ensure rewards are tensors
        dones.append(torch.tensor([done], dtype=torch.bool))  # Ensure dones are tensors
        old_log_probs.append(torch.tensor([old_log_prob], dtype=torch.float32))  # Ensure log probs are tensors
        values.append(value.clone())
        
        # Update state
        state = next_state

    # Convert lists to tensors
    states = torch.stack(states)
    actions = torch.stack(actions)
    rewards = torch.stack(rewards).squeeze()  # Removing unnecessary batch dimension
    dones = torch.stack(dones).squeeze()  # Removing unnecessary batch dimension
    old_log_probs = torch.stack(old_log_probs).squeeze()  # Removing unnecessary batch dimension
    values = torch.stack(values).squeeze()  # Removing unnecessary batch dimension    

    print(f"States: {states.shape}, Actions: {actions.shape}, Rewards: {rewards.shape}, Dones: {dones.shape}, Old Log Probs: {old_log_probs.shape}, Values: {values.shape}")

    return states, actions, rewards, dones, old_log_probs, values

