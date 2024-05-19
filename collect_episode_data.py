# collect_episode_data.py
import torch
from torch.distributions import Categorical
from config import N_QUBITS, N_MOMENTS, N_GATE_CLASSES, N_RULES, N_STEPS,batch_size


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
        action (tuple): 选择的动作索引 (rule_index, qubit_moment_index)。
        old_log_prob (Tensor): 选择动作的对数概率。
    """
    try:
        masked_policy = masked_policy.view(-1)
        if masked_policy.sum() == 0:
            # 如果所有动作都被屏蔽，则随机选择一个允许的动作
            allowed_indices = torch.nonzero(action_mask.mask(env.simulator.circuit, n_gate_classes)).flatten()
            action_index = _choose_random_action(allowed_indices)
            old_log_prob = torch.tensor(0.0)
        else:
            # 从非屏蔽的动作中选择一个动作
            action_dist = Categorical(masked_policy)
            action = action_dist.sample()
            old_log_prob = action_dist.log_prob(action)
            action_index = action.item()

        # 计算动作对应的规则索引和量子比特-时刻索引
        rule_index = action_index // (env.simulator.n_qubits * env.simulator.n_moments)
        qubit_moment_index = action_index % (env.simulator.n_qubits * env.simulator.n_moments)
        # 确保选择的规则索引在有效范围内
        if not (0 <= rule_index < len(env.rules)):
            raise IndexError(f"Selected rule index {rule_index} is out of range. Valid range is [0, {len(env.rules) - 1}].")

        return (rule_index, qubit_moment_index), old_log_prob

    except Exception as e:
        print(f"An error occurred during action selection: {e}")
        raise


def collect_episode_data(agent, env, action_mask, max_steps=N_STEPS, batch_size_r=4):
    """
    收集完整的训练数据集。

    参数：
        agent (CircuitOptimizerAgent): 强化学习代理实例。
        env (QuantumCircuitEnvironment): 环境实例。
        action_mask (ActionMask): ActionMask 实例。
        max_steps (int): 每集的最大步数。
        batch_size_r (int): RL批次大小。
    返回：
        states (Tensor): 收集的状态。
        actions (Tensor): 收集的动作。
        rewards (Tensor): 收集的奖励。
        dones (Tensor): 表示集完成的标志。
        old_log_probs (Tensor): 所采取动作的对数概率。
        values (Tensor): 代理做出的价值预测。
    """
    try:
        # 获取状态张量的形状，并移除批次维度
        #state_shape = env.simulator.get_state().shape[1:]
        # 获取状态张量的形状
        state_shape = env.simulator.get_state().shape

        # 初始化数据容器
        states = torch.empty((batch_size_r, max_steps) + state_shape, dtype=torch.float32)
        actions = torch.empty(batch_size_r, max_steps, 2, dtype=torch.int64)  # (rule_index, qubit_moment_index)
        rewards = torch.empty(batch_size_r, max_steps, dtype=torch.float32)
        dones = torch.empty(batch_size_r, max_steps, dtype=torch.bool)
        old_log_probs = torch.empty(batch_size_r, max_steps, dtype=torch.float32)
        values = torch.empty(batch_size_r, max_steps, dtype=torch.float32)

        step_count = 0
        states[:, 0] = env.reset()

        while step_count < max_steps:
            with torch.no_grad():
                # 代理根据当前状态做出策略和价值预测
                # 选取第 step_count 步的所有批次的状态
                current_states = states[:, step_count]
                # 调整 current_states 的维度以匹配模型输入
                if batch_size == 1:
                    reshaped_states = current_states.squeeze(1).permute(0, 3, 1, 2)
                else:
                    reshaped_states = current_states.reshape(-1, N_QUBITS, N_MOMENTS, N_GATE_CLASSES).permute(0, 3, 1, 2)
                policy, value = agent(reshaped_states)

                policy = policy.view(batch_size_r*batch_size, N_RULES, N_QUBITS * N_MOMENTS)  

                # 应用动作屏蔽
                masked_policy = policy * action_mask.mask(env.simulator.circuit, N_GATE_CLASSES)

                # 确保掩码正确广播或调整形状
                if action_mask.mask(env.simulator.circuit, N_GATE_CLASSES).ndim < policy.ndim:
                    action_mask_adjusted = action_mask.mask(env.simulator.circuit, N_GATE_CLASSES).unsqueeze(0)  # 假设在批次维度上扩展
                else:
                    action_mask_adjusted = action_mask.mask(env.simulator.circuit, N_GATE_CLASSES)

                masked_policy = policy * action_mask_adjusted

                # 确认形状正确
                assert masked_policy.shape == policy.shape, "Shapes do not match after masking."

                # 遍历批次中的每个实例
                for i in range(batch_size_r):
                    action, old_log_prob = select_action(states[i, step_count], masked_policy[i], action_mask, env, N_GATE_CLASSES)
                    # 更新当前实例的状态
                    next_state, reward, done = env.apply_rule(action)
                    states[i, step_count + 1] = next_state
                    rewards[i, step_count] = reward
                    dones[i, step_count] = done
                    old_log_probs[i, step_count] = old_log_prob
                    values[i, step_count] = value

            step_count += 1

            # 如果所有实例都已完成，跳出循环
            if torch.all(dones[:, step_count - 1]):
                break

        # 截断未使用的部分
        if step_count < max_steps:
            states = states[:, :step_count + 1]
            actions = actions[:, :step_count]
            rewards = rewards[:, :step_count]
            dones = dones[:, :step_count]
            old_log_probs = old_log_probs[:, :step_count]
            values = values[:, :step_count]

        return states, actions, rewards, dones, old_log_probs, values

    except Exception as e:
        print(f"An error occurred during data collection: {e}")
        raise