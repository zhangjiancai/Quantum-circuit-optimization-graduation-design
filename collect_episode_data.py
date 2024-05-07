# 引入所需的库和模块
import torch
from torch.distributions import Categorical
from agent import CircuitOptimizerAgent
from environment import QuantumCircuitEnvironment, ActionMask
from rules import RULES
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


# collect_episode_data.py
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


def collect_episode_data(agent, env, action_mask, max_steps=N_STEPS):
    """
    收集完整的训练数据集。

    参数：
        agent (CircuitOptimizerAgent): 强化学习代理实例。
        env (QuantumCircuitEnvironment): 环境实例。
        action_mask (ActionMask): ActionMask 实例。
        max_steps (int): 每集的最大步数。

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
        state_shape = env.simulator.get_state().shape[1:]

        # 初始化数据容器
        states = torch.empty((max_steps,) + state_shape, dtype=torch.float32)
        actions = torch.empty(max_steps, 2, dtype=torch.int64)  # (rule_index, qubit_moment_index)
        rewards = torch.empty(max_steps, dtype=torch.float32)
        dones = torch.empty(max_steps, dtype=torch.bool)
        old_log_probs = torch.empty(max_steps, dtype=torch.float32)
        values = torch.empty(max_steps, dtype=torch.float32)

        step_count = 0
        state = env.reset()

        while step_count < max_steps:
            with torch.no_grad():
                # 代理根据当前状态做出策略和价值预测
                policy, value = agent(state)  
                policy = policy.view(N_RULES, N_QUBITS * N_MOMENTS)  

                # 应用动作屏蔽
                masked_policy = policy * action_mask.mask(env.simulator.circuit, N_GATE_CLASSES)
                action, old_log_prob = select_action(state, masked_policy, action_mask, env, N_GATE_CLASSES)

            # 环境根据选择的动作更新状态，并返回奖励
            next_state, reward, done = env.apply_rule(action)
            state = next_state

            # 填充收集到的数据
            states[step_count] = state
            actions[step_count] = torch.tensor(action)
            rewards[step_count] = reward
            dones[step_count] = done
            old_log_probs[step_count] = old_log_prob
            values[step_count] = value

            step_count += 1
            if done:
                break

        # 截断未使用的部分
        if step_count < max_steps:
            states = states[:step_count]
            actions = actions[:step_count]
            rewards = rewards[:step_count]
            dones = dones[:step_count]
            old_log_probs = old_log_probs[:step_count]
            values = values[:step_count]

        return states, actions, rewards, dones, old_log_probs, values

    except Exception as e:
        print(f"An error occurred during data collection: {e}")
        raise
'''
if __name__ == "__main__":
    agent = CircuitOptimizerAgent(N_QUBITS, N_MOMENTS, N_GATE_CLASSES, N_RULES)

    # 初始化环境并重置状态
    env = QuantumCircuitEnvironment(N_QUBITS, N_MOMENTS, RULES, N_GATE_CLASSES)
    action_mask = ActionMask(N_RULES, N_QUBITS, N_MOMENTS)
    initial_state = env.reset()

    # 收集训练数据
    states, actions, rewards, dones, old_log_probs, values = collect_episode_data(
        agent, env, action_mask, max_steps=N_STEPS
    )

    # 记录测试结果
    print("收集数据测试完成")
    print(f"状态维度：{states.shape}")
    print(f"动作维度：{actions.shape}")
    print(f"奖励：{rewards}")
    print(f"集结束标志：{dones}")
    print(f"旧的对数概率：{old_log_probs}")
    print(f"价值预测：{values}")
'''
