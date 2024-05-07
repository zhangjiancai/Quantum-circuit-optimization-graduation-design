import torch
from torch.distributions import Categorical
import logging
from agent import CircuitOptimizerAgent
from environment import QuantumCircuitEnvironment, ActionMask
from rules import RULES
from config import N_QUBITS, N_MOMENTS, N_GATE_CLASSES, N_RULES, N_STEPS

# 设置日志配置
logging.basicConfig(level=logging.INFO)

# 加载强化学习代理
agent = CircuitOptimizerAgent(N_QUBITS, N_MOMENTS, N_GATE_CLASSES, N_RULES)

# 创建模拟器环境
env = QuantumCircuitEnvironment(N_QUBITS, N_MOMENTS, RULES, N_GATE_CLASSES)
action_mask = ActionMask(N_RULES, N_QUBITS, N_MOMENTS)

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
        action (Tensor): 选择的动作索引。
        old_log_prob (Tensor): 选择动作的对数概率。
    """
    try:
        # 如果所有动作都被禁止，随机选择一个允许的动作
        if masked_policy.sum() == 0:
            allowed_indices = torch.nonzero(action_mask.mask(env.simulator.circuit, n_gate_classes)).flatten()
            if len(allowed_indices) > 0:
                random_index = torch.randint(low=0, high=len(allowed_indices), size=(1,))
                action = allowed_indices[random_index]
            else:
                #raise ValueError("没有可用的动作。")
                old_log_prob = torch.tensor(0.0)
                action = torch.tensor(0)  # 或者选择其他合适的默认处理方式
        else:
            action_dist = Categorical(masked_policy.view(-1))
            action = action_dist.sample()
            old_log_prob = action_dist.log_prob(action)

        logging.debug(f"选择的动作：{action.item()}，对数概率：{old_log_prob.item()}")
        return action, old_log_prob
    except Exception as e:
        logging.error(f"选择动作时出错：{e}")
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
        # 初始化数据容器
        states = torch.empty((max_steps, 1) + env.simulator.get_state().shape[1:], dtype=torch.float32)
        actions = torch.empty(max_steps, dtype=torch.int64)
        rewards = torch.empty(max_steps, dtype=torch.float32)
        dones = torch.empty(max_steps, dtype=torch.bool)
        old_log_probs = torch.empty(max_steps, dtype=torch.float32)
        values = torch.empty(max_steps, dtype=torch.float32)

        step_count = 0
        state = env.reset().squeeze(0)

        while step_count < max_steps:
            try:
                with torch.no_grad():
                    policy, value = agent(state.unsqueeze(0))
                    policy = policy.view(N_RULES, N_QUBITS * N_MOMENTS)

                    # 应用动作屏蔽
                    masked_policy = policy * action_mask.mask(env.simulator.circuit, N_GATE_CLASSES)
                    action, old_log_prob = select_action(state, masked_policy, action_mask, env, N_GATE_CLASSES)

                next_state, reward, done = env.apply_rule(action.item())
                state = next_state

                # 填充数据
                states[step_count] = state.unsqueeze(0)
                actions[step_count] = action
                rewards[step_count] = reward
                dones[step_count] = done
                old_log_probs[step_count] = old_log_prob
                values[step_count] = value

                step_count += 1
                if done:
                    break

            except Exception as e:
                logging.error(f"在第 {step_count} 步的集内出错：{e}")
                break  # 出错时退出循环

        # 截断未使用的部分
        if step_count < max_steps:
            logging.warning(f"集在最大步数前结束：{step_count}/{max_steps}")
        states = states[:step_count]
        actions = actions[:step_count]
        rewards = rewards[:step_count]
        dones = dones[:step_count]
        old_log_probs = old_log_probs[:step_count]
        values = values[:step_count]

        return states, actions, rewards, dones, old_log_probs, values

    except Exception as e:
        logging.error(f"在 collect_episode_data 中出错：{e}")
        raise

if __name__ == "__main__":
    logging.info("收集集数据测试开始")

    # 初始化环境并重置状态
    env = QuantumCircuitEnvironment(N_QUBITS, N_MOMENTS, RULES, N_GATE_CLASSES)
    action_mask = ActionMask(N_RULES, N_QUBITS, N_MOMENTS)
    initial_state = env.reset()

    try:
        # 收集训练数据
        states, actions, rewards, dones, old_log_probs, values = collect_episode_data(
            agent, env, action_mask, max_steps=N_STEPS
        )

        # 记录测试结果
        logging.info("收集集数据测试完成")
        logging.info(f"状态维度：{states.shape}")
        logging.info(f"动作维度：{actions.shape}")
        logging.info(f"奖励：{rewards}")
        logging.info(f"集结束标志：{dones}")
        logging.info(f"旧的对数概率：{old_log_probs}")
        logging.info(f"价值预测：{values}")

    except Exception as e:
        logging.error(f"主进程中出错：{e}")
