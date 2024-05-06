import torch
from torch.distributions import Categorical
from agent import CircuitOptimizerAgent
from environment import QuantumCircuitEnvironment, ActionMask
from optimizer import PPO
from rules import RULES

# 配置
N_QUBITS = 12
N_MOMENTS = 50
N_GATE_CLASSES = 4
N_RULES = len(RULES)
EPOCHS = 1000
STEPS_PER_EPOCH = 600
LEARNING_RATE = 3e-4

# 加载 RL 智能体
agent = CircuitOptimizerAgent(N_QUBITS, N_MOMENTS, N_GATE_CLASSES, N_RULES)
ppo = PPO(agent, lr=LEARNING_RATE)

# 创建模拟器环境
env = QuantumCircuitEnvironment(N_QUBITS, N_MOMENTS, RULES, N_GATE_CLASSES)
action_mask = ActionMask(N_RULES, N_QUBITS, N_MOMENTS)

def collect_episode_data(state, steps):
    """收集一整集的训练数据。"""
    states, actions, rewards, dones, old_log_probs, values = [], [], [], [], [], []

    for _ in range(steps):
        state = state.squeeze(0)  # 移除多余的第一个维度
        with torch.no_grad():
            policy, value = agent(state.unsqueeze(0))  # 重新添加批次维度
            policy, value = agent(state.unsqueeze(0))
            policy = policy.view(N_RULES, N_QUBITS * N_MOMENTS)
            masked_policy = policy * action_mask.mask(env.simulator.circuit,env.simulator.n_gate_classes)
            action_dist = Categorical(masked_policy.flatten())
            action = action_dist.sample()
            old_log_prob = action_dist.log_prob(action)

        next_state, reward, done = env.apply_rule(action.item())

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        old_log_probs.append(old_log_prob)
        values.append(value)

        state = next_state
        if done:
            break

    states = torch.stack(states)
    actions = torch.tensor(actions, dtype=torch.int64)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)
    old_log_probs = torch.stack(old_log_probs)
    values = torch.stack(values)

    return states, actions, rewards, dones, old_log_probs, values

# 训练循环
for epoch in range(EPOCHS):
    state = env.reset()
    states, actions, rewards, dones, old_log_probs, values = collect_episode_data(state, STEPS_PER_EPOCH)
    ppo.update(states, actions, rewards, dones, old_log_probs, values)
    print(f"Epoch {epoch + 1}/{EPOCHS} complete.")

# 保存训练好的模型
torch.save(agent.state_dict(), 'rl_agent.pth')
print("Training Complete. Model saved to 'rl_agent.pth'")
