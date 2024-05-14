# test_agent.py

import torch
from agent import CircuitOptimizerAgent

# 定义模型的参数
n_qubits = 5
n_moments = 10
n_gate_classes = 3
n_rules = 4

# 创建代理模型
agent = CircuitOptimizerAgent(n_qubits, n_moments, n_gate_classes, n_rules)

# 模拟输入数据
batch_size = 8
input_tensor = torch.randn(batch_size, n_gate_classes, n_qubits, n_moments)

# 执行前向传播，获取策略和价值输出
policy, value = agent(input_tensor)

# 打印输出的维度
print("Policy output shape:", policy.shape)  # 应输出 [batch_size, n_rules, n_qubits * n_moments]
print("Value output shape:", value.shape)    # 应输出 [batch_size, 1]
