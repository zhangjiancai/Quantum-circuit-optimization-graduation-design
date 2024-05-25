import torch
import torch.nn as nn
import torch.nn.functional as F
class CircuitOptimizerAgent(nn.Module):
    def __init__(self, num_qubits, num_gate_types, num_transform_rules, num_timesteps):
        super(CircuitOptimizerAgent, self).__init__()
        self.num_qubits = num_qubits
        self.num_gate_types = num_gate_types
        self.num_transform_rules = num_transform_rules
        self.num_timesteps = num_timesteps

        # 3D卷积层
        self.conv1 = nn.Conv3d(2, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(2)

        # 计算池化后的维度
        pooled_timesteps = num_timesteps // 2
        pooled_qubits = num_qubits // 2
        pooled_gate_types = num_gate_types // 2
        pooled_dim = pooled_timesteps * pooled_qubits * pooled_gate_types * 16

        # 策略网络的全连接层
        self.fc1_policy = nn.Linear(pooled_dim, 256)
        self.fc2_policy = nn.Linear(256, num_qubits * num_transform_rules * num_timesteps)

        # 价值网络的全连接层
        self.fc1_value = nn.Linear(pooled_dim, 256)
        self.fc2_value = nn.Linear(256, 1)

    def forward(self, x):
        # 处理单个输入
        x = x.unsqueeze(0)  # 为了兼容网络结构，临时添加批次维度
        x = self.pool(self.relu(self.conv1(x)))
        x = x.squeeze(0)  # 移除批次维度

        # 为全连接层展平数据
        x_flat = x.view(-1)

        # 策略网络
        policy = F.relu(self.fc1_policy(x_flat))
        policy = self.fc2_policy(policy)
        policy = policy.view(self.num_qubits, self.num_transform_rules, self.num_timesteps)
        policy = F.softmax(policy, dim=-1)  # 对num_timesteps应用softmax

        # 重新排列为 (1, num_qubits, num_transform_rules, num_timesteps)
        policy = policy.unsqueeze(0)  # 将概率作为一个明确的维度

        # 价值网络
        value = F.relu(self.fc1_value(x_flat))
        value = self.fc2_value(value)

        return policy, value
