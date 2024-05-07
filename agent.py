# \agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CircuitOptimizerAgent(nn.Module):
    """深度卷积强化学习代理模型，用于量子电路优化。"""
    def __init__(self, n_qubits, n_moments, n_gate_classes, n_rules):
        """
        初始化深度卷积强化学习代理模型。

        参数：
            n_qubits (int): 量子比特数目。
            n_moments (int): 时间步数。
            n_gate_classes (int): 门类型数量。
            n_rules (int): 规则数量。
        """
        super(CircuitOptimizerAgent, self).__init__()
        self.n_qubits = n_qubits
        self.n_moments = n_moments

        # 卷积层
        self.conv1 = nn.Conv2d(n_gate_classes, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # 计算展平后的特征大小
        def conv_output_size(size, kernel_size=3, stride=1, padding=1):
            return ((size - kernel_size + 2 * padding) // stride) + 1

        qubits_out = conv_output_size(conv_output_size(conv_output_size(conv_output_size(self.n_qubits))))
        moments_out = conv_output_size(conv_output_size(conv_output_size(conv_output_size(self.n_moments))))

        flattened_size = 256 * qubits_out * moments_out

        # 策略网络的全连接层
        self.policy_linear = nn.Linear(flattened_size, n_qubits * n_moments * n_rules)

        # 价值网络的全连接层
        self.value_linear = nn.Linear(flattened_size, 1)

    def forward(self, x):
        """前向传播。"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # 展平输出
        x_flat = torch.flatten(x, start_dim=1)

        # 策略网络
        policy = self.policy_linear(x_flat)
        policy = F.softmax(policy, dim=-1)

        # 价值网络
        value = self.value_linear(x_flat)

        return policy, value
'''
# 示例输入
n_qubits = 5
n_moments = 15
n_gate_classes = 10  # 假设的门类别数
n_rules = 8  # 假设的规则数量

# 创建一个 CircuitOptimizerAgent 实例
model = CircuitOptimizerAgent(n_qubits, n_moments, n_gate_classes, n_rules)

# 创建随机输入数据
input_data = torch.randn(1, n_gate_classes, n_qubits, n_moments)  # Batch size 为 1
policy, value = model(input_data)

# 输出形状
print(f"Policy shape: {policy.shape}")
print(f"Value shape: {value.shape}")
'''