import torch
import torch.nn as nn
import torch.nn.functional as F

class CircuitOptimizerAgent(nn.Module):
    """Deep Convolutional RL Agent."""
    def __init__(self, n_qubits, n_moments, n_gate_classes, n_rules):
        super(CircuitOptimizerAgent, self).__init__()
        self.conv1 = nn.Conv2d(n_gate_classes, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Policy head
        self.policy_conv = nn.Conv2d(256, n_rules, kernel_size=1)
        self.policy_linear = nn.Linear(n_qubits * n_moments, n_qubits * n_moments * n_rules)

        # Value head
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.value_linear = nn.Linear(n_qubits * n_moments, 1)

    def forward(self, x):
        """Forward pass through the network."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Policy network
        policy_map = F.relu(self.policy_conv(x))
        policy_map_flat = torch.flatten(policy_map, start_dim=1)
        policy = self.policy_linear(policy_map_flat)
        policy = F.softmax(policy, dim=-1)

        # Value network
        value_map = F.relu(self.value_conv(x))
        value_map_flat = torch.flatten(value_map, start_dim=1)
        value = self.value_linear(value_map_flat)

        return policy, value
