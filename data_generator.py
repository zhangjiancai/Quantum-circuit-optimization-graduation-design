# \data_generator.py

import numpy as np
import torch

class DataGenerator:
    def __init__(self, env, params):
        self.env = env
        self.params = params

    def generate_data(self, batch_size):
        inputs, targets = [], []
        for _ in range(batch_size):
            state = self.env.reset()
            action = self.env.sample_action()
            next_state, reward, done = self.env.step(action)

            # 确保目标维度与网络输出一致
            targets.append([reward, done, 0, 0])
            inputs.append(state)

        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)
'''
import torch.nn as nn
import torch

class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.fc = nn.Linear(4, 4)  # 假设输入和输出维度均为 4

    def forward(self, x):
        return self.fc(x)

# 测试模型
model = SimpleNetwork()
input_tensor = torch.rand((1, 4))  # 假设输入维度是 [1, 4]
output = model(input_tensor)
print(output.shape)  # 输出维度应当是 [1, 4]
'''