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
