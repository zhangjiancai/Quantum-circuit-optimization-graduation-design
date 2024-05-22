import numpy as np
import torch

class DataGenerator:
    def __init__(self, env, params):
        self.env = env
        self.params = params

    def generate_data(self, batch_size):
        # 初始化数据集的数组，四维：batch_size, n_qubits, n_gate_classes, n_moments
        inputs = np.zeros((batch_size, self.env.n_qubits, self.env.n_gate_classes, self.env.n_moments))
        targets = []

        for i in range(batch_size):
            state = self.env.reset()
            action = self.env.sample_action()
            next_state, reward, done = self.env.step(action)

            # state应该是已经是四维数据，检查是否需要转换
            inputs[i, :, :, :] = state.numpy() if isinstance(state, torch.Tensor) else state

            # 根据模型输出调整targets格式
            targets.append([reward, done])  # 根据需要调整

        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)
