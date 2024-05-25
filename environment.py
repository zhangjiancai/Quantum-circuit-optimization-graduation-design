import cirq
import torch
import numpy as np

def initialize_quantum_state(num_qubits, num_gate_types, num_timesteps):
    # 初始化一个四维数组
    quantum_circuit_data = np.zeros((num_qubits, num_gate_types, num_timesteps, 2), dtype=np.float32)

    # 设置随机种子以确保可复现性
    np.random.seed(42)
    activation_probability = 0.1  # 设定任一门在任一位置激活的概率

    # 遍历数组，并根据概率随机激活
    for qubit in range(num_qubits):
        for gate_type in range(num_gate_types):
            for time in range(num_timesteps):
                if np.random.random() < activation_probability:
                    quantum_circuit_data[qubit, gate_type, time, 1] = 1  # 激活该位置的量子门

    # 未激活的通道设置
    quantum_circuit_data[:, :, :, 0] = 1 - quantum_circuit_data[:, :, :, 1]
    return quantum_circuit_data
class QuantumCircuitSimulator:
    def __init__(self, n_qubits, n_moments, n_gate_classes):
        self.n_qubits = n_qubits
        self.n_moments = n_moments
        self.n_gate_classes = n_gate_classes
        self.qubits = [cirq.GridQubit(i, 0) for i in range(n_qubits)]
        self.circuit = cirq.Circuit()
        self.gate_types = [cirq.ops.ZPowGate, cirq.ops.XPowGate, cirq.ops.CNotPowGate, cirq.ops.SwapPowGate]
        self.reset()

    def reset(self):
        # 使用initialize_quantum_state初始化状态
        state_data = initialize_quantum_state(self.n_qubits, self.n_gate_classes, self.n_moments)
        self.state = torch.tensor(state_data).permute(3, 1, 0, 2)  # [channels, depth, height, width]
        return self.state  # 返回初始化的状态

    def add_gate(self, gate, qubits):
        self.circuit.append(gate(*qubits))
        return self.get_state()

    def get_state(self):
        # 直接返回当前状态
        return self.state

    def get_gate_type(self, gate):
        for idx, gate_cls in enumerate(self.gate_types):
            if isinstance(gate, gate_cls):
                return idx
        return None

    def apply_rule(self, rule):
        rule(self)
        reward = self.compute_reward()
        done = self.check_done()
        return self.get_state(), reward, done

    def compute_reward(self):
        depth = len(self.circuit)
        gate_count = sum(len(op.qubits) for op in self.circuit.all_operations())
        reward = -depth - 0.2 * gate_count
        return reward

    def check_done(self):
        """
        检查模拟是否应该结束。
        
        返回:
            done (bool): 如果模拟应该结束，返回 True；否则返回 False。
        """
        # 自定义结束条件：如果电路的深度超过预定的最大深度，或者包含的总门数超过一定数量
        max_depth = 13  # 假设我们设置的最大深度为10
        max_gate_count = 15  # 假设我们允许的最大门数为15

        # 计算当前电路的深度和门数
        current_depth = len(self.circuit)
        current_gate_count = sum(len(op.qubits) for op in self.circuit.all_operations())

        # 检查是否满足结束条件
        if current_depth >= max_depth or current_gate_count >= max_gate_count:
            return True
        return False

class QuantumCircuitEnvironment:
    def __init__(self, n_qubits, n_moments, rules, n_gate_classes):
        self.simulator = QuantumCircuitSimulator(n_qubits, n_moments, n_gate_classes)
        self.rules = rules
        self.reset()

    def reset(self):
        self.state = self.simulator.reset()
        self.done = False
        return self.state

    def apply_rule(self, action):
        rule_index, qubit_index, timestep_index = action
        if not (0 <= rule_index < len(self.rules)):
            raise IndexError(f"Rule index {rule_index} is out of range.")
        rule = self.rules[rule_index]
        state, reward, done = self.simulator.apply_rule(rule)
        self.done = done
        return state, reward, self.done

class ActionMask:
    def __init__(self, n_rules, n_qubits, n_moments):
        self.n_rules = n_rules
        self.n_qubits = n_qubits
        self.n_moments = n_moments

    def mask(self, circuit, gate_classes):
        # 创建一个形状为 (n_qubits, n_rules, n_moments) 的掩码张量
        mask = np.ones((self.n_qubits, self.n_rules, self.n_moments), dtype=bool)
        return torch.tensor(mask, dtype=torch.float32)
