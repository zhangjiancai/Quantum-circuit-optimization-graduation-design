# environment.py

import cirq
import torch
import numpy as np
from config import batch_size
class QuantumCircuitSimulator:
    """Quantum Circuit Simulator using Cirq."""
    def __init__(self, n_qubits, n_moments, n_gate_classes,batch_size):
        """
        初始化一个量子电路模拟器的对象。

        参数:
        n_qubits (int): 量子电路中量子比特的数量。
        n_moments (int): 量子电路中时刻的数量。
        n_gate_classes (int): 量子电路中门类的数量。
        """
        self.n_qubits = n_qubits  # 量子比特数量
        self.n_moments = n_moments  # 电路中时刻的数量
        self.n_gate_classes = n_gate_classes  # 门类的数量
        self.batch_size = batch_size
        # 初始化量子比特列表，使用网格量子比特，并将它们排成一列
        self.qubits = [cirq.GridQubit(i, 0) for i in range(n_qubits)]
        self.circuit = cirq.Circuit()  # 初始化一个空的量子电路
        self.reset()  # 重置电路到初始状态

    def reset(self):
        """Reset simulator state."""
        self.circuit = cirq.Circuit()
        return self.get_state()

    def add_gate(self, gate, qubits):
        """Add a gate to the circuit."""
        self.circuit.append(gate(*qubits))
        return self.get_state()

    def get_state(self):
        """
        返回电路状态作为一个张量（4维）。
    
        方法通过遍历电路中的所有操作，将每个门类型在每个时刻对每个量子比特的作用
        编码为一个4维张量，其中每个维度分别对应量子比特数量、门操作时刻数量、门类型数量以及一个批量维度。
    
        返回:
            - torch.tensor: 表示电路状态的4维张量，维度顺序为[门类型, 量子比特, 门操作时刻]。
        """
        # 初始化一个状态张量，用0填充，表示没有门操作
        state_tensor = np.zeros((self.batch_size, self.n_qubits, self.n_moments, self.n_gate_classes), dtype=np.float32)
        for moment_index, moment in enumerate(self.circuit):
            for op in moment:
                qubit_indices = [q.row for q in op.qubits]
                gate_type = self.get_gate_type(op.gate)
                if gate_type is not None:
                    for qubit_index in qubit_indices:
                        # 填充状态张量
                        state_tensor[:, gate_type, qubit_index, moment_index] = 1.0

        # 返回张量，需要调整维度顺序以符合预期格式
        return torch.tensor(state_tensor).permute(0, 2, 3, 1)  # [批次, 量子比特, 门操作时刻, 门类型]

    #.permute(2, 0, 1)：这个操作交换了张量的第二、第一和第三维度，
    #原始顺序是(n_qubits, n_moments, n_gate_classes)，调整后变为(n_gate_classes, n_qubits, n_moments)。
    #这样做的目的是为了适应神经网络中常见的通道优先的顺序，即先指定颜色通道（在这里是门类型），然后是高度（量子比特），最后是宽度（门操作时刻）。

    #.unsqueeze(0)：这个操作在张量的最前面添加了一个新的维度，将张量从形状(n_gate_classes, n_qubits, n_moments)转换为(1, n_gate_classes, n_qubits, n_moments)。
    #这个新添加的维度通常用于表示批量（batch）大小，即使这里只有一个样本，批量大小也为1。
    #经过这些调整，返回的张量具有形状[batch_size, channels, height, width]，这是一个常见的深度学习模型输入的格式。
    #在这个例子中，batch_size=1，channels=n_gate_classes，height=n_qubits，width=n_moments。

    def get_gate_type(self, gate):
        """
        获取门类型的索引，处理更多属性。
        """
        gate_types = {'cirq.Rz': 0, 'cirq.Ry': 1, 'cirq.PX': 2, 'cirq.CNOT': 3, 'cirq.SWAP': 4}
        gate_name = gate.__class__.__name__
        gate_type = gate_types.get(gate_name)

    # 可能需要根据门的具体参数进一步区分类型
        if gate_name == 'cirq.Rz' or gate_name == 'cirq.Ry':
            if gate.exponent == 1:
                gate_type += 0.1  # 表示这是π操作
            elif gate.exponent == 0.5:
                gate_type += 0.2  # 表示这是π/2操作

        return gate_type


    def apply_rule(self, rule):
        """Apply a transformation rule to the circuit."""
        rule(self)
        reward = self.compute_reward()
        done = self.check_done()
        return self.get_state(), reward, done

    def compute_reward(self):
        """Compute reward based on circuit quality."""
        depth = len(self.circuit)
        gate_count = sum(len(op.qubits) for op in self.circuit.all_operations())
        reward = -depth - 0.2 * gate_count
        return reward

    def check_done(self):
        """Check if optimization is complete."""
        # Implement custom termination criteria...
        return False

class QuantumCircuitEnvironment:
    """RL environment for quantum circuit optimization."""    
    def __init__(self, n_qubits, n_moments, rules, n_gate_classes,batch_size):
        """
        Initialize the quantum circuit environment.
        
        :param n_qubits: Number of qubits in the circuit.
        :param n_moments: Number of moments (time steps) in the circuit.
        :param rules: List of transformation rules applicable to the circuit.
        :param n_gate_classes: Number of distinct gate classes considered in the simulation.
        """
        self.simulator = QuantumCircuitSimulator(n_qubits, n_moments, n_gate_classes,batch_size)
        self.rules = rules
        self.reset()

    def reset(self):
        """Reset environment state."""
        self.state = self.simulator.reset()
        self.done = False
        return self.state

    def apply_rule(self, action):
        """Apply circuit transformation rule."""
        rule_index, _ = action  # 解包元组
        if not (0 <= rule_index < len(self.rules)):
            raise IndexError(f"Rule index {rule_index} is out of range. Valid range is [0, {len(self.rules) - 1}].")
        rule = self.rules[rule_index]
        state, reward, done = self.simulator.apply_rule(rule)
        self.done = done
        return state, reward, self.done

class ActionMask:
    """Helper class for masking illegal actions."""

    def __init__(self, n_rules, n_qubits, n_moments, batch_size=None):
        """
        Initialize the ActionMask instance.

        Parameters:
        n_rules (int): Number of available action rules.
        n_qubits (int): Number of qubits in the circuit.
        n_moments (int): Number of time steps in the circuit.
        batch_size (int, optional): Preset batch size. If None, it needs to be computed from the circuits list.
        """
        self.n_rules = n_rules
        self.n_qubits = n_qubits
        self.n_moments = n_moments
        self.batch_size = batch_size  

    def mask(self, circuits, gate_classes):
        """Compute the action mask based on the current circuit states."""
        if self.batch_size is None:
            batch_size = len(circuits)
        else:
            batch_size = self.batch_size  # Use the preset batch size if available

        mask = np.ones((batch_size, self.n_rules, self.n_qubits * self.n_moments), dtype=bool)

        # Implement masking logic based on valid circuit transformation rules
        # Iterate over the batch and apply the masking logic to each circuit individually
        for i, circuit in enumerate(circuits):
            # Your masking logic here, considering the current circuit `circuit` and its gate classes
            pass

        return torch.tensor(mask, dtype=torch.float32)