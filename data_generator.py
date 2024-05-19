import cirq
import numpy as np

class DataGenerator:
    def __init__(self, n_qubits):
        # 初始化DataGenerator实例，设置量子比特数量。
        self.n_qubits = n_qubits  # 保存量子比特的数量，用于构建电路。
        # 创建一个量子比特列表，每个量子比特都在二维网格的不同行上。
        self.qubits = [cirq.GridQubit(i, 0) for i in range(n_qubits)]
        # 初始化一个空的量子电路，准备添加量子门。
        self.circuit = cirq.Circuit()

    def setup_circuit(self):
        # 为每个量子比特设置电路，添加Rz, Ry, 和X门。
        for qubit in self.qubits:
            # 在每个量子比特上应用Rz(π/2)门。
            self.circuit.append(cirq.Rz(np.pi/2).on(qubit))
            # 在每个量子比特上应用Ry(π/2)门。
            self.circuit.append(cirq.Ry(np.pi/2).on(qubit))
            # 在每个量子比特上应用Pauli X门。
            self.circuit.append(cirq.X.on(qubit))

    def generate_data(self):
        # 设置电路，准备进行模拟。
        self.setup_circuit()  # 构建电路，包括所有预定义的门。
        # 使用Cirq的模拟器来模拟整个电路。
        simulator = cirq.Simulator()
        # 运行模拟器，获取电路的最终状态向量。
        result = simulator.simulate(self.circuit)
        # 返回模拟结果的最终状态向量，作为生成的数据。
        return result.final_state_vector

# 使用方法
data_gen = DataGenerator(5)  # 创建一个包含5个量子比特的数据生成器。
final_state = data_gen.generate_data()  # 生成数据并获取最终的状态向量。
print(final_state)  # 打印最终状态向量。
