import numpy as np

def generate_random_circuits(n_samples, n_qubits, n_gates, gate_classes):
    circuits = []
    for _ in range(n_samples):
        circuit = np.zeros((n_qubits, n_gates, len(gate_classes)), dtype=np.int32)
        for _ in range(n_gates):
            qubit = np.random.randint(0, n_qubits)
            gate_class = np.random.randint(0, len(gate_classes))
            moment = np.random.randint(0, n_gates)
            circuit[qubit, moment, gate_class] = 1
        circuits.append(circuit)
    return circuits
'''
# Example usage
n_samples = 100
n_qubits = 12
n_gates = 50
gate_classes = ['RZ', 'PX', 'CNOT']
random_circuits = generate_random_circuits(n_samples, n_qubits, n_gates, gate_classes)
# 测试代码
if __name__ == '__main__':
    n_samples = 100
    n_qubits = 12
    n_gates = 50
    gate_classes = ['RZ', 'PX', 'CNOT']
    
    random_circuits = generate_random_circuits(n_samples, n_qubits, n_gates, gate_classes)
    
    # 输出部分结果
    print(f"Generated {len(random_circuits)} circuits.")
    print(f"Example Circuit Shape: {random_circuits[0].shape}")
    print(f"Example Circuit Data (first circuit): {random_circuits[0]}")
'''