#rules.py

import cirq
import numpy as np
import torch

def rz_rule(simulator):
    """Apply RZ gate to the first qubit."""
    qubits = simulator.qubits
    gate = cirq.rz(np.pi / 2)
    simulator.add_gate(gate, (qubits[0],))
    return simulator

def x_rule(simulator):
    """Apply X gate to the first qubit."""
    qubits = simulator.qubits
    gate = cirq.X
    simulator.add_gate(gate, (qubits[0],))
    return simulator

def cnot_rule(simulator):
    """Apply CNOT gate between the first and second qubits."""
    qubits = simulator.qubits
    gate = cirq.CNOT
    simulator.add_gate(gate, (qubits[0], qubits[1]))
    return simulator

def swap_rule(simulator):
    """Apply SWAP gate between the first and second qubits."""
    qubits = simulator.qubits
    gate = cirq.SWAP
    simulator.add_gate(gate, (qubits[0], qubits[1]))
    return simulator

def commute_rule(simulator):
    """Swap the order of two adjacent gates if commutable."""
    qubits = simulator.qubits
    if len(simulator.circuit) >= 2:
        op1, op2 = list(simulator.circuit.all_operations())[:2]
        simulator.circuit = cirq.Circuit([op2, op1] + list(simulator.circuit.all_operations())[2:])
    return simulator

def cancel_adjacent_rz(simulator):
    """Cancel adjacent RZ gates if their angles sum to zero."""
    new_ops = []
    for op in simulator.circuit.all_operations():
        if not new_ops or not isinstance(op.gate, cirq.ops.common_gates.RzPowGate):
            new_ops.append(op)
        else:
            prev_op = new_ops.pop()
            if isinstance(prev_op.gate, cirq.ops.common_gates.RzPowGate):
                angle_sum = prev_op.gate._exponent + op.gate._exponent
                if np.isclose(angle_sum % 2, 0):
                    continue  # Cancel out
                else:
                    new_op = cirq.rz(angle_sum * np.pi)(prev_op.qubits[0])
                    new_ops.append(new_op)
            else:
                new_ops.append(prev_op)
                new_ops.append(op)
    simulator.circuit = cirq.Circuit(new_ops)
    return simulator

RULES = [rz_rule, x_rule, cnot_rule, swap_rule, commute_rule, cancel_adjacent_rz]
'''
# 测试规则
if __name__ == '__main__':
    from environment import QuantumCircuitSimulator

    n_qubits = 5
    n_moments = 15
    n_gate_classes = 4

    simulator = QuantumCircuitSimulator(n_qubits, n_moments, n_gate_classes)

    # 测试应用 RZ 规则
    state, reward, done = simulator.apply_rule(rz_rule)
    print(f"State after applying RZ rule: {state.shape}, Reward: {reward}, Done: {done}")

    # 测试应用 CNOT 规则
    state, reward, done = simulator.apply_rule(cnot_rule)
    print(f"State after applying CNOT rule: {state.shape}, Reward: {reward}, Done: {done}")
'''