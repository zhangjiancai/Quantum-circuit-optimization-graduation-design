
import torch
from torch.distributions import Categorical
from agent import CircuitOptimizerAgent
from environment import QuantumCircuitEnvironment, ActionMask
from rules import RULES
from collect_episode_data import collect_episode_data
import numpy as np

# Configuration parameters (Assumed to be defined elsewhere or need to be set appropriately)
N_QUBITS = 5
N_MOMENTS = 10
N_RULES = len(RULES)  # Ensure RULES is defined in the rules.py or imported correctly
N_GATE_CLASSES = 3  # Assuming X, Y, Z gates

# Initialize the agent
agent = CircuitOptimizerAgent(N_QUBITS, N_MOMENTS, N_GATE_CLASSES, N_RULES)

# Load model (Ensure the path to the model is correct)
model_path = 'rl_agent.pth'
agent.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
agent.eval()

# Generate initial quantum circuit data randomly
# This example assumes a function or a method to generate data exists. Here we mock this.
quantum_circuit_data = np.random.rand(N_QUBITS, N_GATE_CLASSES, N_MOMENTS, 2) > 0.9

# Initialize the environment with the quantum circuit
env = QuantumCircuitEnvironment(N_QUBITS, N_MOMENTS, RULES, N_GATE_CLASSES, initial_circuit_data=quantum_circuit_data)
action_mask = ActionMask(N_RULES, N_QUBITS, N_MOMENTS)

# Run the evaluation
initial_state = env.get_state()
result = collect_episode_data(agent, env, action_mask, max_steps=1)
updated_state = env.get_state()

# Output the initial and updated states for comparison
print("Initial State:\n", initial_state)
print("Updated State:\n", updated_state)
