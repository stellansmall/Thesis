#!/usr/bin/env python3
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp


def draw_anisotropic_gate(theta_1, theta_2, theta_3):
    # 1. Define the anisotropic Hamiltonian: theta_1*XX + theta_2*YY + theta_3*ZZ
    op = SparsePauliOp(["XX", "YY", "ZZ"], coeffs=[theta_1, theta_2, theta_3])
    target_unitary = PauliEvolutionGate(op)

    # 2. Add the operator to a 2-qubit circuit
    qc = QuantumCircuit(2)
    qc.append(target_unitary, [0, 1])

    # 3. Synthesize into 3 CNOT gates (CX) and single-qubit rotations
    optimized_circuit = transpile(
        qc, basis_gates=["cz", "rz", "sx", "x"], optimization_level=3
    )

    # 4. Print ASCII art to terminal
    print(optimized_circuit.draw(output="text"))


if __name__ == "__main__":
    # Set sample interaction angles
    t1 = np.pi / 4
    t2 = np.pi / 3
    t3 = np.pi / 6

    draw_anisotropic_gate(t1, t2, t3)
