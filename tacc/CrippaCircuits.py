from qiskit import QuantumCircuit
import numpy as np
from math import pi
from qiskit.circuit.library import n_local
from qiskit.transpiler.passes.analysis import num_qubits
from qiskit.circuits import ParameterVector

def Heisenberg(theta):
    W = QuantumCircuit(2, name="W(θ)")
    W.cx(0,1)
    W.rx(theta/2 - pi/2, 0)
    W.h(0)
    W.rz(theta/2, 1)
    W.cx(0,1)
    W.h(0)
    W.rz(-theta/2, 1)
    W.cx(0,1)
    W.rx(pi/2, 0)
    W.rx(-pi/2, 1) # conjugate of rx(pi/2, 1)
    return W

def V_pma(thetas):
    V = QuantumCircuit(4)
    V.append(Heisenberg(thetas[0]), qargs=[0,1])
    V.append(Heisenberg(thetas[2]), qargs=[2,3])
    V.append(Heisenberg(thetas[1]), qargs=[2,1])
    V.append(Heisenberg(thetas[3]), qargs=[0,3])
    return V


def U_pma(layers):
    U = QuantumCircuit(num_qubits, name="U_PMA")
    params = ParameterVector("θ", 4 * layers)

    idx = 0
    for _ in range(layers):
        U.compose(Heisenberg(params[idx]), qubits=[2, 1], inplace=True)
        U.compose(Heisenberg(params[idx+1]), qubits=[0, 4], inplace=True)
        U.compose(Heisenberg(params[idx+2]), qubits=[0, 1], inplace=True)
        U.compose(Heisenberg(params[idx+3]), qubits=[2, 3], inplace=True)
        idx += 4

def pma_initial(num_qubits=4):
    #IDK what to put here exactly and should probably talk to Friedman about this

def U_ha(layers, qubits=4):
    V = n_local(num_qubits=qubits,
               rotation_blocks="ry",
               entanglement_blocks="cx",
               entanglement="linear",
               reps=layers,
               insert_barriers=True,
               skip_final_rotation_layer=False)
    return V
