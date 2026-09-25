from qiskit import QuantumCircuit
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import n_local
from qiskit.circuits import ParameterVector
from qiskit.transpiler.passes.analysis import num_qubits
import matplotlib.pyplot as plt
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize
from qiskit_ibm_runtime import Session, EstimatorOption, QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_aer import StatevectorSimulator

## Defining the Circuit elements, used to be in a file called CrippaCircuits.py ##

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
    # Initialized states are idealy a superposition of states that have a total spin predicted by the B/J
    # But Crippa didn't implement that. Low values just have one possible configuration of singlets
    # and middle-value have one possible triplet and the rest singlets
    raise NotImplementedError(
            "state initialization isn't implemented yet"
     )

def U_ha(layers, qubits=4):
    V = n_local(num_qubits=qubits,
               rotation_blocks="ry",
               entanglement_blocks="cx",
               entanglement="linear",
               reps=layers,
               insert_barriers=True,
               skip_final_rotation_layer=False)
    return V

## Now we set up the actual experiments by configuring the experimental circuits 
## and running them through a minimization algorith

estimator = StatevectorEstimator()
def define_heisenberg_loop(num_spins, coupling_strength=1, field_strength=1):
    ham_list = []
    circ = QuantumCircuit(num_spins)
    edges = [(i, (i+1) % num_spins) for i in range(num_spins)]
    for edge in edges:
       ham_list.append(("ZZ", edge, coupling_strength/2))
       ham_list.append(("YY", edge, coupling_strength/2))
       ham_list.append(("XX", edge, coupling_strength/2))
    for i in range(num_spins):
        ham_list.append(("Z", [i], field_strength/2))
    hamiltonian = SparsePauliOp.from_sparse_list(ham_list, num_qubits=num_spins)
    return hamiltonian

def cost_func_vqe(params, ansatz, hamiltonian, estimator, iterations_list):
    pub = (ansatz, hamiltonian, params)
    cost = float(estimator.run([pub]).result()[0].data.evs)
    iterations_list.append(cost)
    return cost




def HA(num_spins, backend, estimator_options, layers, coupling_strength, field_strength): #heuristic ansatz approach, params should be a 2d array as in U_ha
    hamiltonian = define_heisenberg_loop(num_spins=num_spins, coupling_strength=coupling_strength, field_strength=field_strength)
    ansatz = QuantumCircuit(num_spins)
    ansatz.compose(U_ha(layers, num_spins), inplace=True) #layers should be 2 or 3 per the paper


    if not isinstance(backend, StatevectorSimulator):
        raise NotImplementedError(
            "Only StatevectorSimulator is supported right now — "
            "real/other backends aren't handled yet."
        )


    # makes sure that something is printed between iterations 
    def callback(xk):
        print(f"Iter {len(iterations)}: cost = {iterations[-1]}")

    iterations = []
    x0 = np.ones(ansatz.num_parameters)#initial guess


    with Session(backend=backend) as session:
        estimator = Estimator(mode=session, options = estimator_options)

        result = minimize(
            cost_func_vqe,
            x0,
            args=(ansatz, hamiltonian, estimator, iterations),
            method="COBYLA", #should probably be made into a function parameter
            options={"maxiter": 500, "disp": True},
            callback=callback
        )
        session.close()
        return result, iterations



def PMA(num_spins, params, coupling_strength, field_strength):
     hamiltonian = define_hamiltonian(num_spins=num_spins, coupling_strength=coupling_strength, field_strength=field_strength)

