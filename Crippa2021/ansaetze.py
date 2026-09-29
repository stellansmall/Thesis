from qiskit import QuantumCircuit
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import n_local
from qiskit.circuit import ParameterVector
import matplotlib.pyplot as plt
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize
from qiskit_aer import StatevectorSimulator
import os
import json
import uuid
import time
from datetime import datetime
# This is the same as CrippaExperiments but I passed it through Gemini and asked it to optimize the code
# for HPC simulation. I looked over it and am pretty confident that it didn't break anything but 
# you can never really trust the LLM and I dont really have version control yet so I'm keeping both at least
# for a bit.

## Defining the Circuit elements, used to be in a file called CrippaCircuits.py ##

def Heisenberg(theta):
    W = QuantumCircuit(2, name="W(θ)")
    W.cx(0,1)
    W.rx(theta/2 - np.pi/2, 0)
    W.h(0)
    W.rz(theta/2, 1)
    W.cx(0,1)
    W.h(0)
    W.rz(-theta/2, 1)
    W.cx(0,1)
    W.rx(np.pi/2, 0)
    W.rx(-np.pi/2, 1) # conjugate of rx(pi/2, 1)
    return W

def V_pma(thetas):
    V = QuantumCircuit(4)
    V.append(Heisenberg(thetas[0]), qargs=[0,1])
    V.append(Heisenberg(thetas[2]), qargs=[2,3])
    V.append(Heisenberg(thetas[1]), qargs=[2,1])
    V.append(Heisenberg(thetas[3]), qargs=[0,3])
    return V

def U_pma(num_qubits, layers):
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
## and running them through a minimization algorithm

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

def HA(num_spins, backend, estimator_options, layers, coupling_strength, field_strength, batch_id=None):
    # Capture start time for DB entry
    start_time = time.time()
    start_timestamp = datetime.now().isoformat()

    hamiltonian = define_heisenberg_loop(num_spins=num_spins, coupling_strength=coupling_strength, field_strength=field_strength)
    ansatz = QuantumCircuit(num_spins)
    ansatz.compose(U_ha(layers, num_spins), inplace=True)

    if not isinstance(backend, StatevectorSimulator):
        raise NotImplementedError(
            "Only StatevectorSimulator is supported right now — "
            "real/other backends aren't handled yet."
        )

    def callback(xk):
        print(f"Iter {len(iterations)}: cost = {iterations[-1]}")

    iterations = []
    x0 = np.ones(ansatz.num_parameters)
    local_estimator = StatevectorEstimator()


    result = minimize(
        cost_func_vqe,
        x0,
        args=(ansatz, hamiltonian, local_estimator, iterations),
        method="COBYLA",
        options={"maxiter": 500, "disp": True},
        callback=callback
    )

    # Capture end time
    end_time = time.time()
    end_timestamp = datetime.now().isoformat()
    comp_time = end_time - start_time

    # 1. Plotting Energy vs Iterations (Saved to disk for HPC compatibility)
    run_id = str(uuid.uuid4())
    plots_dir = os.path.join("output_plots", str(batch_id)) if batch_id else "output_plots"
    os.makedirs("output_plots", exist_ok=True)
    param_tag = f"N{num_spins}_L{layers}_J{coupling_strength}_B{field_strength}"
    plot_filepath = os.path.join(plots_dir, f"HA_energy_plot_{param_tag}_{run_id}.png")

    plt.figure()
    plt.plot(range(1, len(iterations) + 1), iterations, marker='o', linestyle='-', color='b')
    plt.xlabel("Iteration")
    plt.xscale("log")
    plt.ylabel("Energy")
    plt.title(f"Heisenberg VQE Optimization (HA) - {num_spins} Spins")
    plt.grid(True)
    plt.savefig(plot_filepath)
    plt.close() # Close figure to free memory on HPC

    # 2. Local NoSQL Database Document Creation
    db_dir = os.path.join("local_nosql_db", str(batch_id)) if batch_id else "local_nosql_db"
    os.makedirs(db_dir, exist_ok=True)
    db_filepath = os.path.join(db_dir, f"{param_tag}_{run_id}.json")

    # Map implemented parameters to the thesis schema
    db_document = {
        "entry_id": run_id,
        "physical_parameters": {
            "num_spins": num_spins,
            "coupling_strength_J": coupling_strength
            # Note: Jx, Jy, Jz, D omitted as they are unified in the current define_heisenberg_loop
        },
        "external_parameters": {
            "B_z": field_strength
            # Note: Bx and tilt angle θ omitted as they are not implemented in the hamiltonian
        },
        "vqe_configuration": {
            "ansatz_identifier": "HA",
            "ansatz_depth_layers": layers,
            "optimization_algorithm": "COBYLA",
            "max_iterations": 500
        },
        "quantum_run_configuration": {
            "num_gates_pre_transpile": dict(ansatz.count_ops()),
            "shots": "Exact (Statevector)",
            "backend": "Local StatevectorEstimator",
            "measurement_error_mitigation": False
        },
        "results": {
            "best_energy": float(result.fun),
            "best_parameters": result.x.tolist()
            # Note: Density matrix, torque, and parallel/perpendicular magnetization require 
            # additional observables not currently tracked by the basic estimator loop.
        },
        "run_information": {
            "initial_timestamp": start_timestamp,
            "final_timestamp": end_timestamp,
            "computational_time_seconds": comp_time
        }
    }

    # Save to document store
    with open(db_filepath, 'w') as f:
        json.dump(db_document, f, indent=4)

    return result, iterations, plot_filepath, db_filepath


def PMA(num_spins, params, coupling_strength, field_strength):
     hamiltonian = define_heisenberg_loop(num_spins=num_spins, coupling_strength=coupling_strength, field_strength=field_strength)
