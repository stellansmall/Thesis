import json
import os
import time
import uuid
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import StatePreparation, n_local
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import StatevectorSimulator
from scipy.optimize import minimize


def Heisenberg(
    theta,
):  # Creates the fundamental building block of the PMA Ansatz, of which each
    # layer of the ansatz requires 4
    W = QuantumCircuit(2, name="W(θ)")
    W.cx(0, 1)
    W.rx(theta / 2 - np.pi / 2, 0)
    W.h(0)
    W.rz(theta / 2, 1)
    W.cx(0, 1)
    W.h(0)
    W.rz(-theta / 2, 1)
    W.cx(0, 1)
    W.rx(np.pi / 2, 0)
    W.rx(-np.pi / 2, 1)  # conjugate of rx(pi/2, 1)
    return W


def U_pma(bits, layers):
    # Naive D3 symmetry approach: Make all thetas the
    # same
    # Crippa flips his two layers so we full reverse our 3
    # Preserves a symmetry, at least in the long run.

    U = QuantumCircuit(bits, name="U_PMA")
    params = ParameterVector("θ", layers)

    for idx in range(layers):
        U.compose(Heisenberg(params[idx]), qubits=[bits-1, 0], inplace=True)
        for i in range(1, bits-1, 2):
            U.compose(Heisenberg(params[idx]), qubits=[i, i+1], inplace=True)
        for i in range(0, bits-1, 2):
            U.compose(Heisenberg(params[idx]), qubits=[i, i+1], inplace=True)
    return U


def pma_initial(qubits, target_sz=-0.5, k=None):
    """
        Prepares a symmetry-adapted initial Statevector for a 1D spin ring.
        ONLY FOR LOW-FIELD REGIME B/J LOW EST ~0.4 Parameters:
            qubits (int): Number of spins (qubits) in the ring (must be odd).
            target_sz (float): Target total Sz projection (defaults to S=1/2).
            k (int): Momentum quantum number. If None, it automatically calculates
                     the proven ground-state momentum for odd AFM rings.
        """
    # Auto-calculate the mathematically proven ground state momentum
    if k is None:
        k = (qubits - 1) // 2

    # 1. Determine the exact number of 'down' spins (|1>) needed for target Sz
    num_down = int(qubits / 2 - target_sz)

    if num_down < 0 or num_down > qubits or not float(qubits / 2 - target_sz).is_integer():
        raise ValueError(
            f"Target Sz={target_sz} is invalid for {qubits} qubits.")

    # 2. Create the base computational state
    base_state = [1] * num_down + [0] * (qubits - num_down)

    # 3. Initialize an empty complex statevector for the full 2^N Hilbert space
    vector = np.zeros(2**qubits, dtype=complex)

    # 4. Apply the translational projection operator P^(k)
    for m in range(qubits):
        shifted_state = np.roll(base_state, m)
        bitstring = "".join(map(str, shifted_state))
        state_idx = int(bitstring, 2)

        phase = np.exp(2j * np.pi * k * m / qubits)
        vector[state_idx] += phase

        # 5. Normalize the resulting state vector
    norm = np.linalg.norm(vector)
    if norm < 1e-10:
        raise ValueError(
            "Projection yielded a null state. This momentum sector does not exist."
        )
    vector = vector / norm

    # 6. Build and return the Qiskit circuit
    qc = QuantumCircuit(qubits)
    qc.prepare_state(Statevector(vector))

    return qc


def U_ha(layers, qubits=3):  # I think this might be fine to just port straight
    # to 3 spins?
    V = n_local(
        num_qubits=qubits,
        rotation_blocks="ry",
        entanglement_blocks="cx",
        entanglement="linear",
        reps=layers,
        insert_barriers=True,
        skip_final_rotation_layer=False,
    )
    return V


# Now we set up the actual experiments by configuring the experimental circuits
# and running them through a minimization algorithm

estimator = StatevectorEstimator()


# SOI is Spin-orbit Interaction
def define_heisenberg_loop(num_spins, coupling_strength=1, field_strength=1,
                           SOI=1):
    # This should function perfectly fine for odd spins
    ham_list = []
    edges = [(i, (i + 1) % num_spins) for i in range(num_spins)]

    # Heisenberg interaction term
    for edge in edges:
        ham_list.append(
            ("ZZ", edge, coupling_strength / 2)
        )  # Paper we want to match with B/J has H ~ 2JS1S2 but natural
        # implementation uses sigma not S.
        ham_list.append(("YY", edge, coupling_strength / 2))
        ham_list.append(("XX", edge, coupling_strength / 2))

    # Zeeman Term
    for i in range(num_spins):
        ham_list.append(("Z", [i], field_strength / 2))

    # Spin-Orbit / Dzyaloshinskii-Moriya Term: D_z * (s_i x s_j)_z
    for i, j in edges:
        # D_z * (s_i^x s_j^y - s_i^y s_j^x) -> (SOI / 4) * (XY - YX)
        ham_list.append(("XY", [i, j], SOI / 4.0))
        ham_list.append(("YX", [i, j], -SOI / 4.0))

    hamiltonian = SparsePauliOp.from_sparse_list(
        ham_list, num_qubits=num_spins)

    return hamiltonian


def cost_func_vqe(params, ansatz, hamiltonian, estimator, iterations_list):
    pub = (ansatz, hamiltonian, params)
    cost = float(estimator.run([pub]).result()[0].data.evs)
    iterations_list.append(cost)
    return cost


def _execute_and_save_vqe(
    ansatz,
    hamiltonian,
    num_spins,
    layers,
    coupling_strength,
    field_strength,
    spin_orbit_strength,
    ansatz_id,
    batch_id,
    initial_guess=None,
):
    # Capture start time for DB entry
    start_time = time.time()
    start_timestamp = datetime.now().isoformat()

    def callback(xk):
        print(f"Iter {len(iterations)}: cost = {iterations[-1]}")

    iterations = []
    if initial_guess is not None:
        x0 = np.array(initial_guess)
    else:
        x0 = np.ones(ansatz.num_parameters)
    local_estimator = StatevectorEstimator()

    result = minimize(
        cost_func_vqe,
        x0,
        args=(ansatz, hamiltonian, local_estimator, iterations),
        method="COBYLA",
        options={"maxiter": 5000, "disp": True},
        callback=callback,
    )

    # Capture end time
    end_time = time.time()
    end_timestamp = datetime.now().isoformat()
    comp_time = end_time - start_time

    # 1. Plotting Energy vs Iterations (Saved to disk for HPC compatibility)
    run_id = str(uuid.uuid4())
    plots_dir = (
        os.path.join("output_plots", str(batch_id)
                     ) if batch_id else "output_plots"
    )
    os.makedirs(plots_dir, exist_ok=True)

    if ansatz_id == "HA":
        param_tag = f"N{num_spins}_L{layers}_J{coupling_strength}_B{field_strength}_D{spin_orbit_strength}"
        title = f"Frustrated Heisenberg VQE Optimization (HA) - {num_spins} Spins"
    else:
        param_tag = f"N{num_spins}_L{layers}_J{coupling_strength}_B{field_strength}_D{spin_orbit_strength}"
        title = f"Frustrated Heisenberg VQE Optimization (PMA) - {num_spins} Spins"

    plot_filepath = os.path.join(
        plots_dir, f"{ansatz_id}_energy_plot_{param_tag}_{run_id}.png"
    )

    plt.figure()
    plt.plot(
        range(1, len(iterations) + 1), iterations, marker="o", linestyle="-",
        color="b"
    )
    plt.xlabel("Iteration")
    plt.xscale("log")
    plt.ylabel("Energy")
    plt.title(title)
    plt.grid(True)
    plt.savefig(plot_filepath)
    plt.close()  # Close figure to free memory on HPC

    # 2. Local NoSQL Database Document Creation
    db_dir = (
        os.path.join("local_nosql_db", str(batch_id)
                     ) if batch_id else "local_nosql_db"
    )
    os.makedirs(db_dir, exist_ok=True)
    db_filepath = os.path.join(db_dir, f"{param_tag}_{run_id}.json")

    # Map implemented parameters to the thesis schema
    db_document = {
        "entry_id": run_id,
        "physical_parameters": {
            "num_spins": num_spins,
            "coupling_strength_J": coupling_strength,
            "spin_orbit_strength_delta": spin_orbit_strength,
        },
        "external_parameters": {
            "B_z": field_strength
        },
        "vqe_configuration": {
            "ansatz_identifier": ansatz_id,
            "ansatz_depth_layers": layers,
            "optimization_algorithm": "COBYLA",
            "max_iterations": 5000,  # as much as I don't want to make this a
            # variable, its gotta become one at some
            # point
        },
        "quantum_run_configuration": {
            "num_gates_pre_transpile": dict(ansatz.count_ops()),
            "shots": "Exact (Statevector)",
            "backend": "Local StatevectorEstimator",
            "measurement_error_mitigation": False,
        },
        "results": {
            "best_energy": float(result.fun),
            "best_parameters": result.x.tolist(),
            # Note: Density matrix, torque, and parallel/perpendicular
            # magnetization require additional observables not currently
            # tracked by the basic estimator loop.
        },
        "run_information": {
            "initial_timestamp": start_timestamp,
            "final_timestamp": end_timestamp,
            "computational_time_seconds": comp_time,
        },
    }

    # Save to document store
    with open(db_filepath, "w") as f:
        json.dump(db_document, f, indent=4)

    return result, iterations, plot_filepath, db_filepath


def HA(
    num_spins,
    backend,
    estimator_options,
    layers,
    coupling_strength,
    field_strength,
    spin_orbit_strength,
    batch_id=None,
):
    hamiltonian = define_heisenberg_loop(
        num_spins=num_spins,
        coupling_strength=coupling_strength,
        field_strength=field_strength,
        SOI=spin_orbit_strength
    )
    ansatz = QuantumCircuit(num_spins)
    ansatz.compose(U_ha(layers, num_spins), inplace=True)

    if not isinstance(backend, StatevectorSimulator):
        raise NotImplementedError(
            "Only StatevectorSimulator is supported right now — "
            "real/other backends aren't handled yet."
        )

    return _execute_and_save_vqe(
        ansatz=ansatz,
        hamiltonian=hamiltonian,
        num_spins=num_spins,
        layers=layers,
        coupling_strength=coupling_strength,
        field_strength=field_strength,
        spin_orbit_strength=spin_orbit_strength,
        ansatz_id="HA",
        batch_id=batch_id,
    )


def PMA(  # All the PMA Stuff needs to change and this is no exception
    qubits,
    backend,
    estimator_options,
    layers,
    coupling_strength,
    field_strength,
    spin_orbit_strength,
    batch_id=None,
    initial_guess=None,
):
    hamiltonian = define_heisenberg_loop(  # Might need to change this
        num_spins=qubits, coupling_strength=coupling_strength,
        field_strength=field_strength, SOI=spin_orbit_strength
    )
    ansatz = QuantumCircuit(qubits)
    ansatz.compose(
        pma_initial(qubits=qubits),
        inplace=True
    )
    ansatz.compose(U_pma(qubits, layers), inplace=True)

    if not isinstance(backend, StatevectorSimulator):
        raise NotImplementedError(
            "Only StatevectorSimulator is supported right now — "
            "real/other backends aren't handled yet."
        )

    return _execute_and_save_vqe(
        ansatz=ansatz,
        hamiltonian=hamiltonian,
        num_spins=qubits,
        layers=layers,
        coupling_strength=coupling_strength,
        field_strength=field_strength,
        spin_orbit_strength=spin_orbit_strength,
        ansatz_id="PMA",
        batch_id=batch_id,
        initial_guess=initial_guess,
    )
