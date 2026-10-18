import json
import os
import time
import uuid
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate, n_local
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import StatevectorSimulator
from scipy.optimize import minimize


def define_heisenberg_loop(
    num_spins, coupling_strength=1.0, field_strength=1.0, DM=-1.0
):
    """
    Constructs a 1D Heisenberg spin ring Hamiltonian with Zeeman field
    and Dzyaloshinskii-Moriya (DM) spin-orbit interaction.
    """
    ham_list = []
    edges = [(i, (i + 1) % num_spins) for i in range(num_spins)]

    # Heisenberg Exchange: J * (Sx Sx + Sy Sy + Sz Sz)
    for i, j in edges:
        ham_list.append(("ZZ", [i, j], coupling_strength / 2.0))
        ham_list.append(("YY", [i, j], coupling_strength / 2.0))
        ham_list.append(("XX", [i, j], coupling_strength / 2.0))

    # Zeeman Term: B * Sz
    for i in range(num_spins):
        ham_list.append(("Z", [i], field_strength / 2.0))

    # Dzyaloshinskii-Moriya (DM) Spin-Orbit Term: D_z * (Sx Sy - Sy Sx)
    for i, j in edges:
        ham_list.append(("XY", [i, j], DM / 4.0))
        ham_list.append(("YX", [i, j], -DM / 4.0))

    return SparsePauliOp.from_sparse_list(ham_list, num_qubits=num_spins)


def pma_initial(num_spins, target_sz=-0.5, chirality=None):
    """
    Prepares a symmetry-adapted initial Statevector for a 1D spin ring
    projected into total Sz and chirality (momentum) sectors.
    """
    if chirality is None:
        chirality = (num_spins - 1) // 2

    num_down = int(num_spins / 2 - target_sz)
    if (
        num_down < 0
        or num_down > num_spins
        or not float(num_spins / 2 - target_sz).is_integer()
    ):
        raise ValueError(
            f"Target Sz={target_sz} is invalid for {num_spins} spins."
        )

    base_state = [1] * num_down + [0] * (num_spins - num_down)
    vector = np.zeros(2**num_spins, dtype=complex)

    for m in range(num_spins):
        shifted_state = np.roll(base_state, m)
        bitstring = "".join(map(str, shifted_state))
        state_idx = int(bitstring, 2)
        phase = np.exp(2j * np.pi * chirality * m / num_spins)
        vector[state_idx] += phase

    norm = np.linalg.norm(vector)
    if norm < 1e-10:
        raise ValueError(
            "Projection yielded a null state for this symmetry sector."
        )
    vector = vector / norm

    qc = QuantumCircuit(num_spins)
    qc.prepare_state(Statevector(vector))
    return qc


def U_pma(num_spins, layers):
    """
    Symmetry-preserving global Hamiltonian Variational Ansatz (HVA).
    Evolves under the full isotropic ring Hamiltonian to conserve chirality and Sz.
    """
    qc = QuantumCircuit(num_spins, name="U_PMA")
    params = ParameterVector("θ", layers)

    H_ring = define_heisenberg_loop(
        num_spins=num_spins,
        coupling_strength=1.0,
        field_strength=0.0,
        DM=0.0,
    )

    for idx in range(layers):
        evo_gate = PauliEvolutionGate(H_ring, time=params[idx])
        qc.append(evo_gate, range(num_spins))

    return qc


def U_ha(num_spins, layers):
    """Hardware-Efficient Ansatz (n_local RY + CZ/CX)."""
    return n_local(
        num_qubits=num_spins,
        rotation_blocks="ry",
        entanglement_blocks="cx",
        entanglement="linear",
        reps=layers,
        insert_barriers=True,
        skip_final_rotation_layer=False,
    )


def _cost_func_vqe(params, ansatz, hamiltonian, estimator, iterations_list):
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
    target_sz=None,
    chirality=None,
):
    start_time = time.time()
    start_timestamp = datetime.now().isoformat()

    iterations = []
    x0 = (
        np.array(initial_guess)
        if initial_guess is not None
        else np.ones(ansatz.num_parameters)
    )
    estimator = StatevectorEstimator()

    def callback(xk):
        print(f"Iter {len(iterations)}: cost = {iterations[-1]:.6f}")

    result = minimize(
        _cost_func_vqe,
        x0,
        args=(ansatz, hamiltonian, estimator, iterations),
        method="COBYLA",
        options={"maxiter": 5000, "disp": False},
        callback=callback,
    )

    end_time = time.time()
    end_timestamp = datetime.now().isoformat()
    comp_time = end_time - start_time

    run_id = str(uuid.uuid4())[:8]
    param_tag = f"N{num_spins}_L{layers}_J{coupling_strength:.2f}_B{field_strength:.2f}_D{spin_orbit_strength:.3f}"

    # 1. Output Plot Generation
    plots_dir = (
        os.path.join("output_plots", str(batch_id))
        if batch_id
        else "output_plots"
    )
    os.makedirs(plots_dir, exist_ok=True)
    plot_filepath = os.path.join(
        plots_dir, f"{ansatz_id}_energy_{param_tag}_{run_id}.png"
    )

    plt.figure()
    plt.plot(
        range(1, len(iterations) + 1),
        iterations,
        marker="o",
        linestyle="-",
        color="b",
    )
    plt.xlabel("Iteration")
    plt.xscale("log")
    plt.ylabel("Energy (K)")
    plt.title(f"VQE Optimization ({ansatz_id}) - N={num_spins}, L={layers}")
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_filepath)
    plt.close()

    # 2. Document Store Output
    db_dir = (
        os.path.join("local_nosql_db", str(batch_id))
        if batch_id
        else "local_nosql_db"
    )
    os.makedirs(db_dir, exist_ok=True)
    db_filepath = os.path.join(db_dir, f"{param_tag}_{run_id}.json")

    db_document = {
        "entry_id": run_id,
        "physical_parameters": {
            "num_spins": num_spins,
            "coupling_strength_J": coupling_strength,
            "spin_orbit_strength_D": spin_orbit_strength,
            "target_sz": target_sz,
            "chirality": chirality,
        },
        "external_parameters": {"B_z": field_strength},
        "vqe_configuration": {
            "ansatz_identifier": ansatz_id,
            "ansatz_depth_layers": layers,
            "optimization_algorithm": "COBYLA",
            "max_iterations": 5000,
        },
        "quantum_run_configuration": {
            "num_gates_pre_transpile": dict(ansatz.count_ops()),
            "shots": "Exact (Statevector)",
            "backend": "Local StatevectorEstimator",
        },
        "results": {
            "best_energy": float(result.fun),
            "best_parameters": result.x.tolist(),
            "history": iterations,
        },
        "run_information": {
            "initial_timestamp": start_timestamp,
            "final_timestamp": end_timestamp,
            "computational_time_seconds": comp_time,
        },
    }

    with open(db_filepath, "w") as f:
        json.dump(db_document, f, indent=4)

    return result, iterations, plot_filepath, db_filepath


def PMA(
    num_spins,
    backend,
    estimator_options,
    layers,
    coupling_strength,
    field_strength,
    spin_orbit_strength,
    target_sz=-0.5,
    chirality=None,
    batch_id=None,
    initial_guess=None,
):
    hamiltonian = define_heisenberg_loop(
        num_spins=num_spins,
        coupling_strength=coupling_strength,
        field_strength=field_strength,
        DM=spin_orbit_strength,
    )
    ansatz = QuantumCircuit(num_spins)
    ansatz.compose(
        pma_initial(
            num_spins=num_spins, target_sz=target_sz, chirality=chirality
        ),
        inplace=True,
    )
    ansatz.compose(U_pma(num_spins, layers), inplace=True)

    if not isinstance(backend, StatevectorSimulator):
        raise NotImplementedError("Only StatevectorSimulator is supported.")

    return _execute_and_save_vqe(
        ansatz=ansatz,
        hamiltonian=hamiltonian,
        num_spins=num_spins,
        layers=layers,
        coupling_strength=coupling_strength,
        field_strength=field_strength,
        spin_orbit_strength=spin_orbit_strength,
        ansatz_id="PMA",
        batch_id=batch_id,
        initial_guess=initial_guess,
        target_sz=target_sz,
        chirality=chirality,
    )


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
        DM=spin_orbit_strength,
    )
    ansatz = QuantumCircuit(num_spins)
    ansatz.compose(U_ha(num_spins, layers), inplace=True)

    if not isinstance(backend, StatevectorSimulator):
        raise NotImplementedError("Only StatevectorSimulator is supported.")

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
