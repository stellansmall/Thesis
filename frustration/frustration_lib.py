import json
import math
import os
import time
import uuid
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import n_local
from qiskit.primitives import BackendEstimatorV2, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import StatevectorSimulator
from scipy.optimize import minimize


def fold(total_spins: int) -> dict[int, int]:
    """
    Maps logical ring indices to a 1D physical line by interleaving
    the front and back of the chain. Returns a dictionary mapping {logical_idx:
    physical_idx}.
    """
    if total_spins <= 0:
        raise ValueError(f"Invalid total_spins: {total_spins}")

    fold_dict = {}
    for i in range(total_spins):
        if i < total_spins / 2:
            fold_dict[i] = 2 * i
        else:
            fold_dict[i] = 2 * (total_spins - 1 - i) + 1

    return fold_dict


def define_heisenberg_loop(
    num_spins, coupling_strength=1.0, field_strength=1.0, DM=-1.0, is_pma=False
):
    """
    Constructs a 1D Heisenberg spin ring Hamiltonian with Zeeman field
    and Dzyaloshinskii-Moriya (DM) spin-orbit interaction.
    """
    fmap = fold(num_spins)
    ham_list = []
    edges = [(fmap[i], fmap[(i + 1) % num_spins]) for i in range(num_spins)]

    # Heisenberg Exchange: J * (Sx Sx + Sy Sy + Sz Sz)
    for i, j in edges:
        ham_list.append(("ZZ", [i, j], coupling_strength / 2.0))
        ham_list.append(("YY", [i, j], coupling_strength / 2.0))
        ham_list.append(("XX", [i, j], coupling_strength / 2.0))

    # Zeeman Term: B * Sz
    for i in range(num_spins):
        ham_list.append(("Z", [fmap[i]], -field_strength / 2.0))

    # Dzyaloshinskii-Moriya (DM) Spin-Orbit Term: D_z * (Sx Sy - Sy Sx)
    for i, j in edges:
        ham_list.append(("XY", [i, j], DM / 4.0))
        ham_list.append(("YX", [i, j], -DM / 4.0))
    num_qubits = 2 * num_spins if is_pma else num_spins

    return SparsePauliOp.from_sparse_list(ham_list, num_qubits=num_qubits)


def pma_initial(num_spins, chirality=None):

    if num_spins % 2 == 0:
        raise ValueError("num_spins must be an odd number.")

    a = int((num_spins - 1) // 2)
    # used for both total up and
    # chirality momentum number calculations
    # locked into a low-B regime with this but can be changed

    if chirality is None:
        chirality = a * math.ceil(num_spins / 2) % num_spins
        # chirality should never really be none at this point since it's
        # defined in PMA but might as well be sure.

    qc = QuantumCircuit(2 * num_spins, name="PMA_Init")
    fmap = fold(num_spins)

    anc = list(range(num_spins, 2 * num_spins))
    tgt = [fmap[i] for i in range(num_spins)]

    # 1. Inject the single spin flip (Base State)
    qc.x(anc[0])

    # 2. Amplitude Distribution Network (Cascading Beam Splitters)
    for i in range(num_spins - 1):
        # Calculate the probability that needs to remain on the current qubit
        theta = 2 * np.arccos(1 / np.sqrt(num_spins - i))

        # CRY -> CX block to split amplitude to the next spin
        qc.cry(theta, anc[i], anc[i + 1])
        qc.cx(anc[i + 1], anc[i])

    qc.barrier()

    # 3. Momentum Phase Network
    for m in range(num_spins):
        phase = 2 * m * np.pi * chirality / num_spins
        qc.p(phase, anc[m])

    for m in range(a):
        for j in range(num_spins):
            target_idx = tgt[(j + m) % num_spins]
            qc.cx(anc[j], target_idx)

    qc.barrier()

    for j in range(num_spins):
        prev_target_idx = tgt[(j - 1) % num_spins]
        curr_target_idx = tgt[j]

        qc.x(prev_target_idx)
        qc.ccx(prev_target_idx, curr_target_idx, anc[j])
        qc.x(prev_target_idx)

    return qc


def U_pma(num_spins, layers):
    """
    Each layer carries three variational angles (theta_x, theta_y, theta_z),
    shared by every bond in that layer, so the circuit has 3 * layers
    parameters. Ordering is layer-major:
    [θx_0, θy_0, θz_0, θx_1, θy_1, θz_1, ...].
    """
    qc = QuantumCircuit(num_spins, name="U_PMA")
    params = ParameterVector("θ", 3 * layers)

    for i in range(0, layers):  # The order here can and maybe should be
        # changed
        theta = params[3 * i : 3 * i + 3]  # (θx, θy, θz) for this layer
        half = [t / 2 for t in theta]
        qc.append(Trotter(0, half, num_spins), range(num_spins))
        qc.append(Trotter(1, half, num_spins), range(num_spins))
        qc.append(Trotter(2, theta, num_spins), range(num_spins))
        qc.append(Trotter(1, half, num_spins), range(num_spins))
        qc.append(Trotter(0, half, num_spins), range(num_spins))

    return qc


def Trotter(init, thetas, size):
    """thetas = (theta_x, theta_y, theta_z) applied to every bond in this sub-layer."""
    qc = QuantumCircuit(size, name=f"Trotter_{init}")
    fmap = fold(size)
    if init == 0:
        qc.append(interaction_term(*thetas), [fmap[size - 1], fmap[0]])
    else:
        for i in range(init, size, 2):
            qc.append(interaction_term(*thetas), [fmap[i - 1], fmap[i]])
    return qc


def interaction_term(theta_x, theta_y, theta_z):
    """
    exp[-i (θx XX + θy YY + θz ZZ)] on two qubits, up to a global phase.

    This is Crippa's W with theta -> 4*theta on each axis, so the angles are
    the coefficients of the Pauli terms in the exponent. The XX, YY, ZZ terms
    commute, so this is exact for a single bond.
    """
    tx, ty, tz = 4 * theta_x, 4 * theta_y, 4 * theta_z
    W = QuantumCircuit(2, name="H")
    W.cx(0, 1)
    W.rx(tx / 2 - np.pi / 2, 0)
    W.h(0)
    W.rz(tz / 2, 1)
    W.cx(0, 1)
    W.h(0)
    W.rz(-ty / 2, 1)
    W.cx(0, 1)
    W.rx(np.pi / 2, 0)
    W.rx(-np.pi / 2, 1)  # conjugate of rx(pi/2, 1)
    return W


def U_ha(num_spins, layers):
    """Heuristic Ansatz."""
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
    backend=None,
    estimator_options=None,
    optimizer_method="COBYLA",
    initial_guess=None,
    chirality=None,
):
    start_time = time.time()
    start_timestamp = datetime.now().isoformat()

    estimator_options = estimator_options or {}

    # 1. Determine Primitive Estimator & Transpile Circuit if on Real Backend
    is_statevector = backend is None or isinstance(
        backend, StatevectorSimulator)

    if is_statevector:
        estimator = StatevectorEstimator()
        target_circuit = ansatz
        target_hamiltonian = hamiltonian
        backend_name = "Local StatevectorEstimator"
        shots_info = "Exact (Statevector)"
    else:
        # Transpile for Target Hardware / Backend Architecture
        pm = generate_preset_pass_manager(
            optimization_level=2, backend=backend)
        target_circuit = pm.run(ansatz)
        target_hamiltonian = hamiltonian.apply_layout(target_circuit.layout)

        estimator = BackendEstimatorV2(
            backend=backend, options=estimator_options)
        backend_name = getattr(backend, "name", str(backend))
        shots_info = estimator_options.get("default_shots", "Hardware Sampled")

    iterations = []
    x0 = (
        np.array(initial_guess)
        if initial_guess is not None
        else np.ones(ansatz.num_parameters)
    )

    def callback(xk):
        print(f"Iter {len(iterations)}: cost = {iterations[-1]:.6f}")

    result = minimize(
        _cost_func_vqe,
        x0,
        args=(target_circuit, target_hamiltonian, estimator, iterations),
        method=optimizer_method,
        options={"maxiter": 5000, "disp": False},
        callback=callback,
    )

    end_time = time.time()
    end_timestamp = datetime.now().isoformat()
    comp_time = end_time - start_time

    run_id = str(uuid.uuid4())[:8]
    param_tag = f"N{num_spins}_L{layers}_J{coupling_strength:.2f}_B{field_strength:.2f}_D{spin_orbit_strength:.3f}"

    # 2. Output Plot Generation
    plots_dir = (
        os.path.join("output_plots", str(batch_id)
                     ) if batch_id else "output_plots"
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
    plt.title(
        f"VQE Optimization ({ansatz_id}) - N={num_spins}, L={layers}\nBackend: {backend_name}"
    )
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_filepath)
    plt.close()

    # 3. Document Store Output
    db_dir = (
        os.path.join("local_nosql_db", str(batch_id)
                     ) if batch_id else "local_nosql_db"
    )
    os.makedirs(db_dir, exist_ok=True)
    db_filepath = os.path.join(db_dir, f"{param_tag}_{run_id}.json")

    db_document = {
        "entry_id": run_id,
        "physical_parameters": {
            "num_spins": num_spins,
            "coupling_strength_J": coupling_strength,
            "spin_orbit_strength_D": spin_orbit_strength,
            "target_sz": -0.5 if ansatz_id == "PMA" else "none (Heuristic)",
            "chirality": chirality,
        },
        "external_parameters": {"B_z": field_strength},
        "vqe_configuration": {
            "ansatz_identifier": ansatz_id,
            "ansatz_depth_layers": layers,
            "optimization_algorithm": optimizer_method,
            "max_iterations": 5000,
        },
        "quantum_run_configuration": {
            "num_gates_pre_transpile": dict(ansatz.count_ops()),
            "num_gates_post_transpile": dict(target_circuit.count_ops()),
            "shots": shots_info,
            "backend": backend_name,
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
    backend=None,
    estimator_options=None,
    layers=1,
    coupling_strength=1.0,
    field_strength=1.0,
    spin_orbit_strength=-1.0,
    chirality=None,
    batch_id=None,
    optimizer_method="COBYLA",
    initial_guess=None,
):
    if chirality is None:
        chirality = int((0.5 * (num_spins - 1) *
                     math.ceil(num_spins / 2))) % num_spins

    if initial_guess is not None and len(initial_guess) != 3 * layers:
        raise ValueError(
            f"initial_guess needs 3 * layers = {3 * layers} entries "
            f"([θx, θy, θz] per layer), got {len(initial_guess)}."
        )

    hamiltonian = define_heisenberg_loop(
        num_spins=num_spins,
        coupling_strength=coupling_strength,
        field_strength=field_strength,
        DM=spin_orbit_strength,
        is_pma=True,
    )
    ansatz = QuantumCircuit(2 * num_spins)
    ansatz.compose(
        pma_initial(num_spins=num_spins, chirality=chirality),
        inplace=True,
    )
    ansatz.compose(U_pma(num_spins, layers), qubits=range(num_spins), inplace=True)

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
        backend=backend,
        estimator_options=estimator_options,
        optimizer_method=optimizer_method,
        initial_guess=initial_guess,
        chirality=chirality,
    )


def HA(
    num_spins,
    backend=None,
    estimator_options=None,
    layers=1,
    coupling_strength=1.0,
    field_strength=1.0,
    spin_orbit_strength=-1.0,
    batch_id=None,
    optimizer_method="COBYLA",
    initial_guess=None,
):
    hamiltonian = define_heisenberg_loop(
        num_spins=num_spins,
        coupling_strength=coupling_strength,
        field_strength=field_strength,
        DM=spin_orbit_strength,
        is_pma=False,
    )
    ansatz = QuantumCircuit(num_spins)
    ansatz.compose(U_ha(num_spins, layers), inplace=True)

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
        backend=backend,
        estimator_options=estimator_options,
        optimizer_method=optimizer_method,
        initial_guess=initial_guess,
    )
