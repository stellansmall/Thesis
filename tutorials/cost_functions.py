from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import n_local
original_circuit = QuantumCircuit(1)
original_circuit.h(0)

H = SparsePauliOp(["X", "Z"], [2, -1])

aux_circuits = []
for pauli in H.paulis:
    aux_circ = original_circuit.copy()
    aux_circ.barrier()
    if str(pauli) == "X":
        aux_circ.h(0)
    elif str(pauli) == "Y":
        aux_circ.sdg(0)
        aux_circ.h(0)
    else:
         aux_circ.id(0)
    aux_circ.measure_all()
    aux_circuits.append(aux_circ)

print(original_circuit)

print(aux_circuits[0])
print(aux_circuits[1])

from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.result import QuasiDistribution
import numpy as np

shots = 10000
sampler = StatevectorSampler()
job = sampler.run(aux_circuits, shots=shots)

expvals = []
for index, pauli in enumerate(H.paulis):
    data_pub = job.result()[index].data
    bitstrings = data_pub.meas.get_bitstrings()
    counts = data_pub.meas.get_counts()
    quasi_dist = QuasiDistribution(
        {outcome: freq / shots for outcome, freq in counts.items()}
    )

    val = 0

    if str(pauli) == "X":
        val += -1 * quasi_dist.get(1,0)
        val += 1 * quasi_dist.get(0,0)
    if str(pauli) == "Y":
        val += -1 * quasi_dist.get(1,0)
        val += 1 * quasi_dist.get(0,0)
    if str(pauli)== "Z":
        val += 1 * quasi_dist.get(0,0)
        val += -1 * quasi_dist.get(1,0)
    expvals.append(val)

print("Sampler results:")
for pauli, expval in zip(H.paulis, expvals):
    print(f" >> Expected value of {str(pauli)}: {expval:.5f}")

total_expval = np.sum(H.coeffs * expvals).real
print(f" >> Total expected value: {total_expval:.5f}")

observables = [
    *H.paulis,
    H,
]

estimator = StatevectorEstimator()

job = estimator.run([(original_circuit, observables)])
estimator_expvals = job.result()[0].data.evs

print("Estimator results:")
for obs, expval in zip(observables, estimator_expvals):
    if obs is not H:
        print(f" >> Expected value of {str(obs)}: {expval:.5f}")
    else:
        print(f" >> Total expected value: {expval:.5f}")

def cost_func_vqe(params, circuit, hamiltonian, estimator):

    pub = (circuit, hamiltonian, params)
    cost = estimator.run([pub]).result()[0].data.evs
    return cost

observable = SparsePauliOp.from_list([("XX", 1), ("YY", -3)])

reference_circuit = QuantumCircuit(2)
reference_circuit.x(0)

variational_form = n_local(
    2,
    rotation_blocks=["rz", "ry"],
    entanglement_blocks="cx",
    entanglement="linear",
    reps = 1,
)
ansatz = reference_circuit.compose(variational_form)

theta_list = (2 * np.pi * np.random.rand(1, 8)).tolist()
print(ansatz.decompose())

estimator = StatevectorEstimator()
cost = cost_func_vqe(theta_list, ansatz, observable, estimator)
print(cost)
