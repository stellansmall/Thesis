from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService

J = 1.0
hx = -0.5

obs = SparsePauliOp.from_list([("ZZ", J), ("XI", hx), ("IX", hx)])

qc = QuantumCircuit(2)

qc.h(0)
qc.cx(0, 1)
qc.x(1)
qc.z(0)

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=127)
print(backend.name)

target = backend.target
pm = generate_preset_pass_manager(target=target, optimization_level=3)
qc_isa = pm.run(qc)
obs_isa = obs.apply_layout(layout=qc_isa.layout)

print(qc_isa)

from qiskit_ibm_runtime import EstimatorV2 as Estimator

estimator = Estimator(mode=backend)

pubs = [(qc_isa, obs_isa)]
job = estimator.run([[qc_isa, obs_isa]])
res = job.result()

print(res[0].data.evs)

