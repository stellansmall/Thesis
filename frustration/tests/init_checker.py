import math
import sys

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def prepare_consecutive_ones_state(N: int, k_mom=None) -> QuantumCircuit:
    if N % 2 == 0:
        raise ValueError("N must be an odd number.")

    k = (N - 1) // 2
    if k_mom is None:
        a = 0.5 * (N - 1)
        k_mom = a * math.ceil(N / 2) % N

    qc = QuantumCircuit(2 * N)
    anc = list(range(0, N))
    tgt = list(range(N, 2 * N))

    qc.x(anc[0])
    for i in range(N - 1):
        theta = 2 * np.arccos(1 / np.sqrt(N - i))
        qc.cry(theta, anc[i], anc[i + 1])
        qc.cx(anc[i + 1], anc[i])

    qc.barrier()

    for m in range(N):
        phase = -2 * m * np.pi * k_mom / N
        qc.p(phase, anc[m])

    for m in range(k):
        for j in range(N):
            target_idx = tgt[(j + m) % N]
            qc.cx(anc[j], target_idx)

    qc.barrier()

    for j in range(N):
        prev_target_idx = tgt[(j - 1) % N]
        curr_target_idx = tgt[j]

        qc.x(prev_target_idx)
        qc.ccx(prev_target_idx, curr_target_idx, anc[j])
        qc.x(prev_target_idx)

    return qc


if __name__ == "__main__":
    # 1. Parse command line argument for N
    N = 5  # Default value
    if len(sys.argv) > 1:
        try:
            N = int(sys.argv[1])
        except ValueError:
            print("Error: N must be an integer.")
            sys.exit(1)

    if N % 2 == 0:
        print("Error: N must be an odd number.")
        sys.exit(1)

    # 2. Calculate parameters and exact theoretical expectations
    a = 0.5 * (N - 1)
    k_mom = a * math.ceil(N / 2) % N
    expected_mag = 1.0 / np.sqrt(N)
    expected_phase_step = -2 * np.pi * k_mom / N

    # 3. Generate state
    qc = prepare_consecutive_ones_state(N, k_mom=k_mom)
    ideal_state = Statevector(qc)
    state_dict = ideal_state.to_dict()

    valid_states = []

    # 4. Extract and reverse strings to physical order
    for bitstring, amplitude in state_dict.items():
        magnitude = np.abs(amplitude)
        if magnitude > 1e-6:
            phase = np.angle(amplitude)
            tgt_bits = bitstring[:N][::-1]
            anc_bits = bitstring[N:][::-1]
            valid_states.append((tgt_bits, anc_bits, magnitude, phase))

    # 5. Find the shift 'm' to sort by
    def get_shift(tgt_string):
        for i in range(len(tgt_string)):
            if tgt_string[i] == "1" and tgt_string[i - 1] == "0":
                return i
        return 0

    valid_states.sort(key=lambda x: get_shift(x[0]))

    # 6. Dynamic formatting for output table
    col_width = max(N, 7)

    print(f"\nTarget State Verification (N={N}, k_mom={k_mom})")
    print(f"Expected Magnitude:      {expected_mag:.4f}")
    print(
        f"Expected Phase Shift:    {expected_phase_step:.4f} radians per site")
    print("-" * (col_width * 2 + 30))
    print(
        f"{'Target':<{col_width}} | {'Ancilla':<{col_width}} | {'Magnitude':<9} | {'Phase (rad)':<10}"
    )
    print("-" * (col_width * 2 + 30))

    for tgt_bits, anc_bits, magnitude, phase in valid_states:
        print(
            f"{tgt_bits:<{col_width}} | {anc_bits:<{col_width}} | {magnitude:.4f}    | {phase:.4f}"
        )
