#!/usr/bin/env python3
"""Numerical check: interaction_term(θx, θy, θz) == exp[-i(θx XX + θy YY + θz ZZ)] up to global phase."""
import itertools

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator, SparsePauliOp

N_RANDOM = 200
RANGE = 2 * np.pi          # random angles drawn uniformly from [-RANGE, RANGE]
TOL = 1e-10
SEED = 0


def interaction_term(theta_x, theta_y, theta_z):
    W = QuantumCircuit(2, name="H")
    W.cx(0, 1)
    W.rx(theta_x / 2 - np.pi / 2, 0)
    W.h(0)
    W.rz(theta_z / 2, 1)
    W.cx(0, 1)
    W.h(0)
    W.rz(-theta_y / 2, 1)
    W.cx(0, 1)
    W.rx(np.pi / 2, 0)
    W.rx(-np.pi / 2, 1)  # conjugate of rx(pi/2, 1)
    return W


def reference(tx, ty, tz):
    op = SparsePauliOp(["XX", "YY", "ZZ"], coeffs=np.array([tx, ty, tz]))
    return Operator(PauliEvolutionGate(op, time=1.0)).data


def compare(tx, ty, tz):
    """Return (max deviation after removing global phase, global phase)."""
    U = Operator(interaction_term(tx, ty, tz)).data
    V = reference(tx, ty, tz)
    k = np.unravel_index(np.argmax(np.abs(V)), V.shape)   # phase from largest entry
    phase = U[k] / V[k]
    phase /= abs(phase)
    return np.max(np.abs(U - phase * V)), np.angle(phase)


def main():
    rng = np.random.default_rng(SEED)
    special = [0.0, np.pi / 8, np.pi / 4, np.pi / 2, np.pi, -np.pi / 3, 2 * np.pi]
    corner = list(itertools.product(special, repeat=3))
    random = [tuple(rng.uniform(-RANGE, RANGE, 3)) for _ in range(N_RANDOM)]

    results = {"corner": [], "random": []}
    for name, pts in (("corner", corner), ("random", random)):
        for p in pts:
            err, ph = compare(*p)
            results[name].append((err, ph, p))

    print(f"{'set':<8}{'# tested':>10}{'max error':>14}{'phases (rad)':>20}   status")
    print("-" * 64)
    all_ok = True
    for name, rows in results.items():
        errs = np.array([r[0] for r in rows])
        phs = np.array([r[1] for r in rows])
        ok = errs.max() < TOL
        all_ok &= ok
        print(f"{name:<8}{len(rows):>10}{errs.max():>14.2e}"
              f"{f'[{phs.min():.3f}, {phs.max():.3f}]':>20}   {'PASS' if ok else 'FAIL'}")

    worst = max((r for rows in results.values() for r in rows), key=lambda r: r[0])
    print(f"\nWorst case: error {worst[0]:.2e} at "
          f"(θx, θy, θz) = ({worst[2][0]:.4f}, {worst[2][1]:.4f}, {worst[2][2]:.4f})")
    print(f"π/4 = {np.pi / 4:.3f}; a constant phase at that value means the global phase is angle-independent.")
    print("\nOVERALL:", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
