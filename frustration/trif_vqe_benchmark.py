#! /usr/bin/env python3
import sys
import numpy as np
from qiskit_aer import StatevectorSimulator

from frustration_lib import define_heisenberg_loop, PMA

# --- Physical Parameters ---
num_spins = 3
J_code = 2.5             # J_Trif = 5.0 K -> J_code = 2.5 K
D_z = 1.0 / np.sqrt(3)   # Calibrated DM strength for Δ_SO = 1.0 K
layers = 1
initial_guess = [0.0] * layers

# --- Define the 19-Task Parameter Grid ---
grid_panel1 = [
    {"panel": 1, "field": 0.0, "k": 1, "sz": -0.5, "label": "k=1, Sz=-1/2"},
    {"panel": 1, "field": 0.0, "k": 1, "sz": 0.5, "label": "k=1, Sz=+1/2"},
    {"panel": 1, "field": 0.0, "k": -1, "sz": -0.5, "label": "k=-1, Sz=-1/2"},
    {"panel": 1, "field": 0.0, "k": -1, "sz": 0.5, "label": "k=-1, Sz=+1/2"},
]

grid_panel2 = [
    {"panel": 2, "field": b, "k": 1, "sz": -0.5, "label": f"B_code={b:.2f}"}
    for b in [0.25, 0.5, 0.75, 1.0]
]

grid_panel3 = [
    {"panel": 3, "field": b, "k": 1, "sz": -0.5, "label": f"B_tesla={b*0.75:.2f}"}
    for b in np.arange(0.0, 1.05, 0.1)
]

parameter_grid = grid_panel1 + grid_panel2 + grid_panel3


def main():
    try:
        task_id = int(sys.argv[1])
    except IndexError:
        print("Error: Please provide a SLURM task ID (0-18).")
        sys.exit(1)

    batch_id = sys.argv[2] if len(sys.argv) > 2 else "trif_bench"

    if task_id >= len(parameter_grid):
        print(f"Task ID {task_id} exceeds grid size ({len(parameter_grid)}).")
        sys.exit(0)

    task = parameter_grid[task_id]
    print(f"Running Task {task_id} [Panel {task['panel']}]: {task['label']}")

    backend = StatevectorSimulator()
    estimator_options = {}

    # Execute PMA using your updated top-level PMA function
    result, iterations, plot_path, db_path = PMA(
        qubits=num_spins,
        backend=backend,
        estimator_options=estimator_options,
        layers=layers,
        coupling_strength=J_code,
        field_strength=task["field"],
        spin_orbit_strength=D_z,
        batch_id=batch_id,
        initial_guess=initial_guess,
        target_sz=task["sz"],
        k=task["k"],
    )

    print(f"Task {task_id} complete. Data saved to {db_path}")


if __name__ == "__main__":
    main()
