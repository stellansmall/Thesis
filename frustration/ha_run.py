#! /usr/bin/env python3
import itertools
import sys
import matplotlib

matplotlib.use("Agg")
from qiskit_aer import StatevectorSimulator
from frustration_lib import HA


def main():
    # Non-variational input parameters for ODD spins
    num_spins_list = [3, 5]
    layers_list = [2, 3]  # HA requires 2-3 layers for reasonable convergence
    coupling_strengths = [1.0]
    field_strengths = [0.25, 0.4, 1.0, 1.5, 2.4, 3.2, 2.8]
    spin_orbit_strengths = [0.1]  # DM interaction strength (SOI)

    # Generate parameter grid
    parameter_grid = list(
        itertools.product(
            num_spins_list,
            layers_list,
            coupling_strengths,
            field_strengths,
            spin_orbit_strengths,
        )
    )

    try:
        task_id = int(sys.argv[1])
    except IndexError:
        print("Error: Please provide a SLURM task ID.")
        sys.exit(1)

    batch_id = sys.argv[2] if len(sys.argv) > 2 else None

    if task_id >= len(parameter_grid):
        print(
            f"Task ID {task_id} exceeds parameter grid size ({len(parameter_grid)})."
        )
        sys.exit(0)

    num_spins, layers, coupling, field, spin_orbit = parameter_grid[task_id]

    print(
        f"Running HA Task {task_id}: Spins={num_spins}, Layers={layers}, "
        f"J={coupling}, Bz={field}, D_z={spin_orbit}"
    )

    backend = StatevectorSimulator()
    estimator_options = {}

    # Execute HA function
    result, iterations, plot_path, db_path = HA(
        num_spins=num_spins,
        backend=backend,
        estimator_options=estimator_options,
        layers=layers,
        coupling_strength=coupling,
        field_strength=field,
        spin_orbit_strength=spin_orbit,
        batch_id=batch_id,
    )

    print(f"Task {task_id} complete. Data saved to {db_path} and {plot_path}")


if __name__ == "__main__":
    main()
