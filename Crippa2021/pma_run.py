#! /usr/bin/env python3
import sys
import itertools
import matplotlib
matplotlib.use('Agg')
from qiskit_aer import StatevectorSimulator
from ansaetze import *

def main():
    #non-variational input parameters
    num_spins_list = [4]
    layers_list = [2]
    coupling_strengths = [1.0]
    field_strengths = [0.4, 3.2, 5]

    #generate all lists of parameters
    parameter_grid = list(itertools.product(
        num_spins_list,
        layers_list,
        coupling_strengths,
        field_strengths,
    ))

    # Get the SLURM array task ID from command line arguments
    #NOTE: AI Generated, dont really know what this does and wasn't in andys code
    #just looked it up: slurm gives each run a task id which is a number 0-53
    #each one corresponds to a configuration of non-var parameters.
    try:
        task_id = int(sys.argv[1])
    except IndexError:
        print("Error: Please provide a task ID.")
        sys.exit(1)

    # Optional second argument: a batch identifier (e.g. $SLURM_ARRAY_JOB_ID)
    # used to keep separate sbatch submissions in their own subfolders.
    batch_id = sys.argv[2] if len(sys.argv) > 2 else None

    if task_id >= len(parameter_grid):
        print(f"Task ID {task_id} exceeds parameter grid size ({len(parameter_grid)}).")
        sys.exit(0)

    num_spins, layers, coupling, field = parameter_grid[task_id]

    print(f"Running Task {task_id}: Spins={num_spins}, Layers={layers}, J={coupling}, Bz={field}")

    backend = StatevectorSimulator()
    estimator_options = {}

    # 4. Execute the function
    result, iterations, plot_path, db_path = PMA(
        backend,
        estimator_options,
        layers,
        coupling,
        field,
        batch_id
    )

    print(f"Task {task_id} complete. Data saved to {db_path} and {plot_path}")

if __name__ == "__main__":
    main()

