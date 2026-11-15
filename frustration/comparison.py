#!/usr/bin/env python3
import math
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
from qiskit_aer import StatevectorSimulator

from frustration_lib import HA, PMA

matplotlib.use("Agg")


def main():
    coupling = 5.0
    spin_orbit = -1.0 / math.sqrt(3)

    # Task 0 -> Graph 1 (3 spins); Task 1 -> Graph 2 (5 spins)
    configs = [
        (3, [0.0, 0.25]),
        (5, [0.0, 0.25, 0.5]),
    ]

    try:
        task_id = int(sys.argv[1])
    except IndexError:
        print("Error: Please provide a SLURM task ID.")
        sys.exit(1)

    batch_id = sys.argv[2] if len(sys.argv) > 2 else None

    if task_id >= len(configs):
        print(f"Task ID {task_id} exceeds config list size ({len(configs)}).")
        sys.exit(0)

    num_spins, field_strengths = configs[task_id]
    backend = StatevectorSimulator()
    pma_energies = []
    ha_energies = []

    for run_idx, field in enumerate(field_strengths, start=1):
        print(f"Task {task_id} | Run {run_idx}: N={num_spins}, Bz={field} K")

        pma_res, _, _, _ = PMA(
            num_spins=num_spins,
            backend=backend,
            estimator_options={},
            layers=1,
            coupling_strength=coupling,
            field_strength=field,
            spin_orbit_strength=spin_orbit,
            batch_id=batch_id,
            initial_guess=[0.0] * 3,
        )
        ha_res, _, _, _ = HA(
            num_spins=num_spins,
            backend=backend,
            estimator_options={},
            layers=3,
            coupling_strength=coupling,
            field_strength=field,
            spin_orbit_strength=spin_orbit,
            batch_id=batch_id,
        )

        pma_energies.append(float(pma_res.fun))
        ha_energies.append(float(ha_res.fun))

    runs = list(range(1, len(field_strengths) + 1))
    plots_dir = os.path.join("output_plots", str(batch_id)) if batch_id else "output_plots"
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f"comparison_N{num_spins}_PMA_vs_HA.png")

    plt.figure()
    plt.plot(runs, pma_energies, marker="o", label="PMA")
    plt.plot(runs, ha_energies, marker="s", label="HA")
    plt.xticks(runs, [f"{r} ($B_z={b}$ K)" for r, b in zip(runs, field_strengths)])
    plt.xlabel("Run #")
    plt.xscale("log")
    plt.ylabel("Energy (K)")
    plt.title(f"PMA vs. HA Energy ({num_spins} Spins, J={coupling} K, D={spin_orbit:.3f} K)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"Task {task_id} complete. Graph saved to {plot_path}")


if __name__ == "__main__":
    main()
