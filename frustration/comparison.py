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
    # Base physical parameters from pma_run.py
    coupling = 5.0
    spin_orbit = -1.0 / math.sqrt(3)

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

    if task_id != 0:
        print(f"Task ID {task_id} skipped (all runs execute in task 0).")
        sys.exit(0)

    backend = StatevectorSimulator()
    estimator_options = {}
    plots_dir = os.path.join("output_plots", str(batch_id)) if batch_id else "output_plots"
    os.makedirs(plots_dir, exist_ok=True)

    colors = ["tab:blue", "tab:orange", "tab:green"]

    for num_spins, field_strengths in configs:
        plt.figure(figsize=(9, 5.5))

        for color, field in zip(colors, field_strengths):
            print(
                f"Running PMA (L=1) & HA (L=3): Spins={num_spins}, "
                f"J={coupling}, Bz={field}, D_z={spin_orbit}"
            )

            _, pma_iters, _, _ = PMA(
                num_spins=num_spins,
                backend=backend,
                estimator_options=estimator_options,
                layers=1,
                coupling_strength=coupling,
                field_strength=field,
                spin_orbit_strength=spin_orbit,
                batch_id=batch_id,
                initial_guess=[0.0] * 3,
            )
            _, ha_iters, _, _ = HA(
                num_spins=num_spins,
                backend=backend,
                estimator_options=estimator_options,
                layers=3,
                coupling_strength=coupling,
                field_strength=field,
                spin_orbit_strength=spin_orbit,
                batch_id=batch_id,
            )

            plt.plot(
                range(1, len(pma_iters) + 1),
                pma_iters,
                linestyle="-",
                color=color,
                linewidth=1.8,
                label=f"PMA ($B_z = {field}$ K)",
            )
            plt.plot(
                range(1, len(ha_iters) + 1),
                ha_iters,
                linestyle="--",
                color=color,
                linewidth=1.8,
                label=f"HA ($B_z = {field}$ K)",
            )

        plot_path = os.path.join(plots_dir, f"overlaid_N{num_spins}_PMA_vs_HA.png")
        plt.xlabel("Iteration (Run #)")
        plt.xscale("log")
        plt.ylabel("Energy (K)")
        plt.title(
            f"VQE Optimization Overlay (PMA L=1 vs. HA L=3) - N={num_spins}\n"
            f"J={coupling} K, D_z={spin_orbit:.3f} K"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        print(f"Saved overlaid graph to {plot_path}")


if __name__ == "__main__":
    main()