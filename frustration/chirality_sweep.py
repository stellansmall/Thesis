#!/usr/bin/env python3
import math
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
from qiskit_aer import StatevectorSimulator

from frustration_lib import HA, PMA

matplotlib.use("Agg")

# --- Physical Parameters ---
num_spins = 5
coupling = 5.0
field = 0.0  # IN KELVIN
spin_orbit = -1.0 / math.sqrt(3)

pma_layers = 1
ha_layers = 3  # same depth as comparison.py

# Chirality is defined mod num_spins, so the full set is 0..N-1.
chiralities = list(range(num_spins))

# Horizontal reference line (K). Leave as None to skip drawing it.
REFERENCE_ENERGY = -9.864
REFERENCE_LABEL = "XDiag Simulated Value"


def main():
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

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    plt.figure(figsize=(9, 5.5))

    # --- PMA: one run per chirality ---
    for color, chirality in zip(colors, chiralities):
        print(
            f"Running PMA (L={pma_layers}): Spins={num_spins}, "
            f"J={coupling}, Bz={field}, D_z={spin_orbit}, C={chirality}"
        )

        result, pma_iters, _, _ = PMA(
            num_spins=num_spins,
            backend=backend,
            estimator_options=estimator_options,
            layers=pma_layers,
            coupling_strength=coupling,
            field_strength=field,
            spin_orbit_strength=spin_orbit,
            chirality=chirality,
            batch_id=batch_id,
            initial_guess=[0.0] * 3 * pma_layers,
        )
        print(f"PMA C={chirality}: best energy = {result.fun:.6f} K")

        plt.plot(
            range(1, len(pma_iters) + 1),
            pma_iters,
            linestyle="-",
            color=color,
            linewidth=1.8,
            label=f"PMA (C = {chirality})",
        )

    # --- HA: same Hamiltonian, no symmetry adaptation ---
    print(
        f"Running HA (L={ha_layers}): Spins={num_spins}, "
        f"J={coupling}, Bz={field}, D_z={spin_orbit}"
    )

    result, ha_iters, _, _ = HA(
        num_spins=num_spins,
        backend=backend,
        estimator_options=estimator_options,
        layers=ha_layers,
        coupling_strength=coupling,
        field_strength=field,
        spin_orbit_strength=spin_orbit,
        batch_id=batch_id,
    )
    print(f"HA: best energy = {result.fun:.6f} K")

    plt.plot(
        range(1, len(ha_iters) + 1),
        ha_iters,
        linestyle="--",
        color="black",
        linewidth=1.8,
        label=f"HA (L = {ha_layers})",
    )

    # --- Horizontal reference line ---
    if REFERENCE_ENERGY is not None:
        plt.axhline(
            REFERENCE_ENERGY,
            linestyle=":",
            color="gray",
            linewidth=1.5,
            label=REFERENCE_LABEL,
        )

    plot_path = os.path.join(plots_dir, f"chirality_scan_N{num_spins}_PMA_vs_HA.png")
    plt.xlabel("Iteration (Run #)")
    plt.xscale("log")
    plt.ylabel("Energy (K)")
    plt.title(
        f"VQE Optimization (PMA L={pma_layers}, all chiralities vs. HA L={ha_layers}) - N={num_spins}\n"
        f"J={coupling} K, B_z={field} K, D_z={spin_orbit:.3f} K"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved overlaid graph to {plot_path}")


if __name__ == "__main__":
    main()
