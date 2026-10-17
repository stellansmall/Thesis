#! /usr/bin/env python3
import glob
import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from frustration_lib import define_heisenberg_loop

# --- Physical Parameters for Energy Offset ---
num_spins = 3
J_code = 2.5
D_z = 1.0 / np.sqrt(3)

# Calculate S=1/2 mean energy at B=0 for Trif et al. alignment
H_op = define_heisenberg_loop(num_spins, J_code, 0.0, D_z)
E_shift = np.mean(np.linalg.eigvalsh(H_op.to_matrix())[:4])

# --- Retrieve Job ID from Command-Line ---
job_id = sys.argv[1] if len(sys.argv) > 1 else "trif_bench"
db_dir = os.path.join("local_nosql_db", job_id)
json_files = glob.glob(os.path.join(db_dir, "*.json"))

if not json_files:
    print(f"Error: No JSON records found in {db_dir}")
    sys.exit(1)

# --- Load Records ---
records = []
for filepath in json_files:
    with open(filepath, "r") as f:
        records.append(json.load(f))

# Define parameter grid ordering matching trif_vqe_benchmark.py

grid_panel1 = [
    {"field": 0.0, "C": 1, "sz": -0.5, "label": "C=1, Sz=-1/2"},
    {"field": 0.0, "C": 1, "sz": 0.5, "label": "C=1, Sz=+1/2"},
    {"field": 0.0, "C": -1, "sz": -0.5, "label": "C=-1, Sz=-1/2"},
    {"field": 0.0, "C": -1, "sz": 0.5, "label": "C=-1, Sz=+1/2"},
]

grid_panel2 = [
    {"field": b, "label": f"B_code = {b:.2f} K"}
    for b in [0.25, 0.5, 0.75, 1.0]
]

grid_panel3_fields = np.arange(0.0, 1.05, 0.1)

# --- Setup Plot ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f"Trif et al. VQE Benchmark (Cu3 Triangle, J={J_code*2}K, Δ_SO=1K)",
    fontsize=14,
)

# Panel 1: Zero-Field Splitting
ax1 = axes[0]
ax1.set_title("Panel 1: Zero-Field Splitting (B=0 T)")
ax1.set_xlabel("VQE Iteration")
ax1.set_ylabel("Energy (K)")

for target in grid_panel1:
    # Match JSON record by magnetic field value
    rec = next(
        (
            r
            for r in records
            if abs(r["external_parameters"]["B_z"] - target["field"]) < 1e-5
            and r["physical_parameters"].get("Initialized Sz (PMA only)") == target["sz"]
            and r["physical_parameters"].get("Initialized Chirality (PMA only)") == target["C"]
            # Additional matching metadata if saved in JSON
        ),
        None,
    )
    if rec:
        # Note: iterations are read from JSON best_energy if history isn't logged,
        # or loaded directly from your saved run histories.
        final_e = rec["results"]["best_energy"] - E_shift
        ax1.axhline(
            final_e, linestyle="-", label=f"{target['label']} ({final_e:.2f}K)"
        )

ax1.axhline(-0.5, color="k", linestyle="--", alpha=0.5, label="Trif Exact (-0.5 K)")
ax1.axhline(0.5, color="k", linestyle="--", alpha=0.5, label="Trif Exact (+0.5 K)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: Low-Field Minimums
ax2 = axes[1]
ax2.set_title("Panel 2: VQE Convergence (Targeting Ground State)")
ax2.set_xlabel("VQE Iteration")
ax2.set_ylabel("Energy (K)")

for target in grid_panel2:
    rec = next(
        (
            r
            for r in records
            if abs(r["external_parameters"]["B_z"] - target["field"]) < 1e-5
        ),
        None,
    )
    if rec:
        final_e = rec["results"]["best_energy"] - E_shift
        ax2.axhline(final_e, label=f"{target['label']}: {final_e:.3f} K")

ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel 3: Energy vs Magnetic Field
ax3 = axes[2]
ax3.set_title("Panel 3: Ground State Energy vs. Magnetic Field")
ax3.set_xlabel("B (Tesla)")
ax3.set_ylabel("Energy (K)")

b_teslas = []
energies = []

for b_val in grid_panel3_fields:
    rec = next(
        (
            r
            for r in records
            if abs(r["external_parameters"]["B_z"] - b_val) < 1e-5
        ),
        None,
    )
    if rec:
        b_teslas.append(b_val * 0.75)  # Conversion B_tesla = (3/4) * B_code
        energies.append(rec["results"]["best_energy"] - E_shift)

ax3.plot(
    b_teslas,
    energies,
    marker="o",
    linestyle="-",
    color="b",
    label="VQE Ground State",
)
ax3.legend()
ax3.grid(True, alpha=0.3)

# --- Save Summary Plot ---
output_dir = os.path.join("output_plots", job_id)
os.makedirs(output_dir, exist_ok=True)
plot_path = os.path.join(output_dir, "trif_summary_plot.png")

plt.tight_layout()
plt.savefig(plot_path)
print(f"Summary plot successfully generated and saved to: {plot_path}")
