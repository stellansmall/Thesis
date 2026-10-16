import numpy as np
from frustration_lib import define_heisenberg_loop

# --- Physical Parameters (Trif et al. Conventions) ---
J_trif = 5.0  # Exchange coupling J in Kelvin (Trif convention)
J_code = J_trif / 2.0  # J_Crippa = J_Trif / 2 = 2.5 K
D_z = 1.0 / np.sqrt(3)  # Calibrated DM strength (~0.57735 K) for Δ_SO = 1.0 K

# --- 1. Build & Diagonalize Hamiltonian at B = 0 ---
hamiltonian_op = define_heisenberg_loop(
    num_spins=3, coupling_strength=J_code, field_strength=0.0, DM=D_z
)
H_matrix = hamiltonian_op.to_matrix()
raw_energies = np.linalg.eigvalsh(H_matrix)

# --- 2. Align Energy Origin (Trif et al. Offset) ---
# S = 1/2 manifold comprises the lowest 4 levels
s12_mean = np.mean(raw_energies[:4])
energies = raw_energies - s12_mean

# --- 3. Output Eigenenergy Analysis ---
print(f"--- Exact Diagonalization Spectrum at B = 0 T ---")
print(f"Lower S=1/2 Doublet (E1, E2) : {energies[0]:.4f} K, {energies[1]:.4f} K")
print(f"Upper S=1/2 Doublet (E3, E4) : {energies[2]:.4f} K, {energies[3]:.4f} K")
print(
    f"Calculated Δ_SO Gap         : {energies[2] - energies[0]:.4f} K (Target: 1.0 K)"
)
print(f"\nS=3/2 Quadruplet (E5..E8)  : {energies[4:]} K")
print(
    f"S=3/2 Exchange Gap          : {energies[4]:.4f} K (Target: 3J/2 = 7.5 K)"
)
