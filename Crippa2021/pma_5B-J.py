#!/usr/bin/env python3
import sys
import json
import os
import matplotlib
matplotlib.use('Agg')
from qiskit_aer import StatevectorSimulator
from ansaetze import *

def main():
    if len(sys.argv) < 2:
        print("Usage: python pma_restart.py <path_to_3.2_json_file> [batch_id]")
        sys.exit(1)
        
    json_path = sys.argv[1]
    batch_id = sys.argv[2] if len(sys.argv) > 2 else "restart_run"
    
    # Extract the previously converged parameters
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            initial_guess = data['results']['best_parameters']
            layers = data['vqe_configuration']['ansatz_depth_layers']
            coupling = data['physical_parameters']['coupling_strength_J']
    except Exception as e:
        print(f"Error loading JSON: {e}")
        sys.exit(1)
        
    field_strength = 5.0 # Target B for this new run
    
    print(f"Restarting from {json_path}")
    print(f"Targeting Spins=4, Layers={layers}, J={coupling}, Bz={field_strength}")
    
    backend = StatevectorSimulator()
    estimator_options = {}
    
    # Execute PMA with the extracted parameters as the initial guess
    result, iterations, plot_path, db_path = PMA(
        backend,
        estimator_options,
        layers,
        coupling,
        field_strength,
        batch_id,
        initial_guess=initial_guess
    )
    
    print(f"Restart run complete. Data saved to {db_path} and {plot_path}")

if __name__ == "__main__":
    main()
