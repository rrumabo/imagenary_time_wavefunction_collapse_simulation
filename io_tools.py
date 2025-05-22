import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def save_results(results, config):
    """
    Save all simulation arrays and a CSV for dissipation rate.
    """
    output_dir = config.get('output_dir', 'simulation_output')
    os.makedirs(output_dir, exist_ok=True)

    for key in results:
        np.save(os.path.join(output_dir, f"{key}.npy"), results[key])

    energy_dissipation = np.gradient(results['energy_history'], config['d_tau'])
    df = pd.DataFrame({
        "Tau": results['tau'],
        "Energy": results['energy_history'],
        "Energy Dissipation Rate": energy_dissipation
    })
    df.to_csv(os.path.join(output_dir, "energy_dissipation_rate.csv"), index=False)

def plot_diagnostics(results, x, config):
    """
    Generate and save diagnostic plots.
    """
    tau = results['tau']
    output_dir = config.get('output_dir', 'simulation_output')

    # Entropy & Energy
    plt.figure()
    plt.plot(tau, results['entropy_history'], label="Entropy")
    plt.plot(tau, results['energy_history'], label="Energy")
    plt.legend(); plt.grid()
    plt.title("Entropy & Energy")
    plt.savefig(os.path.join(output_dir, "entropy_energy.png"))
    plt.close()

    # dE/dτ and T(τ)
    dE = np.gradient(results['energy_history'], config['d_tau'])
    plt.figure()
    plt.plot(tau, dE, label="dE/dτ")
    plt.plot(tau, results['T_computed'], label="T(τ)")
    plt.legend(); plt.grid()
    plt.title("Dissipation & Effective Temperature")
    plt.savefig(os.path.join(output_dir, "temperature_dissipation.png"))
    plt.close()

    # Heatmap |ψ(x,τ)|²
    snapshots = np.abs(results['snapshots'])**2
    snapshot_interval = config['snapshot_interval']
    snapshot_times = np.arange(0, len(snapshots)) * snapshot_interval * config['d_tau']
    plt.figure()
    plt.imshow(snapshots, extent=[x[0], x[-1], snapshot_times[-1], snapshot_times[0]],
               aspect='auto', cmap='viridis')
    plt.colorbar(label='|ψ(x,τ)|²')
    plt.title("Wavefunction Collapse")
    plt.savefig(os.path.join(output_dir, "collapse_heatmap.png"))
    plt.close()

    # ⟨x⟩ and ⟨x²⟩
    plt.figure()
    plt.plot(tau, results['x_mean_history'], label="⟨x⟩")
    plt.plot(tau, results['x2_mean_history'], label="⟨x²⟩")
    plt.legend(); plt.grid()
    plt.title("Position Expectation Values")
    plt.savefig(os.path.join(output_dir, "expectation_values.png"))
    plt.close()
