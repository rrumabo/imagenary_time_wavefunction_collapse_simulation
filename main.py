import numpy as np
import json
import argparse
from numpy.fft import fftfreq
from evolution import evolve_wavefunction
from io_tools import save_results, plot_diagnostics

def run_simulation(config):
    L = config['L']
    N = config['N']
    dx = L / N
    x = np.linspace(-L/2, L/2, N)
    k = fftfreq(N, d=dx) * 2 * np.pi

    x0 = config.get('x0', -5.0)
    sigma0 = config.get('sigma0', 1.0)
    p0 = config.get('p0', 0.0)
    normalization = (1 / (sigma0 * np.sqrt(np.pi)))**0.5
    psi = normalization * np.exp(-(x - x0)**2 / (2 * sigma0**2)) * np.exp(1j * p0 * x)

    psi_final, results = evolve_wavefunction(psi, config, x, k)
    return x, psi_final, results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="Path to JSON config", default=None)
    args = parser.parse_args()

    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        x, psi_final, results = run_simulation(config)
        save_results(results, config)
        plot_diagnostics(results, x, config)
    else:
        T_values = [0.5, 1.0, 1.5]
        alpha_values = [1.0, 2.0, 3.0]
        for T in T_values:
            for alpha in alpha_values:
                config = {
                    'T_param': T, 'alpha': alpha, 'd_tau': 0.001,
                    'num_steps': 2000, 'L': 20.0, 'N': 1024,
                    'imaginary_time': True, 'snapshot_interval': 200,
                    'output_dir': f"simulation_T{T}_alpha{alpha}",
                    'x0': -5.0, 'sigma0': 1.0, 'p0': 0.0, 'm': 1.0
                }
                x, psi_final, results = run_simulation(config)
                save_results(results, config)
                plot_diagnostics(results, x, config)

if __name__ == "__main__":
    main()
