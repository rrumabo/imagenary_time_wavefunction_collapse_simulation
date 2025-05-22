import numpy as np
from numpy.fft import fft
from scipy.ndimage import gaussian_filter1d
from utils import compute_kinetic, compute_entropy, compute_mu

def evolve_wavefunction(psi, config, x, k):
    """
    Evolve the wavefunction ψ using entropy-driven imaginary time dynamics.
    Tracks: entropy, energy, norm, ⟨x⟩, ⟨x²⟩, T(τ), and snapshots.
    """
    num_steps = config['num_steps']
    d_tau = config['d_tau']
    T_param = config['T_param']
    alpha = config['alpha']
    snapshot_interval = config.get('snapshot_interval', 200)
    imaginary_time = config.get('imaginary_time', True)

    norm_history, entropy_history, energy_history = [], [], []
    x_mean_history, x2_mean_history, snapshots = [], [], []

    for n in range(num_steps):
        S, rho = compute_entropy(psi, x)
        kinetic = compute_kinetic(psi, k)
        log_rho = np.log(np.maximum(rho, 1e-12))
        ent_force = alpha * T_param * (log_rho + 1) * psi
        mu = compute_mu(psi, kinetic, ent_force, x)

        if imaginary_time:
            psi = psi - d_tau * (kinetic + ent_force - mu * psi)
        else:
            psi = psi - 1j * d_tau * (kinetic + ent_force - mu * psi)

        psi /= np.sqrt(np.trapezoid(np.abs(psi)**2, x))
        norm_history.append(np.trapezoid(np.abs(psi)**2, x))
        S, rho = compute_entropy(psi, x)
        entropy_history.append(S)

        psi_k = fft(psi)
        kinetic_energy = np.trapezoid(np.abs(psi_k)**2 * (k**2) / (2 * config.get('m', 1.0)), k)
        V_eff = alpha * T_param * (log_rho + 1)
        potential_energy = np.trapezoid(rho * V_eff, x)
        total_energy = kinetic_energy + potential_energy
        energy_history.append(total_energy)

        x_mean = np.trapezoid(x * rho, x)
        x2_mean = np.trapezoid(x**2 * rho, x***_
