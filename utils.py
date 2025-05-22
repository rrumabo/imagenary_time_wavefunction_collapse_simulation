import numpy as np
from numpy.fft import fft, ifft
from scipy.integrate import trapezoid

def compute_kinetic(psi, k):
    """
    Compute the kinetic term H₀ψ = -½ ∂²ψ/∂x² using FFT.
    """
    return -0.5 * ifft(k**2 * fft(psi))

def compute_entropy(psi, x):
    """
    Compute Shannon entropy: S = -∫ρ lnρ dx, where ρ = |ψ|².
    """
    rho = np.abs(psi)**2
    log_rho = np.log(np.maximum(rho, 1e-12))
    S = -trapezoid(rho * log_rho, x)
    return S, rho

def compute_mu(psi, kinetic, ent_force, x):
    """
    Compute μ(τ) to enforce norm conservation:
    μ = ∫ Re[ψ*(H₀ψ + F_entropy)] dx
    """
    integrand = np.real(np.conjugate(psi) * (kinetic + ent_force))
    return trapezoid(integrand, x)
