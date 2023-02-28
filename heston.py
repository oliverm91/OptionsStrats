from typing import List
import numpy as np
from scipy.integrate import quad

def characteristic_function(phi: float, params: List[float], spot: float, r: float, q: float, term: float, type_flag: int) -> complex:
    initial_vol, kappa, long_term_vol, vol_vol, rho = params
    rho_sigma = rho * vol_vol
    phi_i = phi * 1j
    rho_sigma_phi_i = rho_sigma * phi_i
    vol_vol_2 = vol_vol ** 2
    if type_flag==1:
        u = 0.5
        b = kappa - rho_sigma
    else: 
        u = -0.5
        b = kappa
    
    kappa_theta = kappa*long_term_vol
    d = np.sqrt((rho_sigma_phi_i-b)**2 - vol_vol_2*(2*u*phi_i-phi**2))
    g = (b-rho_sigma_phi_i+d)/(b-rho_sigma_phi_i-d)
    exp_d_term = np.exp(d*term)
    getd = g*exp_d_term
    C = (r-q)*phi_i*term + (kappa_theta/vol_vol_2) * ((b - rho_sigma_phi_i + d) * term - 2 * np.log((1-getd)/(1-g)))
    D = ((b-rho_sigma_phi_i+d)/vol_vol_2)*(1-exp_d_term)/(1-getd)
    
    return np.exp(C + D*initial_vol + phi_i*np.log(spot))

def characteristic_function2(phi: float, params: List[float], spot: float, r: float, q: float, term: float, type_flag: int) -> complex:
    initial_vol, kappa, long_term_vol, vol_vol, rho = params
    rho_sigma = rho * vol_vol
    phi_i = phi * 1j
    rho_sigma_phi_i = rho_sigma * phi_i
    vol_vol_2 = vol_vol ** 2
    if type_flag==1:
        u = 0.5
        b = kappa - rho_sigma
    else: 
        u = -0.5
        b = kappa
    
    kappa_theta = kappa*long_term_vol
    d = np.sqrt((rho_sigma_phi_i-b)**2 - vol_vol_2*(2*u*phi_i-phi**2))
    g = (b-rho_sigma_phi_i+d)/(b-rho_sigma_phi_i-d)
    exp_d_term = np.exp(d*term)
    getd = g*exp_d_term
    C = (r-q)*phi_i*term + (kappa_theta/vol_vol_2) * ((b - rho_sigma_phi_i + d) * term - 2 * np.log((1-getd)/(1-g)))
    D = ((b-rho_sigma_phi_i+d)/vol_vol_2)*(1-exp_d_term)/(1-getd)
    
    return np.exp(C + D*initial_vol + phi_i*np.log(spot))

def integral_function(u: float, params: List[float], spot: float, r: float, q: float, term: float, K: float, type_flag: int) -> float:
    complex_integrand = np.exp(-1j*u*np.log(K))*characteristic_function(u, params, spot, r, q, term, type_flag) / (1j * u)
    return np.real(complex_integrand)

def get_value_put(params: List[float], spot: float, r: float, q: float, term: float, K: float, type_flag: int) -> float:
    ifun = lambda u: integral_function(u, params, spot, r, q, term, K, type_flag)
    return 0.5 + (1/np.pi) * quad(ifun, 0, 50)[0]

def get_value_call(params: List[float], spot: float, r: float, q: float, term: float, K: float) -> float:
    P1 = get_value_put(params, spot, r, q, term, K, 1)
    P2 = get_value_put(params, spot, r, q, term, K, 2)
    return spot * P1 - K * np.exp(-(r-q)*term) * P2

def get_value_call_montecarlo(params: List[float], spot: float, r: float, q: float, term: float, K: float, steps: int=252, n_sims: int=10_000) -> float:
    initial_vol, kappa, long_term_vol, vol_vol, rho = params
    A = np.linalg.cholesky([[1, rho], [rho, 1]])
    s = spot * np.ones(n_sims)
    v = initial_vol * np.ones(n_sims)
    dt = term / steps
    sqrt_dt = np.sqrt(dt)
    mu = r - q
    dxs = np.random.standard_normal((steps, 2, n_sims))
    dxs = (dxs - dxs.mean(axis=2)[:,:,None])/dxs.std(axis=2)[:,:,None] # Re-Standardize
    for dx in dxs: # dx dim [2, n_sims]
        # Correlate random numbers
        dz = A@dx
        # Recover standardarization of second number after correlation
        dz[1] /= dz[1].std()
        # Move step
        v_plus = v*(v>0)
        s *= np.exp((mu - 0.5*v_plus**2)*dt + np.sqrt(v_plus)*dz[0]*sqrt_dt)
        v += kappa * (long_term_vol - v_plus)*dt + vol_vol*np.sqrt(v_plus)*dz[1]*sqrt_dt

    flows = (s - K) * (s > K)
    pv = np.mean(flows * np.exp(-r * term))
    return pv