from typing import Tuple
import numpy as np
from scipy.stats import norm


def bsm_value_call(s: float, k: float, sigma: float, r: float, q: float, t: float) -> float:
    sigma_sqrt_t = sigma * np.sqrt(t)
    d1 = (np.log(s/k) + (r - q + np.power(sigma,2) / 2) * t) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    return s * np.exp(-q*t) * norm.cdf(d1) - k * np.exp(-r*t) * norm.cdf(d2)

def bsm_value_put(s: float, k: float, sigma: float, r: float, q: float, t: float) -> float:
    sigma_sqrt_t = sigma * np.sqrt(t)
    d1 = (np.log(s/k) + (r - q - np.power(sigma,2) / 2) * t) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    return k * np.exp(-r*t) * norm.cdf(-d2) - s * np.exp(-q*t) * norm.cdf(-d1)

def bsm_iv_call(bsm_value: float, s: float, k: float, r: float, q: float, t: float, initial_sigma_guess: float, min_variation: float=0.00001) -> float:
    logsk = np.log(s/k)
    mu = r - q
    s_discounted = s * np.exp(-q*t)
    k_discounted = k * np.exp(-r*t)
    sqrt_t = np.sqrt(t)
    def bsm_value_vega(sigma: float) -> Tuple[float, float]:
        sigma_sqrt_t = sigma * sqrt_t
        d1 = (logsk + (mu - np.power(sigma, 2) / 2) * t) / sigma_sqrt_t
        d2 = d1 - sigma_sqrt_t
        return s_discounted * norm.cdf(d1) - k_discounted * norm.cdf(d2), s_discounted * sqrt_t * norm.pdf(d1)

    sigma_guess = initial_sigma_guess
    adjustment = 1000000000

    while np.abs(adjustment) > min_variation:
        estimated_value, vega = bsm_value_vega(sigma_guess)
        adjustment = (estimated_value - bsm_value) / vega
        sigma_guess -= adjustment

    return sigma_guess

def bsm_iv_put(bsm_value: float, s: float, k: float, r: float, q: float, t: float, initial_sigma_guess: float, min_variation: float=0.00001) -> float:
    logsk = np.log(s/k)
    mu = r - q
    s_discounted = s * np.exp(-q*t)
    k_discounted = k * np.exp(-r*t)
    sqrt_t = np.sqrt(t)
    def bsm_value_vega(sigma: float) -> Tuple[float,float]:
        sigma_sqrt_t = sigma * sqrt_t
        d1 = (logsk + (mu - np.power(sigma, 2) / 2) * t) / sigma_sqrt_t
        d2 = d1 - sigma_sqrt_t
        return  k_discounted * norm.cdf(-d2) - s_discounted * norm.cdf(-d1), s_discounted * sqrt_t * norm.pdf(d1)

    sigma_guess = initial_sigma_guess
    adjustment = 1000000000
    while np.abs(adjustment) > min_variation:
        estimated_value, vega = bsm_value_vega(sigma_guess)
        adjustment = (estimated_value - bsm_value) / vega
        sigma_guess -= adjustment
    
    return sigma_guess

def get_value_call_montecarlo(s: float, k: float, sigma: float, r: float, q: float, t: float, steps_per_year: int=252, n_sims: int=50_000) -> float:
    steps = steps_per_year * t
    s = s * np.ones(n_sims)
    dt = t / steps
    sqrt_dt = np.sqrt(dt)
    mu = r-q
    dxs = np.random.standard_normal((steps, n_sims))
    for dx in dxs:
        dz = (dx - np.mean(dx)) / np.std(dx)
        s *= np.exp((mu-0.5*sigma**2)*dt+sigma*sqrt_dt*dz)

    flows = (s - k) * (s > k)
    pv = np.mean(flows * np.exp(-r * t))
    return pv