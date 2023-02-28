from typing import List
import numpy as np
np.random.seed(1)
from read_market_data import get_volatility_surface
import bsm
import heston
from vols import VolatilitySurface
import warnings
warnings.filterwarnings("ignore")

spot = 800
T = 0.75

r, q = 9.65 / 100, 4.65 / 100#6.26 / 100
vol_surf = get_volatility_surface(spot)
delta_call = 0.5
strike = vol_surf.get_strike_from_volatility_point(spot, T, delta_call)
bsm_volatility = vol_surf.get_vol(T, delta_call)

# Heston Parameters
initial_vol = 0.06
long_term_vol = 0.035
kappa = 1
vol_vol = 0.8
rho = 0.3
initial_parameters = [initial_vol, kappa, long_term_vol, vol_vol, rho]

hv = heston.get_value_call(initial_parameters, spot, r, q, T, strike)

import scipy.optimize as optimize
def calibrate_single_point(strike, T, bsm_vol, spot, r, q, initial_params):
    def error(params):
        hv = heston.get_value_call(params, spot, r, q, T, strike)
        h_iv = bsm.bsm_iv_call(hv, spot, strike, r, q, T, 0.1)
        return (100*(bsm_vol - h_iv)) ** 2
    
    bounds = optimize.Bounds(lb=np.array([0.0004, 0, 0.0004, 0.02, -1]), ub=np.array([np.inf, np.inf, np.inf, np.inf, 1]))
    result = optimize.minimize(error, initial_params, bounds=bounds, method='SLSQP')
    if result.success:
        return result.x
    else:
        print(result.x)
        raise ValueError(result.message)

def calibrate_full_surface(spot: float, vol_surface: VolatilitySurface, initial_params: List[float]):
    def error(params: List[float]):
        error = 0
        volatility_smiles = vol_surface.volatility_smiles
        for smile in volatility_smiles:
            t, r, q = smile.term, smile.r, smile.q
            for vol_point in smile.volatility_points:
                vol = vol_point.volatility
                delta_call = vol_point.delta_call
                k = vol_surface.get_strike_from_volatility_point(spot, t, delta_call)
                heston_value = heston.get_value_call(params, spot, r, q, t, k)
                heston_iv = bsm.bsm_iv_call(heston_value, spot, k, r, q, t, 0.1)
                error += (100*(vol - heston_iv))**2

        return error
    
    bounds = optimize.Bounds(lb=np.array([0.0004, 0, 0.0004, 0.02, -1]), ub=np.array([2, 10, 2, 10, 1]))
    result = optimize.minimize(error, initial_params, bounds=bounds, method='SLSQP')
    if result.success:
        return result.x
    else:
        print(result.x)
        raise ValueError(result.message)

print('\nCalibrating single point...')
calibrated_params = calibrate_single_point(strike, T, bsm_volatility, spot, r, q, initial_parameters)

bsm_value = bsm.bsm_value_call(spot, strike, bsm_volatility, r, q, T)
bsm_value_mc = bsm.get_value_call_montecarlo(bsm_volatility, spot, r, q, T, strike)
heston_value_closed = heston.get_value_call(initial_parameters, spot, r, q, T, strike)
heston_closed_IV = bsm.bsm_iv_call(heston_value_closed, spot, strike, r, q, T, 0.1)
heston_value_mc = heston.get_value_call_montecarlo(initial_parameters, spot, r, q, T, strike)

print(f'\n\nMontecarlo BSM value: {round(bsm_value_mc, 4)}')
print(f'Closed form BSM value: {round(bsm_value, 4)}')
print(f'BSM vol used: {round(100*bsm_volatility,2)}')
print('--------------------------')
print('Uncalibrated Heston values')
print('--------------------------')
print(f'Montecarlo Heston value: {round(heston_value_mc, 4)}')
print(f'Closed form Heston value: {round(heston_value_closed, 4)}')
print(f'Closed form BSM IV: {round(100*heston_closed_IV, 2)}')

calib_heston_value_closed = heston.get_value_call(calibrated_params, spot, r, q, T, strike)
calib_heston_closed_IV = bsm.bsm_iv_call(calib_heston_value_closed, spot, strike, r, q, T, 0.1)
calib_heston_value_mc = heston.get_value_call_montecarlo(calibrated_params, spot, r, q, T, strike)
print('--------------------------')
print('Single point calibrated Heston values')
print('--------------------------')
print(f'Montecarlo Heston value: {round(calib_heston_value_mc, 4)}')
print(f'Closed form Heston value: {round(calib_heston_value_closed, 4)}')
print(f'Closed form BSM IV: {round(100*calib_heston_closed_IV, 2)}')
print('--------------------------')
print(f'Single point calibrated Heston parameters (v0, kappa, theta, volvol, rho): {calibrated_params}')

print('Full surface calibration...')
import time
t0 = time.time()
full_calibrated_params = calibrate_full_surface(spot, vol_surf, initial_parameters)
print(f'Calibration time: {round(time.time()-t0, 0)} seconds.')
f_calib_heston_value_closed = heston.get_value_call(full_calibrated_params, spot, r, q, T, strike)
f_calib_heston_closed_IV = bsm.bsm_iv_call(f_calib_heston_value_closed, spot, strike, r, q, T, 0.1)
f_calib_heston_value_mc = heston.get_value_call_montecarlo(full_calibrated_params, spot, r, q, T, strike)
print('--------------------------')
print('Full surface calibrated Heston values')
print('--------------------------')
print(f'Montecarlo Heston value: {round(f_calib_heston_value_mc, 4)}')
print(f'Closed form Heston value: {round(f_calib_heston_value_closed, 4)}')
print(f'Closed form BSM IV: {round(100*f_calib_heston_closed_IV, 2)}')
print('--------------------------')
print(f'Full surface calibrated Heston parameters (v0, kappa, theta, volvol, rho): {full_calibrated_params}')