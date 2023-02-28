import heston
import bsm

#market
S, r, q, bsm_vol = 800, 0.09, 0.05, 0.24

#option config
K, T = 850, 2

# Params
v0, kappa, theta, sigma, rho = bsm_vol*bsm_vol, 2, bsm_vol*bsm_vol, 0.4, 0.3
params = [v0, kappa, theta, sigma, rho]

# Calculations
montecarlo_value = heston.get_value_call_montecarlo(params, S, r, q, T, K)
mc_iv = bsm.bsm_iv_call(montecarlo_value, S, K, r, q, T, 0.1)
closed_form_value = heston.get_value_call(params, S, r, q, T, K)
cf_iv = bsm.bsm_iv_call(closed_form_value, S, K, r, q, T, 0.1)

print(f'BSM Vol: {round(bsm_vol*100,0)}')
print(f'Montecarlo Heston IV: {round(mc_iv*100,0)}.')
print(f'Closed form Heston IV: {round(cf_iv*100,0)}.')