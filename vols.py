from typing import List
import numpy as np
from scipy.stats import norm


class VolatilityPoint:
    def __init__(self, delta_call: float, volatility: float):
        self.delta_call = delta_call
        self.volatility = volatility

class VolatilitySmile:
    def __init__(self, term: float, volatility_points: List[VolatilityPoint], r: float, q: float):
        self.term = term
        self.volatility_points = volatility_points
        self.volatility_points.sort(key=lambda x: x.delta_call)
        self.r = r
        self.q = q
        self.vol_dict = {vs.delta_call: vs.volatility for vs in self.volatility_points}
    
    def interpolate_vol(self, delta: float) -> float:
        return np.interp(delta, [vp.delta_call for vp in self.volatility_points], [vp.volatility for vp in self.volatility_points])

class VolatilitySurface:
    def __init__(self, volatility_smiles: List[VolatilitySmile]):
        self.volatility_smiles = volatility_smiles
        self.volatility_smiles.sort(key=lambda x: x.term)
        self.vol_dict = {vs.term: vs for vs in self.volatility_smiles}
    
    def get_vol_smile(self, yf: float) -> VolatilitySmile:
        yfs = [vs.term for vs in self.volatility_smiles]
        min_yf, max_yf = min(yfs), max(yfs)
        if min_yf <= yf <= max_yf:
            if yf in yfs:
                return [vs for vs in self.volatility_smiles if vs.term == yf][0]
            else:
                x1 = max([t for t in yfs if t <= yf])
                x2 = min([t for t in yfs if t > yf])
                x1_prop = (x2 - yf) / (x2 - x1)
                x2_prop = (yf - x1) / (x2 - x1)
                vs1 = self.vol_dict[x1]
                vs2 = self.vol_dict[x2]
                vs12 = zip(vs1.volatility_points, vs2.volatility_points)
                v_points = [VolatilityPoint(vp1.delta_call, vp1.volatility*x1_prop + vp2.volatility*x2_prop) for vp1, vp2 in vs12]
                vs = VolatilitySmile(yf, v_points, vs1.r * x1_prop + vs2.r * x2_prop, vs1.q * x1_prop + vs2.q * x2_prop)
                return vs

        else:
            raise(f'Term volatility interpolation out of borders. yf to interpolate: {yf}. Range: [{min_yf}, {max_yf}].')
        
    def get_vol(self, yf: float, delta_call: float) -> float:
        vs = self.get_vol_smile(yf)
        sigma = vs.interpolate_vol(delta_call)
        return sigma
    
    def get_strike_from_volatility_point(self, spot: float, yf: float, delta_call: float) -> float:
        vs = self.get_vol_smile(yf)
        sigma = vs.interpolate_vol(delta_call)
        r, q = vs.r,  vs.q
        return spot * np.exp(-(norm.ppf(np.exp(q * yf) * delta_call) * sigma*np.sqrt(yf) - (r - q - np.power(sigma, 2) / 2) * delta_call))