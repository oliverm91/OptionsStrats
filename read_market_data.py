from typing import Dict
from vols import VolatilitySurface, VolatilitySmile, VolatilityPoint
import pandas as pd
import math

def row_to_volatility_smile(row: Dict[str, float], spot: float) -> VolatilitySmile:
    r = float(row['r']) / 100
    days = int(row['days'])
    ptfwd = float(row['ptfwd'])
    d90 = float(row['90'])
    d75 = float(row['75'])
    d50 = float(row['50'])
    d25 = float(row['25'])
    d10 = float(row['10'])
    delta_dict = {0.9:d90, 0.75: d75, 0.5: d50, 0.25: d25, 0.1: d10}

    yf = days / 360
    s_t = spot + ptfwd
    wf_r = math.pow(1 + r, yf)
    wf_q = (spot / s_t) * wf_r
    r = math.log(wf_r) / yf
    q = math.log(wf_q) / yf

    vs = VolatilitySmile(yf, [VolatilityPoint(d, v / 100) for d, v in delta_dict.items()], r, q)
    return vs

def get_volatility_surface(spot: float) -> VolatilitySurface:
    filename = 'mkt_data.xlsx'
    df = pd.read_excel(filename)
    vol_smiles = []
    for i in range(len(df)):
        row = {}
        row['days'] = df['Days'].iat[i]
        row['ptfwd'] = df['Pt Fwd'].iat[i]
        row['r'] = df['r'].iat[i]
        row['90'] = df[90].iat[i]
        row['75'] = df[75].iat[i]
        row['50'] = df[50].iat[i]
        row['25'] = df[25].iat[i]
        row['10'] = df[10].iat[i]
        vs = row_to_volatility_smile(row, spot)
        vol_smiles.append(vs)

    vol_surface = VolatilitySurface(vol_smiles)
    return vol_surface