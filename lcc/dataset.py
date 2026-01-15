# lcc/dataset.py
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BINDER_FAMILIES, FAMILY_PROTOTYPES

def _closest_family(slag_frac: float, fly_frac: float) -> str:
    best_key, best_dist = None, 1e9
    for k, (s, f) in FAMILY_PROTOTYPES.items():
        d = (slag_frac - s)**2 + (fly_frac - f)**2
        if d < best_dist:
            best_key, best_dist = k, d
    assert best_key is not None
    return best_key

def build_df() -> pd.DataFrame:
    """
    Build the dataset dataframe exactly like your current script:
    - base rows
    - append T3 rows
    - add EC_exp (padded with NaN)
    - compute slag/fly fractions and nearest family label
    """
    data2 = {
        "Water/Binder": [
            0.57, 0.57, 0.57, 0.58, 0.57, 0.57, 0.57, 0.55, 0.54,
            0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.44, 0.47,
            0.42, 0.41, 0.42, 0.42, 0.42, 0.42, 0.41, 0.40, 0.39,
            0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34,
            0.66, 0.66, 0.64, 0.66, 0.66, 0.66, 0.66, 0.66, 0.66
        ],
        "Free Water": [
            191, 192, 193, 197, 191, 190, 191, 182, 180,
            189, 192, 191, 196, 191, 191, 191, 173, 184,
            188, 189, 190, 195, 188, 195, 189, 184, 183,
            191, 192, 193, 197, 193, 193, 192, 193, 193,
            194, 195, 189, 197, 194, 194, 193, 194, 193
        ],
        "Cement": [
            338, 255, 205, 171, 220, 167, 118, 135, 101,
            386, 294, 234, 199, 254, 195, 136, 159, 119,
            461, 346, 280, 233, 297, 233, 163, 190, 141,
            574, 432, 348, 290, 372, 286, 199, 228, 171,
            295, 222, 177, 149, 191, 147, 103, 117, 88
        ],
        "GGBFS": [
            0, 0, 0, 0, 118, 167, 218, 135, 135,
            0, 0, 0, 0, 137, 195, 253, 159, 159,
            0, 0, 0, 0, 161, 233, 304, 190, 187,
            0, 0, 0, 0, 200, 286, 369, 228, 228,
            0, 0, 0, 0, 103, 147, 190, 117, 118
        ],
        "Fly Ash": [
            0, 84, 137, 171, 0, 0, 0, 68, 101,
            0, 98, 156, 199, 0, 0, 0, 79, 119,
            0, 117, 186, 233, 0, 0, 0, 95, 141,
            0, 144, 232, 290, 0, 0, 0, 114, 171,
            0, 74, 118, 149, 0, 0, 0, 59, 88
        ],
        "Strength3d": [
            31.5, 24.3, 20.5, 16.2, 19.0, 12.5, 11.2, 13.4, 11.9,
            36.2, 30.6, 27.2, 22.3, 24.0, 18.7, 15.9, 20.7, 15.2,
            48.4, 40.4, 35.6, 31.3, 28.9, 23.0, 17.0, 24.9, 20.7,
            55.3, 45.0, 45.3, 44.5, 37.8, 32.8, 23.0, 28.6, 25.3,
            21.5, 16.2, 13.6, 11.6, 12.5, 9.2, 7.9, 9.3, 7.8
        ],
        "Strength7d": [
            38.4, 30.0, 28.9, 21.6, 26.4, 22.6, 21.9, 23.8, 21.8,
            44.3, 39.5, 34.3, 29.4, 35.0, 29.1, 22.3, 29.5, 25.4,
            57.6, 51.7, 44.3, 41.0, 39.9, 36.1, 30.7, 37.5, 31.1,
            68.9, 61.7, 59.1, 52.0, 54.7, 47.0, 36.7, 41.8, 39.3,
            26.4, 21.1, 18.6, 15.9, 20.5, 17.2, 14.9, 15.0, 14.0
        ],
        "Strength28d": [
            49.6, 45.6, 36.7, 34.0, 43.7, 42.4, 37.0, 44.3, 37.6,
            54.1, 54.0, 48.8, 44.0, 55.3, 47.4, 41.4, 51.9, 47.1,
            70.7, 66.5, 60.0, 58.5, 62.7, 58.8, 47.4, 58.6, 55.6,
            83.3, 78.6, 74.5, 72.9, 72.7, 67.3, 56.4, 61.9, 61.8,
            34.6, 30.6, 28.9, 24.3, 36.8, 35.3, 29.8, 31.3, 28.1
        ],
    }

    # ---- Add T3 mixes ----
    T3_MIX_ROWS = [
        (212.3707, 231.6680, 173.7510, 173.7510, 28.0, 48.0, 70.0),
        (204.5452, 174.3984, 130.7988, 130.7988, 18.9, 31.70, 51.50),
        (195.6861, 130.4574, 97.84306, 97.84306, 9.4, 16.6, 32.5),
    ]
    for w, c, s, fa, f3, f7, f28 in T3_MIX_ROWS:
        binder_total = c + s + fa
        wb = w / binder_total
        data2["Water/Binder"].append(float(wb))
        data2["Free Water"].append(float(w))
        data2["Cement"].append(float(c))
        data2["GGBFS"].append(float(s))
        data2["Fly Ash"].append(float(fa))
        data2["Strength3d"].append(float(f3))
        data2["Strength7d"].append(float(f7))
        data2["Strength28d"].append(float(f28))

    ec_list = [
        358, 280, 235, 206, 275, 233, 199, 197, 168,
        403, 318, 268, 233, 310, 269, 222, 226, 190,
        470, 366, 307, 267, 366, 306, 258, 261, 218,
        581, 453, 375, 327, 431, 373, 306, 306, 255,
        299, 227, 188, 163, 223, 192, 159, 155, 129
    ]

    df = pd.DataFrame(data2)
    df["EC_exp"] = ec_list + [np.nan] * (len(df) - len(ec_list))

    binder_sum = (df["Cement"] + df["GGBFS"] + df["Fly Ash"]).replace(0, 1e-9)
    df["_slag_frac"] = df["GGBFS"] / binder_sum
    df["_fly_frac"]  = df["Fly Ash"] / binder_sum
    df["_family"] = [_closest_family(s, f) for s, f in zip(df["_slag_frac"], df["_fly_frac"])]

    return df
